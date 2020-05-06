#include "setup_store.h"
#include <iostream>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
#include <igl/readOBJ.h>
#include <igl/jet.h>
#include <igl/png/readPNG.h>
#include <igl/png/writePNG.h>
#include <igl/volume.h>
#include <igl/slice.h>
#include <igl/boundary_facets.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/destroy_shader_program.h>
#include <igl/opengl/create_shader_program.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/remove_unreferenced.h>
#include <igl/list_to_matrix.h>
#include <imgui/imgui.h>
#include <igl/null.h>
#include <json.hpp>
#include <Eigen/SparseCholesky>
#include <LBFGS.h>


#include <linear_tetmesh_B.h>
#include <fixed_point_constraint_matrix.h>
#include <emu_to_lame.h>
#include <linear_tetmesh_mass_matrix.h>

#include <sstream>
#include <iomanip>
// #include <omp.h>

#include "store.h"
#include "read_config_files.h"
#include "muscle_energy_gradient.h"
#include "stablenh_energy_gradient.h"
#include "construct_kkt_system.h"
#include "acap_solve.h"
#include "newton_solve.h"



using namespace Eigen;
using Store = exact::Store;
using namespace LBFGSpp;

void exact::setupStore(Store& store){
	std::cout<<"---Read Mesh"<<std::endl;
		exact::read_config_files(store);

	std::cout<<"---Record Mesh Setup Info"<<std::endl;
		std::cout<<"EIGEN:"<<EIGEN_MAJOR_VERSION<<EIGEN_MINOR_VERSION<<std::endl;
		std::cout<<"V size: "<<store.V.rows()<<std::endl;
		std::cout<<"T size: "<<store.T.rows()<<std::endl;
		std::cout<<"F size: "<<store.F.rows()<<std::endl;
		store.joutput["info"]["Vsize"] = store.V.rows();
		store.joutput["info"]["Tsize"] = store.T.rows();
		store.joutput["info"]["Fsize"] = store.F.rows();
		store.joutput["info"]["NumModes"] = NUM_MODES;
		store.joutput["info"]["NumThreads"] = Eigen::nbThreads();

	std::cout<<"---Set variables"<<std::endl;
		Eigen::VectorXd q0 = VectorXd::Zero(3*store.V.rows());
		for(int i=0; i<store.V.rows(); i++){
			q0[3*i+0] = store.V(i,0); 
			q0[3*i+1] = store.V(i,1); 
			q0[3*i+2] = store.V(i,2);   
	    }
		sim::linear_tetmesh_B(store.B, store.V, store.T);
		store.Fvec = store.B*q0;
		store.Fvec0 = store.B*q0;
		store.x0 = q0;

	std::cout<<"---Set boundary conditions"<<std::endl;
		double minY = store.V.col(1).maxCoeff();
		std::vector<int> idxs;
		for(int i=0; i<store.V.rows(); i++){
			if(fabs(store.V.row(i)[1] - minY)<1e-2)
				idxs.push_back(i);
		}
		Eigen::Map<Eigen::VectorXi> indx(idxs.data(), idxs.size());
		sim::fixed_point_constraint_matrix(store.P, store.V, indx);

		store.b = q0 - store.P.transpose()*store.P*q0;

	std::cout<<"---Physics parameters"<<std::endl;
		igl::volume(store.V, store.T, store.rest_tet_vols);
		store.muscle_mag = VectorXd::Zero(store.T.rows());
		//YM, poissons
		store.eY = 1e10*VectorXd::Ones(store.T.rows());
		store.eP = 0.49*VectorXd::Ones(store.T.rows());
		store.muscle_mag = VectorXd::Zero(store.T.rows());
		VectorXd densities = 150*VectorXd::Ones(store.T.rows()); //kg per m^3

		for(int m=0; m<store.muscle_tets.size(); m++){
			for(int t=0; t<store.muscle_tets[m].size(); t++){
				//no tendons for now, add later
				store.eY[store.muscle_tets[m][t]] = 60000;
				densities[store.muscle_tets[m][t]] = 1000;//kg per m^3
			}
		}
		store.eY = 60000*VectorXd::Ones(store.T.rows());

		std::string inputfile = store.jinput["data"];
		sim::linear_tetmesh_mass_matrix(store.M, store.V, store.T, densities, store.rest_tet_vols);


	std::cout<<"--Hessians, gradients"<<std::endl;
		store.H_n.resize(store.Fvec.size(), store.Fvec.size());
		store.H_m.resize(store.Fvec.size(), store.Fvec.size());
		store.grad_n = VectorXd::Zero(store.Fvec.size());
		store.grad_m = VectorXd::Zero(store.Fvec.size());

		exact::muscle::hessian(store, store.Fvec, store.H_m);
		exact::stablenh::hessian(store, store.Fvec, store.H_n);
		exact::muscle::gradient(store, store.Fvec, store.grad_m);
		exact::stablenh::gradient(store, store.Fvec, store.grad_n);



	std::cout<<"--ACAP solve constraint"<<std::endl;
		store.H_a = store.P*(store.B.transpose()*store.B)*store.P.transpose();
		//x = P'*((P*B'*B*P')\(P*B'*F - P*B'*B*b)) + b;
		Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> ACAP;
		ACAP.compute(store.H_a);
		store.x = store.P.transpose()*ACAP.solve(store.P*store.B.transpose()*(store.Fvec - store.B*store.b)) + store.b;


	std::cout<<"--Modes"<<std::endl;
		SparseMatrix<double, Eigen::RowMajor> G(store.P.rows(), store.P.rows());
		G.setIdentity();

	std::cout<<"--Other constants"<<std::endl;
		MatrixXd Ai = MatrixXd(G.transpose()*store.H_a*G).inverse();
		MatrixXd Vtilde = store.B*store.P.transpose()*G;
		MatrixXd J = MatrixXd::Identity(store.Fvec.size(), store.Fvec.size()) - Vtilde*Ai*Vtilde.transpose();
		VectorXd d = J*(store.B*(store.b + store.P.transpose()*store.P*q0));


	std::cout<<"--Woodbury solve setup"<<std::endl;
		SparseMatrix<double, Eigen::RowMajor> H = store.H_m + store.H_n;


	std::cout<<"--Write out"<<std::endl;
		//H_m
		// igl::writeDMAT(inputfile+"/H_m.dmat", MatrixXd(store.H_m));
		// igl::writeDMAT(inputfile+"/grad_m.dmat", MatrixXd(store.grad_m));
		// igl::writeDMAT(inputfile+"/H_n.dmat", MatrixXd(store.H_n));
		// igl::writeDMAT(inputfile+"/grad_n.dmat", MatrixXd(store.grad_n));
		store.printState(0, "woodbury", store.x);
		for(int it = 1; it<100; it++){
			exact::newton_solve(store, store.Fvec, d, Ai, Vtilde, J, (30000/100)*it);
			VectorXd c = store.b + store.P.transpose()*store.P*q0;
			exact::acap_solve(store.x, ACAP, store.P, store.B, store.Fvec, c);
			store.printState(it, "woodbury", store.x);
		}


	std::cout<<"---Viewer parameters"<<std::endl;
		{
			VectorXd vertex_wise_YM =Eigen::VectorXd::Zero(store.V.rows());
			store.elogVY = Eigen::VectorXd::Zero(store.V.rows());
			// volume associated with each vertex
			Eigen::VectorXd Vvol = Eigen::VectorXd::Zero(store.V.rows());
			Eigen::VectorXd Tvol;
			igl::volume(store.V,store.T,Tvol);
			// loop over tets
			for(int i = 0;i<store.T.rows();i++)
			{
				const double vol4 = Tvol(i)/4.0;
				for(int j = 0;j<4;j++)
				{
					Vvol(store.T(i,j)) += vol4;
					store.elogVY(store.T(i,j)) += vol4*log10(store.eY(i));
					vertex_wise_YM(store.T(i,j)) = store.eY(i);
				}
			}
			// loop over vertices to divide to take average
			for(int i = 0;i<store.V.rows();i++)
			{
				store.elogVY(i) /= Vvol(i);
			}

			VectorXd surface_youngs = VectorXd::Zero(store.F.rows());

			for(int ff =0 ; ff<store.F.rows(); ff++){
				Vector3i face = store.F.row(ff);
				Vector3d face_material_YM(vertex_wise_YM(face(0)), vertex_wise_YM(face(1)), vertex_wise_YM(face(2)));
				double minYM = face_material_YM.minCoeff();
				surface_youngs(ff) = minYM;

			}

			igl::writeDMAT(inputfile +"/surface_mesh_youngs.dmat", surface_youngs, true);
		}



}
