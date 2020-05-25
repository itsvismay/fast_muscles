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
#include <linear_tetmesh_mass_matrix.h>

#include <sstream>
#include <iomanip>
#include <Eigen/Sparse>
#ifdef __linux__
#include <Eigen/PardisoSupport>
#endif
#include <unsupported/Eigen/SparseExtra>


#include "store.h"
#include "read_config_files.h"
#include "muscle_energy_gradient.h"
#include "stablenh_energy_gradient.h"
#include "construct_kkt_system.h"
#include "acap_solve.h"
#include "newton_solve.h"
#include "setup_hessian_modes.h"
#include "project_bone_F.h"
#include "bone_vertices_projection.h"
#include "passlambda.h"



using namespace Eigen;
using Store = exact::Store;
using namespace LBFGSpp;
using namespace std;

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
		

	std::cout<<"---Set boundary conditions"<<std::endl;
		cout<<"If it fails here, make sure indexing is within bounds"<<endl;
	    std::set<int> fix_verts_set;
	    for(int ii=0; ii<store.fix_bones.size(); ii++){
	        int bone_ind = store.bone_name_index_map[store.fix_bones[ii]];
	        fix_verts_set.insert(store.T.row(store.bone_tets[bone_ind][0])[0]);
	        fix_verts_set.insert(store.T.row(store.bone_tets[bone_ind][0])[1]);
	        fix_verts_set.insert(store.T.row(store.bone_tets[bone_ind][0])[2]);
	        fix_verts_set.insert(store.T.row(store.bone_tets[bone_ind][0])[3]);
	    }
		std::vector<int> idxs;
		idxs.assign(fix_verts_set.begin(), fix_verts_set.end());
		std::sort (idxs.begin(), idxs.end());
		Eigen::Map<Eigen::VectorXi> indx(idxs.data(), idxs.size());
		// sim::fixed_point_constraint_matrix(store.P, store.V, indx);

	cout<<"---Set Vertex Constraint Matrix"<<endl;
		SparseMatrix<double, Eigen::RowMajor> Y;
		exact::bone_vertices_projection(store, Y);

	std::cout<<"---Project Bone F, remove fixed bones"<<std::endl;
		//bone dF map
		//start off all as -1
		VectorXi bone_or_muscle = -1*Eigen::VectorXi::Ones(store.T.rows());
		// // // assign bone tets to 1 dF for each bone, starting at 0...bone_tets.size()
		for(int i=0; i<store.bone_tets.size(); i++){
	    	for(int j=0; j<store.bone_tets[i].size(); j++){
	    		bone_or_muscle[store.bone_tets[i][j]] = i;
	    	}
	    }

	    //assign muscle tets, dF per element, starting at bone_tets.size()
	    int muscle_ind = store.bone_tets.size();
	    for(int i=0; i<store.T.rows(); i++){
	    	if(bone_or_muscle[i]<-1e-8){
	    		bone_or_muscle[i] = muscle_ind;
	    		muscle_ind +=1;
	    	}
	    }
		
	   
		SparseMatrix<double, Eigen::RowMajor> ProjectF, RemFixedBones;
		//exact::project_bone_F(store, bone_or_muscle, ProjectF, RemFixedBones);
		ProjectF.resize(9*store.T.rows(), 9*store.T.rows());
		ProjectF.setIdentity();
		//ProjectF -> 9T x 9|muscles elements| + 9|bones|

	std::cout<<"---Set variables"<<std::endl;
		Eigen::VectorXd q0 = VectorXd::Zero(3*store.V.rows());
		for(int i=0; i<store.V.rows(); i++){
			q0[3*i+0] = store.V(i,0); 
			q0[3*i+1] = store.V(i,1); 
			q0[3*i+2] = store.V(i,2);   
	    }

	    SparseMatrix<double, Eigen::RowMajor> B;
		sim::linear_tetmesh_B(B, store.V, store.T);
		store.B = ProjectF.transpose()*B;
		VectorXd Fvec = VectorXd::Zero(ProjectF.cols());
		for(int t=0; t<Fvec.size()/9; t++){
			Fvec[9*t + 0] = 1;
			Fvec[9*t + 4] = 1;
			Fvec[9*t + 8] = 1;
		}
		VectorXd Fvec0 = Fvec;
		store.x0 = q0;
		store.b = q0 - Y*Y.transpose()*q0;

	std::cout<<"---Physics parameters"<<std::endl;
		igl::volume(store.V, store.T, store.rest_tet_vols);
		for(int i=0; i<store.bone_tets.size(); i++){
			double bone_vol = 0;
	    	for(int j=0; j<store.bone_tets[i].size(); j++){
	    		bone_vol += store.rest_tet_vols[store.bone_tets[i][j]];
	    	}
	    	store.bone_vols.push_back(bone_vol);
	    }
		store.muscle_mag = VectorXd::Zero(store.T.rows());
		//YM, poissons
		store.eY = 1e10*VectorXd::Ones(store.T.rows());
		store.eP = 0.49*VectorXd::Ones(store.T.rows());
		store.muscle_mag = VectorXd::Zero(store.T.rows());
		VectorXd densities = 150*VectorXd::Ones(store.T.rows()); //kg per m^3

		for(int m=0; m<store.muscle_tets.size(); m++){
			for(int t=0; t<store.muscle_tets[m].size(); t++){
				//no tendons for now, add later
				if(store.relativeStiffness[store.muscle_tets[m][t]]>1){
					store.eY[store.muscle_tets[m][t]] = 4.5e8;
				}else{
					store.eY[store.muscle_tets[m][t]] = 60000;
				}
				densities[store.muscle_tets[m][t]] = 1000;//kg per m^3
			}
		}

		std::string inputfile = store.jinput["data"];
		sim::linear_tetmesh_mass_matrix(store.M, store.V, store.T, densities, store.rest_tet_vols);


	std::cout<<"--Hessians, gradients"<<std::endl;
		store.H_n.resize(Fvec.size(), Fvec.size());
		store.H_m.resize(Fvec.size(), Fvec.size());
		store.grad_n = VectorXd::Zero(Fvec.size());
		store.grad_m = VectorXd::Zero(Fvec.size());

		exact::stablenh::gradient(store.grad_n, Fvec, store.T, store.eY, store.eP, store.rest_tet_vols);
		exact::stablenh::hessian(store.H_n, Fvec, store.T, store.eY, store.eP, store.rest_tet_vols);
		exact::muscle::gradient(store.grad_m, Fvec, store.T, store.rest_tet_vols, store.Uvec);
		exact::muscle::hessian(store.H_m, Fvec, store.T, store.rest_tet_vols, store.Uvec);



	std::cout<<"--ACAP solve constraint"<<std::endl;
		store.H_a = Y.transpose()*(B.transpose()*B)*Y;
		SparseMatrix<double> H_a_colmajor = store.H_a;
		//x = P'*((P*B'*B*P')\(P*B'*F - P*B'*B*b)) + b;
		#ifdef __linux__
		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> ACAP;
		ACAP.pardisoParameterArray()[2] = Eigen::nbThreads();
		#else
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ACAP;	
		#endif
		ACAP.compute(H_a_colmajor);

		store.ACAP.compute(H_a_colmajor);
		VectorXd x = Y*ACAP.solve(Y.transpose()*B.transpose()*(ProjectF*Fvec - B*store.b)) + store.b;


	std::cout<<"--Modes"<<std::endl;
		MatrixXd temp1;
		VectorXd eigenvalues;
		int num_modes = 150;
		std::string outputfile = store.jinput["output"];
		igl::readDMAT(outputfile+"/"+std::to_string(num_modes)+"modes.dmat", temp1);
		if(temp1.rows() == 0){			
			SparseMatrix<double> FindMyModes = Y.transpose()*B.transpose()*(store.H_n + 100000*store.H_m)*B*Y;
			SparseMatrix<double> FindMyModesMass = Y.transpose()*store.M*Y;
			Eigen::saveMarket(FindMyModes, outputfile+"/FindMyModes.txt");
			Eigen::saveMarket(FindMyModesMass, outputfile+"/FindMyModesMass.txt");
			
			exact::setup_hessian_modes(FindMyModes, temp1, eigenvalues, num_modes);
			igl::writeDMAT(outputfile+"/"+std::to_string(num_modes)+"modes.dmat", temp1);
			igl::writeDMAT(outputfile+"/"+std::to_string(num_modes)+"eigs.dmat", eigenvalues);
			// exit(0);
		}else{
			//read eigenvalues (for the woodbury solve)
			igl::readDMAT(outputfile+"/"+std::to_string(num_modes)+"eigs.dmat", eigenvalues);
		}
		
		MatrixXd G = temp1;

	


	std::cout<<"--Woodbury solve setup"<<std::endl;

		SparseMatrix<double, Eigen::RowMajor> H = store.H_m + store.H_n;
		MatrixXd Ai = MatrixXd(G.transpose()*store.H_a*G).inverse();
		MatrixXd Vtilde = B*Y*G;
		VectorXd Bc = B*(store.b + Y*Y.transpose()*q0);
		SparseMatrix<double> wId9T(9*store.T.rows(), 9*store.T.rows());
		wId9T.setIdentity();

		MatrixXd wVAi = Vtilde*Ai;
		MatrixXd wHiV(H.rows(), Vtilde.cols());
		MatrixXd wHiVAi(H.rows(), Ai.cols());
		MatrixXd wC(Ai.rows(), Ai.cols());
		MatrixXd wPhi(wHiV.rows(), wHiV.cols() + Vtilde.cols());
		MatrixXd wHPhi(H.rows(), wPhi.cols());
		MatrixXd wL = MatrixXd::Zero(2*Ai.rows(), 2*Ai.cols());
		MatrixXd wIdL = MatrixXd::Identity(Ai.rows(), Ai.cols());
		MatrixXd wQ(wL.rows(), wL.cols());

		wPhi.block(0, Vtilde.cols(), Vtilde.rows(), Vtilde.cols()) = Vtilde;
		wL.block(0,0, Ai.rows(), Ai.rows()) = MatrixXd::Zero(Ai.rows(), Ai.cols()); //TL
		wL.block(0, Ai.cols(), Ai.rows(), Ai.cols()) = -Ai; //BL
		wL.block(Ai.rows(), 0, Ai.rows(), Ai.cols()) = -Ai; //TR

		VectorXd d1 = Vtilde.transpose()*Bc;
		VectorXd d2 = Ai*d1;
		VectorXd d3 = Vtilde*d2;
		VectorXd d = Bc - d3;


	// std::cout<<"--Write out"<<std::endl;
		//H_m
		// igl::writeDMAT(outputfile+"/PF.dmat", MatrixXd(ProjectF));
		// igl::writeDMAT(outputfile+"/P.dmat", MatrixXd(store.P));
		// igl::writeDMAT(outputfile+"/Y.dmat", MatrixXd(Y));
		// exit(0);
		// igl::writeDMAT(inputfile+"/H_m.dmat", MatrixXd(store.H_m));
		// igl::writeDMAT(inputfile+"/grad_m.dmat", MatrixXd(store.grad_m));
		// igl::writeDMAT(inputfile+"/H_n.dmat", MatrixXd(store.H_n));
		// igl::writeDMAT(inputfile+"/grad_n.dmat", MatrixXd(store.grad_n));
		
			store.printState(0, "wood", x);
			VectorXd c = store.b + Y*Y.transpose()*q0;
			double tol = store.jinput["nm_tolerance"];
			exact::newton_solve(Fvec, 
								x,
								tol,
								store.T,
								store.eY,
								store.eP,
								store.Uvec,
								store.rest_tet_vols,
								bone_or_muscle, 
								ProjectF, 
								d, 
								Ai, 
								Vtilde, 
								50000,
								Y, 
								B, 
								c,
								store.bone_tets, 
								wId9T, wVAi, wHiV, wHiVAi, wC, wPhi, wHPhi, wL, wIdL, wQ,
								store);

			exact::acap_solve(x, ProjectF, ACAP, Y, B, Fvec, c);
			store.printState(1, "wood", x);
		


	// std::cout<<"---Viewer parameters"<<std::endl;
	// 	{
	// 		VectorXd vertex_wise_YM =Eigen::VectorXd::Zero(store.V.rows());
	// 		store.elogVY = Eigen::VectorXd::Zero(store.V.rows());
	// 		// volume associated with each vertex
	// 		Eigen::VectorXd Vvol = Eigen::VectorXd::Zero(store.V.rows());
	// 		Eigen::VectorXd Tvol;
	// 		igl::volume(store.V,store.T,Tvol);
	// 		// loop over tets
	// 		for(int i = 0;i<store.T.rows();i++)
	// 		{
	// 			const double vol4 = Tvol(i)/4.0;
	// 			for(int j = 0;j<4;j++)
	// 			{
	// 				Vvol(store.T(i,j)) += vol4;
	// 				store.elogVY(store.T(i,j)) += vol4*log10(store.eY(i));
	// 				vertex_wise_YM(store.T(i,j)) = store.eY(i);
	// 			}
	// 		}
	// 		// loop over vertices to divide to take average
	// 		for(int i = 0;i<store.V.rows();i++)
	// 		{
	// 			store.elogVY(i) /= Vvol(i);
	// 		}

	// 		VectorXd surface_youngs = VectorXd::Zero(store.F.rows());

	// 		for(int ff =0 ; ff<store.F.rows(); ff++){
	// 			Vector3i face = store.F.row(ff);
	// 			Vector3d face_material_YM(vertex_wise_YM(face(0)), vertex_wise_YM(face(1)), vertex_wise_YM(face(2)));
	// 			double minYM = face_material_YM.minCoeff();
	// 			surface_youngs(ff) = minYM;

	// 		}

	// 		igl::writeDMAT(inputfile +"/surface_mesh_youngs.dmat", surface_youngs, true);
	// 	}



}
