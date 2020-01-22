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

#include <sstream>
#include <iomanip>
// #include <omp.h>

#include "store.h"
#include "read_config_files.h"
#include "vertex_bc.h"
#include "discontinuous_edge_vectors.h"
#include "discontinuous_centroids_matrix.h"
#include "cont_to_discont_tets.h"
#include "construct_kkt_system.h"
#include "get_min_max_verts.h"
#include "muscle_energy_gradient.h"
#include "stablenh_energy_gradient.h"
#include "acap_solve_energy_gradient.h"
#include "draw_disc_mesh_functions.h"
#include "dfmatrix_vector_swap.h"
#include "newton_solver.h"
#include "joint_constraint_matrix.h"
#include "fixed_bones_projection_matrix.h"
#include "bone_elem_def_grad_projection_matrix.h"
#include "setup_hessian_modes.h"

using namespace Eigen;
using Store = famu::Store;

void famu::setupStore(Store& store){
	   	igl::Timer timer;

		famu::read_config_files(store, 
								store.V, 
								store.T, 
								store.F, 
								store.Uvec, 
								store.bone_name_index_map, 
								store.muscle_name_index_map, 
								store.joint_bones_verts, 
								store.bone_tets, 
								store.muscle_tets, 
								store.fix_bones,
								store.script_bones,
								store.relativeStiffness,
								store.contract_muscles,
								store.muscle_steps,
								store.jinput);  
		store.alpha_arap = store.jinput["alpha_arap"];
		store.alpha_neo = store.jinput["alpha_neo"];
		


		
	cout<<"---Record Mesh Setup Info"<<endl;
		cout<<"V size: "<<store.V.rows()<<endl;
		cout<<"T size: "<<store.T.rows()<<endl;
		cout<<"F size: "<<store.F.rows()<<endl;
		store.joutput["info"]["Vsize"] = store.V.rows();
		store.joutput["info"]["Tsize"] = store.T.rows();
		store.joutput["info"]["Fsize"] = store.F.rows();
		store.joutput["info"]["NumModes"] = NUM_MODES;
		store.joutput["info"]["NumThreads"] = Eigen::nbThreads();
		store.joutput["info"]["acap_alpha"] = store.alpha_arap;


		store.jinput["number_modes"] = NUM_MODES;

	cout<<"---Set Fixed Vertices"<<endl;
		// store.mfix = famu::getMaxVerts(store.V, 1);
		// store.mmov = {};//famu::getMinVerts(store.V, 1);
		cout<<"If it fails here, make sure indexing is within bounds"<<endl;
	    std::set<int> fix_verts_set;
	    for(int ii=0; ii<store.fix_bones.size(); ii++){
	        cout<<store.fix_bones[ii]<<endl;
	        int bone_ind = store.bone_name_index_map[store.fix_bones[ii]];
	        fix_verts_set.insert(store.T.row(store.bone_tets[bone_ind][0])[0]);
	        fix_verts_set.insert(store.T.row(store.bone_tets[bone_ind][0])[1]);
	        fix_verts_set.insert(store.T.row(store.bone_tets[bone_ind][0])[2]);
	        fix_verts_set.insert(store.T.row(store.bone_tets[bone_ind][0])[3]);
	    }
	    store.mfix.assign(fix_verts_set.begin(), fix_verts_set.end());
	    std::sort (store.mfix.begin(), store.mfix.end());
	
	cout<<"---Set Mesh Params"<<store.x.size()<<endl;
		//YM, poissons
		store.eY = 1e10*VectorXd::Ones(store.T.rows());
		store.eP = 0.49*VectorXd::Ones(store.T.rows());
		store.muscle_mag = VectorXd::Zero(store.T.rows());
		for(int m=0; m<store.muscle_tets.size(); m++){
			for(int t=0; t<store.muscle_tets[m].size(); t++){
				if(store.relativeStiffness[store.muscle_tets[m][t]]>1){
					store.eY[store.muscle_tets[m][t]] = 4.5e8;
				}else{
					store.eY[store.muscle_tets[m][t]] = 60000;
				}
			}
		}

        std::string inputfile = store.jinput["data"];
        
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

		//bone dF map
		//start off all as -1
		store.bone_or_muscle = -1*Eigen::VectorXi::Ones(store.T.rows());
		if(store.jinput["reduced"]){

			// // // assign bone tets to 1 dF for each bone, starting at 0...bone_tets.size()
			for(int i=0; i<store.bone_tets.size(); i++){
		    	for(int j=0; j<store.bone_tets[i].size(); j++){
		    		store.bone_or_muscle[store.bone_tets[i][j]] = i;
		    	}
		    }

		    //assign muscle tets, dF per element, starting at bone_tets.size()
		    int muscle_ind = store.bone_tets.size();
		    for(int i=0; i<store.T.rows(); i++){
		    	if(store.bone_or_muscle[i]<-1e-8){
		    		store.bone_or_muscle[i] = muscle_ind;
		    		muscle_ind +=1;
		    	}
		    }
		}
	    else{
			for(int i=0; i<store.T.rows(); i++){
				store.bone_or_muscle[i] = i;
			}
	    }

	    //Rest state volumes
	    store.rest_tet_volume = VectorXd::Ones(store.T.rows());
		for(int i =0; i<store.T.rows(); i++){
			Vector3d p1 = store.V.row(store.T.row(i)[0]); 
			Vector3d p2 = store.V.row(store.T.row(i)[1]); 
			Vector3d p3 = store.V.row(store.T.row(i)[2]);
			Vector3d p4 = store.V.row(store.T.row(i)[3]); 
			
			Matrix3d Dm;
			Dm.col(0) = p1 - p4;
			Dm.col(1) = p2 - p4;
			Dm.col(2) = p3 - p4;
			double density = 1000;
			double undef_vol = (1.0/6)*fabs(Dm.determinant());
			store.rest_tet_volume[i] = undef_vol;
		}
		for(int i=0; i<store.bone_tets.size(); i++){
			double bone_vol = 0;
	    	for(int j=0; j<store.bone_tets[i].size(); j++){
	    		bone_vol += store.rest_tet_volume[store.bone_tets[i][j]];
	    	}
	    	store.bone_vols.push_back(bone_vol);
	    }

	cout<<"---Setup continuous mesh"<<store.x.size()<<endl;
		store.x0.resize(3*store.V.rows());
		for(int i=0; i<store.V.rows(); i++){
			store.x0[3*i+0] = store.V(i,0); 
			store.x0[3*i+1] = store.V(i,1); 
			store.x0[3*i+2] = store.V(i,2);   
	    }
	    store.dx = VectorXd::Zero(3*store.V.rows());


	cout<<"---Cont. to Discont. matrix"<<store.x.size()<<endl;
		famu::cont_to_discont_tets(store.S, store.T, store.V);
	    
    cout<<"---Set Vertex Constraint Matrices"<<store.x.size()<<endl;
		famu::vertex_bc(store.mmov, store.mfix, store.UnconstrainProjection, store.ConstrainProjection, store.V);

	cout<<"---Set Discontinuous Tet Centroid vector matrix"<<store.x.size()<<endl;
		famu::discontinuous_edge_vectors(store, store.D, store._D, store.T, store.muscle_tets);

	cout<<"---Set Centroid Matrix"<<store.x.size()<<endl;
		famu::discontinuous_centroids_matrix(store.C, store.T);

	cout<<"---Set Disc T and V"<<store.x.size()<<endl;
		famu::setDiscontinuousMeshT(store.T, store.discT);
		igl::boundary_facets(store.discT, store.discF);
		store.discV.resize(4*store.T.rows(), 3);

	cout<<"---Set Joints Constraint Matrix"<<store.x.size()<<endl;
		cout<<"fixed bones"<<endl;
		famu::fixed_bones_projection_matrix(store, store.Y);
		cout<<"scripted bones"<<endl;
		famu::scripted_bones_projection_matrix(store, store.ScriptBonesY);

	    store.x = VectorXd::Zero(store.Y.cols());
		famu::joint_constraint_matrix(store, store.JointConstraints);

		famu::bone_def_grad_projection_matrix(store, store.ProjectF, store.RemFixedBones);
		famu::bone_acap_deformation_constraints(store, store.Bx, store.Bf, store.Bsx);
	    store.lambda2 = VectorXd::Zero(store.Bf.rows());
	
	cout<<"---ACAP Solve KKT setup"<<store.x.size()<<endl;
		SparseMatrix<double, Eigen::RowMajor> KKT_left, KKT_left0, KKT_left1, KKT_left2;
		store.YtStDtDSY = (store.D*store.S*store.Y).transpose()*(store.D*store.S*store.Y);
		double jointSlack = 0;
		double boneSlack = 0;
		if(store.jinput["script_bones"].size()==0){
			jointSlack = 0;
			boneSlack = -1e-3;
		}else{
			jointSlack = -1e-4;
			boneSlack = 0;
		}
		famu::construct_kkt_system_left(store.YtStDtDSY, store.JointConstraints, KKT_left, jointSlack);


		//---------------SPRINGS
		// if(springk>0){
		// 	SparseMatrix<double, Eigen::RowMajor> PY = springk*store.ContactP*store.Y;
		// 	famu::construct_kkt_system_left(KKT_left, PY, KKT_left1, -1);
		// 	famu::construct_kkt_system_left(KKT_left1, store.Bx,  KKT_left2, boneSlack); 
		// }else{
			famu::construct_kkt_system_left(KKT_left, store.Bx,  KKT_left2, boneSlack);
		// }

		// MatrixXd Hkkt = MatrixXd(KKT_left2);
		#ifdef __linux__
		store.ACAP_KKT_SPLU.pardisoParameterArray()[2] = Eigen::nbThreads(); 
		#endif

		store.ACAP_KKT_SPLU.analyzePattern(KKT_left2);
		store.ACAP_KKT_SPLU.factorize(KKT_left2);

		if(store.ACAP_KKT_SPLU.info()!=Success){
			cout<<"1. ACAP Jacobian solve failed"<<endl;
			cout<<"2. numerical issue: "<<(store.ACAP_KKT_SPLU.info()==NumericalIssue)<<endl;
			cout<<"3. invalid input: "<<(store.ACAP_KKT_SPLU.info()==InvalidInput)<<endl;

			exit(0);
		}
		store.acap_solve_result.resize(KKT_left2.rows());
		store.acap_solve_rhs = VectorXd::Zero(KKT_left2.rows());

	cout<<"---2nd ACAP Solve KKT setup"<<store.x.size()<<endl;
		SparseMatrix<double, Eigen::RowMajor> KKT2_0, KKT2_1;
		if(store.jinput["script_bones"].size()==0){
			jointSlack = -1e-4;
			boneSlack = 0;
			famu::construct_kkt_system_left(store.YtStDtDSY, store.JointConstraints, KKT2_1, jointSlack);
			famu::construct_kkt_system_left(KKT2_1, store.Bx, KKT2_0, boneSlack);

			store.ACAP_KKT_SPLU2.analyzePattern(KKT2_0);
			store.ACAP_KKT_SPLU2.factorize(KKT2_0);
		}else{
			store.ACAP_KKT_SPLU2.analyzePattern(KKT_left2);
			store.ACAP_KKT_SPLU2.factorize(KKT_left2);
		}

		// MatrixXd Hkkt = MatrixXd(KKT_left2);
		#ifdef __linux__
		store.ACAP_KKT_SPLU2.pardisoParameterArray()[2] = Eigen::nbThreads(); 
		#endif


		

		if(store.ACAP_KKT_SPLU2.info()!=Success){
			cout<<"1. SBY ACAP Jacobian solve failed"<<endl;
			cout<<"2. numerical issue: "<<(store.ACAP_KKT_SPLU2.info()==NumericalIssue)<<endl;
			cout<<"3. invalid input: "<<(store.ACAP_KKT_SPLU2.info()==InvalidInput)<<endl;

			exit(0);
		}

		store.acap_solve_result2.resize(KKT2_0.rows());
		store.acap_solve_rhs2 = VectorXd::Zero(KKT2_0.rows());
	
	cout<<"---Setup dFvec and dF"<<endl;
		cout<<store.ProjectF.cols()<<endl;
		cout<<store.RemFixedBones.rows()<<endl;
		store.dFvec = VectorXd::Zero(store.ProjectF.cols());
		for(int t=0; t<store.dFvec.size()/9; t++){
			store.dFvec[9*t + 0] = 1;
			store.dFvec[9*t + 4] = 1;
			store.dFvec[9*t + 8] = 1;
		}
		store.I0 = store.dFvec;
		store.BfI0 = store.Bf*store.dFvec;
		store.tot_Fc = VectorXd::Zero(3*store.V.rows());
		

	cout<<"---Setup Fast ACAP energy"<<endl;
		store.StDtDS = (store.D*store.S).transpose()*(store.D*store.S);
		store.DSY = store.D*store.S*store.Y;
		store.DSx0 = store.D*store.S*store.x0;
		famu::dFMatrix_Vector_Swap(store.DSx0_mat, store.DSx0);
		

		store.x0tStDtDSx0 = store.DSx0.transpose()*store.DSx0;
		store.x0tStDtDSY = store.DSx0.transpose()*store.DSY;
		store.x0tStDt_dF_DSx0 = store.DSx0.transpose()*store.DSx0_mat*store.ProjectF;
		store.YtStDt_dF_DSx0 = (store.DSY).transpose()*store.DSx0_mat*store.ProjectF;
		store.x0tStDt_dF_dF_DSx0 = (store.DSx0_mat*store.ProjectF).transpose()*store.DSx0_mat*store.ProjectF;

		famu::muscle::setupFastMuscles(store);


	cout<<"--- Setup Modes"<<endl;
	if(store.jinput["woodbury"]){

        MatrixXd temp1;
        if(store.JointConstraints.rows() != 0){
			MatrixXd nullJ;
			igl::null(MatrixXd(store.JointConstraints), nullJ);
			store.NullJ = nullJ.sparseView();
		}else{
			store.NullJ.resize(store.Y.cols(), store.Y.cols());
			store.NullJ.setIdentity();
		}
		std::string outputfile = store.jinput["output"];
        SparseMatrix<double> NjtYtStDtDSYNj = store.NullJ.transpose()*store.Y.transpose()*store.S.transpose()*store.D.transpose()*store.D*store.S*store.Y*store.NullJ;
        igl::readDMAT(outputfile+"/"+to_string((int)store.jinput["number_modes"])+"modes.dmat", temp1);
        if(temp1.rows() == 0){
        	// igl::writeDMAT(outputfile+"/"+"NjtYtStDtDSYNj.dmat", Eigen::MatrixXd(NjtYtStDtDSYNj));
			famu::setup_hessian_modes(store, NjtYtStDtDSYNj, temp1);
			igl::writeDMAT(outputfile+"/"+to_string((int)store.jinput["number_modes"])+"modes.dmat", temp1);
			igl::writeDMAT(outputfile+"/"+to_string((int)store.jinput["number_modes"])+"eigs.dmat", store.eigenvalues);
		}else{
			//read eigenvalues (for the woodbury solve)
			igl::readDMAT(outputfile+"/"+to_string((int)store.jinput["number_modes"])+"eigs.dmat", store.eigenvalues);
		}
		store.G = store.NullJ*temp1;
	}
	

	cout<<"--- ACAP Hessians"<<endl;
		famu::acap::setJacobian(store);
		
		store.denseNeoHess = MatrixXd::Zero(store.dFvec.size(), 9);
		store.neoHess.resize(store.dFvec.size(), store.dFvec.size());
		famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess);

		store.denseMuscleHess = MatrixXd::Zero(store.dFvec.size(), 9);
		store.muscleHess.resize(store.dFvec.size(), store.dFvec.size());
		famu::muscle::fastHessian(store, store.muscleHess, store.denseMuscleHess);

		store.denseAcapHess = MatrixXd::Zero(store.dFvec.size(), 9);
		store.acapHess.resize(store.dFvec.size(), store.dFvec.size());
		famu::acap::fastHessian(store, store.acapHess, store.denseAcapHess);
		

		SparseMatrix<double> hessFvec = store.neoHess + store.acapHess + store.muscleHess;
		store.NM_SPLU.analyzePattern(hessFvec);
		store.NM_SPLU.factorize(hessFvec);

			
	if(store.jinput["woodbury"]){
		cout<<"--- Setup woodbury matrices"<<endl;
			double aa = store.jinput["alpha_arap"];
			cout<<store.RemFixedBones.rows()<<", "<<store.RemFixedBones.cols()<<endl;
			cout<<store.YtStDt_dF_DSx0.rows()<<", "<<store.YtStDt_dF_DSx0.cols()<<endl;
			store.WoodB = -store.RemFixedBones*store.YtStDt_dF_DSx0.transpose()*store.G;
			store.WoodD = -1*store.WoodB.transpose();
			store.WoodB *= aa;

			store.InvC = store.eigenvalues.asDiagonal();
			store.WoodC = store.eigenvalues.asDiagonal().inverse();
			for(int i=0; i<store.RemFixedBones.rows()/9; i++){
				LDLT<Matrix9d> InvA;
				store.vecInvA.push_back(InvA);
			}
	}

	//GInv(L)G = UCV

    cout<<"---Setup TMP Vars"<<endl;
    	famu::discontinuousV(store);
    	store.acaptmp_sizex = store.x;
		store.acaptmp_sizedFvec1= store.dFvec;
		store.acaptmp_sizedFvec2 = store.dFvec;

	cout<<"---Set External Forces"<<endl;
		Eigen::VectorXd yext;
		famu::acap::external_forces(store, yext, true);
		store.ContactForce = VectorXd::Zero(store.dFvec.size());
		if(store.jinput["springk"]!=0){
			Eigen::VectorXd temp = Eigen::VectorXd::Zero(3*store.V.rows());
			Eigen::MatrixXd DR = Eigen::MatrixXd::Zero(store.V.rows(), store.V.cols());
			famu::acap::mesh_collisions(store, DR);
			
			for(int i=0; i<store.V.rows(); i++){
				temp[3*i+0] = DR(i,0); 
				temp[3*i+1] = DR(i,1); 
				temp[3*i+2] = DR(i,2);
				if(DR.row(i).norm()>1e-7){
					store.draw_points.push_back(i);
				}
		    }
		    std::vector<int> blank;
		    famu::vertex_bc(blank, store.draw_points, store.UnPickBoundaryForCollisions, store.PickBoundaryForCollisions, store.V);
		}
		

	cout<<"---Setup Muscle Activations"<<endl;
		famu::muscle::set_muscle_mag(store, 0);

}
