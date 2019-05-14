#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
#include <igl/readOBJ.h>
#include <igl/jet.h>
#include <igl/slice.h>
#include <igl/boundary_facets.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/list_to_matrix.h>
#include <imgui/imgui.h>
#include <igl/null.h>
#include <json.hpp>
#include <Eigen/SparseCholesky>

#include <sstream>
#include <iomanip>

#include "famu/store.h"
#include "famu/read_config_files.h"
#include "famu/vertex_bc.h"
#include "famu/discontinuous_edge_vectors.h"
#include "famu/discontinuous_centroids_matrix.h"
#include "famu/cont_to_discont_tets.h"
#include "famu/construct_kkt_system.h"
#include "famu/get_min_max_verts.h"
#include "famu/muscle_energy_gradient.h"
#include "famu/stablenh_energy_gradient.h"
#include "famu/acap_solve_energy_gradient.h"
#include "famu/draw_disc_mesh_functions.h"
#include "famu/dfmatrix_vector_swap.h"
#include "famu/newton_solver.h"
#include "famu/joint_constraint_matrix.h"
#include "famu/fixed_bones_projection_matrix.h"
#include "famu/bone_elem_def_grad_projection_matrix.h"
#include "famu/setup_hessian_modes.h"

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

using Store = famu::Store;
json j_input;


int main(int argc, char *argv[])
{
	std::cout<<"-----Configs-------"<<std::endl;
    	igl::Timer timer;
		std::ifstream input_file("../input/input.json");
		input_file >> j_input;

		famu::Store store;
		store.jinput = j_input;

		famu::read_config_files(store.V, 
								store.T, 
								store.F, 
								store.Uvec, 
								store.bone_name_index_map, 
								store.muscle_name_index_map, 
								store.joint_bones_verts, 
								store.bone_tets, 
								store.muscle_tets, 
								store.fix_bones, 
								store.relativeStiffness, 
								store.jinput);  
		store.alpha_arap = store.jinput["alpha_arap"];
		store.alpha_neo = store.jinput["alpha_neo"];
		std::vector<int> contract_muscles = store.jinput["contract_muscles_at_index"];
		store.contract_muscles = contract_muscles;


	cout<<"---Record Mesh Setup Info"<<endl;
		cout<<"V size: "<<store.V.rows()<<endl;
		cout<<"T size: "<<store.T.rows()<<endl;
		cout<<"F size: "<<store.F.rows()<<endl;
		if(argc>1){
			j_input["number_modes"] =  stoi(argv[1]);
			j_input["number_rot_clusters"] =  stoi(argv[2]);
			j_input["number_skinning_handles"] =  stoi(argv[3]);
		}
		store.jinput["number_modes"] = NUM_MODES;
		cout<<"NSH: "<<j_input["number_skinning_handles"]<<endl;
		cout<<"NRC: "<<j_input["number_rot_clusters"]<<endl;
		cout<<"MODES: "<<j_input["number_modes"]<<endl;
		std::string outputfile = j_input["output"];
		std::string namestring = to_string((int)j_input["number_modes"])+"modes"+to_string((int)j_input["number_rot_clusters"])+"clusters"+to_string((int)j_input["number_skinning_handles"])+"handles";
		igl::boundary_facets(store.T, store.F);

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
					store.eY[store.muscle_tets[m][t]] = 1.2e9;
				}else{
					store.eY[store.muscle_tets[m][t]] = 60000;
				}
				store.muscle_mag[store.muscle_tets[m][t]] = j_input["muscle_starting_strength"];
			}
		}
		igl::writeDMAT("youngs_per_tet.dmat", store.eY);

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
		store.discV.resize(4*store.T.rows(), 3);

	cout<<"---Set Joints Constraint Matrix"<<store.x.size()<<endl;
		famu::fixed_bones_projection_matrix(store, store.Y);
	    store.x = VectorXd::Zero(store.Y.cols());
		famu::joint_constraint_matrix(store, store.JointConstraints);

		famu::bone_def_grad_projection_matrix(store, store.ProjectF, store.PickBoneF);
		if(store.JointConstraints.rows() != 0){
			MatrixXd nullJ;
			igl::null(MatrixXd(store.JointConstraints), nullJ);
			store.NullJ = nullJ.sparseView();
		}else{
			store.NullJ.resize(store.Y.cols(), store.Y.cols());
			store.NullJ.setIdentity();
		}
	
		famu::bone_acap_deformation_constraints(store, store.Bx, store.Bf);
	    store.lambda2 = VectorXd::Zero(store.Bf.rows());
	

	cout<<"---ACAP Solve KKT setup"<<store.x.size()<<endl;
		SparseMatrix<double> KKT_left;
		store.YtStDtDSY = (store.D*store.S*store.Y).transpose()*(store.D*store.S*store.Y);
		famu::construct_kkt_system_left(store.YtStDtDSY, store.JointConstraints, KKT_left);

		SparseMatrix<double> KKT_left2;
		famu::construct_kkt_system_left(KKT_left, store.Bx,  KKT_left2, -1e-3); 
		// MatrixXd Hkkt = MatrixXd(KKT_left2);
		

		store.ACAP_KKT_SPLU.analyzePattern(KKT_left2);
		store.ACAP_KKT_SPLU.factorize(KKT_left2);

		if(store.ACAP_KKT_SPLU.info()!=Success){
			cout<<"1. ACAP Jacobian solve failed"<<endl;
			cout<<"2. numerical issue: "<<(store.ACAP_KKT_SPLU.info()==NumericalIssue)<<endl;
			cout<<"3. invalid input: "<<(store.ACAP_KKT_SPLU.info()==InvalidInput)<<endl;

			exit(0);
		}


	cout<<"---Setup dFvec and dF"<<endl;
		store.dFvec = VectorXd::Zero(store.ProjectF.cols());
		for(int t=0; t<store.dFvec.size()/9; t++){
			store.dFvec[9*t + 0] = 1;
			store.dFvec[9*t + 4] = 1;
			store.dFvec[9*t + 8] = 1;
		}
		store.BfI0 = store.Bf*store.dFvec;
		store.acap_solve_result.resize(KKT_left2.rows());
		store.acap_solve_rhs = VectorXd::Zero(KKT_left2.rows());



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
        MatrixXd temp1;
        SparseMatrix<double> NjtYtStDtDSYNj = store.NullJ.transpose()*store.Y.transpose()*store.S.transpose()*store.D.transpose()*store.D*store.S*store.Y*store.NullJ;
        igl::readDMAT(outputfile+"/"+to_string((int)store.jinput["number_modes"])+"modes.dmat", temp1);
        if(temp1.rows() == 0 && !store.jinput["sparseJac"]){
			famu::setup_hessian_modes(store, NjtYtStDtDSYNj, temp1);
		}else{
			//read eigenvalues (for the woodbury solve)
			igl::readDMAT(outputfile+"/"+to_string((int)store.jinput["number_modes"])+"eigs.dmat", store.eigenvalues);
		}
		store.G = store.NullJ*temp1;
		// cout<<store.G.rows()<<", "<<store.G.cols()<<endl;

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
			store.WoodB = -store.YtStDt_dF_DSx0.transpose()*store.G;
			store.WoodD = -1*store.WoodB.transpose();
			

			store.InvC = store.eigenvalues.asDiagonal();
			store.WoodC = store.eigenvalues.asDiagonal().inverse();
	}


    cout<<"---Setup Solver"<<endl;




	cout<<"--- Write Meshes"<<endl;
		// int run =0;
	   //  for(int run=0; run<store.jinput["QS_steps"]; run++){
	   //      VectorXd y = store.Y*store.x;
	   //      Eigen::Map<Eigen::MatrixXd> newV(y.data(), store.V.cols(), store.V.rows());
	   //      std::string datafile = j_input["data"];
	   //      ostringstream out;
	   //      out << std::internal << std::setfill('0') << std::setw(3) << run;
	   //      igl::writeOBJ(outputfile+"/"+"animation"+out.str()+".obj",(newV.transpose()+store.V), store.F);
	   //      igl::writeDMAT(outputfile+"/"+"animation"+out.str()+".dmat",(newV.transpose()+store.V));
	        
	   //      cout<<"     ---Quasi-Newton Step Info"<<endl;
		  //       double fx = 0;
				// timer.start();
				// int niters = 0;
				// niters = famu::newton_static_solve(store);
				// timer.stop();
				// cout<<"+++QS Step iterations: "<<niters<<", secs: "<<timer.getElapsedTimeInMicroSec()<<endl;
        	
	        
	   //      store.muscle_mag *= 1.5;
	   //  }
	   //  exit(0);

	std::cout<<"-----Display-------"<<std::endl;
    igl::opengl::glfw::Viewer viewer;
    int kkkk =0;
    double tttt =0;
    // MatrixXd modes = store.Y*store.G;
    // viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & viewer){   
    //     if(viewer.core.is_animating){
    //         if(kkkk < store.G.cols()){
    //             VectorXd x = 50*sin(tttt)*modes.col(kkkk) + store.x0;
    //             Eigen::Map<Eigen::MatrixXd> newV(x.data(), store.V.cols(), store.V.rows());
    //             viewer.data().set_mesh(newV.transpose(), store.F);
    //             tttt+= 0.1;
    //         }
    // 	}
    //     return false;
    // };

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers){   
        std::cout<<"Key down, "<<key<<std::endl;
        viewer.data().clear();

        if(key =='K'){
        	kkkk ++;
        }

        if(key=='A'){
        	store.muscle_mag *= 1.5;
        	famu::muscle::setupFastMuscles(store);
        	famu::muscle::fastHessian(store, store.muscleHess, store.denseMuscleHess);
        }
        
        if(key==' '){
        
   //  		store.dFvec[9+0] = 0.7071;
   //  		store.dFvec[9+1] = 0.7071;
   //  		store.dFvec[9+2] = 0;
   //  		store.dFvec[9+3] = -0.7071;
   //  		store.dFvec[9+4] = 0.7071;
   //  		store.dFvec[9+5] = 0;
   //  		store.dFvec[9+6] = 0;
   //  		store.dFvec[9+7] = 0;
   //  		store.dFvec[9+8] = 1;
   //  		timer.start();
			// famu::acap::solve(store, store.dFvec);        	
			// timer.stop();
			// cout<<"+++Microsecs per solve: "<<timer.getElapsedTimeInMicroSec()<<endl;

			double fx = 0;
			timer.start();
			int niters = 0;
		
			niters = famu::newton_static_solve(store);
			timer.stop();
			double totaltime = timer.getElapsedTimeInMicroSec();
			cout<<"Full NM per iter: "<<totaltime/niters<<endl;
			cout<<"Total time: "<<totaltime<<endl;
			cout<<"Total its: "<<niters<<endl;
			cout<<"+++++ QS Iteration +++++"<<endl;
        	
        }

        if(key=='D'){
            
            // Draw disc mesh
            famu::discontinuousV(store);
            for(int m=0; m<store.discT.rows()/10; m++){
                int t= 10*m;
                Vector4i e = store.discT.row(t);
                
                Matrix<double, 1,3> p0 = store.discV.row(e[0]);
                Matrix<double, 1,3> p1 = store.discV.row(e[1]);
                Matrix<double, 1,3> p2 = store.discV.row(e[2]);
                Matrix<double, 1,3> p3 = store.discV.row(e[3]);

                viewer.data().add_edges(p0,p1,Eigen::RowVector3d(1,0,1));
                viewer.data().add_edges(p0,p2,Eigen::RowVector3d(1,0,1));
                viewer.data().add_edges(p0,p3,Eigen::RowVector3d(1,0,1));
                viewer.data().add_edges(p1,p2,Eigen::RowVector3d(1,0,1));
                viewer.data().add_edges(p1,p3,Eigen::RowVector3d(1,0,1));
                viewer.data().add_edges(p2,p3,Eigen::RowVector3d(1,0,1));
            }
        
        }
        VectorXd y = store.Y*store.x;
        Eigen::Map<Eigen::MatrixXd> newV(y.data(), store.V.cols(), store.V.rows());
        viewer.data().set_mesh((newV.transpose()+store.V), store.F);
        igl::writeOBJ("ACAP_unred.obj", (newV.transpose()+store.V), store.F);
        
        if(key=='V' || key=='S' || key=='E' || key =='C'){
            MatrixXd COLRS;
            VectorXd zz = VectorXd::Ones(store.V.rows());

            if(key=='C'){

	            for(int m=0; m<store.T.rows(); m++){
	                Matrix3d Dm;
	                for(int i=0; i<3; i++){
	                    Dm.col(i) = store.V.row(store.T.row(m)[i]) - store.V.row(store.T.row(m)[3]);
	                }
	                Matrix3d m_InvRefShapeMatrix = Dm.inverse();
	                
	                Matrix3d Ds;
	                for(int i=0; i<3; i++)
	                {
	                    Ds.col(i) = y.segment<3>(3*store.T.row(m)[i]) - y.segment<3>(3*store.T.row(m)[3]);
	                }

	                Matrix3d F = Matrix3d::Identity() + Ds*m_InvRefShapeMatrix;

	                double snorm = (F.transpose()*F - Matrix3d::Identity()).norm();
	               
	                zz[store.T.row(m)[0]] += snorm;
	                zz[store.T.row(m)[1]] += snorm;
	                zz[store.T.row(m)[2]] += snorm; 
	                zz[store.T.row(m)[3]] += snorm;
	            }
            }

        	if(key=='V'){
            //Display tendon areas
	            for(int i=0; i<store.T.rows(); i++){
	                zz[store.T.row(i)[0]] = store.relativeStiffness[i];
	                zz[store.T.row(i)[1]] = store.relativeStiffness[i];
	                zz[store.T.row(i)[2]] = store.relativeStiffness[i];
	                zz[store.T.row(i)[3]] = store.relativeStiffness[i];
	            }
            }
            
            if(key=='S'){
            	//map strains
	            VectorXd fulldFvec = store.ProjectF*store.dFvec;
	            for(int m=0; m<store.T.rows(); m++){
	                Matrix3d F = Map<Matrix3d>(fulldFvec.segment<9>(9*m).data()).transpose();
	                double snorm = (F.transpose()*F - Matrix3d::Identity()).norm();
	               
	                zz[store.T.row(m)[0]] += snorm;
	                zz[store.T.row(m)[1]] += snorm;
	                zz[store.T.row(m)[2]] += snorm; 
	                zz[store.T.row(m)[3]] += snorm;
	            }
            }

            if(key=='E'){
            	//map ACAP energy over the meseh
            	VectorXd ls = store.DSY*store.x + store.DSx0;
            	VectorXd rs = store.DSx0_mat*store.ProjectF*store.dFvec;
            	for(int i=0; i<store.T.rows(); i++){
            		double enorm = (ls.segment<12>(12*i) - rs.segment<12>(12*i)).norm();

            		zz[store.T.row(i)[0]] += enorm;
	                zz[store.T.row(i)[1]] += enorm;
	                zz[store.T.row(i)[2]] += enorm; 
	                zz[store.T.row(i)[3]] += enorm;

            	}
            }
	            // cout<<zz.transpose()<<endl;


            igl::jet(zz, true, COLRS);
            viewer.data().set_colors(COLRS);
        }
        

        for(int i=0; i<store.mfix.size(); i++){
        	viewer.data().add_points((newV.transpose().row(store.mfix[i]) + store.V.row(store.mfix[i])), Eigen::RowVector3d(1,0,0));
        }

        for(int i=0; i<store.mmov.size(); i++){
        	viewer.data().add_points((newV.transpose().row(store.mmov[i]) + store.V.row(store.mmov[i])), Eigen::RowVector3d(0,1,0));
        }
        
        return false;
    };

	viewer.data().set_mesh(store.V, store.F);
    viewer.data().show_lines = false;
    viewer.data().invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data().face_based = true;
    viewer.core.background_color = Eigen::Vector4f(1,1,1,0);
    // viewer.data().set_colors(SETCOLORSMAT);

    viewer.launch();

}