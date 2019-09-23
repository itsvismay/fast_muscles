#include <igl/opengl/glfw/Viewer.h>
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
  int fancy_data_index,debug_data_index,discontinuous_data_index;
	std::cout<<"-----Configs-------"<<std::endl;
		std::string inputfile;
		int num_threads = 1;
		if(argc<1){
			cout<<"Run as: ./famu input.json <threads>"<<endl;
			exit(0);
		}
		if(argc==3){
			num_threads = std::stoi(argv[2]);
			#ifdef __linux__
			omp_set_num_threads(num_threads);
			#endif
			std::ifstream input_file(argv[1]);
			input_file >> j_input;
		}else if(argc==4){
			num_threads = std::stoi(argv[3]);
			#ifdef __linux__
			omp_set_num_threads(num_threads);
			#endif
			std::ifstream input_file(argv[2]);
			input_file >> j_input;

		}
		Eigen::initParallel();
	
		
    	igl::Timer timer;

		famu::Store store;
		store.jinput = j_input;

		famu::read_config_files(store);  
		store.alpha_arap = store.jinput["alpha_arap"];
		store.alpha_neo = store.jinput["alpha_neo"];
		


	cout<<"---Record Mesh Setup Info"<<endl;
		cout<<"V size: "<<store.V.rows()<<endl;
		cout<<"T size: "<<store.T.rows()<<endl;
		cout<<"F size: "<<store.F.rows()<<endl;
		store.jinput["number_modes"] = NUM_MODES;
		std::string outputfile = j_input["output"];
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
        {
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
            }
          }
          // loop over vertices to divide to take average
          for(int i = 0;i<store.V.rows();i++)
          {
            store.elogVY(i) /= Vvol(i);
          }
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
		famu::fixed_bones_projection_matrix(store, store.Y);
	    store.x = VectorXd::Zero(store.Y.cols());
		famu::joint_constraint_matrix(store, store.JointConstraints);

		famu::bone_def_grad_projection_matrix(store, store.ProjectF, store.PickBoneF);
		famu::bone_acap_deformation_constraints(store, store.Bx, store.Bf);
	    store.lambda2 = VectorXd::Zero(store.Bf.rows());

	    // std::vector<std::pair<int, int>> springs;
	    // std::vector<int> bcMuscle1 = getMinVerts_Axis_Tolerance(store.T, store.V, 2, 1e-1, store.muscle_tets[0]);
	    // std::vector<int> bcMuscle2 = getMaxVerts_Axis_Tolerance(store.T, store.V, 2, 1e-1, store.muscle_tets[1]);
	    // // famu::make_closest_point_springs(store.T, store.V, store.muscle_tets[1],  bcMuscle1, springs);
	    // famu::make_closest_point_springs(store.T, store.V, store.muscle_tets[0],  bcMuscle2, springs);

	    // famu::penalty_spring_bc(springs, store.ContactP, store.V);



	cout<<"---ACAP Solve KKT setup"<<store.x.size()<<endl;
		SparseMatrix<double, Eigen::RowMajor> KKT_left, KKT_left1;
		store.YtStDtDSY = (store.D*store.S*store.Y).transpose()*(store.D*store.S*store.Y);
		famu::construct_kkt_system_left(store.YtStDtDSY, store.JointConstraints, KKT_left);

		double k = store.jinput["springk"];
		// SparseMatrix<double, Eigen::RowMajor> PY = k*store.ContactP*store.Y;
		// famu::construct_kkt_system_left(KKT_left, PY, KKT_left1, -1);


		SparseMatrix<double, Eigen::RowMajor> KKT_left2;
		famu::construct_kkt_system_left(KKT_left, store.Bx,  KKT_left2, -1e-3); 
		// MatrixXd Hkkt = MatrixXd(KKT_left2);
		#ifdef __linux__
		store.ACAP_KKT_SPLU.pardisoParameterArray()[2] = num_threads; 
		#endif

		store.ACAP_KKT_SPLU.analyzePattern(KKT_left2);
		store.ACAP_KKT_SPLU.factorize(KKT_left2);

		if(store.ACAP_KKT_SPLU.info()!=Success){
			cout<<"1. ACAP Jacobian solve failed"<<endl;
			cout<<"2. numerical issue: "<<(store.ACAP_KKT_SPLU.info()==NumericalIssue)<<endl;
			cout<<"3. invalid input: "<<(store.ACAP_KKT_SPLU.info()==InvalidInput)<<endl;

			exit(0);
		}
		
	cout<<"---Setup dFvec and dF"<<endl;
		store.boneDOFS = VectorXd::Zero(3*store.bone_tets.size());
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
        SparseMatrix<double> NjtYtStDtDSYNj = store.NullJ.transpose()*store.Y.transpose()*store.S.transpose()*store.D.transpose()*store.D*store.S*store.Y*store.NullJ;
        igl::readDMAT(outputfile+"/"+to_string((int)store.jinput["number_modes"])+"modes.dmat", temp1);
        if(temp1.rows() == 0){
			famu::setup_hessian_modes(store, NjtYtStDtDSYNj, temp1);
		}else{
			//read eigenvalues (for the woodbury solve)
			igl::readDMAT(outputfile+"/"+to_string((int)store.jinput["number_modes"])+"eigs.dmat", store.eigenvalues);
		}
		store.G = store.NullJ*temp1;
	}
	

	cout<<"--- ACAP Hessians"<<endl;
		famu::acap::setJacobian(store);

		store.dRdW.resize(store.dFvec.size() - 6*store.bone_tets.size(), store.dFvec.size());
		store.dRdW0.resize(store.dFvec.size()- 6*store.bone_tets.size(), store.dFvec.size());
		store.dRdW.setZero();
		store.dRdW0.setZero();
		vector<Trip> dRdW_trips;
		store.dRdW0.setZero();
		//fill in the rest of dRdW as mxm Id
		for(int t =0; t<store.dFvec.size()-9*store.bone_tets.size(); t++){
			//fill it in backwards, bottom right to top left.
			dRdW_trips.push_back(Trip( store.dRdW.rows() - t -1, store.dRdW.cols() - t -1, 1));
		}
		store.dRdW0.setFromTriplets(dRdW_trips.begin(), dRdW_trips.end());
		famu::acap::updatedRdW(store);


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
			for(int i=0; i<store.dFvec.size()/9; i++){
				LDLT<Matrix9d> InvA;
				store.vecInvA.push_back(InvA);
			}

	}


    cout<<"---Setup TMP Vars"<<endl;
    	famu::discontinuousV(store);
    	store.acaptmp_sizex = store.x;
		store.acaptmp_sizedFvec1= store.dFvec;
		store.acaptmp_sizedFvec2 = store.dFvec;


		store.dFvec[9+0] = 0.7071;
	    store.dFvec[9+1] = 0.7071;
	    store.dFvec[9+2] = 0;
	    store.dFvec[9+3] = -0.7071;
	    store.dFvec[9+4] = 0.7071;
	    store.dFvec[9+5] = 0;
	    store.dFvec[9+6] = 0;
	    store.dFvec[9+7] = 0;
	    store.dFvec[9+8] = 1;
	    famu::acap::solve(store, store.dFvec);
	    famu::acap::updatedRdW(store);	

		cout<<"ACAP Energy: "<<famu::acap::energy(store, store.dFvec, store.boneDOFS)<<"-"<<famu::acap::fastEnergy(store,store.dFvec)<<endl;
		cout<<"ACAP fd Grad"<<endl;
		VectorXd dEdF = VectorXd::Zero(store.dFvec.size());
		famu::acap::fastGradient(store, dEdF);
		VectorXd fdgrad = famu::acap::fd_gradient(store);
		VectorXd grad = store.dRdW*dEdF;
		cout<<(fdgrad.transpose() - grad.segment<20>(0).transpose()).squaredNorm()<<endl;
		// cout<<"ACAP Hess:"<<endl;
		// MatrixXd testH = MatrixXd(store.acapHess);
		// MatrixXd fdH = famu::acap::fd_hessian(store);
	// // cout<<fdH<<endl<<endl<<endl;
	// // cout<<testH.block<20,20>(0,0)<<endl<<endl;
	// cout<<"Norm:"<<(testH.block<20,20>(0,0) - fdH).squaredNorm()<<endl;
	// cout<<"ACAP dxdF:"<<endl;
	// MatrixXd testJac = MatrixXd(store.JacdxdF);
	// MatrixXd fdJac = famu::acap::fd_dxdF(store);
	// cout<<testJac.block<15,15>(0,0)<<endl<<endl;
	// cout<<fdJac.block<15,15>(0,0)<<endl<<endl;
	// cout<<(testJac.block<15,15>(0,0) - fdJac.block<15,15>(0,0)).squaredNorm()<<endl;

	exit(0);


	cout<<"--- Write Meshes"<<endl;
		// double fx = 0;
		// int niters = 0;
		// niters = famu::newton_static_solve(store);

		// VectorXd y = store.Y*store.x;
		// Eigen::Map<Eigen::MatrixXd> newV(y.data(), store.V.cols(), store.V.rows());
		// igl::writeOBJ(outputfile+"/EMU"+to_string(store.T.rows())+"-Alpha:"+to_string(store.alpha_arap)+".obj", (newV.transpose()+store.V), store.F);
		// exit(0);

	cout<<"--- External Forces Hard Coded Contact Matrices"<<endl;
	    // famu::acap::adjointMethodExternalForces(store);
	

	std::cout<<"-----Display-------"<<std::endl;
    	igl::opengl::glfw::Viewer viewer;
    	int currentStep = 0;
    	viewer.callback_post_draw= [&](igl::opengl::glfw::Viewer & viewer) {
	    
	    // std::stringstream out_file;
	    // //render out current view
	    // // Allocate temporary buffers for 1280x800 image
	    // Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(1920,1280);
	    // Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(1920,1280);
	    // Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(1920,1280);
	    // Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(1920,1280);
	    
	    // // Draw the scene in the buffers
	    // viewer.core.draw_buffer(viewer.data(),false,R,G,B,A);
	    
	    // // Save it to a PNG
	    // out_file<<"out_"<<std::setfill('0') << std::setw(5) <<currentStep<<".png";
	    // igl::png::writePNG(R,G,B,A,out_file.str());
	    // currentStep += 1;
	    return false;
	};

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers){   
        std::cout<<"Key down, "<<key<<std::endl;
        // given data show it as colors on debug mesh
        const auto set_colors_from_data = [&](const Eigen::VectorXd & zz)
        {
          MatrixXd COLRS;
          igl::jet(zz, true, COLRS);
          viewer.data_list[debug_data_index].set_colors(COLRS);
        };
        // If debug mesh is currently visible, turn it off and turn on fancy
        // mesh and return true; otherwise return false.
        const auto hide_debug = [&]()->bool
        {
          if(viewer.data_list[debug_data_index].show_faces)
          {
            viewer.data_list[debug_data_index].show_faces = false;
            viewer.data_list[fancy_data_index].show_faces = true;
            std::cout<<"hiding debug..."<<std::endl;
            return true;
          }
          viewer.data_list[debug_data_index].show_faces = true;
          viewer.data_list[fancy_data_index].show_faces = false;
          return false;
        };
        switch(key)
        {
          case ' ':
          {
          
            //store.dFvec[9+0] = 0.7071;
            //store.dFvec[9+1] = 0.7071;
            //store.dFvec[9+2] = 0;
            //store.dFvec[9+3] = -0.7071;
            //store.dFvec[9+4] = 0.7071;
            //store.dFvec[9+5] = 0;
            //store.dFvec[9+6] = 0;
            //store.dFvec[9+7] = 0;
            //store.dFvec[9+8] = 1;
            //timer.start();
            // famu::acap::solve(store, store.dFvec);        	
            // timer.stop();
            // cout<<"+++Microsecs per solve: "<<timer.getElapsedTimeInMicroSec()<<endl;

            double fx = 0;
            int niters = 0;
            niters = famu::newton_static_solve(store);

            VectorXd y = store.Y*store.x;
        	Eigen::Map<Eigen::MatrixXd> newV(y.data(), store.V.cols(), store.V.rows());
            igl::writeOBJ(outputfile+"EMU"+to_string(store.T.rows())+".obj", (newV.transpose()+store.V), store.F);
            viewer.data_list[fancy_data_index].set_vertices((newV.transpose()+store.V));
            viewer.data_list[debug_data_index].set_vertices((newV.transpose()+store.V));
            return true;
          }
          case 'A':
          case 'a':
          {
            store.muscle_mag *= 1.5;
            famu::muscle::setupFastMuscles(store);
            famu::muscle::fastHessian(store, store.muscleHess, store.denseMuscleHess);
            return true;
          }
          case 'C':
          case 'c':
          {
            std::cout<<"C..."<<std::endl;
            if(!hide_debug())
            {
              std::cout<<" C..."<<std::endl;
              VectorXd zz = VectorXd::Ones(store.V.rows());
              // probably want to have this visualization update with each press
              // of space ' '... I'd consider having a little lambda that will
              // update the geometry _and_ any active visualizations. Might want
              // to have an enum or something to tell which debug visualization
              // is active.
              VectorXd y = store.Y*store.x;
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
              set_colors_from_data(zz);
            }
            return true;
          }
          case 'D':
          case 'd':
          {
            viewer.data_list[discontinuous_data_index].show_lines =
              !viewer.data_list[discontinuous_data_index].show_lines;
            if(viewer.data_list[discontinuous_data_index].show_lines)
            {
              famu::discontinuousV(store);
              viewer.data_list[discontinuous_data_index].set_vertices(store.discV);
            }
            return true;
          }
          case 'E':
          case 'e':
          {
            if(!hide_debug())
            {
              VectorXd zz = VectorXd::Ones(store.V.rows());
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
              set_colors_from_data(zz);
            }
            return true;
          }
          case 'S':
          case 's':
          {
            if(!hide_debug())
            {
              VectorXd zz = VectorXd::Ones(store.V.rows());
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
              set_colors_from_data(zz);
            }
            return true;
          }
          case 'V':
          case 'v':
          {
            if(!hide_debug())
            {
              VectorXd zz = VectorXd::Ones(store.V.rows());
              //Display tendon areas
              for(int i=0; i<store.T.rows(); i++){
                zz[store.T.row(i)[0]] = store.relativeStiffness[i];
                zz[store.T.row(i)[1]] = store.relativeStiffness[i];
                zz[store.T.row(i)[2]] = store.relativeStiffness[i];
                zz[store.T.row(i)[3]] = store.relativeStiffness[i];
              }
              set_colors_from_data(zz);
            }
            return true;
          }
        }


        // viewer.data().add_points( (store.ContactP1.transpose()*(store.Y*store.x + store.x0)).transpose() , Eigen::RowVector3d(1,0,0));
        // viewer.data().add_points( (store.ContactP2.transpose()*(store.Y*store.x + store.x0)).transpose() , Eigen::RowVector3d(0,1,0));
        viewer.data().points = Eigen::MatrixXd(0,6);
        viewer.data().lines = Eigen::MatrixXd(0,9);
        // for(int i=0; i<springs.size(); i++){
        // 	viewer.data().add_points(viewer.data_list[debug_data_index].V.row(springs[i].first), Eigen::RowVector3d(1,0,0));
        // 	viewer.data().add_points(viewer.data_list[debug_data_index].V.row(springs[i].second), Eigen::RowVector3d(1,0,0));
        // 	viewer.data().add_edges(viewer.data_list[debug_data_index].V.row(springs[i].first),viewer.data_list[debug_data_index].V.row(springs[i].second),Eigen::RowVector3d(1,0,0));
        // }
    

        //for(int i=0; i<store.mmov.size(); i++){
        //	viewer.data().add_points((newV.transpose().row(store.mmov[i]) + store.V.row(store.mmov[i])), Eigen::RowVector3d(0,1,0));
        //}
 
        // return false indicates that keystroke was not used and should be
        // passed on to viewer to handle
        return false;
    };

  fancy_data_index = viewer.selected_data_index;
  viewer.data_list[fancy_data_index].set_mesh(store.V, store.F);
  viewer.data_list[fancy_data_index].show_lines = false;
  viewer.data_list[fancy_data_index].invert_normals = true;
  viewer.data_list[fancy_data_index].set_face_based(false);
  viewer.append_mesh();
  debug_data_index = viewer.selected_data_index;
  viewer.data_list[debug_data_index].set_mesh(store.V, store.F);
  viewer.data_list[debug_data_index].show_faces = false;
  viewer.data_list[debug_data_index].invert_normals = true;
  viewer.data_list[debug_data_index].show_lines = false;
  viewer.append_mesh();
  discontinuous_data_index = viewer.selected_data_index;
  viewer.data_list[discontinuous_data_index].set_mesh(store.discV, store.discF);
  viewer.data_list[discontinuous_data_index].show_lines = true;
  viewer.data_list[discontinuous_data_index].show_faces = false;
  // set fancy rendered mesh to be selected.
  viewer.selected_data_index = fancy_data_index;


  // must be called before messing with shaders
  viewer.launch_init(true,false);
  std::cout<<R"(
fd_famu:
  C,c  Show continuous mesh's strain
  D,d  Toggle discontinous mesh wireframe
  E,e  Show ACAP energy (interpolated on the continuous mesh)
  S,s  Show discontinuous mesh's strain (interpolated on the continuous mesh)
  V,v  Tendon vs. muscle vis
)";

  // Send Young's modulus data in via color channel
  {
    Eigen::MatrixXd C(store.V.rows(),3);
    for(int i = 0;i<store.V.rows();i++)
    {
      if(store.elogVY(i) < 0.5*(60000 + 1.2e9))
      {
        C.row(i) = Eigen::RowVector3d(1,0,0);
      }else if(store.elogVY(i) < 0.5*(1.2e9 + 1.0e10))
      {
        C.row(i) = Eigen::RowVector3d(0.99,0.99,1);
      }else
      {
        C.row(i) = Eigen::RowVector3d(0.85,0.85,0.8);
      }
    }
    viewer.data_list[fancy_data_index].set_colors(store.elogVY.replicate(1,3));
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;
    // @Vismay, perhaps include this path in the json?
    igl::png::readPNG(store.jinput["material"],R,G,B,A);
    viewer.data_list[fancy_data_index].set_texture(R,G,B,A);
    viewer.data_list[fancy_data_index].show_texture = true;
    // must be called before messing with shaders
    viewer.data_list[fancy_data_index].meshgl.init();
    igl::opengl::destroy_shader_program(
      viewer.data_list[fancy_data_index].meshgl.shader_mesh);
    {
      std::string mesh_vertex_shader_string =
R"(#version 150
uniform mat4 view;
uniform mat4 proj;
uniform mat4 normal_matrix;
in vec3 position;
in vec3 normal;
// Color
in vec3 Kd;
// Young's modulus
out float elogY;
out vec3 normal_eye;

void main()
{
  normal_eye = normalize(vec3 (normal_matrix * vec4 (normal, 0.0)));
  gl_Position = proj * view * vec4(position, 1.0);
  elogY = Kd.r;
})";

      std::string mesh_fragment_shader_string =
R"(#version 150
in vec3 normal_eye;
// Young's modulus
in float elogY;
out vec4 outColor;
uniform sampler2D tex;
void main()
{
  vec2 uv = normalize(normal_eye).xy * vec2(0.5/3.0,0.5);
  float t_tendon = clamp( (elogY-4.7782)/(9.0792-4.7782) , 0.0 , 1.0);
  float t_bone =   clamp( (elogY-9.0092)/(10.000-9.0792) , 0.0 , 1.0);
  outColor = mix(
      texture(tex, uv + vec2(0.5/3.0,0.5)),
      texture(tex, uv + vec2(1.5/3.0,0.5)),
      t_tendon);
  outColor = mix( outColor,   texture(tex, uv + vec2(2.5/3.0,0.5)),t_bone);
  //outColor.a = 1.0;
})";

      igl::opengl::create_shader_program(
        mesh_vertex_shader_string,
        mesh_fragment_shader_string,
        {},
        viewer.data_list[fancy_data_index].meshgl.shader_mesh);
    }
  }



  viewer.core.is_animating = false;
  viewer.core.background_color = Eigen::Vector4f(1,1,1,0);

  viewer.launch_rendering(true);
  viewer.launch_shut();

}
