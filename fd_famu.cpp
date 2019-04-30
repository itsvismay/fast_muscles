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
#include <json.hpp>
#include <LBFGS.h>

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
#include "famu/full_solver.h"


using namespace Eigen;
using namespace std;
using json = nlohmann::json;
using namespace LBFGSpp;
using Store = famu::Store;
json j_input;
double alpha_arap = 1e6;
typedef Eigen::Triplet<double> Trip;


int main(int argc, char *argv[])
{
	std::cout<<"-----Configs-------"<<std::endl;
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

	cout<<"---Record Mesh Setup Info"<<endl;
		cout<<"V size: "<<store.V.rows()<<endl;
		cout<<"T size: "<<store.T.rows()<<endl;
		cout<<"F size: "<<store.F.rows()<<endl;
		if(argc>1){
			j_input["number_modes"] =  stoi(argv[1]);
			j_input["number_rot_clusters"] =  stoi(argv[2]);
			j_input["number_skinning_handles"] =  stoi(argv[3]);
		}
		cout<<"NSH: "<<j_input["number_skinning_handles"]<<endl;
		cout<<"NRC: "<<j_input["number_rot_clusters"]<<endl;
		cout<<"MODES: "<<j_input["number_modes"]<<endl;
		std::string outputfile = j_input["output"];
		std::string namestring = to_string((int)j_input["number_modes"])+"modes"+to_string((int)j_input["number_rot_clusters"])+"clusters"+to_string((int)j_input["number_skinning_handles"])+"handles";
		igl::boundary_facets(store.T, store.F);

	cout<<"---Set Fixed Vertices"<<endl;
		store.mfix = famu::getMaxVerts(store.V, 1);
		store.mmov = {};//famu::getMinVerts(store.V, 1);
		// cout<<"If it fails here, make sure indexing is within bounds"<<endl;
	 //    std::set<int> fix_verts_set;
	 //    for(int ii=0; ii<fix_bones.size(); ii++){
	 //        cout<<fix_bones[ii]<<endl;
	 //        int bone_ind = bone_name_index_map[fix_bones[ii]];
	 //        fix_verts_set.insert(T.row(bone_tets[bone_ind][0])[0]);
	 //        fix_verts_set.insert(T.row(bone_tets[bone_ind][0])[1]);
	 //        fix_verts_set.insert(T.row(bone_tets[bone_ind][0])[2]);
	 //        fix_verts_set.insert(T.row(bone_tets[bone_ind][0])[3]);
	 //    }
	 //    std::vector<int> mfix, mmov;
	 //    mfix.assign(fix_verts_set.begin(), fix_verts_set.end());
	 //    std::sort (mfix.begin(), mfix.end());
	
	cout<<"---Set YM and Poissons"<<endl;
		store.eY = 9.1e9*VectorXd::Ones(store.T.rows());
		store.eP = 0.49*VectorXd::Ones(store.T.rows());
		store.muscle_mag = VectorXd::Zero(store.T.rows());
		for(int m=0; m<store.muscle_tets.size(); m++){
			for(int t=0; t<store.muscle_tets[m].size(); t++){
				store.eY[store.muscle_tets[m][t]] = 60000;
				store.muscle_mag[store.muscle_tets[m][t]] = j_input["muscle_starting_strength"];
			}
		}

    cout<<"---Set Vertex Constraint Matrices"<<endl;
		famu::vertex_bc(store.mmov, store.mfix, store.UnconstrainProjection, store.ConstrainProjection, store.V);

	cout<<"---Set Discontinuous Tet Centroid vector matrix"<<endl;
		famu::discontinuous_edge_vectors(store.D, store._D, store.T);

	cout<<"---Cont. to Discont. matrix"<<endl;
		famu::cont_to_discont_tets(store.S, store.T, store.V);

	cout<<"---Set Centroid Matrix"<<endl;
		famu::discontinuous_centroids_matrix(store.C, store.T);

	cout<<"---Set Disc T and V"<<endl;
		famu::setDiscontinuousMeshT(store.T, store.discT);
		store.discV.resize(4*store.T.rows(), 3);


	cout<<"---ACAP Solve KKT setup"<<endl;
		SparseMatrix<double> KKT_left;
		store.StDtDS = (store.D*store.S).transpose()*(store.D*store.S);
		SparseMatrix<double> CPt = store.ConstrainProjection.transpose();
		famu::construct_kkt_system_left(store.StDtDS, CPt, KKT_left);
		store.SPLU.analyzePattern(KKT_left);
		store.SPLU.factorize(KKT_left);

	cout<<"---Setup dFvec and dF"<<endl;
		store.dFvec = VectorXd::Zero(9*store.T.rows());
		for(int t=0; t<store.T.rows(); t++){
			store.dFvec[9*t + 0] = 1;
			store.dFvec[9*t + 4] = 1;
			store.dFvec[9*t + 8] = 1;
		}
		store.dF.resize(12*store.T.rows(), 12*store.T.rows());

	cout<<"---Setup continuous mesh"<<endl;
		store.x0.resize(3*store.V.rows());
		for(int i=0; i<store.V.rows(); i++){
			store.x0[3*i+0] = store.V(i,0); 
			store.x0[3*i+1] = store.V(i,1); 
			store.x0[3*i+2] = store.V(i,2);   
	    }
	    store.x = VectorXd::Zero(3*store.V.rows());
	    store.dx = VectorXd::Zero(3*store.V.rows());
    	for(int i=0; i<store.mmov.size(); i++){
			store.dx[3*store.mmov[i]+1] -= 1;
		}

	cout<<"---Setup Fast ACAP energy"<<endl;
		store.DSx0 = store.D*store.S*store.x0;
		famu::dFMatrix_Vector_Swap(store.DSx0_mat, store.DSx0);
		
		store.StDt_dF_DSx0 = store.S.transpose()*store.D.transpose()*store.DSx0_mat;
		store.x0tStDt_dF_DSx0 = store.DSx0.transpose()*store.DSx0_mat;
		store.x0tStDt_dF_dF_DSx0 = store.DSx0_mat.transpose()*store.DSx0_mat;

		SparseMatrix<double> mat_uvec;
		famu::muscle::setupFastMuscles(store, mat_uvec);
		store.fastMuscles = mat_uvec.transpose()*mat_uvec;

    cout<<"---Setup Solver"<<endl;
	    int DIM = store.dFvec.size();
	    famu::FullSolver fullsolver(DIM, &store);
	    LBFGSParam<double> param;
	    param.epsilon = 1e-1;
        param.delta = 1e-5;
        param.past = 5;
	    
	    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
	    LBFGSSolver<double> solver(param);


	std::cout<<"-----Display-------"<<std::endl;
    igl::opengl::glfw::Viewer viewer;
    igl::Timer timer;
    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers){   
 
        std::cout<<"Key down, "<<key<<std::endl;
        viewer.data().clear();

        
        if(key==' '){
			double fx = 0;
		    timer.start();
 			int niters = solver.minimize(fullsolver, store.dFvec, fx);
 			timer.stop();
 			cout<<"+++QS Step iterations: "<<niters<<", secs: "<<timer.getElapsedTimeInMicroSec()<<endl;
        	
        }

        if(key=='D'){
            
            // Draw disc mesh
            std::cout<<std::endl;
            famu::discontinuousV(store);
            for(int m=0; m<store.T.rows(); m++){
                int t= m;
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

        Eigen::Map<Eigen::MatrixXd> newV(store.x.data(), store.V.cols(), store.V.rows());
        viewer.data().set_mesh((newV.transpose()+store.V), store.F);
        igl::writeOBJ("ACAP_unred.obj", (newV.transpose()+store.V), store.F);
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