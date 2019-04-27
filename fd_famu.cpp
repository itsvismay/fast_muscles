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
#include <igl/Timer.h>
#include<Eigen/LU>
#include <sstream>
#include <iomanip>


#include "famu/read_config_files.h"
#include "famu/vertex_bc.h"
#include "famu/discontinuous_edge_vectors.h"
#include "famu/cont_to_discont_tets.h"
#include "famu/construct_kkt_system.h"
#include "famu/get_min_max_verts.h"


using namespace Eigen;
using namespace std;
using json = nlohmann::json;
json j_input;
typedef Eigen::Triplet<double> Trip;

void setDF(VectorXd& dFvec, SparseMatrix<double>& dF){
	dF.setZero();
	std::vector<Trip> dF_trips;
	for(int t=0; t<dFvec.size()/9; t++){
		for(int j=0; j<4; j++){
            dF_trips.push_back(Trip(3*j+12*t + 0, 3*j+12*t + 0, dFvec(9*t+0)));
            dF_trips.push_back(Trip(3*j+12*t + 0, 3*j+12*t + 1, dFvec(9*t+1)));
            dF_trips.push_back(Trip(3*j+12*t + 0, 3*j+12*t + 2, dFvec(9*t+2)));
            dF_trips.push_back(Trip(3*j+12*t + 1, 3*j+12*t + 0, dFvec(9*t+3)));
            dF_trips.push_back(Trip(3*j+12*t + 1, 3*j+12*t + 1, dFvec(9*t+4)));
            dF_trips.push_back(Trip(3*j+12*t + 1, 3*j+12*t + 2, dFvec(9*t+5)));
            dF_trips.push_back(Trip(3*j+12*t + 2, 3*j+12*t + 0, dFvec(9*t+6)));
            dF_trips.push_back(Trip(3*j+12*t + 2, 3*j+12*t + 1, dFvec(9*t+7)));
            dF_trips.push_back(Trip(3*j+12*t + 2, 3*j+12*t + 2, dFvec(9*t+8)));
        }
	}
	dF.setFromTriplets(dF_trips.begin(), dF_trips.end());
}


double EnergyMuscle(){

}

double EnergyStableNH(){

}

double EnergyACAP(VectorXd& x, SparseMatrix<double>& dF, SparseMatrix<double>& D, SparseMatrix<double>& S, VectorXd& x0){
	return 0.5*(D*S*x - dF*D*S*x0).squaredNorm();
}

double Energy(VectorXd& x, SparseMatrix<double>& dF, SparseMatrix<double>& D, SparseMatrix<double>& S, VectorXd& x0){

	return EnergyACAP(x, dF, D, S, x0);
}

VectorXd fdGradEnergy(VectorXd& dFvec, VectorXd& x, SparseMatrix<double>& dF, SparseMatrix<double>& D, SparseMatrix<double>& S, VectorXd& x0){
	std::vector<double> grad;
	for(int i=0; i<dFvec.size(); i++){
		dFvec[i] += 0.0001;
		setDF(dFvec, dF);
		double Eleft = Energy(x, dF, D, S, x0);
		dFvec[i] -= 0.0002;
		setDF(dFvec, dF);
		double Eright = Energy(x, dF, D, S, x0);
		dFvec[i] += 0.0001;
		grad.push_back((Eleft - Eright)/(0.0001));
	}
	VectorXd mgrad;
	igl::list_to_matrix(grad, mgrad);
}

void fdGradEnergyACAP(){

}

void fdGradEnergyStableNH(){
	
}

void fdGradEnergyMuscle(){
	
}

void ACAPSolve(VectorXd& x, VectorXd& dx, SparseMatrix<double>& ConstrainProjection, SparseMatrix<double>& dF, SparseMatrix<double>& D, SparseMatrix<double>& S, VectorXd& x0, SparseLU<SparseMatrix<double>>& SPLU){
	VectorXd constrains = ConstrainProjection*dx + ConstrainProjection*x;
	VectorXd top = (D*S).transpose()*dF*D*S*x0;
	VectorXd KKT_right;
	KKT_right<<top, constrains;

	VectorXd result = SPLU.solve(KKT_right);
	x = result.head(top.size());
	cout<<x.transpose()<<endl;
}


int main(int argc, char *argv[])
{

	std::cout<<"-----Configs-------"<<std::endl;
	std::ifstream input_file("../input/input.json");
	input_file >> j_input;

	MatrixXd V;
	MatrixXi T;
	MatrixXi F;    
	MatrixXd Uvec;
	std::vector<int> mov = {};

	std::vector<std::string> fix_bones = {};
	std::vector<VectorXi> bone_tets = {};
	std::vector<VectorXi> muscle_tets = {};
	std::map<std::string, int> bone_name_index_map;
	std::map<std::string, int> muscle_name_index_map;
	std::vector< std::pair<std::vector<std::string>, MatrixXd>> joint_bones_verts;
	VectorXd relativeStiffness;
	famu::read_config_files(V, T, F, Uvec, bone_name_index_map, muscle_name_index_map, joint_bones_verts, bone_tets, muscle_tets, fix_bones, relativeStiffness, j_input);  

	cout<<"---Record Mesh Setup Info"<<endl;
	cout<<"V size: "<<V.rows()<<endl;
	cout<<"T size: "<<T.rows()<<endl;
	cout<<"F size: "<<F.rows()<<endl;
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
	igl::boundary_facets(T, F);

	cout<<"---Set Fixed Vertices"<<endl;
	std::vector<int> mfix = famu::getMaxVerts(V, 1);
	std::vector<int> mmov = famu::getMinVerts(V, 1);
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

    cout<<"---Set Vertex Constraint Matrices"<<endl;
	SparseMatrix<double> mFree, mConstrained;
	famu::vertex_bc(mmov, mfix, mFree, mConstrained, V);

	cout<<"---Set Discontinuous Tet Centroid vector matrix"<<endl;
	SparseMatrix<double> mD, m_D;
	famu::discontinuous_edge_vectors(mD, m_D, T);

	cout<<"---Cont. to Discont. matrix"<<endl;
	SparseMatrix<double> mS;
	famu::cont_to_discont_tets(mS, T, V);

	cout<<"---ACAP Solve KKT setup"<<endl;
	SparseMatrix<double> KKT_right;
	SparseMatrix<double> H = (mD*mS).transpose()*(mD*mS);
	SparseMatrix<double> ConstrainProjection = mFree.transpose();
	famu::construct_kkt_system_left(H, ConstrainProjection, KKT_right);
	SparseLU<SparseMatrix<double>>  ACAPSparseSolver;
	ACAPSparseSolver.analyzePattern(KKT_right);
	ACAPSparseSolver.factorize(KKT_right);

	cout<<"---Setup dFvec and dF"<<endl;
	VectorXd dFvec = VectorXd::Zero(9*T.rows());
	for(int t=0; t<T.rows(); t++){
		dFvec[9*t + 0] = 1;
		dFvec[9*t + 4] = 1;
		dFvec[9*t + 8] = 1;
	}
	SparseMatrix<double> dF(12*T.rows(), 12*T.rows());
	setDF(dFvec, dF);

	cout<<"---Setup continuous mesh"<<endl;
	VectorXd x0(3*V.rows());
	for(int i=0; i<V.rows(); i++){
		x0[3*i+0] = V(i,0); x0[3*i+1] = V(i,1); x0[3*i+2] = V(i,2);   
    }
    VectorXd x = x0;
    VectorXd dx = VectorXd::Zero(3*V.rows());
    for(int i=0; i<mmov.size(); i++){
    	dx[3*mmov[i]+1] -= 0.5;
    }





	std::cout<<"-----Display-------"<<std::endl;
    igl::opengl::glfw::Viewer viewer;

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers){   
 
        std::cout<<"Key down, "<<key<<std::endl;
        viewer.data().clear();

        Eigen::Map<Eigen::MatrixXd> newV(x.data(), V.cols(), V.rows());
        viewer.data().set_mesh(newV.transpose(), F);
        
        if(key==' '){
 			// ACAPSolve( x, dx, ConstrainProjection, dF, mD, mS, x0, ACAPSparseSolver);
 			cout<<"Energy ACAP"<<endl;
        	cout<<EnergyACAP( x, dF, mD, mS, x0)<<endl;
        }

        for(int i=0; i<mfix.size(); i++){
        	viewer.data().add_points(V.row(mfix[i]), Eigen::RowVector3d(1,0,0));
        }

        for(int i=0; i<mmov.size(); i++){
        	viewer.data().add_points(V.row(mmov[i]), Eigen::RowVector3d(1,1,0));
        }
        
        return false;
    };

	viewer.data().set_mesh(V,F);
    viewer.data().show_lines = false;
    viewer.data().invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data().face_based = true;
    viewer.core.background_color = Eigen::Vector4f(1,1,1,0);
    // viewer.data().set_colors(SETCOLORSMAT);

    viewer.launch();

}