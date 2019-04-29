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
#include <LBFGS.h>
#include <Eigen/LU>
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
using namespace LBFGSpp;
json j_input;
double alpha_arap = 1e6;
typedef Eigen::Triplet<double> Trip;
struct Store{
	json jinput;
	std::string checkchange="";
	MatrixXd V, discV;
	MatrixXi T, discT, F;
	MatrixXd Uvec;
	std::vector<std::string> fix_bones = {};
	std::vector<VectorXi> bone_tets = {};
	std::vector<VectorXi> muscle_tets = {};
	std::map<std::string, int> bone_name_index_map;
	std::map<std::string, int> muscle_name_index_map;
	std::vector< std::pair<std::vector<std::string>, MatrixXd>> joint_bones_verts;
	VectorXd relativeStiffness;

	std::vector<int> fixverts, movverts;
	std::vector<int> mfix, mmov;

	SparseMatrix<double> dF;
	SparseMatrix<double> D, _D;
	SparseMatrix<double> C;
	SparseMatrix<double> S;
	SparseMatrix<double> ConstrainProjection, UnconstrainProjection;
	SparseMatrix<double> StDtDS;

	VectorXd dFvec;
	VectorXd x, dx, x0;

	SparseLU<SparseMatrix<double>> SPLU;
	
	//Fast Terms
	VectorXd DSx0;
	SparseMatrix<double> DSx0_mat;
	SparseMatrix<double> StDt_dF_DSx0;
	VectorXd x0tStDt_dF_DSx0;
	SparseMatrix<double> x0tStDt_dF_dF_DSx0;


};

void dFMatrix_Vector_Swap(SparseMatrix<double>& mat, VectorXd& vec){
    std::vector<Trip> mat_trips;

    for (int i=0; i<vec.size()/12; i++){

    	for(int j=0; j<4; j++){
	    	Vector3d seg = vec.segment<3>(12*i + 3*j);
	        mat_trips.push_back(Trip(12*i+3*j, 9*i+0, seg[0]));
	        mat_trips.push_back(Trip(12*i+3*j, 9*i+1, seg[0]));
	        mat_trips.push_back(Trip(12*i+3*j, 9*i+2, seg[0]));

	        mat_trips.push_back(Trip(12*i+3*j+1, 9*i+3, seg[1]));
	        mat_trips.push_back(Trip(12*i+3*j+1, 9*i+4, seg[1]));
	        mat_trips.push_back(Trip(12*i+3*j+1, 9*i+5, seg[1]));

	        mat_trips.push_back(Trip(12*i+3*j+2, 9*i+6, seg[2]));
	        mat_trips.push_back(Trip(12*i+3*j+2, 9*i+7, seg[2]));
	        mat_trips.push_back(Trip(12*i+3*j+2, 9*i+8, seg[2]));
	        
    	}
    }

    mat.resize(vec.size(), 9*(vec.size()/12));
    mat.setFromTriplets(mat_trips.begin(), mat_trips.end());

}

void setC(SparseMatrix<double>& mC, MatrixXi& mT){
        mC.resize(12*mT.rows(), 12*mT.rows());

        vector<Trip> triplets;
        triplets.reserve(3*16*mT.rows());

        for(int i=0; i<mT.rows(); i++){
            for(int j=0; j<3; j++){
                triplets.push_back(Trip(12*i+0+j, 12*i+0+j, 1.0/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+3+j, 1.0/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+6+j, 1.0/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+9+j, 1.0/4));

                triplets.push_back(Trip(12*i+3+j, 12*i+0+j, 1/4.0));
                triplets.push_back(Trip(12*i+3+j, 12*i+3+j, 1/4.0));
                triplets.push_back(Trip(12*i+3+j, 12*i+6+j, 1/4.0));
                triplets.push_back(Trip(12*i+3+j, 12*i+9+j, 1/4.0));

                triplets.push_back(Trip(12*i+6+j, 12*i+0+j, 1/4.0));
                triplets.push_back(Trip(12*i+6+j, 12*i+3+j, 1/4.0));
                triplets.push_back(Trip(12*i+6+j, 12*i+6+j, 1/4.0));
                triplets.push_back(Trip(12*i+6+j, 12*i+9+j, 1/4.0));

                triplets.push_back(Trip(12*i+9+j, 12*i+0+j, 1/4.0));
                triplets.push_back(Trip(12*i+9+j, 12*i+3+j, 1/4.0));
                triplets.push_back(Trip(12*i+9+j, 12*i+6+j, 1/4.0));
                triplets.push_back(Trip(12*i+9+j, 12*i+9+j, 1/4.0));
            }
        }   

        mC.setFromTriplets(triplets.begin(), triplets.end());
}

void setDiscontinuousMeshT(MatrixXi& mT, MatrixXi& discT){
    discT.resize(mT.rows(), 4);
    for(int i=0; i<mT.rows(); i++){
        discT(i, 0) = 4*i+0; 
        discT(i, 1) = 4*i+1; 
        discT(i, 2) = 4*i+2; 
        discT(i, 3) = 4*i+3;
    }
}

//m_D, mS, mC, dF, x0, x, mT, discV
void discontinuousV(Store& store){
    //discV.resize(4*mT.rows(), 3);
    VectorXd DAx = store._D*store.S*(store.x+store.x0);
    VectorXd CAx = store.C*store.S*(store.x+store.x0);
    VectorXd newx = store.dF*DAx+ CAx;

	for(int t =0; t<store.T.rows(); t++){
        store.discV(4*t+0, 0) = newx[12*t+0];
        store.discV(4*t+0, 1) = newx[12*t+1];
        store.discV(4*t+0, 2) = newx[12*t+2];
        store.discV(4*t+1, 0) = newx[12*t+3];
        store.discV(4*t+1, 1) = newx[12*t+4];
        store.discV(4*t+1, 2) = newx[12*t+5];
        store.discV(4*t+2, 0) = newx[12*t+6];
        store.discV(4*t+2, 1) = newx[12*t+7];
        store.discV(4*t+2, 2) = newx[12*t+8];
        store.discV(4*t+3, 0) = newx[12*t+9];
        store.discV(4*t+3, 1) = newx[12*t+10];
        store.discV(4*t+3, 2) = newx[12*t+11];
    }
}

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

double EnergyMuscle(VectorXd& dFvec){
	double MuscleEnergy = 0;
	for(int t=0; t<dFvec.size()/9; t++){
		Matrix3d F = Map<Matrix3d>(dFvec.segment<9>(9*t).data()).transpose();
		Vector3d y = Vector3d::UnitY();
		Vector3d z = F*y;
		double W = 0.5*1000000*(z.dot(z));
		MuscleEnergy += W;
	}
	return MuscleEnergy;
}

void GradMuscle(Store& store, VectorXd& dFvec){

}

double EnergyStableNH(VectorXd& dFvec){
	double stableNHEnergy = 0;

	double youngsModulus = 6e5;
	double poissonsRatio = 0.49;
	double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
	double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));

	for(int t=0; t<dFvec.size()/9; t++){
		Matrix3d F = Map<Matrix3d>(dFvec.segment<9>(9*t).data()).transpose();
		double I1 = (F.transpose()*F).trace();
		double J = F.determinant();
		double alpha = (1 + (C1/D1) - (C1/(D1*4)));
		double W = 0.5*C1*(I1 -3) + 0.5*D1*(J-alpha)*(J-alpha) - 0.5*C1*log(I1 + 1);
		stableNHEnergy += W;
	}

	return stableNHEnergy;
}

void GradStableNH(Store& store, VectorXd& dFvec){
	
}


double EnergyACAP(Store& store){
	SparseMatrix<double> DS = store.D*store.S;
	double E1 =  0.5*(store.D*store.S*(store.x+store.x0) - store.dF*store.DSx0).squaredNorm();

	double E2 = 0.5*store.x.transpose()*store.StDtDS*store.x;
	double E3 = store.x0.transpose()*store.StDtDS*store.x;
	double E4 = 0.5*store.x0.transpose()*store.StDtDS*store.x0;
	double E5 = -store.x.transpose()*DS.transpose()*store.dF*DS*store.x0;
	double E6 = -store.x0.transpose()*DS.transpose()*store.dF*DS*store.x0;
	double E7 = 0.5*(store.dF*store.DSx0).transpose()*(store.dF*store.DSx0);
	double E8 = E2+E3+E4+E5+E6+E7;
	// cout<<"E1: "<<E1<<endl;
	// cout<<"E8: "<<E8<<endl;
	// assert(fabs(E1-E8)<1e-5);
	return E1;
}

double fastEnergyACAP(Store& store){

	double E2 = 0.5*store.x.transpose()*store.StDtDS*store.x;
	double E3 = store.x0.transpose()*store.StDtDS*store.x;
	double E4 = 0.5*store.x0.transpose()*store.StDtDS*store.x0;
	double E5 = -store.x.transpose()*store.StDt_dF_DSx0*store.dFvec;
	double E6 = -store.x0tStDt_dF_DSx0.dot(store.dFvec);
	double E7 = 0.5*store.dFvec.transpose()*store.x0tStDt_dF_dF_DSx0*store.dFvec;
	double E9 = E2+E3+E4+E5+E6+E7;
	return E9;
}

void GradFastACAP(Store& store){

}

double Energy(Store& store){

	double EM = EnergyMuscle(store.dFvec);
	double ENH = EnergyStableNH(store.dFvec);
	double EACAP = alpha_arap*fastEnergyACAP(store);

	return EM + ENH + EACAP;
}

VectorXd fdGradEnergy(Store& store){
	VectorXd fake = VectorXd::Zero(store.dFvec.size());
	double eps = 0.00001;
	for(int i=0; i<store.dFvec.size(); i++){
		store.dFvec[i] += 0.5*eps;
		// setDF(store.dFvec, store.dF);
		double Eleft = Energy(store);
		store.dFvec[i] -= 0.5*eps;

		store.dFvec[i] -= 0.5*eps;
		// setDF(store.dFvec, store.dF);
		double Eright = Energy(store);
		store.dFvec[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	return fake;
}

void fdGradEnergyACAP(){

}

void fdGradEnergyStableNH(){
	
}

void fdGradEnergyMuscle(){
	
}

void ACAPSolve(Store& store){
	VectorXd constrains = store.ConstrainProjection.transpose()*store.dx;
	VectorXd top = store.StDt_dF_DSx0*store.dFvec - store.StDtDS*store.x0;
	VectorXd KKT_right(top.size() + constrains.size());
	KKT_right<<top, constrains;

	VectorXd result = store.SPLU.solve(KKT_right);
	store.x = result.head(top.size());
}



class FullSolver
{
private:
	Store* store;
	bool mtest;


public:

	 FullSolver(int n_, Store* istore, bool test = false){

	 	store = istore;
	 	mtest = test;

	}

	double operator()(const VectorXd& dFvec, VectorXd& graddFvec, bool computeGrad = true)
	{
		store->dFvec = dFvec;
		setDF(store->dFvec, store->dF);

		ACAPSolve(*store);
		
		double EM = EnergyMuscle(store->dFvec);
		double ENH = EnergyStableNH(store->dFvec);
		// double EACAP = alpha_arap*EnergyACAP(store->);
		double EACAP = alpha_arap*fastEnergyACAP(*store);
		// cout<<EACAP<<endl;
		// cout<<EfastACAP<<endl;
		// assert(fabs(EACAP-EfastACAP)<1);
		double E = EM + ENH + EACAP;

		if(computeGrad){
			VectorXd grad = fdGradEnergy(*store);


			if(graddFvec.size() != grad.size()){
				cout<<"whats wrong here"<<endl;
				cout<<graddFvec.size()<<endl;
				cout<<grad.size()<<endl;
				cout<<store->dFvec.size()<<endl;
				exit(0);
			}
			for(int i=0; i<store->dFvec.size(); i++){
				graddFvec[i] = grad[i];
			}
			cout<<"---BFGS Info---"<<endl;
			cout<<"	Total Grad N: "<<grad.norm()<<endl;
		}
		cout<<"	ACAP Energy: "<<EACAP<<endl;
		cout<<"	ENH Energy: "<<ENH<<endl;
		cout<<"	EM Energy: "<<EM<<endl;
		return E;
	}
};



int main(int argc, char *argv[])
{
	std::cout<<"-----Configs-------"<<std::endl;
		std::ifstream input_file("../input/input.json");
		input_file >> j_input;

		Store store;
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
		store.mfix = {0};//famu::getMaxVerts(store.V, 1);
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

    cout<<"---Set Vertex Constraint Matrices"<<endl;
		famu::vertex_bc(store.mmov, store.mfix, store.UnconstrainProjection, store.ConstrainProjection, store.V);

	cout<<"---Set Discontinuous Tet Centroid vector matrix"<<endl;
		famu::discontinuous_edge_vectors(store.D, store._D, store.T);

	cout<<"---Cont. to Discont. matrix"<<endl;
		famu::cont_to_discont_tets(store.S, store.T, store.V);

	cout<<"---Set Centroid Matrix"<<endl;
		setC(store.C, store.T);

	cout<<"---Set Disc T and V"<<endl;
		setDiscontinuousMeshT(store.T, store.discT);
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
		setDF(store.dFvec, store.dF);

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
		dFMatrix_Vector_Swap(store.DSx0_mat, store.DSx0);
		
		store.StDt_dF_DSx0 = store.S.transpose()*store.D.transpose()*store.DSx0_mat;
		store.x0tStDt_dF_DSx0 = store.DSx0.transpose()*store.DSx0_mat;
		store.x0tStDt_dF_dF_DSx0 = store.DSx0_mat.transpose()*store.DSx0_mat;


    cout<<"---Setup Solver"<<endl;
	    int DIM = store.dFvec.size();
	    FullSolver fullsolver(DIM, &store);
	    LBFGSParam<double> param;
	    param.epsilon = 1e-1;
        param.delta = 1e-5;
        param.past = 1;
	    
	    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
	    LBFGSSolver<double> solver(param);


	std::cout<<"-----Display-------"<<std::endl;
    igl::opengl::glfw::Viewer viewer;

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers){   
 
        std::cout<<"Key down, "<<key<<std::endl;
        viewer.data().clear();

        
        if(key==' '){
			// for(int i=0; i<store.mmov.size(); i++){
			// 	store.dx[3*store.mmov[i]+1] -= 1;
			// }

		    double fx = 0;
 			solver.minimize(fullsolver, store.dFvec, fx);

 			// ACAPSolve(store);
        	
        }

        if(key=='D'){
            
            // Draw disc mesh
            std::cout<<std::endl;
            discontinuousV(store);
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