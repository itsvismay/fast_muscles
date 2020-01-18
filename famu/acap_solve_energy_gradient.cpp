#include "acap_solve_energy_gradient.h"
#include "store.h"
#include <iostream>
using namespace Eigen;
using Store = famu::Store;
#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include <igl/massmatrix.h>
#include <igl/consistent_penetrations.h>
#include<Eigen/LU>

using namespace std;
using Store = famu::Store;


std::vector<Eigen::Triplet<double>> to_Triplets(Eigen::SparseMatrix<double, Eigen::RowMajor> & M){
	std::vector<Eigen::Triplet<double>> v;
	for(int i = 0; i < M.outerSize(); i++){
		for(typename Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(M,i); it; ++it){	
			v.emplace_back(it.row(),it.col(),it.value());
		}
	}
	return v;
}

void famu::acap::mass_matrix_mesh(Eigen::SparseMatrix<double, Eigen::RowMajor> &M, Eigen::Ref<const Eigen::MatrixXi> T, double density, Eigen::Ref<const Eigen::VectorXd> v0) {
	M.setZero();
    std::vector<Eigen::Triplet<double>> Mtrips;
    for(int i=0; i<T.rows(); i++){
    	Eigen::Vector4i element = T.row(i);
  
    	Eigen::Matrix<double, 12,12> local_M;
    	famu::acap::mass_matrix_linear_tetrahedron(local_M, density, v0(i));
    	//assemble local into global
    	for(int ei=0; ei<element.size(); ei++){
    		for(int ej=0; ej<element.size(); ej++){
		    	Mtrips.push_back(Eigen::Triplet<double>(3*element(ei)+0, 3*element(ej)+0, local_M(3*ei, 3*ej)));
		  		Mtrips.push_back(Eigen::Triplet<double>(3*element(ei)+1, 3*element(ej)+1, local_M(3*ei+1, 3*ej+1)));
		  		Mtrips.push_back(Eigen::Triplet<double>(3*element(ei)+2, 3*element(ej)+2, local_M(3*ei+2, 3*ej+2)));
    		}
    	}


    }
    M.setFromTriplets(Mtrips.begin(), Mtrips.end());

}
void famu::acap::mass_matrix_linear_tetrahedron(Eigen::Matrix<double, 12,12> &M, double density, double volume) {
	
	//integral is here
	//https://www.sciencedirect.com/topics/engineering/consistent-mass-matrix
	M<<2,0,0,1,0,0,1,0,0,1,0,0,
	   0,2,0,0,1,0,0,1,0,0,1,0,
	   0,0,2,0,0,1,0,0,1,0,0,1,
	   1,0,0,2,0,0,1,0,0,1,0,0,
	   0,1,0,0,2,0,0,1,0,0,1,0,
	   0,0,1,0,0,2,0,0,1,0,0,1,
	   1,0,0,1,0,0,2,0,0,1,0,0,
	   0,1,0,0,1,0,0,2,0,0,1,0,
	   0,0,1,0,0,1,0,0,2,0,0,1,
	   1,0,0,1,0,0,1,0,0,2,0,0,
	   0,1,0,0,1,0,0,1,0,0,2,0,
	   0,0,1,0,0,1,0,0,1,0,0,2;
	M *= (density*volume)/20;
	   
}

void famu::acap::mesh_collisions(Store& store, Eigen::MatrixXd& DR){
	DR.setZero();
	VectorXd y = store.Y*store.x;
	Eigen::Map<Eigen::MatrixXd> newV(y.data(), store.V.cols(), store.V.rows());
	Eigen::MatrixXd newVt = newV.transpose()+ store.V;
	for(int i=0; i<store.contact_muscle_T_F.size(); i++){
		for(int j=0; j<store.contact_muscle_T_F.size(); j++){
			if(i!=j){
				MatrixXd DRi;
				igl::embree::consistent_penetrations(newVt,
												store.contact_muscle_T_F[i].first,
												newVt,
												store.contact_muscle_T_F[j].second,
												DRi);
				DR += DRi;
			}

		}

		for(int j=0; j<store.contact_bone_T_F.size(); j++){
			MatrixXd DRi;						
			igl::embree::consistent_penetrations(newVt,
											store.contact_muscle_T_F[i].first,
											newVt,
											store.contact_bone_T_F[j].second,
											DRi);
			DR += DRi;
		}
	}
	return;
}

double famu::acap::energy(Store& store){
	SparseMatrix<double, RowMajor> DS = store.D*store.S;
	double E1 =  0.5*(store.D*store.S*(store.x+store.x0) - store.dF*store.DSx0).squaredNorm();

	double E2 = 0.5*store.x.transpose()*store.StDtDS*store.x;
	double E3 = store.x0.transpose()*store.StDtDS*store.x;
	double E4 = 0.5*store.x0.transpose()*store.StDtDS*store.x0;
	double E5 = -store.x.transpose()*DS.transpose()*store.dF*DS*store.x0;
	double E6 = -store.x0.transpose()*DS.transpose()*store.dF*DS*store.x0;
	double E7 = 0.5*(store.dF*store.DSx0).transpose()*(store.dF*store.DSx0);
	double E8 = E2+E3+E4+E5+E6+E7;
	assert(fabs(E1 - E8)< 1e-6);
	return E1;
}

double famu::acap::fastEnergy(Store& store, VectorXd& dFvec){
	double E1 = 0.5*store.x0tStDtDSx0;
	double E2 = store.x0tStDtDSY.dot(store.x);
	store.acaptmp_sizex = store.YtStDtDSY*store.x;
	double E3 = 0.5*store.x.transpose()*store.acaptmp_sizex;
	double E4 = -store.x0tStDt_dF_DSx0.dot(dFvec);

	store.acaptmp_sizedFvec1 = store.YtStDt_dF_DSx0*dFvec;
	double E5 = -store.x.transpose()*store.acaptmp_sizedFvec1;

	store.acaptmp_sizedFvec2 = store.x0tStDt_dF_dF_DSx0*dFvec;
	double E6 = 0.5*dFvec.transpose()*store.acaptmp_sizedFvec2;
 	
 	// double E7 = store.x.transpose()*store.YtMg;
 	double E7 = dFvec.transpose()*(store.ContactForce+store.ConstantGravityForce);

 	double k = store.jinput["springk"];
 	double E8 = 0;//0.5*k*temp1.dot(temp1);
	
	double E9 = E1+E2+E3+E4+E5+E6+E7+E8;
	
	double aa = store.jinput["alpha_arap"];
	return E9*aa;
}

void famu::acap::fastGradient(Store& store, VectorXd& grad){
	grad = -store.x0tStDt_dF_DSx0;
	grad += -store.x.transpose()*store.YtStDt_dF_DSx0;
	grad += store.dFvec.transpose()*store.x0tStDt_dF_dF_DSx0;
	grad -= store.Bf.transpose()*store.lambda2;
	double aa = store.jinput["alpha_arap"];
	grad *= aa;
	grad += store.ContactForce;
	grad += store.ConstantGravityForce;
}

void famu::acap::fastHessian(Store& store, SparseMatrix<double, RowMajor>& hess, Eigen::MatrixXd& denseHess, bool include_dense){
	hess.setZero();
	hess = store.x0tStDt_dF_dF_DSx0; //PtZtZP


	if(include_dense){
		//else compute dense jacobian based hessian
		SparseMatrix<double, RowMajor> temp = store.YtStDt_dF_DSx0.transpose()*store.JacdxdF;
		hess -= temp;
	}
	double aa = store.jinput["alpha_arap"];
	hess *= aa;


}


void famu::acap::solve(Store& store, VectorXd& dFvec, bool solve1){
	if(solve1){
		store.acap_solve_rhs.setZero();
		store.acap_solve_rhs.head(store.x.size()) = store.YtStDt_dF_DSx0*dFvec - store.x0tStDtDSY;
		store.acap_solve_rhs.tail(store.BfI0.size()) = store.Bf*store.dFvec - store.BfI0;;
		store.acap_solve_result = store.ACAP_KKT_SPLU.solve(store.acap_solve_rhs);
		store.x = store.acap_solve_result.head(store.x.size());
		store.lambda2 = store.acap_solve_result.tail(store.BfI0.size());	
	
	}else{
		store.acap_solve_rhs.setZero();
		store.acap_solve_rhs.head(store.x.size()) = store.YtStDt_dF_DSx0*dFvec - store.x0tStDtDSY;
		store.acap_solve_rhs.tail(store.BfI0.size()) = store.Bf*store.dFvec - store.BfI0;;
		store.acap_solve_result = store.ACAP_KKT_SPLU2.solve(store.acap_solve_rhs);
		store.x = store.acap_solve_result.head(store.x.size());
		store.lambda2 = store.acap_solve_result.tail(store.BfI0.size());
	}

}

void famu::acap::external_forces(Store& store, VectorXd& f_ext, bool first){
	
		SparseMatrix<double> adjointP; 
		adjointP.resize(store.YtStDt_dF_DSx0.rows(), store.YtStDt_dF_DSx0.rows()+store.JointConstraints.rows()+store.Bf.rows());
		adjointP.setZero();
		for(int i=0; i<adjointP.rows(); i++){
			adjointP.coeffRef(i,i) = 1;
		}
		if(first){

		std::vector<Trip> YtStDt_dF_DSx0_trips = to_Triplets(store.YtStDt_dF_DSx0);
		std::vector<Trip> JointConstraints_trips = to_Triplets(store.JointConstraints);
		std::vector<Trip> Bf_trips = to_Triplets(store.Bf);
		std::vector<Trip> all_trips;
		for(int i=0; i<YtStDt_dF_DSx0_trips.size(); i++){
			int row = YtStDt_dF_DSx0_trips[i].row();
			int col = YtStDt_dF_DSx0_trips[i].col();
			double val = YtStDt_dF_DSx0_trips[i].value();
			all_trips.push_back(Trip(row, col, val));
		}


		for(int i=0; i<Bf_trips.size(); i++){
			int row = Bf_trips[i].row() + store.JointConstraints.rows() + store.YtStDt_dF_DSx0.rows();
			int col = Bf_trips[i].col();
			double val = Bf_trips[i].value();
			all_trips.push_back(Trip(row, col, val));
		}
		store.KKT_right.resize(store.YtStDt_dF_DSx0.rows() + store.JointConstraints.rows() + store.Bf.rows(), store.YtStDt_dF_DSx0.cols());
		store.KKT_right.setFromTriplets(all_trips.begin(), all_trips.end());

		// //SET full mass matrix
		//SET gravity g vector
		double density = 1;
		VectorXd mg = VectorXd::Zero(3*store.V.rows());
		double gravity = store.jinput["gravity"];
		for(int i=0; i<store.T.rows(); i++){
			for(int j=0; j<4; j++){
				mg[3*store.T.row(i)[j]+0] = 0;
				mg[3*store.T.row(i)[j]+1] = gravity;
				mg[3*store.T.row(i)[j]+2] = 0;
			}
		}

		SparseMatrix<double, Eigen::RowMajor> spMass(mg.size(), mg.size());
		famu::acap::mass_matrix_mesh(spMass, store.T, density, store.rest_tet_volume);
		 f_ext =  store.Y.transpose()*spMass*mg;
		
	}



	VectorXd rhs = adjointP.transpose()*f_ext;
	VectorXd newrhs = store.ACAP_KKT_SPLU.solve(rhs);

	if(first){
		store.ConstantGravityForce = store.KKT_right.transpose()*newrhs;
	}else{
		store.ContactForce = store.KKT_right.transpose()*newrhs;
	}


}

void famu::acap::setJacobian(Store& store, bool include_dense){
	if(!include_dense){
		return;
	}
	//Sparse jacobian
	MatrixXd result;
	if(result.rows()==0){
		//DENSE REDUCED JAC
		MatrixXd top = MatrixXd(store.YtStDt_dF_DSx0);
		MatrixXd zer = MatrixXd::Zero(store.JointConstraints.rows(), top.cols());
		MatrixXd bone_def = MatrixXd(store.Bf);
		MatrixXd KKT_right(top.rows() + zer.rows()+ bone_def.rows(), top.cols());
		KKT_right<<top,zer, bone_def;

		result = store.ACAP_KKT_SPLU.solve(KKT_right).topRows(top.rows());
		if(result!=result){
			cout<<"ACAP Jacobian result has nans"<<endl;
			exit(0);
		}
		
	}
	SparseMatrix<double, RowMajor> spRes = (result).sparseView();
	store.JacdxdF = spRes;

}