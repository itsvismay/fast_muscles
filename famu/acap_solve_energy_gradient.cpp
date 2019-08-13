#include "acap_solve_energy_gradient.h"
#include "store.h"
#include <iostream>
using namespace Eigen;
using Store = famu::Store;
#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
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
 	
 	double E7 = 0;

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
	// grad += store.ContactForce;
}

void famu::acap::fastHessian(Store& store, SparseMatrix<double, RowMajor>& hess, Eigen::MatrixXd& denseHess){
	hess.setZero();
	hess = store.jinput["alpha_arap"]*store.x0tStDt_dF_dF_DSx0; //PtZtZP


	if(store.jinput["woodbury"]){
		//if woodbury, store PtZtZP as dense block diag hessian
	
	}else{
		//else compute dense jacobian based hessian
		SparseMatrix<double, RowMajor> temp = store.YtStDt_dF_DSx0.transpose()*store.JacdxdF;
		hess -= temp;
	}

}

VectorXd famu::acap::fd_gradient(Store& store){
	VectorXd fake = VectorXd::Zero(store.dFvec.size());
	VectorXd dFvec = store.dFvec;
	double eps = 0.00001;
	for(int i=0; i<dFvec.size(); i++){
		dFvec[i] += 0.5*eps;
		double Eleft = fastEnergy(store, dFvec);
		dFvec[i] -= 0.5*eps;

		dFvec[i] -= 0.5*eps;
		double Eright = fastEnergy(store, dFvec);
		dFvec[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	return fake;
}

MatrixXd famu::acap::fd_hessian(Store& store){
	MatrixXd fake = MatrixXd::Zero(store.dFvec.size(), store.dFvec.size());
	VectorXd dFvec = store.dFvec;
	double eps = 1e-3;
	double E0 = fastEnergy(store, dFvec);
	for(int i=0; i<11; i++){
		for(int j=0; j<11; j++){
			dFvec[i] += eps;
			dFvec[j] += eps;
			double Eij = fastEnergy(store, dFvec);
			dFvec[i] -= eps;
			dFvec[j] -= eps;

			dFvec[i] += eps;
			double Ei = fastEnergy(store, dFvec);
			dFvec[i] -=eps;

			dFvec[j] += eps;
			double Ej = fastEnergy(store, dFvec);
			dFvec[j] -=eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	return fake;
}

void famu::acap::solve(Store& store, VectorXd& dFvec){
	store.acap_solve_rhs.setZero();
	store.acap_solve_rhs.head(store.x.size()) = store.YtStDt_dF_DSx0*dFvec - store.x0tStDtDSY;
	// store.acap_solve_rhs.tail(store.BfI0.size()) = store.Bf*store.dFvec - store.BfI0;;

	store.acap_solve_result = store.ACAP_KKT_SPLU.solve(store.acap_solve_rhs);
	store.x = store.acap_solve_result.head(store.x.size());
	// store.lambda2 = store.acap_solve_result.tail(store.BfI0.size());	

}

void famu::acap::adjointMethodExternalForces(Store& store){

	SparseMatrix<double> adjointP; 
	adjointP.resize(store.YtStDt_dF_DSx0.rows(), store.YtStDt_dF_DSx0.rows()+store.JointConstraints.rows()+store.Bf.rows());
	adjointP.setZero();
	for(int i=0; i<adjointP.rows(); i++){
		adjointP.coeffRef(i,i) = 1;
	}

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

	for(int i=0; i<JointConstraints_trips.size(); i++){
		int row = JointConstraints_trips[i].row() + store.YtStDt_dF_DSx0.rows();
		int col = JointConstraints_trips[i].col();
		double val = JointConstraints_trips[i].value();
		all_trips.push_back(Trip(row, col, val));
	}

	for(int i=0; i<Bf_trips.size(); i++){
		int row = Bf_trips[i].row() + store.JointConstraints.rows() + store.YtStDt_dF_DSx0.rows();
		int col = Bf_trips[i].col();
		double val = Bf_trips[i].value();
		all_trips.push_back(Trip(row, col, val));
	}
	SparseMatrix<double> KKT_right(store.YtStDt_dF_DSx0.rows() + store.JointConstraints.rows() + store.Bf.rows(), store.YtStDt_dF_DSx0.cols());
	KKT_right.setFromTriplets(all_trips.begin(), all_trips.end());

	VectorXd t0 = store.Y*store.x + store.x0;
	double k = store.jinput["springk"];
	VectorXd temp = k*adjointP.transpose()*(store.Y.transpose()*store.ContactP.transpose())*(store.ContactP*t0);


	// VectorXd temp1 = store.ACAP_KKT_SPLU.solve(temp);
	// store.ContactForce = KKT_right.transpose()*temp1;


	////HESSIAN
	// SparseMatrix<double, RowMajor> dE2dxdx = store.Y.transpose()*store.ContactP.transpose()*store.ContactP*store.Y;
	// cout<<dE2dxdx.rows()<<", "<<dE2dxdx.cols()<<endl;
	// store.ContactHess = k*store.JacdxdF.transpose()*dE2dxdx*store.JacdxdF;

}

void famu::acap::setJacobian(Store& store){

	//Sparse jacobian
	MatrixXd result;
	igl::readDMAT("jacKKT.dmat", result);

	if(result.rows()==0){
		//DENSE REDUCED JAC
		MatrixXd top = MatrixXd(store.YtStDt_dF_DSx0);
		MatrixXd zer = MatrixXd(store.JointConstraints.rows(), top.cols());
		MatrixXd bone_def = MatrixXd(store.Bf);
		MatrixXd KKT_right(top.rows() + zer.rows()+ bone_def.rows(), top.cols());
		KKT_right<<top,zer, bone_def;

		result = store.ACAP_KKT_SPLU.solve(KKT_right).topRows(top.rows());
		if(result!=result){
			cout<<"ACAP Jacobian result has nans"<<endl;
			exit(0);
		}
		
		igl::writeDMAT("jacKKT.dmat", result);

	}
	SparseMatrix<double, RowMajor> spRes = (result).sparseView();
	store.JacdxdF = spRes;
	cout<<"jac dims: "<<store.JacdxdF.rows()<<", "<<store.JacdxdF.cols()<<endl;

}