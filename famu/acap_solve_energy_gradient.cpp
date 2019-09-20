#include "acap_solve_energy_gradient.h"
#include "store.h"
#include <iostream>
using namespace Eigen;
using Store = famu::Store;
#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include<Eigen/LU>
#include <unsupported/Eigen/MatrixFunctions>
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


double famu::acap::energy(Store& store, VectorXd& dFvec, VectorXd& boneDOFS){
	VectorXd DSx0 = store.D*store.S*store.x0;
	VectorXd DSx = store.D*store.S*(store.Y*store.x + store.x0);
	VectorXd F = dFvec;
	double E = 0;
	//Energy for muscles
	for(int m=0; m<store.muscle_tets.size(); m++){
		for(int i=0; i<store.muscle_tets[m].size(); i++){
			int t = store.muscle_tets[m][i];
			int f_index = store.bone_or_muscle[t];

			Matrix3d F = Map<Matrix3d>(dFvec.segment<9>(9*f_index).data()).transpose();

			E += 0.5*(DSx.segment<3>(12*t + 0) - F*DSx0.segment<3>(12*t + 0)).squaredNorm();
			E += 0.5*(DSx.segment<3>(12*t + 3) - F*DSx0.segment<3>(12*t + 3)).squaredNorm();
			E += 0.5*(DSx.segment<3>(12*t + 6) - F*DSx0.segment<3>(12*t + 6)).squaredNorm();
			E += 0.5*(DSx.segment<3>(12*t + 9) - F*DSx0.segment<3>(12*t + 9)).squaredNorm();
		
		}
	}

	//Energy for bones (single rotation cluster)
	for(int m=0; m<store.bone_tets.size(); m++){
		int f_index = m;
		Matrix3d R0 = Map<Matrix3d>(dFvec.segment<9>(9*f_index).data()).transpose();
		double wX = boneDOFS(3*m + 0);
		double wY = boneDOFS(3*m + 1);
		double wZ = boneDOFS(3*m + 2);
		Matrix3d cross;
        cross<<0, -wZ, wY,
                wZ, 0, -wX,
                -wY, wX, 0;
        Matrix3d Rot = cross.exp();
		Matrix3d R = R0*Rot;
		Matrix3d Rt = R.transpose();
		F.segment<9>(9*m) = Map<Vector9d>(Rt.data());

		for(int i=0; i<store.bone_tets[m].size(); i++){
			int t = store.bone_tets[m][i];

			E += 0.5*(DSx.segment<3>(12*t + 0) - R*DSx0.segment<3>(12*t + 0)).squaredNorm();
			E += 0.5*(DSx.segment<3>(12*t + 3) - R*DSx0.segment<3>(12*t + 3)).squaredNorm();
			E += 0.5*(DSx.segment<3>(12*t + 6) - R*DSx0.segment<3>(12*t + 6)).squaredNorm();
			E += 0.5*(DSx.segment<3>(12*t + 9) - R*DSx0.segment<3>(12*t + 9)).squaredNorm();

		}

	}

	double E7 = store.lambda2.transpose()*store.Bx*store.x;
 	E7 -= store.lambda2.transpose()*store.Bf*F;
 	E7 += store.lambda2.transpose()*store.BfI0;
 	
	return E+E7;
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

 	double E7 = store.lambda2.transpose()*store.Bx*store.x;
 	E7 -= store.lambda2.transpose()*store.Bf*store.dFvec;
 	E7 += store.lambda2.transpose()*store.BfI0;
 	double k = store.jinput["springk"];
 	double E8 = 0;//0.5*k*temp1.dot(temp1);
	
	double E9 = E1+E2+E3+E4+E5+E6+E7+E8;
	
	double aa = store.jinput["alpha_arap"];
	return E9*aa;
}

void famu::acap::updatedRdW(Store& store){
	vector<Trip> dRdW_trips;
	store.dRdW.setZero();
	//SO3 BONES -> F(w) = R0 exp(J(w)) -> dE/dw = dE/dF * dF/dw
	for(int b =0; b < store.bone_tets.size(); b++){
		Eigen::Matrix3d R0 = Map<Eigen::Matrix3d>(store.dFvec.segment<9>(9*b).data()).transpose();
		
		Eigen::Matrix3d Jx = store.cross_prod_mat(1,0,0);
		Eigen::Matrix3d Jy = store.cross_prod_mat(0,1,0);
		Eigen::Matrix3d Jz = store.cross_prod_mat(0,0,1);

		Eigen::Matrix3d dRdw1 = R0*Jx;
		Eigen::Matrix3d dRdw2 = R0*Jy;
		Eigen::Matrix3d dRdw3 = R0*Jz;

		Eigen::Matrix3d dRdw1t = dRdw1.transpose();
		Eigen::Matrix3d dRdw2t = dRdw2.transpose();
		Eigen::Matrix3d dRdw3t = dRdw3.transpose();
		Eigen::Matrix<double,3, 9> dRdw;
		dRdw.row(0) = Map<Vector9d>(dRdw1t.data());
		dRdw.row(1) = Map<Vector9d>(dRdw2t.data());
		dRdw.row(2) = Map<Vector9d>(dRdw3t.data());
		
		for(int ii=0; ii<3; ii++){
			for(int jj=0; jj<9; jj++){
				dRdW_trips.push_back(Trip(3*b+ii,9*b+jj, dRdw(ii, jj)));
			}
		}
	}

	//fill in the rest of dRdW as mxm Id
	for(int t =0; t<store.dFvec.size()-9*store.bone_tets.size(); t++){
		//fill it in backwards, bottom right to top left.
		dRdW_trips.push_back(Trip( store.dRdW.rows() - t -1, store.dRdW.cols() - t -1, 1));
	}
	store.dRdW.setFromTriplets(dRdW_trips.begin(), dRdW_trips.end());
}

void famu::acap::fastGradient(Store& store, VectorXd& dEdF){
	dEdF = -store.x0tStDt_dF_DSx0;
	dEdF += -store.x.transpose()*store.YtStDt_dF_DSx0;
	dEdF += store.dFvec.transpose()*store.x0tStDt_dF_dF_DSx0;
	dEdF -= store.Bf.transpose()*store.lambda2;

	double aa = store.jinput["alpha_arap"];
	dEdF *= aa;
}

void famu::acap::fastHessian(Store& store, SparseMatrix<double, RowMajor>& hess, Eigen::MatrixXd& denseHess){
	hess.setZero();
	hess = store.alpha_arap*store.x0tStDt_dF_dF_DSx0; //PtZtZP


	SparseMatrix<double, RowMajor> temp = store.YtStDt_dF_DSx0.transpose()*store.JacdxdF;
	hess -= temp;

}

void famu::acap::setupWoodbury(Store& store){
	double aa = store.jinput["alpha_arap"];
	store.WoodB = -1*store.YtStDt_dF_DSx0.transpose()*store.G;
	store.WoodD = -1*store.WoodB.transpose();
	store.WoodB *= aa;

	store.InvC = store.eigenvalues.asDiagonal();
	store.WoodC = store.eigenvalues.asDiagonal().inverse();
	for(int i=0; i<store.dFvec.size()/9; i++){
		LDLT<Matrix9d> InvA;
		store.vecInvA.push_back(InvA);
	}

	// cout<<"WoodB: "<<store.WoodB.rows()<<", "<<store.WoodB.cols()<<endl;
	// cout<<"dRdW: "<<store.dRdW.rows()<<", "<<store.dRdW.cols()<<endl;
	store.dRdW_WoodB = store.dRdW*store.WoodB;
	store.dRdW_WoodD = store.WoodD*store.dRdW.transpose();

	
}

VectorXd famu::acap::fd_gradient(Store& store){
	VectorXd fake = VectorXd::Zero(20);
	double eps = 0.000001;
	for(int i=0; i<store.boneDOFS.size(); i++){
		store.boneDOFS[i] += 0.5*eps;
		double Eleft = energy(store, store.dFvec, store.boneDOFS);
		store.boneDOFS[i] -= 0.5*eps;

		store.boneDOFS[i] -= 0.5*eps;
		double Eright = energy(store, store.dFvec, store.boneDOFS);
		store.boneDOFS[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	int j= 0;
	for(int i=9*store.bone_tets.size(); i<9*store.bone_tets.size() + 20 - store.boneDOFS.size(); i++){

		store.dFvec[i] += 0.5*eps;
		double Eleft = energy(store, store.dFvec, store.boneDOFS);
		store.dFvec[i] -= 0.5*eps;

		store.dFvec[i] -= 0.5*eps;
		double Eright = energy(store, store.dFvec, store.boneDOFS);
		store.dFvec[i] += 0.5*eps;
		fake[store.boneDOFS.size()+j] = (Eleft - Eright)/eps;
		j+=1;
	}
	// for(int i=0; i<20; i++){

	// 	store.dFvec[i] += 0.5*eps;
	// 	double Eleft = energy(store, store.dFvec, store.boneDOFS);
	// 	store.dFvec[i] -= 0.5*eps;

	// 	store.dFvec[i] -= 0.5*eps;
	// 	double Eright = energy(store, store.dFvec, store.boneDOFS);
	// 	store.dFvec[i] += 0.5*eps;
	// 	fake[i] = (Eleft - Eright)/eps;
	// }
	return fake;
}

MatrixXd famu::acap::fd_hessian(Store& store){
	MatrixXd fake = MatrixXd::Zero(20,20);
	double eps = 1e-4;
	double E0 = energy(store, store.dFvec, store.boneDOFS);

	// for(int i=0; i<store.boneDOFS.size(); i++){
	// 	for(int j=0; j<store.boneDOFS.size(); j++){
	// 		store.boneDOFS[i] += eps;
	// 		store.boneDOFS[j] += eps;
	// 		double Eij = energy(store, store.dFvec, store.boneDOFS);
	// 		store.boneDOFS[i] -= eps;
	// 		store.boneDOFS[j] -= eps;

	// 		store.boneDOFS[i] += eps;
	// 		double Ei = energy(store, store.dFvec, store.boneDOFS);
	// 		store.boneDOFS[i] -= eps;

	// 		store.boneDOFS[j] += eps;
	// 		double Ej = energy(store, store.dFvec, store.boneDOFS);
	// 		store.boneDOFS[j] -= eps;

	// 		fake(i, j) = ((Eij - Ei - Ej + E0)/(eps*eps));
	// 	}
	// }
	// int jj= 0;
	// int ii =0;
	// for(int i=9*store.bone_tets.size(); i<9*store.bone_tets.size() + 20 - store.boneDOFS.size(); i++){
	// 	jj =0;
	// 	for(int j=9*store.bone_tets.size(); j<9*store.bone_tets.size() + 20 - store.boneDOFS.size(); j++){
	// 		store.dFvec[i] += eps;
	// 		store.dFvec[j] += eps;
	// 		double Eij = energy(store, store.dFvec, store.boneDOFS);
	// 		store.dFvec[i] -= eps;
	// 		store.dFvec[j] -= eps;

	// 		store.dFvec[i] += eps;
	// 		double Ei = energy(store, store.dFvec, store.boneDOFS);
	// 		store.dFvec[i] -=eps;

	// 		store.dFvec[j] += eps;
	// 		double Ej = energy(store, store.dFvec, store.boneDOFS);
	// 		store.dFvec[j] -=eps;

	// 		fake(store.boneDOFS.size()+ii, store.boneDOFS.size()+jj) = ((Eij - Ei - Ej + E0)/(eps*eps));

	// 		jj += 1;
	// 	}

	// 	ii += 1;
	// }

	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			store.dFvec[i] += eps;
			store.dFvec[j] += eps;
			double Eij = energy(store, store.dFvec, store.boneDOFS);
			store.dFvec[i] -= eps;
			store.dFvec[j] -= eps;

			store.dFvec[i] += eps;
			double Ei = energy(store, store.dFvec, store.boneDOFS);
			store.dFvec[i] -=eps;

			store.dFvec[j] += eps;
			double Ej = energy(store, store.dFvec, store.boneDOFS);
			store.dFvec[j] -=eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	return fake;
}

void famu::acap::solve(Store& store, VectorXd& dFvec){
	store.acap_solve_rhs.setZero();
	store.acap_solve_rhs.head(store.x.size()) = store.YtStDt_dF_DSx0*dFvec - store.x0tStDtDSY;
	store.acap_solve_rhs.tail(store.BfI0.size()) = store.Bf*store.dFvec - store.BfI0;

	store.acap_solve_result = store.ACAP_KKT_SPLU.solve(store.acap_solve_rhs);
	store.x = store.acap_solve_result.head(store.x.size());
	store.lambda2 = store.acap_solve_result.tail(store.BfI0.size());	

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


	VectorXd temp1 = store.ACAP_KKT_SPLU.solve(temp);
	store.ContactForce = KKT_right.transpose()*temp1;


	////HESSIAN
	SparseMatrix<double, RowMajor> dE2dxdx = store.Y.transpose()*store.ContactP.transpose()*store.ContactP*store.Y;
	cout<<dE2dxdx.rows()<<", "<<dE2dxdx.cols()<<endl;
	store.ContactHess = k*store.JacdxdF.transpose()*dE2dxdx*store.JacdxdF;

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