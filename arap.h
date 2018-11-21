#ifndef ARAP
#define ARAP

#include <igl/polar_svd.h>
#include "mesh.h"
#include<Eigen/SparseLU>


using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;

class Arap
{

protected:
	PartialPivLU<MatrixXd>  ARAPKKTSolver;
	VectorXd PAx0, UtPAx0;
	MatrixXd Exx, Erx, Err, Exs, Ers;

public:
	Arap(Mesh& m){
		int r_size = m.T().rows();
		int z_size = m.x0().size();
		int s_size = m.s().size();
		int t_size = m.T().rows();

		SparseMatrix<double> spExx = (m.P()*m.A()).transpose()*(m.P()*m.A());
		Exx = MatrixXd(spExx);
		Err = MatrixXd::Zero(r_size, r_size);
		Exs = MatrixXd::Zero(z_size, s_size);
		Ers = MatrixXd::Zero(r_size, s_size);

		PAx0 = m.P()*m.A()*m.x0();
		UtPAx0 = m.GU().transpose()*PAx0;

		MatrixXd C = MatrixXd(m.AB().transpose());

		MatrixXd col1(Exx.rows()+C.rows(), Exx.cols());
		MatrixXd col2(Exx.rows()+C.rows(), C.rows());
		// std::cout<<"Exx "<<Exx.rows()<<", "<<Exx.cols()<<std::endl;
		// std::cout<<"C "<<C.rows()<<", "<<C.cols()<<std::endl;
		// std::cout<<"col1 "<<col1.rows()<<", "<<col1.cols()<<std::endl;
		// std::cout<<"col2 "<<col2.rows()<<", "<<col2.cols()<<std::endl;

		col1<< Exx, C;
		col2<<C.transpose(), MatrixXd::Zero(C.rows(), C.rows());
		MatrixXd KKTmat(Exx.rows()+C.rows(), Exx.rows()+C.rows());
		KKTmat<<col1,col2;
		ARAPKKTSolver.compute(KKTmat);
			
	}

	inline double Energy(Mesh& m){
		VectorXd PAx = m.P()*m.A()*m.x();
		VectorXd FPAx0 = m.xbar();
		return 0.5*(PAx - FPAx0).squaredNorm();
	}

	inline double Energy(Mesh& m, VectorXd& z, SparseMatrix<double>& R, SparseMatrix<double>& S, SparseMatrix<double>& U){
		VectorXd PAx = m.P()*m.A()*z;
		VectorXd FPAx0 = R*U*S*U.transpose()*m.P()*m.A()*m.x0();
		return 0.5*(PAx - FPAx0).squaredNorm();
	}

	VectorXd FDGrad(Mesh& m){
		//DEDs = dEds + dEdx*dxds + dEdR*dRds

		//dEds
		VectorXd dEds = VectorXd::Zero(m.s().size());
		VectorXd& s = m.s();

		double E0 = Energy(m);
		double eps = 1e-5;
		for(int i=0; i<dEds.size(); i++){
			s[i] += eps;
			m.setGlobalF(false, true, false);
			double Ei = Energy(m);
			dEds[i] = (Ei - E0)/eps;
			s[i] -= eps;
		}
		m.setGlobalF(false, true, false);

		//dEdx
		//already implemented

		//dEdR
		

		return dEds;
	}

	void Jacobians(Mesh& m){
		Hessians(m, Exx, Erx, Err, Exs, Ers);
		// lhs_left
		//return dEds, dgds, drds
	}

	void Hessians(Mesh& m, MatrixXd& Exx, MatrixXd& Erx, MatrixXd& Err, MatrixXd& Exs, MatrixXd& Ers){
		//Exx is constant

		// Erx = 
	}

	void Gradients(Mesh& m){

		VectorXd Ex = dEdx(m);
	}

	VectorXd dEdx(Mesh& m){
		VectorXd PAx = m.P()*m.A()*m.x();
		VectorXd FPAx0 = m.xbar();
		VectorXd res = (m.P()*m.A()).transpose()*(PAx - FPAx0);
		return res;
	}

	void itT(Mesh& m){
		VectorXd FPAx0 = m.xbar();
		VectorXd deltaABtx = m.AB().transpose()*m.dx();
		VectorXd AtPtFPAx0 = (m.P()*m.A()).transpose()*FPAx0;
		VectorXd AtPtPAx0 = (m.P()*m.A()).transpose()*(m.P()*m.A()*m.x0());

		VectorXd gb = AtPtFPAx0 - AtPtPAx0;
		VectorXd gd(gb.size()+deltaABtx.size());
		gd<<gb,deltaABtx;
		VectorXd gu = ARAPKKTSolver.solve(gd).head(gb.size());
		m.dx(gu);
	}

	void itR(Mesh& m){
		VectorXd PAx = m.P()*m.A()*m.x();
		VectorXd USUtPAx0 = m.GU()*m.GS()*UtPAx0;
		MatrixXd ePAx(4,3);
		MatrixXd eUSUtPAx0(4,3);
		MatrixXd& mR =m.Rclusters();
		for (int i=0; i<m.T().rows(); i++){
			ePAx.row(0) = PAx.segment<3>(12*i);
			ePAx.row(1) = PAx.segment<3>(12*i+3);
			ePAx.row(2) = PAx.segment<3>(12*i+6);
			ePAx.row(3) = PAx.segment<3>(12*i+9);

			eUSUtPAx0.row(0) = USUtPAx0.segment<3>(12*i);
			eUSUtPAx0.row(1) = USUtPAx0.segment<3>(12*i+3);
			eUSUtPAx0.row(2) = USUtPAx0.segment<3>(12*i+6);
			eUSUtPAx0.row(3) = USUtPAx0.segment<3>(12*i+9);

			Matrix3d F = ePAx.transpose()*eUSUtPAx0;
			Matrix3d ri,ti,ui,vi;
     		Vector3d _;
      		igl::polar_svd(F,ri,ti,ui,_,vi);
      		mR.block<3,3>(3*i,0) = ri;
		}
	}

	void minimize(Mesh& m){
		// print("		+ ARAP minimize");
		
		VectorXd Ex0 = dEdx(m);
		for(int i=0; i< 100; i++){
			itT(m);
			itR(m);
			m.setGlobalF(true, false, false);
			
			VectorXd Ex = dEdx(m);
		
			if ((Ex - Ex0).norm()<1e-7){
				// std::cout<<"		- ARAP minimize "<<i<<std::endl;
				return;
			}
			Ex0 = Ex;
		}
	}


	void createKKTMatrix(SparseMatrix<double>& Full, SparseMatrix<double> & TL, SparseMatrix<double> & BL, SparseMatrix<double> & TR, SparseMatrix<double> & BR){
		// SparseMatrix<double> KKTmat(Exx.rows()+C.rows(), Exx.rows()+C.rows());
		// KKTmat.setZero();
		// std::vector<Trip> ExxTrips = to_triplets(Exx);
		// std::vector<Trip> CTrips = to_triplets(C);
		// std::vector<Trip> CtTrips = to_triplets(m.AB());
		// for(int i=0; i<CTrips.size(); i++){
		// 	int row = CTrips[i].row();
		// 	int col = CTrips[i].col();
		// 	int val = CTrips[i].value();
		// 	ExxTrips.push_back(Trip(row+Exx.rows(), col, val));

		// 	ExxTrips.push_back(Trip(col, row+Exx.cols(), val));
		// }
		// ExxTrips.insert(ExxTrips.end(),CTrips.begin(), CTrips.end());
		// KKTmat.setFromTriplets(ExxTrips.begin(), ExxTrips.end());
		
		// std::vector<Eigen::Triplet<double>> v;
		// for(int i = 0; i < M.outerSize(); i++)
		// 	for(typename Eigen::SparseMatrix<double>::InnerIterator it(M,i); it; ++it)
		// 		v.emplace_back(it.row(),it.col(),it.value());
		// return v;
	}
	template<class T>
    inline void print(T a){ std::cout<<a<<std::endl; }

};

#endif