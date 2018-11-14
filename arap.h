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
	SparseMatrix<double> Exx;
	SparseLU<SparseMatrix<double>>  ARAPKKTSolver;
	VectorXd PAx0, UtPAx0;
public:
	Arap(Mesh& m){ 
		Exx = (m.P()*m.A()).transpose()*(m.P()*m.A());
		PAx0 = m.P()*m.A()*m.x0();
		UtPAx0 = m.GU().transpose()*PAx0;

		SparseMatrix<double> C = m.AB().transpose();
		SparseMatrix<double> KKTmat(Exx.rows()+C.rows(), Exx.rows()+C.rows());
		KKTmat.setZero();
		std::vector<Trip> ExxTrips = to_triplets(Exx);
		std::vector<Trip> CTrips = to_triplets(C);
		std::vector<Trip> CtTrips = to_triplets(m.AB());
		for(int i=0; i<CTrips.size(); i++){
			int row = CTrips[i].row();
			int col = CTrips[i].col();
			int val = CTrips[i].value();
			ExxTrips.push_back(Trip(row+Exx.rows(), col, val));

			ExxTrips.push_back(Trip(col, row+Exx.cols(), val));
		}
		ExxTrips.insert(ExxTrips.end(),CTrips.begin(), CTrips.end());
		KKTmat.setFromTriplets(ExxTrips.begin(), ExxTrips.end());
		
		
		ARAPKKTSolver.analyzePattern(KKTmat);
		ARAPKKTSolver.factorize(KKTmat);

	}

	inline double Energy(Mesh& m){
		VectorXd PAx = m.P()*m.A()*m.x();
		VectorXd FPAx0 = m.xbar();
		return 0.5*PAx.transpose()*FPAx0;
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
		print(" + ARAP minimize");
		
		VectorXd Ex0 = dEdx(m);
		for(int i=0; i< 100; i++){
			itT(m);
			itR(m);
			m.setGlobalF(true, false, false);
			
			VectorXd Ex = dEdx(m);
			if ((Ex - Ex0).norm()){
				print(" - ARAP minimize");
				return;
			}
			Ex0 = Ex;
		}
	}


	std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> & M){
		std::vector<Eigen::Triplet<double>> v;
		for(int i = 0; i < M.outerSize(); i++)
			for(typename Eigen::SparseMatrix<double>::InnerIterator it(M,i); it; ++it)
				v.emplace_back(it.row(),it.col(),it.value());
		return v;
	}
	template<class T>
    inline void print(T a){ std::cout<<a<<std::endl; }

};

#endif