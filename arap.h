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
		int s_size = m.red_s().size();
		int t_size = m.T().rows();

		Exx = (m.P()*m.A()*m.G()).transpose()*(m.P()*m.A()*m.G());
		print(Exx);
		Err = MatrixXd::Zero(r_size, r_size);
		Exs = MatrixXd::Zero(z_size, s_size);
		Ers = MatrixXd::Zero(r_size, s_size);

		PAx0 = m.P()*m.A()*m.x0();
		UtPAx0 = m.GU().transpose()*PAx0;

		MatrixXd CG = MatrixXd(m.AB().transpose())*m.G();
		print(CG);

		MatrixXd KKTmat = MatrixXd::Zero(Exx.rows()+CG.rows(), Exx.rows()+CG.rows());
		KKTmat.block(0,0, Exx.rows(), Exx.cols()) = Exx;
		KKTmat.block(Exx.rows(), 0, CG.rows(), CG.cols()) = CG;
		KKTmat.block(0, Exx.cols(), CG.cols(), CG.rows()) = CG.transpose();
		ARAPKKTSolver.compute(KKTmat);
			
	}

	inline double Energy(Mesh& m){
		VectorXd PAx = m.P()*m.A()*m.x();
		VectorXd FPAx0 = m.xbar();
		return 0.5*(PAx - FPAx0).squaredNorm();
	}

	inline double Energy(Mesh& m, VectorXd& z, SparseMatrix<double>& R, SparseMatrix<double>& S, SparseMatrix<double>& U){
		VectorXd PAx = m.P()*m.A()*m.G()*z;
		VectorXd FPAx0 = R*U*S*U.transpose()*m.P()*m.A()*m.x0();
		return 0.5*(PAx - FPAx0).squaredNorm();
	}

	VectorXd FDGrad(Mesh& m){
		//DEDs = dEds + dEdx*dxds + dEdR*dRds

		//dEds
		VectorXd dEds = VectorXd::Zero(m.red_s().size());
		VectorXd& s = m.red_s();

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
		VectorXd dEds1 = VectorXd::Zero(m.red_s().size());
		VectorXd Ex = dEdx(m);

		//dEdR
		VectorXd dEds2 = VectorXd::Zero(m.red_s().size());
		VectorXd negPAx = -1*m.P()*m.A()*m.x();
		VectorXd USUtPAx0 = m.GU()*m.GS()*UtPAx0;
		MatrixXd dEdR = negPAx*USUtPAx0.transpose();
		
		//dxds dRds
		VectorXd z0 = m.x();
		MatrixXd dxds(m.red_s().size(), z0.size());
		vector<SparseMatrix<double>> dRds_r3;

		for(int i=0; i<m.red_s().size(); i++){
			// dxds_right.setZero();
			// dxds_left.setZero();

			m.red_s()[i] += 0.5*eps;
			m.setGlobalF(false, true, false);
			minimize(m);
			VectorXd dxds_left = m.dx();
			SparseMatrix<double> dRds_left = m.GR();
			m.red_s()[i] -= 0.5*eps;
			m.setGlobalF(false, true, false);
			minimize(m);
			
			m.red_s()[i] -= 0.5*eps;
			m.setGlobalF(false, true, false);
			minimize(m);
			VectorXd dxds_right = m.dx();
			SparseMatrix<double> dRds_right = m.GR();
			m.red_s()[i] += 0.5*eps;
			m.setGlobalF(false, true, false);
			minimize(m);

			dxds.row(i) = (dxds_left - dxds_right)/eps;
			SparseMatrix<double> dRdsi =(dRds_left - dRds_right)/eps; 
			dRds_r3.push_back(dRdsi);
			dEds2[i] = (dRdsi.cwiseProduct(dEdR)).sum();
		}
		dEds1 = dxds*Ex;
		return dEds + dEds1 + dEds2;
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

	void dSds(){

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
		VectorXd GtAtPtFPAx0 = (m.P()*m.A()*m.G()).transpose()*FPAx0;
		VectorXd GtAtPtPAx0 = (m.P()*m.A()*m.G()).transpose()*(m.P()*m.A()*m.x0());

		VectorXd gb = GtAtPtFPAx0 - GtAtPtPAx0;
		VectorXd gd(gb.size()+deltaABtx.size());
		gd<<gb,deltaABtx;
		VectorXd gu = ARAPKKTSolver.solve(gd).head(gb.size());
		m.red_x(gu);
	}

	void itR(Mesh& m){
		VectorXd PAx = m.P()*m.A()*m.x();
		VectorXd USUtPAx0 = m.GU()*m.GS()*UtPAx0;
		VectorXd& mr =m.red_r();
		std::map<int, std::vector<int>>& c_e_map = m.r_cluster_elem_map();
		for (int i=0; i<mr.size()/9; i++){
			std::vector<int> cluster_elem = c_e_map[i];
			MatrixXd ePAx(4*cluster_elem.size(),3);
			MatrixXd eUSUtPAx0(4*cluster_elem.size(),3);
			for(int c=0; c<cluster_elem.size(); c++){
				ePAx.row(4*c+0) = PAx.segment<3>(12*cluster_elem[c]);
				ePAx.row(4*c+1) = PAx.segment<3>(12*cluster_elem[c]+3);
				ePAx.row(4*c+2) = PAx.segment<3>(12*cluster_elem[c]+6);
				ePAx.row(4*c+3) = PAx.segment<3>(12*cluster_elem[c]+9);
				
				eUSUtPAx0.row(4*c+0) = USUtPAx0.segment<3>(12*cluster_elem[c]);
				eUSUtPAx0.row(4*c+1) = USUtPAx0.segment<3>(12*cluster_elem[c]+3);
				eUSUtPAx0.row(4*c+2) = USUtPAx0.segment<3>(12*cluster_elem[c]+6);
				eUSUtPAx0.row(4*c+3) = USUtPAx0.segment<3>(12*cluster_elem[c]+9);
			}


			Matrix3d F = ePAx.transpose()*eUSUtPAx0;
			Matrix3d ri,ti,ui,vi;
     		Vector3d _;
      		igl::polar_svd(F,ri,ti,ui,_,vi);
      		mr[9*i+0] = ri(0,0);
      		mr[9*i+1] = ri(0,1);
      		mr[9*i+2] = ri(0,2);
      		mr[9*i+3] = ri(1,0);
      		mr[9*i+4] = ri(1,1);
      		mr[9*i+5] = ri(1,2);
      		mr[9*i+6] = ri(2,0);
      		mr[9*i+7] = ri(2,1);
      		mr[9*i+8] = ri(2,2);
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


	template<class T>
    inline void print(T a){ std::cout<<a<<std::endl; }

};

#endif