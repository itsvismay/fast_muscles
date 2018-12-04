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
	PartialPivLU<MatrixXd>  aARAPKKTSolver;
	VectorXd aPAx0, aUtPAx0, aEr, aEs;
	MatrixXd aExx, aExr, aErr, aExs, aErs, aPAG;
	SparseMatrix<double> aPA;

	std::vector<SparseMatrix<double>> aDR;
	std::vector<vector<Trip>> aDS;

public:
	Arap(Mesh& m){
		int r_size = m.red_r().size();
		int z_size = m.red_x().size();
		int s_size = m.red_s().size();
		int t_size = m.T().rows();

		aExx = (m.P()*m.A()*m.G()).transpose()*(m.P()*m.A()*m.G());
		aErr = MatrixXd::Zero(r_size, r_size);
		aExs = MatrixXd::Zero(z_size, s_size);
		aErs = MatrixXd::Zero(r_size, s_size);
		aExr = MatrixXd::Zero(z_size, r_size);
		aEr = VectorXd::Zero(r_size);
		aEs = VectorXd::Zero(s_size);

		aPAx0 = m.P()*m.A()*m.x0();
		aUtPAx0 = m.GU().transpose()*aPAx0;
		aPAG = m.P()*m.A()*m.G();
		aPA = m.P()*m.A();

		MatrixXd CG = MatrixXd(m.AB().transpose())*m.G();

		MatrixXd KKTmat = MatrixXd::Zero(aExx.rows()+CG.rows(), aExx.rows()+CG.rows());
		KKTmat.block(0,0, aExx.rows(), aExx.cols()) = aExx;
		KKTmat.block(aExx.rows(), 0, CG.rows(), CG.cols()) = CG;
		KKTmat.block(0, aExx.cols(), CG.cols(), CG.rows()) = CG.transpose();
		aARAPKKTSolver.compute(KKTmat);

		setupRedSparseDRdr(m);//one time pre-processing
		setupRedSparseDSds(m);//one time pre-processing
		// setupFastErTerms(m);
		dEdr(m);
	}

	inline double Energy(Mesh& m){
		VectorXd PAx = aPA*m.x();
		VectorXd FPAx0 = m.GR()*m.GU()*m.GS()*m.GU().transpose()*aPA*m.x0();
		double En= 0.5*(PAx - FPAx0).squaredNorm();
		return En;
	}

	inline double Energy(Mesh& m, VectorXd& z, SparseMatrix<double>& R, SparseMatrix<double>& S, SparseMatrix<double>& U){
		VectorXd PAx = m.P()*m.A()*(m.G()*z + m.x0());
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
		VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;
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
		// Hessians(m, Exx, Erx, Err, Exs, Ers);
		// lhs_left
		//return dEds, dgds, drds
	}

	void Hessians(Mesh& m, MatrixXd& Exx, MatrixXd& Erx, MatrixXd& Err, MatrixXd& Exs, MatrixXd& Ers){
		//Exx is constant

		// Erx = 
	}

	void Gradients(Mesh& m){

		VectorXd Ex = dEdx(m);

		// fastEr(m);

	}

	VectorXd dEdx(Mesh& m){
		VectorXd PAx = m.P()*m.A()*m.x();
		VectorXd FPAx0 = m.xbar();
		VectorXd res = (aPA*m.G()).transpose()*(PAx - FPAx0);
		return res;
	}

	VectorXd dEdr(Mesh& m){
		VectorXd PAg = aPA*(m.G()*m.red_x() + m.x0());
		VectorXd FPAx0 = m.GF()*aPA*m.x0();
		VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;

		aEr.setZero();
		for(int i=0; i<aEr.size(); i++){
			auto v = to_triplets(aDR[i]);
			for(int k=0; k<v.size(); k++){
				aEr[i] += -1*PAg[v[k].row()]*USUtPAx0[v[k].col()]*v[k].value() + FPAx0[v[k].row()]*USUtPAx0[v[k].col()]*v[k].value();
			}
		}
		return aEr;
	}

	VectorXd dEds(Mesh& m){
		SparseMatrix<double> RU =m.GR()*m.GU(); 
		VectorXd UtRtRUSUtPAx0 = (RU).transpose()*RU*m.GS()*aUtPAx0;
		VectorXd UtRtPAx = (RU).transpose()*aPA*m.x();

		aEs.setZero();
		for(int i=0; i<aEs.size(); i++){
			std::vector<Trip> v = aDS[i];
			for(int k=0; k<aDS[i].size(); k++){
				aEs[i] += (UtRtRUSUtPAx0[v[k].row()]*aUtPAx0[v[k].col()] - UtRtPAx[v[k].row()]*aUtPAx0[v[k].col()])*v[k].value();
			}
		}
		return aEs;
	}

	MatrixXd& Exr(Mesh& m){
		VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;
		
		aExr.setZero();
		for(int i=0; i<aExr.rows(); i++){
			for(int j=0; j<aExr.cols(); j++){
				auto v = to_triplets(aDR[j]);
				for(int k=0; k<v.size(); k++){
					aExr(i,j) += -1*v[k].value()*(aPAG(v[k].row(), i)*USUtPAx0[v[k].col()]);
				}
			}
		}
		
		return aExr;

	}

	MatrixXd& Exs(Mesh& m){
		MatrixXd GtAtPtRU = aPAG.transpose()*(m.GR()*m.GU());

		aExs.setZero();
		for(int i=0; i<GtAtPtRU.rows(); i++){
			for(int j=0; j<aExs.cols(); j++){
				std::vector<Trip> v = aDS[j];
				for(int k=0; k<v.size(); k++){
					aExs(i,j) += -1*v[k].value()*(GtAtPtRU(i, v[k].row())*aUtPAx0[v[k].col()]);
				}
			}
		}
		return aExs;
	}

	MatrixXd& Err(Mesh& m){	
		VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;

		MatrixXd TEMP = MatrixXd::Zero(12*m.T().rows(), aErr.rows());

		aErr.setZero();
		for(int j=0; j<aErr.cols(); j++){
			auto v = to_triplets(aDR[j]);
			for(int k=0; k<v.size(); k++){
				TEMP(v[k].row(),j) += v[k].value()*(USUtPAx0[v[k].col()]);
			}
		}

		for(int i=0; i<aErr.rows(); i++){
			auto v = to_triplets(aDR[i]);
			for(int j=0; j<TEMP.cols(); j++){
				for(int k=0; k<v.size(); k++){
					aErr(i,j) += v[k].value()*(USUtPAx0[v[k].col()]*TEMP(v[k].row(), j));
				}	
			}
		}

		return aErr;
	}

	MatrixXd& Ers(Mesh& m){
		VectorXd PAg = aPA*m.x();
		VectorXd FPAx0 = m.GF()*aPA*m.x();
		SparseMatrix<double> RUt = (m.GR()*m.GU()).transpose();
		VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;

		print("temp1");
		MatrixXd TEMP1 = MatrixXd::Zero(12*m.T().rows(), aErs.rows());
		for(int i=0; i<TEMP1.rows(); i++){
			for(int j=0; j<TEMP1.cols(); j++){
				auto v = to_triplets(aDR[j]);
				for(int k=0; k<v.size(); k++){
					TEMP1(i,j) += v[k].value()*(-1*m.GU().coeff(v[k].row(), i)*PAg[v[k].col()]);
				}
			}
		}

		print("temp2");
		MatrixXd TEMP2 = MatrixXd::Zero(12*m.T().rows(), aErs.rows());
		for(int i=0; i<TEMP2.rows(); i++){
			for(int j=0; j<TEMP2.cols(); j++){
				auto v = to_triplets(aDR[j]);
				for(int k=0; k<v.size(); k++){
					TEMP2(i,j) += v[k].value()*(m.GU().coeff(v[k].row(), i)*FPAx0[v[k].col()] + RUt.coeff(i, v[k].row())*USUtPAx0[v[k].col()]);
				}
			}
		}
		aErs.setZero();

		print("mid1");
		for(int i=0; i<aErs.cols(); i++){
			std::vector<Trip> v = aDS[i];
			for(int j=0; j<TEMP1.cols(); j++){
				for(int k=0; k<v.size(); k++){
					aErs(j, i) += v[k].value()*(aUtPAx0[v[k].col()]*TEMP1(v[k].row(), j));
				}
			}
		}
		print("mid2");
		for(int i=0; i<aErs.cols(); i++){
			std::vector<Trip> v = aDS[i];
			for(int j=0; j<TEMP2.cols(); j++){
				for(int k=0; k<v.size(); k++){
					aErs(j, i) += v[k].value()*(aUtPAx0[v[k].col()]*TEMP2(v[k].row(), j));
				}
			}
		}



		print("ERS");
		print(aErs);

		return aErs;
	}

	void setupRedSparseDRdr(Mesh& m){
		Matrix3d r1; r1<<1,0,0,0,0,0,0,0,0;
		Matrix3d r2; r2<<0,1,0,0,0,0,0,0,0;
		Matrix3d r3; r3<<0,0,1,0,0,0,0,0,0;
		Matrix3d r4; r4<<0,0,0,1,0,0,0,0,0;
		Matrix3d r5; r5<<0,0,0,0,1,0,0,0,0;
		Matrix3d r6; r6<<0,0,0,0,0,1,0,0,0;
		Matrix3d r7; r7<<0,0,0,0,0,0,1,0,0;
		Matrix3d r8; r8<<0,0,0,0,0,0,0,1,0;
		Matrix3d r9; r9<<0,0,0,0,0,0,0,0,1;
		std::map<int, std::vector<int>>& c_e_map = m.r_cluster_elem_map();
		//iterator through rotation clusters
		for(int t=0; t<m.red_r().size()/9;t++){
			SparseMatrix<double> Ident(4*c_e_map[t].size(), 4*c_e_map[t].size());
            Ident.setIdentity();
            SparseMatrix<double>& B = m.RotBLOCK()[t];
			
			SparseMatrix<double> block1 = Eigen::kroneckerProduct(Ident, r1);
			aDR.push_back(B*block1*B.transpose());
			SparseMatrix<double> block2 = Eigen::kroneckerProduct(Ident, r2);
			aDR.push_back(B*block2*B.transpose());
			SparseMatrix<double> block3 = Eigen::kroneckerProduct(Ident, r3);
			aDR.push_back(B*block3*B.transpose());
			SparseMatrix<double> block4 = Eigen::kroneckerProduct(Ident, r4);
			aDR.push_back(B*block4*B.transpose());
			SparseMatrix<double> block5 = Eigen::kroneckerProduct(Ident, r5);
			aDR.push_back(B*block5*B.transpose());
			SparseMatrix<double> block6 = Eigen::kroneckerProduct(Ident, r6);
			aDR.push_back(B*block6*B.transpose());
			SparseMatrix<double> block7 = Eigen::kroneckerProduct(Ident, r7);
			aDR.push_back(B*block7*B.transpose());
			SparseMatrix<double> block8 = Eigen::kroneckerProduct(Ident, r8);
			aDR.push_back(B*block8*B.transpose());
			SparseMatrix<double> block9 = Eigen::kroneckerProduct(Ident, r9);
			aDR.push_back(B*block9*B.transpose());
		}
	}

	void setupRedSparseDSds(Mesh& m){
	
		for(int i=0; i<m.red_s().size()/6; i++){
			VectorXd sWx = m.sW().col(6*i+0); 

			vector<Trip> sx;
			vector<Trip> sy;
			vector<Trip> sz;
			vector<Trip> s01;
			vector<Trip> s02;
			vector<Trip> s12;
			for(int j=0; j<m.T().rows(); j++){
				sx.push_back(Trip(12*j+0, 12*j+0, sWx[6*i]));
				sx.push_back(Trip(12*j+3, 12*j+3, sWx[6*i]));
				sx.push_back(Trip(12*j+6, 12*j+6, sWx[6*i]));
				sx.push_back(Trip(12*j+9, 12*j+9, sWx[6*i]));

				sy.push_back(Trip(12*j+0+1, 12*j+0+1, sWx[6*i]));
				sy.push_back(Trip(12*j+3+1, 12*j+3+1, sWx[6*i]));
				sy.push_back(Trip(12*j+6+1, 12*j+6+1, sWx[6*i]));
				sy.push_back(Trip(12*j+9+1, 12*j+9+1, sWx[6*i]));

				sz.push_back(Trip(12*j+0+2, 12*j+0+2, sWx[6*i]));
				sz.push_back(Trip(12*j+3+2, 12*j+3+2, sWx[6*i]));
				sz.push_back(Trip(12*j+6+2, 12*j+6+2, sWx[6*i]));
				sz.push_back(Trip(12*j+9+2, 12*j+9+2, sWx[6*i]));

				s01.push_back(Trip(12*j+0, 12*j+0+1, sWx[6*i]));
				s01.push_back(Trip(12*j+3, 12*j+3+1, sWx[6*i]));
				s01.push_back(Trip(12*j+6, 12*j+6+1, sWx[6*i]));
				s01.push_back(Trip(12*j+9, 12*j+9+1, sWx[6*i]));
				s01.push_back(Trip(12*j+0+1, 12*j+0, sWx[6*i]));
				s01.push_back(Trip(12*j+3+1, 12*j+3, sWx[6*i]));
				s01.push_back(Trip(12*j+6+1, 12*j+6, sWx[6*i]));
				s01.push_back(Trip(12*j+9+1, 12*j+9, sWx[6*i]));
				
				s02.push_back(Trip(12*j+0, 12*j+0+2, sWx[6*i]));
				s02.push_back(Trip(12*j+3, 12*j+3+2, sWx[6*i]));
				s02.push_back(Trip(12*j+6, 12*j+6+2, sWx[6*i]));
				s02.push_back(Trip(12*j+9, 12*j+9+2, sWx[6*i]));
				s02.push_back(Trip(12*j+0+2, 12*j+0, sWx[6*i]));
				s02.push_back(Trip(12*j+3+2, 12*j+3, sWx[6*i]));
				s02.push_back(Trip(12*j+6+2, 12*j+6, sWx[6*i]));
				s02.push_back(Trip(12*j+9+2, 12*j+9, sWx[6*i]));

				s12.push_back(Trip(12*j+0+1, 12*j+0+2, sWx[6*i]));
				s12.push_back(Trip(12*j+3+1, 12*j+3+2, sWx[6*i]));
				s12.push_back(Trip(12*j+6+1, 12*j+6+2, sWx[6*i]));
				s12.push_back(Trip(12*j+9+1, 12*j+9+2, sWx[6*i]));
				s12.push_back(Trip(12*j+0+2, 12*j+0+1, sWx[6*i]));
				s12.push_back(Trip(12*j+3+2, 12*j+3+1, sWx[6*i]));
				s12.push_back(Trip(12*j+6+2, 12*j+6+1, sWx[6*i]));
				s12.push_back(Trip(12*j+9+2, 12*j+9+1, sWx[6*i]));
			}
			aDS.push_back(sx);
			aDS.push_back(sy);
			aDS.push_back(sz);
			aDS.push_back(s01);
			aDS.push_back(s02);
			aDS.push_back(s12);
		}
	}


	void itT(Mesh& m){
		VectorXd FPAx0 = m.xbar();
		VectorXd deltaABtx = m.AB().transpose()*m.dx();
		VectorXd GtAtPtFPAx0 = (m.P()*m.A()*m.G()).transpose()*FPAx0;
		VectorXd GtAtPtPAx0 = (m.P()*m.A()*m.G()).transpose()*(m.P()*m.A()*m.x0());

		VectorXd gb = GtAtPtFPAx0 - GtAtPtPAx0;
		VectorXd gd(gb.size()+deltaABtx.size());
		gd<<gb,deltaABtx;
		VectorXd gu = aARAPKKTSolver.solve(gd).head(gb.size());
		m.red_x(gu);
	}

	void itR(Mesh& m){
		VectorXd PAx = aPA*m.x();
		VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;
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

	std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> & M){
		std::vector<Eigen::Triplet<double>> v;
		for(int i = 0; i < M.outerSize(); i++){
			for(typename Eigen::SparseMatrix<double>::InnerIterator it(M,i); it; ++it){	
				v.emplace_back(it.row(),it.col(),it.value());
			}
		}
		return v;
	}

	inline MatrixXd& Exx(){ return aExx; }
	template<class T>
    inline void print(T a){ std::cout<<a<<std::endl; }

};

#endif