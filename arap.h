#ifndef ARAP
#define ARAP

#include <igl/polar_svd.h>
#include "mesh.h"
#include<Eigen/SparseLU>
#include <iostream>
#include <string>
// #include <UtilitiesEigen.h>


using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;

class Arap
{

protected:
	FullPivLU<MatrixXd>  aARAPKKTSolver;
	VectorXd aPAx0, aUtPAx0, aEr, aEs, aEx, aDEDs;
	MatrixXd aExx, aExr, aErr, aExs, aErs, aPAG;
	SparseMatrix<double> aPA;

	std::vector<vector<Trip>> aDS, aDR, aDDR;

public:
	Arap(Mesh& m){
		int r_size = m.red_w().size();
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

		setupRedSparseDSds(m);//one time pre-processing
		setupRedSparseDRdr(m);
		setupRedSparseDDRdrdr(m);
		// setupFastErTerms(m);
		dEdr(m);
	}

	double Energy(Mesh& m){
		VectorXd PAx = aPA*(m.G()*m.red_x() + m.x0());
		VectorXd FPAx0 = m.GR()*m.GU()*m.GS()*m.GU().transpose()*aPA*m.x0();
		double En= 0.5*(PAx - FPAx0).squaredNorm();
		return En;
	}

	double Energy(Mesh& m, VectorXd& z, SparseMatrix<double>& R, SparseMatrix<double>& S, SparseMatrix<double>& U){
		VectorXd PAx = m.P()*m.A()*(m.G()*z + m.x0());
		VectorXd FPAx0 = R*U*S*U.transpose()*m.P()*m.A()*m.x0();
		return 0.5*(PAx - FPAx0).squaredNorm();
	}

	VectorXd Jacobians(Mesh& m){
		setupRedSparseDRdr(m);
		setupRedSparseDDRdrdr(m);

		Hessians(m);
		
		MatrixXd lhs_left(aExx.rows()+aExr.cols(), aExx.cols());
		lhs_left<<aExx, aExr.transpose();

		MatrixXd lhs_right(aExr.rows() + aErr.rows() , aExr.cols());
		lhs_right<<aExr, aErr; 

		MatrixXd rhs(aExs.rows()+aErs.rows(), aExs.cols());
		rhs<<-1*aExs, -1*aErs;
		
		MatrixXd CG = MatrixXd(m.AB().transpose())*m.G();

		MatrixXd col1(lhs_left.rows()+CG.rows(), lhs_left.cols());
		col1<<lhs_left, CG;

		MatrixXd col2(lhs_right.rows()+CG.rows(), lhs_right.cols());
		col2<<lhs_right,MatrixXd::Zero(CG.rows(), lhs_right.cols());

		MatrixXd col3(CG.cols()+CG.rows()+aErr.rows(), CG.rows());
		col3<<CG.transpose(),MatrixXd::Zero(CG.rows()+aErr.rows(), CG.rows());

		MatrixXd KKT_constrains(rhs.rows() + CG.rows(), rhs.cols());
		KKT_constrains<<rhs,MatrixXd::Zero(CG.rows(), rhs.cols());

		MatrixXd JacKKT(col1.rows(), col1.rows());
		JacKKT<<col1, col2, col3;

		Gradients(m);

		// std::ofstream ExxFile("Exx.mat");
		// if (ExxFile.is_open())
		// {
		// 	ExxFile << aExx;
		// }
		// ExxFile.close();

		// std::ofstream ExrFile("Exr.mat");
		// if (ExrFile.is_open())
		// {
		// 	ExrFile << aExr;
		// }
		// ExrFile.close();

		// std::ofstream ErrFile("Err.mat");
		// if (ErrFile.is_open())
		// {
		// 	ErrFile << aErr;
		// }
		// ErrFile.close();

		// std::ofstream ExsFile("Exs.mat");
		// if (ExsFile.is_open())
		// {
		// 	ExsFile << aExs;
		// }
		// ExsFile.close();

		// std::ofstream ErsFile("Ers.mat");
		// if (ErsFile.is_open())
		// {
		// 	ErsFile << aErs;
		// }
		// ErsFile.close();
		
		// std::ofstream ExFile("Ex.mat");
		// if (ExFile.is_open())
		// {
		// 	ExFile << aEx;
		// }
		// ExFile.close();

		// std::ofstream ErFile("Er.mat");
		// if (ErFile.is_open())
		// {
		// 	ErFile << aEr;
		// }
		// ErFile.close();

		// std::ofstream EsFile("Es.mat");
		// if (EsFile.is_open())
		// {
		// 	EsFile << aEs;
		// }
		// EsFile.close();


		MatrixXd results = JacKKT.fullPivLu().solve(KKT_constrains).topRows(rhs.rows());

		MatrixXd dgds = results.topRows(aExx.rows());
		MatrixXd drds = results.bottomRows(aErr.rows());
		// print("DGDS");
		// print(dgds);
		// print("DRDS");
		// print(drds);
		aDEDs = dgds.transpose()*aEx + drds.transpose()*aEr + aEs;

		return aDEDs;
	}

	void Hessians(Mesh& m){
		// print("+Hessians");
		//Exx is constant

		// Exr
		Exr(m);

		//Err
		Err(m);

		//Exs
		Exs(m);

		//Ers
		Ers(m);

		// print("-Hessians");
	}

	void Gradients(Mesh& m){
		// print("+ Gradients");
		aEx = dEdx(m);
		aEr = dEdr(m);
		aEs = dEds(m);
		// print("- Gradients");
	}

	VectorXd dEdx(Mesh& m){
		VectorXd PAx = aPA*(m.G()*m.red_x() + m.x0());;
		VectorXd FPAx0 = m.xbar();
		VectorXd res = (aPA*m.G()).transpose()*(PAx - FPAx0);
		return res;
	}

	VectorXd dEdr(Mesh& m){
		setupRedSparseDRdr(m);
		VectorXd PAg = aPA*(m.G()*m.red_x() + m.x0());
		VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;

		aEr.setZero();
		for(int i=0; i<aEr.size(); i++){
			auto v = aDR[i];
			for(int k=0; k<v.size(); k++){
				aEr[i] += -1*PAg[v[k].row()]*USUtPAx0[v[k].col()]*v[k].value();
			}
		}
		return aEr;
	}

	VectorXd dEds(Mesh& m){
		SparseMatrix<double> RU =m.GR()*m.GU(); 
		VectorXd SUtPAx0 = m.GS()*aUtPAx0;
		VectorXd UtRtPAx = (RU).transpose()*aPA*(m.G()*m.red_x() + m.x0());;

		aEs.setZero();
		for(int i=0; i<aEs.size(); i++){
			std::vector<Trip> v = aDS[i];
			for(int k=0; k<aDS[i].size(); k++){
				aEs[i] -= UtRtPAx[v[k].row()]*aUtPAx0[v[k].col()]*v[k].value();
				aEs[i] += SUtPAx0[v[k].row()]*aUtPAx0[v[k].col()]*v[k].value();
			}
		}
		return aEs;
	}

	MatrixXd& Exr(Mesh& m){
		setupRedSparseDRdr(m);
		VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;
		
		aExr.setZero();
		for(int i=0; i<aExr.rows(); i++){
			for(int j=0; j<aExr.cols(); j++){
				auto v = aDR[j];
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
		setupRedSparseDDRdrdr(m);
		VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;
		VectorXd PAx = aPA*(m.G()*m.red_x() + m.x0());


		aErr.setZero();
		for(int i=0; i<aErr.rows()/3; i++){
			auto v00 = aDDR[9*i+0];
			auto v01 = aDDR[9*i+1];
			auto v02 = aDDR[9*i+2];
			auto v11 = aDDR[9*i+4];
			auto v12 = aDDR[9*i+5];
			auto v22 = aDDR[9*i+8];
			for(int k=0; k<v00.size(); k++){
				aErr(3*i+0,3*i+0) += -1*v00[k].value()*(PAx[v00[k].row()]*USUtPAx0[v00[k].col()]);
			}
			for(int k=0; k<v11.size(); k++){
				aErr(3*i+1,3*i+1) += -1*v11[k].value()*(PAx[v11[k].row()]*USUtPAx0[v11[k].col()]);
			}
			for(int k=0; k<v22.size(); k++){
				aErr(3*i+2,3*i+2) += -1*v22[k].value()*(PAx[v22[k].row()]*USUtPAx0[v22[k].col()]);
			}
			for(int k=0; k<v01.size(); k++){
				aErr(3*i+0,3*i+1) += -1*v01[k].value()*(PAx[v01[k].row()]*USUtPAx0[v01[k].col()]);
			}
			for(int k=0; k<v02.size(); k++){
				aErr(3*i+0,3*i+2) += -1*v02[k].value()*(PAx[v02[k].row()]*USUtPAx0[v02[k].col()]);
			}
			for(int k=0; k<v12.size(); k++){
				aErr(3*i+1,3*i+2) += -1*v12[k].value()*(PAx[v12[k].row()]*USUtPAx0[v12[k].col()]);
			}
			
			aErr(3*i+1, 3*i+0) = aErr(3*i+0, 3*i+1);
			aErr(3*i+2, 3*i+0) = aErr(3*i+0, 3*i+2);
			aErr(3*i+2, 3*i+1) = aErr(3*i+1, 3*i+2);
				
		}



		return aErr;
	}

	MatrixXd& Ers(Mesh& m){
		setupRedSparseDRdr(m);
		VectorXd PAg = aPA*(m.G()*m.red_x() + m.x0());;

		MatrixXd TEMP1 = MatrixXd::Zero(12*m.T().rows(), aErs.rows());
		for(int i=0; i<TEMP1.rows(); i++){
			for(int j=0; j<TEMP1.cols(); j++){
				auto v = aDR[j];
				for(int k=0; k<v.size(); k++){
					TEMP1(i,j) += v[k].value()*(-1*m.GU().coeff(v[k].col(), i)*PAg[v[k].row()]);
				}
			}
		}

		aErs.setZero();
		for(int i=0; i<aErs.cols(); i++){
			std::vector<Trip> v = aDS[i];
			for(int j=0; j<TEMP1.cols(); j++){
				for(int k=0; k<v.size(); k++){
					aErs(j, i) += v[k].value()*(aUtPAx0[v[k].row()]*TEMP1(v[k].col(), j));
				}
			}
		}



		return aErs;
	}

	void setupRedSparseDRdr(Mesh& m){
		aDR.clear();
		Matrix3d Jx = cross_prod_mat(1,0,0);
		Matrix3d Jy = cross_prod_mat(0,1,0);
		Matrix3d Jz = cross_prod_mat(0,0,1);

		std::map<int, std::vector<int>>& c_e_map = m.r_cluster_elem_map();
		//iterator through rotation clusters
		for(int t=0; t<m.red_w().size()/3;t++){
			SparseMatrix<double> Ident(4*c_e_map[t].size(), 4*c_e_map[t].size());
            Ident.setIdentity();
            SparseMatrix<double>& B = m.RotBLOCK()[t];
            VectorXd mred_r = m.red_r();
			Matrix3d R0; 
			R0<< mred_r[9*t+0], mred_r[9*t+1], mred_r[9*t+2],
				mred_r[9*t+3], mred_r[9*t+4], mred_r[9*t+5],
				mred_r[9*t+6], mred_r[9*t+7], mred_r[9*t+8];

			Matrix3d r1 = R0*Jx;
			Matrix3d r2 = R0*Jy;
			Matrix3d r3 = R0*Jz;
			
			SparseMatrix<double> block1 = Eigen::kroneckerProduct(Ident, r1);
			SparseMatrix<double> block2 = Eigen::kroneckerProduct(Ident, r2);
			SparseMatrix<double> block3 = Eigen::kroneckerProduct(Ident, r3);
			SparseMatrix<double> slice1 = B*block1*B.transpose();
			SparseMatrix<double> slice2 = B*block2*B.transpose();
			SparseMatrix<double> slice3 = B*block3*B.transpose();
			aDR.push_back(to_triplets(slice1));
			aDR.push_back(to_triplets(slice2));
			aDR.push_back(to_triplets(slice3));
		
		}
	}

	void setupRedSparseDDRdrdr(Mesh& m){
		aDDR.clear();
		Matrix3d Jx = cross_prod_mat(1,0,0);
		Matrix3d Jy = cross_prod_mat(0,1,0);
		Matrix3d Jz = cross_prod_mat(0,0,1);

		std::map<int, std::vector<int>>& c_e_map = m.r_cluster_elem_map();
		for(int t=0; t<m.red_w().size()/3; t++){
			SparseMatrix<double> Ident(4*c_e_map[t].size(), 4*c_e_map[t].size());
            Ident.setIdentity();
            SparseMatrix<double>& B = m.RotBLOCK()[t];
			VectorXd mred_r = m.red_r();
			Matrix3d R0; 
			R0<< mred_r[9*t+0], mred_r[9*t+1], mred_r[9*t+2],
				mred_r[9*t+3], mred_r[9*t+4], mred_r[9*t+5],
				mred_r[9*t+6], mred_r[9*t+7], mred_r[9*t+8];

			Matrix3d r1 = R0*0.5*(Jx*Jx + Jx*Jx);
			Matrix3d r2 = R0*0.5*(Jx*Jy + Jy*Jx);
			Matrix3d r3 = R0*0.5*(Jx*Jz + Jz*Jx);
			Matrix3d r4 = R0*0.5*(Jy*Jx + Jx*Jy);
			Matrix3d r5 = R0*0.5*(Jy*Jy + Jy*Jy);
			Matrix3d r6 = R0*0.5*(Jy*Jz + Jz*Jy);
			Matrix3d r7 = R0*0.5*(Jz*Jx + Jx*Jz);
			Matrix3d r8 = R0*0.5*(Jz*Jy + Jy*Jz);
			Matrix3d r9 = R0*0.5*(Jz*Jz + Jz*Jz);

			SparseMatrix<double> block1 = Eigen::kroneckerProduct(Ident, r1);
			SparseMatrix<double> block2 = Eigen::kroneckerProduct(Ident, r2);
			SparseMatrix<double> block3 = Eigen::kroneckerProduct(Ident, r3);
			SparseMatrix<double> block4 = Eigen::kroneckerProduct(Ident, r4);
			SparseMatrix<double> block5 = Eigen::kroneckerProduct(Ident, r5);
			SparseMatrix<double> block6 = Eigen::kroneckerProduct(Ident, r6);
			SparseMatrix<double> block7 = Eigen::kroneckerProduct(Ident, r7);
			SparseMatrix<double> block8 = Eigen::kroneckerProduct(Ident, r8);
			SparseMatrix<double> block9 = Eigen::kroneckerProduct(Ident, r9);


			SparseMatrix<double> slice1 = B*block1*B.transpose();
			SparseMatrix<double> slice2 = B*block2*B.transpose();
			SparseMatrix<double> slice3 = B*block3*B.transpose();
			SparseMatrix<double> slice4 = B*block4*B.transpose();
			SparseMatrix<double> slice5 = B*block5*B.transpose();
			SparseMatrix<double> slice6 = B*block6*B.transpose();
			SparseMatrix<double> slice7 = B*block7*B.transpose();
			SparseMatrix<double> slice8 = B*block8*B.transpose();
			SparseMatrix<double> slice9 = B*block9*B.transpose();
			

			aDDR.push_back(to_triplets(slice1));
			aDDR.push_back(to_triplets(slice2));
			aDDR.push_back(to_triplets(slice3));
			aDDR.push_back(to_triplets(slice4));
			aDDR.push_back(to_triplets(slice5));
			aDDR.push_back(to_triplets(slice6));
			aDDR.push_back(to_triplets(slice7));
			aDDR.push_back(to_triplets(slice8));
			aDDR.push_back(to_triplets(slice9));
		}	
	}

	void setupRedSparseDSds(Mesh& m){
	
		for(int i=0; i<m.red_s().size()/6; i++){
			VectorXd sWx = m.sW().col(6*i+0); 

			vector<Trip> sx = {};
			vector<Trip> sy = {};
			vector<Trip> sz = {};
			vector<Trip> s01 = {};
			vector<Trip> s02 = {};
			vector<Trip> s12 = {};
			for(int j=0; j<m.sW().rows()/6; j++){
				sx.push_back(Trip(12*j+0, 12*j+0, sWx[6*j]));
				sx.push_back(Trip(12*j+3, 12*j+3, sWx[6*j]));
				sx.push_back(Trip(12*j+6, 12*j+6, sWx[6*j]));
				sx.push_back(Trip(12*j+9, 12*j+9, sWx[6*j]));

				sy.push_back(Trip(12*j+0+1, 12*j+0+1, sWx[6*j]));
				sy.push_back(Trip(12*j+3+1, 12*j+3+1, sWx[6*j]));
				sy.push_back(Trip(12*j+6+1, 12*j+6+1, sWx[6*j]));
				sy.push_back(Trip(12*j+9+1, 12*j+9+1, sWx[6*j]));

				sz.push_back(Trip(12*j+0+2, 12*j+0+2, sWx[6*j]));
				sz.push_back(Trip(12*j+3+2, 12*j+3+2, sWx[6*j]));
				sz.push_back(Trip(12*j+6+2, 12*j+6+2, sWx[6*j]));
				sz.push_back(Trip(12*j+9+2, 12*j+9+2, sWx[6*j]));

				s01.push_back(Trip(12*j+0, 12*j+0+1, sWx[6*j]));
				s01.push_back(Trip(12*j+3, 12*j+3+1, sWx[6*j]));
				s01.push_back(Trip(12*j+6, 12*j+6+1, sWx[6*j]));
				s01.push_back(Trip(12*j+9, 12*j+9+1, sWx[6*j]));
				s01.push_back(Trip(12*j+0+1, 12*j+0, sWx[6*j]));
				s01.push_back(Trip(12*j+3+1, 12*j+3, sWx[6*j]));
				s01.push_back(Trip(12*j+6+1, 12*j+6, sWx[6*j]));
				s01.push_back(Trip(12*j+9+1, 12*j+9, sWx[6*j]));
				
				s02.push_back(Trip(12*j+0, 12*j+0+2, sWx[6*j]));
				s02.push_back(Trip(12*j+3, 12*j+3+2, sWx[6*j]));
				s02.push_back(Trip(12*j+6, 12*j+6+2, sWx[6*j]));
				s02.push_back(Trip(12*j+9, 12*j+9+2, sWx[6*j]));
				s02.push_back(Trip(12*j+0+2, 12*j+0, sWx[6*j]));
				s02.push_back(Trip(12*j+3+2, 12*j+3, sWx[6*j]));
				s02.push_back(Trip(12*j+6+2, 12*j+6, sWx[6*j]));
				s02.push_back(Trip(12*j+9+2, 12*j+9, sWx[6*j]));

				s12.push_back(Trip(12*j+0+1, 12*j+0+2, sWx[6*j]));
				s12.push_back(Trip(12*j+3+1, 12*j+3+2, sWx[6*j]));
				s12.push_back(Trip(12*j+6+1, 12*j+6+2, sWx[6*j]));
				s12.push_back(Trip(12*j+9+1, 12*j+9+2, sWx[6*j]));
				s12.push_back(Trip(12*j+0+2, 12*j+0+1, sWx[6*j]));
				s12.push_back(Trip(12*j+3+2, 12*j+3+1, sWx[6*j]));
				s12.push_back(Trip(12*j+6+2, 12*j+6+1, sWx[6*j]));
				s12.push_back(Trip(12*j+9+2, 12*j+9+1, sWx[6*j]));
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
		VectorXd PAx = aPA*(m.G()*m.red_x() + m.x0());
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
		
			if ((Ex - Ex0).norm()<1e-12){
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

	MatrixXd& Exx(){ return aExx; }
	MatrixXd& Exr(){ return aExr; }
	MatrixXd& Exs(){ return aExs; }
	MatrixXd& Ers(){ return aErs; }
	MatrixXd& Err(){ return aErr; }
	VectorXd& Er() { return aEr; }
	VectorXd& Es() { return aEs; }
	VectorXd& Ex() { return aEx; }


	template<class T>
    void print(T a){ std::cout<<a<<std::endl; }

    MatrixXd RodriguesRotation(double wX, double wY, double wZ, double wlen){
        double c = cos(wlen);
        double s = sin(wlen);
        double c1 = 1 - c;
        Matrix3d Rot;
        Rot<< c + wX*wX*c1, -wZ*s + wX*wY*c1, wY*s + wX*wZ*c1,
            wZ*s + wX*wY*c1, c + wY*wY*c1, -wX*s + wY*wZ*c1,
            -wY*s + wX*wZ*c1, wX*s + wY*wZ*c1, c + wZ*wZ*c1;
        return Rot;
    }

    MatrixXd cross_prod_mat(double wX, double wY, double wZ){
        Matrix3d cross;
        cross<<0, -wZ, wY,
        		wZ, 0, -wX,
        		-wY, wX, 0;
        return cross;
    }

};

#endif