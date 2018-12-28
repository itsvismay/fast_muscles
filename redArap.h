#ifndef RED_ARAP
#define RED_ARAP

#include <igl/polar_svd.h>
#include "mesh.h"
#include<Eigen/LU>
#include <iostream>
#include <string>


using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;

class Reduced_Arap
{

protected:
	FullPivLU<MatrixXd> ajacLU;
	FullPivLU<MatrixXd>  aARAPKKTSolver;
	VectorXd aPAx0, aUtPAx0, aEr, aEs, aEx, aDEDs, aFPAx0;
	MatrixXd aExx, aExr, aErr, aExs, aErs, aPAG, aCG, aJacKKT, aJacConstrains, aTEMP1, aGtAtPtRU;

	std::vector<vector<Trip>> aDS, aDR, aDDR;

	std::vector<MatrixXd> aConstItRTerms;
	std::vector<VectorXd> aUSUtPAx_E;
	std::vector<MatrixXd> aConstErTerms;
	double jacLUanalyzed = false;

	SparseMatrix<double> adjointP;

public:
	Reduced_Arap(Mesh& m){
		int r_size = m.red_w().size();
		int z_size = m.red_x().size();
		int s_size = m.red_s().size();
		int t_size = m.T().rows();

		aFPAx0.resize(12*m.T().rows());

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

		aGtAtPtRU = MatrixXd::Zero(m.G().cols(), 12*m.T().rows());// aPAG.transpose()*(m.GR()*m.GU());
		aTEMP1 = MatrixXd::Zero(12*m.T().rows(), aErs.rows());
		aCG = MatrixXd(m.AB().transpose())*m.G();

		MatrixXd KKTmat = MatrixXd::Zero(aExx.rows()+aCG.rows(), aExx.rows()+aCG.rows());
		KKTmat.block(0,0, aExx.rows(), aExx.cols()) = aExx;
		KKTmat.block(aExx.rows(), 0, aCG.rows(), aCG.cols()) = aCG;
		KKTmat.block(0, aExx.cols(), aCG.cols(), aCG.rows()) = aCG.transpose();
		aARAPKKTSolver.compute(KKTmat);

		aJacKKT.resize(z_size+r_size+aCG.rows(), z_size+r_size+aCG.rows());
		aJacConstrains.resize(z_size+r_size+aCG.rows() ,s_size);


		setupRedSparseDSds(m);//one time pre-processing
		setupRedSparseDRdr(m);
		setupRedSparseDDRdrdr(m);

		// setupFastItRTerms(m);
		// setupFastErTerms(m);
		// setupFastEsTerms(m);
		// setupFastUSUtPAx0Terms(m);

		print("rarap 6");
		setupAdjointP();
	}

	void setupAdjointP(){
		adjointP.resize(aExx.rows()+aErr.rows(), aExx.rows()+aErr.rows()+aCG.rows());
		for(int i=0; i<aExx.rows()+aErr.rows(); i++){
			adjointP.coeffRef(i,i) = 1;
		}
	}

	double Energy(Mesh& m){
		VectorXd PAx = aPAG*m.red_x() + aPAx0;
		double En= 0.5*(PAx - aFPAx0).squaredNorm();
		return En;
	}

	double Energy(Mesh& m, VectorXd& z, VectorXd& redw, VectorXd& redr, VectorXd& reds, VectorXd& redu){
		VectorXd ms = m.sW()*reds;
		VectorXd mr = 
		VectorXd PAx = aPAG*z + aPAx0;
		VectorXd FPAx0(PAx.size());
		for(int i=0; i<m.T().rows(); i++){
            Matrix3d ri = Map<Matrix3d>(m.red_r().segment<9>(9*i).data()).transpose();
            Matrix3d r;
            Vector3d w;
            w<<redw(3*i+0),redw(3*i+1),redw(3*i+2);
            double wlen = w.norm();
            if (wlen>1e-9){
                double wX = w(0);
                double wY = w(1);
                double wZ = w(2);
                Matrix3d cross;
                cross<<0, -wZ, wY,
                        wZ, 0, -wX,
                        -wY, wX, 0;
                Matrix3d Rot = cross.exp();
                r = ri*Rot;
            }else{
                r = ri;
            }
            
            Matrix3d u = Map<Matrix3d>(redu.segment<9>(9*i).data()).transpose();
            Matrix3d s;
            s<< ms[6*i + 0], ms[6*i + 3], ms[6*i + 4],
                ms[6*i + 3], ms[6*i + 1], ms[6*i + 5],
                ms[6*i + 4], ms[6*i + 5], ms[6*i + 2];

            Matrix3d rusut = r*u*s*u.transpose();
            FPAx0.segment<3>(12*i+0) = rusut*aPAx0.segment<3>(12*i+0);
            FPAx0.segment<3>(12*i+3) = rusut*aPAx0.segment<3>(12*i+3);
            FPAx0.segment<3>(12*i+6) = rusut*aPAx0.segment<3>(12*i+6);
            FPAx0.segment<3>(12*i+9) = rusut*aPAx0.segment<3>(12*i+9);
        }
		return 0.5*(PAx - FPAx0).squaredNorm();
	}

	VectorXd Jacobians(Mesh& m){
		// VectorXd& Ex, VectorXd& Er,  VectorXd& Es, 
		// 		SparseMatrix<double>& Exx, SparseMatrix<double>& Erx, 
		// 		SparseMatrix<double>& Err, SparseMatrix<double>& Exs, SparseMatrix<double>& Ers
		
		int h = Hessians(m);

		int gg = Gradients(m);


		//Dense
		aJacKKT.setZero();
		aJacConstrains.setZero();
		//col1
		aJacKKT.block(0,0,aExx.rows(), aExx.cols()) = Exx();
		aJacKKT.block(aExx.rows(), 0, aExr.cols(), aExr.rows()) = Exr().transpose();
		aJacKKT.block(aExx.rows()+aExr.cols(), 0, aCG.rows(), aCG.cols()) = aCG;
		//col2
		aJacKKT.block(0,aExx.cols(),aExr.rows(), aExr.cols()) = Exr();
		aJacKKT.block(aExr.rows(), aExx.cols(), aErr.rows(), aErr.cols()) = Err();
		// // //col3
		aJacKKT.block(0, aExx.cols()+aExr.cols(), aCG.cols(), aCG.rows())= aCG.transpose();
		// //rhs
		aJacConstrains.block(0,0, aExs.rows(), aExs.cols()) = Exs();
		aJacConstrains.block(aExs.rows(), 0, aErs.rows(), aErs.cols()) = Ers();

		
		// std::ofstream ExxFile("Exx.mat");
		// if (ExxFile.is_open())
		// {
		// 	ExxFile << aExx;
		// }
		// ExxFile.close();

		// print("before LU");		
		VectorXd ExEr(aEx.size()+aEr.size());
		ExEr<<aEx,aEr;
		VectorXd PtExEr = adjointP.transpose()*ExEr;
		VectorXd g = aJacKKT.fullPivLu().solve(PtExEr);
		aDEDs = aJacConstrains.transpose()*g + aEs;
		return aDEDs;
	}

	int Hessians(Mesh& m){
		// print("		+Hessians");

		// Exr
		// print("		Exr");
		Exr(m);

		//Err
		// print("		Err");
		Err(m);

		//Exs
		// print("		Exs");
		Exs(m);

		//Ers
		// print("		Ers");
		Ers(m);

		// print("		-Hessians");		
		return 1;
	}

	int Gradients(Mesh& m){
		// print("			+ Gradients");

		m.constTimeFPAx0(aFPAx0);
		aEx = dEdx(m);
		
		// print("		Er");
		aEr = dEdr(m);
		// print("		Es");
		aEs = dEds(m);

		// print("			- Gradients");
		return 1;
	}

	MatrixXd& Exr(Mesh& m){
		Matrix3d Jx = cross_prod_mat(1,0,0);
		Matrix3d Jy = cross_prod_mat(0,1,0);
		Matrix3d Jz = cross_prod_mat(0,0,1);

		VectorXd ms = m.sW()*m.red_s();
		VectorXd& mred_r = m.red_r();
		for(int t=0; t<m.T().rows(); t++){
			//Tet
			Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];
			
			Matrix3d R0; 
			R0<< mred_r[9*t+0], mred_r[9*t+1], mred_r[9*t+2],
				mred_r[9*t+3], mred_r[9*t+4], mred_r[9*t+5],
				mred_r[9*t+6], mred_r[9*t+7], mred_r[9*t+8];
			Matrix3d r1 = R0*Jx;
			Matrix3d r2 = R0*Jy;
			Matrix3d r3 = R0*Jz;

			for(int e=0; e<4; e++){
				//Vert on tet
				for(int a =0; a<3; a++){
					Matrix3d p1 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+0)*(u*s*aUtPAx0.segment<3>(12*t+0)).transpose();
					Matrix3d p2 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+3)*(u*s*aUtPAx0.segment<3>(12*t+3)).transpose();
					Matrix3d p3 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+6)*(u*s*aUtPAx0.segment<3>(12*t+6)).transpose();
					Matrix3d p4 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+9)*(u*s*aUtPAx0.segment<3>(12*t+9)).transpose();
					

					double Exar1 = p1.cwiseProduct(r1).sum() + p2.cwiseProduct(r1).sum() + p3.cwiseProduct(r1).sum() + p4.cwiseProduct(r1).sum();
					double Exar2 = p1.cwiseProduct(r2).sum() + p2.cwiseProduct(r2).sum() + p3.cwiseProduct(r2).sum() + p4.cwiseProduct(r2).sum();
					double Exar3 = p1.cwiseProduct(r3).sum() + p2.cwiseProduct(r3).sum() + p3.cwiseProduct(r3).sum() + p4.cwiseProduct(r3).sum();

					aExr.row(3*m.T().row(t)[e]+a)[3*t+0] = Exar1;
					aExr.row(3*m.T().row(t)[e]+a)[3*t+1] = Exar2;
					aExr.row(3*m.T().row(t)[e]+a)[3*t+2] = Exar3;
				}
			}
		}
		
		return aExr;
	}

	MatrixXd& Exs(Mesh& m){

		for(int t =0; t<m.T().rows(); t++){
			//Tet
			Matrix3d r = Map<Matrix3d>(m.red_r().segment<9>(9*m.r_elem_cluster_map()[t]).data()).transpose();
			Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
			for(int e=0; e<4; e++){
				//Vert on tet
				for(int a=0; a<3; a++){
					//x, y, or z axis
					Vector3d GtAtPtRU_row1 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+0).transpose()*r*u;
					Vector3d GtAtPtRU_row2 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+3).transpose()*r*u;
					Vector3d GtAtPtRU_row3 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+6).transpose()*r*u;
					Vector3d GtAtPtRU_row4 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+9).transpose()*r*u;
					Matrix3d p1 = GtAtPtRU_row1*aUtPAx0.segment<3>(12*t+0).transpose();
					Matrix3d p2 = GtAtPtRU_row2*aUtPAx0.segment<3>(12*t+3).transpose();
					Matrix3d p3 = GtAtPtRU_row3*aUtPAx0.segment<3>(12*t+6).transpose();
					Matrix3d p4 = GtAtPtRU_row4*aUtPAx0.segment<3>(12*t+9).transpose();

					double Exas1 = p1(0,0) + p2(0,0) + p3(0,0) + p4(0,0);
					double Exas2 = p1(1,1) + p2(1,1) + p3(1,1) + p4(1,1);
					double Exas3 = p1(2,2) + p2(2,2) + p3(2,2) + p4(2,2);
					double Exas4 = p1(0,1) + p2(0,1) + p3(0,1) + p4(0,1)+ p1(1,0) + p2(1,0) + p3(1,0) + p4(1,0);
					double Exas5 = p1(0,2) + p2(0,2) + p3(0,2) + p4(0,2)+ p1(2,0) + p2(2,0) + p3(2,0) + p4(2,0);
					double Exas6 = p1(2,1) + p2(2,1) + p3(2,1) + p4(2,1)+ p1(1,2) + p2(1,2) + p3(1,2) + p4(1,2);
					aExs.row(3*m.T().row(t)[e]+a)[6*t+0] = Exas1;
					aExs.row(3*m.T().row(t)[e]+a)[6*t+1] = Exas2;
					aExs.row(3*m.T().row(t)[e]+a)[6*t+2] = Exas3;
					aExs.row(3*m.T().row(t)[e]+a)[6*t+3] = Exas4;
					aExs.row(3*m.T().row(t)[e]+a)[6*t+4] = Exas5;
					aExs.row(3*m.T().row(t)[e]+a)[6*t+5] = Exas6;
				}
			}
		}
		return aExs;
	}

	MatrixXd& Err(Mesh& m){	
		// VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;
		VectorXd PAx = aPAG*m.red_x() + aPAx0;
		VectorXd ms = m.sW()*m.red_s();
		VectorXd USUtPAx0 = VectorXd::Zero(12*m.T().rows());

		for(int t =0; t<m.T().rows(); t++){
			Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];
			for(int j=0; j<4; j++){
				USUtPAx0.segment<3>(12*t+3*j) = u*s*aUtPAx0.segment<3>(12*t+3*j);
			}
		}

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
		VectorXd PAg = aPAG*m.red_x() + aPAx0;;

		aTEMP1.setZero();
		for(int i=0; i<aTEMP1.rows(); i++){
			for(int j=0; j<aTEMP1.cols(); j++){
				for(int k=0; k<aDR[j].size(); k++){
					aTEMP1(i,j) += aDR[j][k].value()*(-1*m.GU().coeff(aDR[j][k].col(), i)*PAg[aDR[j][k].row()]);
				}
			}
		}

		aErs.setZero();
		for(int i=0; i<aErs.cols(); i++){
			for(int j=0; j<aTEMP1.cols(); j++){
				for(int k=0; k<aDS[i].size(); k++){
					aErs(j, i) += aDS[i][k].value()*(aUtPAx0[aDS[i][k].row()]*aTEMP1(aDS[i][k].col(), j));
				}
			}
		}



		return aErs;
	}
	
	VectorXd dEds(Mesh& m){
		// SparseMatrix<double> RU = m.GR()*m.GU(); 
		// VectorXd SUtPAx0 = m.GS()*aUtPAx0;
		// VectorXd UtRtPAx = (RU).transpose()*aPA*(m.G()*m.red_x() + m.x0());
		VectorXd ms = m.sW()*m.red_s();
		VectorXd PAx = aPAG*m.red_x() + aPAx0;

		
		VectorXd UtRtPAx1 = VectorXd::Zero(12*m.T().rows());
		VectorXd SUtPAx01 = VectorXd::Zero(12*m.T().rows());
		for(int t =0; t<m.T().rows(); t++){
			Matrix3d rt = Map<Matrix3d>(m.red_r().segment<9>(9*m.r_elem_cluster_map()[t]).data());
			Matrix3d ut = Map<Matrix3d>(m.red_u().segment<9>(9*t).data());
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];

			for(int j=0; j<4; j++){
				Vector3d p = PAx.segment<3>(12*t+3*j);
				UtRtPAx1.segment<3>(12*t+3*j) = ut*rt*p;
				SUtPAx01.segment<3>(12*t+3*j) = s*aUtPAx0.segment<3>(12*t+3*j);
			}
		}
	
		aEs.setZero();
		for(int i=0; i<aEs.size(); i++){
			std::vector<Trip> v = aDS[i];
			for(int k=0; k<aDS[i].size(); k++){
				int t= v[k].row()/12;
				aEs[i] -= (UtRtPAx1[v[k].row()])*aUtPAx0[v[k].col()]*v[k].value();
				aEs[i] += SUtPAx01[v[k].row()]*aUtPAx0[v[k].col()]*v[k].value();
			}
		}
		return aEs;
	}

	VectorXd dEdr(Mesh& m){
		VectorXd PAg = aPAG*m.red_x() + aPAx0;
		// VectorXd USUtPAx0 = m.GU()*m.GS()*aUtPAx0;
		VectorXd USUtPAx0 = VectorXd::Zero(12*m.T().rows());
		VectorXd ms = m.sW()*m.red_s();
		for(int t =0; t<m.T().rows(); t++){
			Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];
			for(int j=0; j<4; j++){
				USUtPAx0.segment<3>(12*t+3*j) = u*s*aUtPAx0.segment<3>(12*t+3*j);
			}
		}

		aEr.setZero();
		for(int i=0; i<aEr.size(); i++){
			for(int k=0; k<aDR[i].size(); k++){
				aEr[i] += -1*PAg[aDR[i][k].row()]*USUtPAx0[aDR[i][k].col()]*aDR[i][k].value();
			}
		}
		return aEr;
	}

	VectorXd dEdx(Mesh& m){
		VectorXd PAx = aPAG*m.red_x() + aPAx0;
		VectorXd res = (aPAG).transpose()*(PAx - aFPAx0);
		return res;
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

	bool itT(Mesh& m){
		VectorXd deltaABtx = m.AB().transpose()*m.dx();
		VectorXd AtPtFPAx0 = (aPAG).transpose()*aFPAx0;
		VectorXd AtPtPAx0 = (aPAG).transpose()*(aPAx0);
		VectorXd gb = AtPtFPAx0 - AtPtPAx0;
		VectorXd gd(gb.size()+deltaABtx.size());
		gd<<gb,deltaABtx;
		VectorXd result = aARAPKKTSolver.solve(gd);
		VectorXd gu = result.head(gb.size());
		m.red_x(gu);
		return false;
	}

	void itR(Mesh& m, VectorXd& USUtPAx0){
		VectorXd PAx = aPAG*m.red_x() + aPAx0;
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

	int minimize(Mesh& m){
		// print("	+ ARAP minimize");
		VectorXd ms = m.red_s();
		VectorXd USUtPAx0 = VectorXd::Zero(12*m.T().rows());
		for(int t =0; t<m.T().rows(); t++){
			Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];
			for(int j=0; j<4; j++){
				USUtPAx0.segment<3>(12*t+3*j) = u*s*aUtPAx0.segment<3>(12*t+3*j);
			}
		}
		m.constTimeFPAx0(aFPAx0);
		double previous5ItE = Energy(m);
		double oldE = Energy(m);
		// VectorXd E0 = m.B().transpose()*dEdx(m);
		for(int i=1; i< 1000; i++){
			bool converged = itT(m);
			itR(m, USUtPAx0);
			m.constTimeFPAx0(aFPAx0);
			double newE = Energy(m);
			cout<<i<<",";
			if((newE - oldE)>1e-8){
				print("Reduced_Arap::minimize() error. ARAP should monotonically decrease.");
				print(i);
				print(oldE);
				print(newE);
				exit(0);
			}
			oldE = newE;
			// if(converged){
			// 	std::cout<<"		- ARAP minimize "<<i<<", "<<std::endl;
			// 	break;
			// }
			if (i%5==0){
				if(fabs(newE - previous5ItE)<1e-10){
					if(i>1000){
						// print(m.red_s().transpose());
						// exit(0);
					}
					std::cout<<"		- Red_ARAP minimize "<<i<<", "<<(newE - previous5ItE)<<std::endl;
					return i;
				}
				previous5ItE = newE;
			}
			
			// VectorXd Ex = m.B().transpose()*dEdx(m);
			// if((Ex - E0).norm()<1e-8){
			// 	std::cout<<"		- ARAP minimize "<<i<<", "<<(Ex - E0).norm()<<std::endl;
			// 	return i;
			// }
			// E0 = Ex;
		}
		// VectorXd Ex = dEdx(m);
		// print("WHy is it not converging?");
		// print("redx\n");
		// print(m.red_x().transpose());
		// print("1\n");
		// VectorXd PAx = aPA*m.red_x() + aPAx0;
		// print((m.AB().transpose()*PAx).transpose());
		// print("2\n");
		// print((m.AB().transpose()*aFPAx0).transpose());
		// print("3\n");
		// VectorXd res = (aPA).transpose()*(PAx - aFPAx0);
		// // print(((aPA).transpose()*(aPA*m.red_x() + aPAx0)).transpose());
		// print("4\n");
		// print(((aPA).transpose()*(aFPAx0)).transpose());
		// print("5\n");
		// print(Ex.transpose());
		// print("6\n");
		// print((m.AB().transpose()*Ex).transpose());
		// print(m.red_s());		
		std::cout<<"		- ARAP never converged "<<Energy(m)-previous5ItE<<std::endl;
		// exit(0);
		return 1000;
	}

	MatrixXd Exx(){ return aExx; }
	MatrixXd Exr(){ return aExr; }
	MatrixXd Exs(){ return aExs; }
	MatrixXd Ers(){ return aErs; }
	MatrixXd Err(){ return aErr; }
	VectorXd Er() { return aEr; }
	VectorXd Es() { return aEs; }
	VectorXd Ex() { return aEx; }
	VectorXd& FPAx0() { return aFPAx0; }

	std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> & M){
		std::vector<Eigen::Triplet<double>> v;
		for(int i = 0; i < M.outerSize(); i++){
			for(typename Eigen::SparseMatrix<double>::InnerIterator it(M,i); it; ++it){	
				v.emplace_back(it.row(),it.col(),it.value());
			}
		}
		return v;
	}

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