#ifndef RED_UARAP
#define RED_UARAP

#include <igl/polar_svd.h>
#include "mesh.h"
#include <Eigen/LU>
#include <iostream>
#include <string>
#include <igl/Timer.h>


using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;
typedef Matrix<double, 12, 1> Vector12d;
typedef Matrix<double, 9, 1> Vector9d;


class Reduced_UArap
{

protected:
	SparseLU<SparseMatrix<double>>  aARAPKKTSparseSolver;
	SparseLU<SparseMatrix<double>> ajacLU;

	FullPivLU<MatrixXd>  aARAPKKTSolver;
	Eigen::VectorXd aPAx0, aEr, aEr_max, aEs, aEs_max, aEx, aDEDs, aFPAx0;
	SparseMatrix<double>  aExs_max, aExr_max, aErr_max, aErs_max, aPA;
	MatrixXd aExx, aExs, aExr, aErr, aErs, aPAG, aCG;
	SparseMatrix<double> aJacKKTSparse, aJacConstrainsSparse;
	MatrixXd aJacKKT, aJacConstrains;


	MatrixXd aPAx0DS;
	std::vector<Trip> aExx_trips, aExr_max_trips, aExr_trips, aErr_trips, aErs_trips, aErr_max_trips, aExs_trips, aExs_max_trips, aErs_max_trips, aC_trips;
	double jacLUanalyzed = false;

	SparseMatrix<double> adjointP, a_Wr, a_Ww;
	std::vector<MatrixXd> aePAx;
	std::vector<MatrixXd> aeUSUtPAx0;

	double aPAx0squaredNorm =0;
	VectorXd aPAx0tPAG;
	MatrixXd aPAx0tRSPAx0;
	MatrixXd aRSPAx0tRSPAx0;

	std::vector<MatrixXd> aFASTARAPDenseTerms;
	std::vector<MatrixXd> aFASTARAPSparseTerms;
	std::vector<MatrixXd> aFASTARAPCubeGtAtPtRSPAx0;
	std::vector<MatrixXd> aFASTARAPItRTerms1;
	std::vector<VectorXd> aFASTARAPItRTerms2;
	MatrixXd aFastEsTerm1;
	MatrixXd aFastEsTerm2;
	std::vector<MatrixXd> aFastEsTerm3s;

	igl::Timer atimer;



public:
	Reduced_UArap(Mesh& m){
		int r_size = m.red_w().size();
		int z_size = m.red_x().size();
		int s_size = m.red_s().size();
		int t_size = m.T().rows();
		int v_size = m.V().rows();

		aFPAx0.resize(12*m.T().rows());
		aFPAx0.setZero();

		aExx = (m.P()*m.A()*m.G()).transpose()*(m.P()*m.A()*m.G());

		print("arap 2");
		aErr_max.resize(3*t_size, 3*t_size);
		aErs_max.resize(3*t_size, 6*t_size);
		aExr_max.resize(3*v_size, 3*t_size);
		aExs_max.resize(3*v_size, 6*t_size);
		aEr_max.resize(3*t_size);
		aEs_max.resize(6*t_size);

		aExs.resize(z_size, s_size);
		aErr.resize(r_size, r_size);
		aErs.resize(r_size, s_size);
		aExr.resize(z_size, r_size);
		aEr.resize(r_size);
		aEs.resize(s_size);

		aPAx0 = m.P()*m.A()*m.x0();
		aPA = m.P()*m.A();
		aPAG = m.P()*m.A()*m.G();
		aCG = m.AB().transpose()*m.G();
		print("rarap 4");
		MatrixXd& YC = m.JointY();
		MatrixXd ARAPKKTmat = MatrixXd::Zero(aExx.rows() + YC.rows(), aExx.cols() + YC.rows());
		ARAPKKTmat.block(0,0,aExx.rows(), aExx.cols()) = aExx;
		ARAPKKTmat.block(0, aExx.cols(), YC.cols(), YC.rows()) = YC.transpose();
		ARAPKKTmat.block(aExx.rows(), 0, YC.rows(), YC.cols()) = YC;
		aARAPKKTSolver.compute(ARAPKKTmat);

		print("rarap 5");
		aJacKKT.resize(z_size+r_size+aCG.rows(), z_size+r_size+aCG.rows());
		aJacConstrains.resize(z_size+r_size+aCG.rows() ,s_size);
		print("rarap 6");
		setupAdjointP();
		print("pre-processing");
		m.constTimeFPAx0(aFPAx0);
		setupWrWw(m);
		// setupFASTARAPTerms(m);
		// setupFastPAx0DSTerm(m);
		// setupFastEsTerms(m);
		// setupFastEnergyTerms(m);
		setupFastItR(m);

		print("Jacobian solve pre-processing");
		aJacKKT.block(0,0,aExx.rows(), aExx.cols()) = Exx();
		aJacKKT.block(aExx.rows()+aExr.cols(), 0, aCG.rows(), aCG.cols()) = aCG;
		aJacKKT.block(0, aExx.cols()+aExr.cols(), aCG.cols(), aCG.rows())= aCG.transpose();
	}

	void setupFastItR(Mesh& m){
		std::map<int, std::vector<int>>& c_e_map = m.r_cluster_elem_map();
		for (int i=0; i<m.red_w().size()/3; i++){
			std::vector<int> cluster_elem = c_e_map[i];
			aePAx.push_back(MatrixXd::Zero(4*cluster_elem.size(), 3));
			aeUSUtPAx0.push_back(MatrixXd::Zero(4*cluster_elem.size(), 3));
		}
	}
	

	void setupAdjointP(){
		adjointP.resize(aExx.rows()+aErr.rows(), aExx.rows()+aErr.rows()+aCG.rows());
		for(int i=0; i<aExx.rows()+aErr.rows(); i++){
			adjointP.coeffRef(i,i) = 1;
		}
	}

	void setupWrWw(Mesh& m){
		std::vector<Trip> wr_trips;
		std::vector<Trip> ww_trips;

		std::map<int, std::vector<int>>& c_e_map = m.r_cluster_elem_map();
		for (int i=0; i<m.red_r().size()/9; i++){
			std::vector<int> cluster_elem = c_e_map[i];
			for(int e=0; e<cluster_elem.size(); e++){
				wr_trips.push_back(Trip(9*cluster_elem[e]+0, 9*i+0, 1));
				wr_trips.push_back(Trip(9*cluster_elem[e]+1, 9*i+1, 1));
				wr_trips.push_back(Trip(9*cluster_elem[e]+2, 9*i+2, 1));
				wr_trips.push_back(Trip(9*cluster_elem[e]+3, 9*i+3, 1));
				wr_trips.push_back(Trip(9*cluster_elem[e]+4, 9*i+4, 1));
				wr_trips.push_back(Trip(9*cluster_elem[e]+5, 9*i+5, 1));
				wr_trips.push_back(Trip(9*cluster_elem[e]+6, 9*i+6, 1));
				wr_trips.push_back(Trip(9*cluster_elem[e]+7, 9*i+7, 1));
				wr_trips.push_back(Trip(9*cluster_elem[e]+8, 9*i+8, 1));
				
				ww_trips.push_back(Trip(3*cluster_elem[e]+0, 3*i+0, 1));
				ww_trips.push_back(Trip(3*cluster_elem[e]+1, 3*i+1, 1));
				ww_trips.push_back(Trip(3*cluster_elem[e]+2, 3*i+2, 1));
			
			}
		}

		a_Wr.resize( 9*m.T().rows(), m.red_r().size());
		a_Wr.setFromTriplets(wr_trips.begin(), wr_trips.end());

		a_Ww.resize( 3*m.T().rows(), m.red_w().size());
		a_Ww.setFromTriplets(ww_trips.begin(), ww_trips.end());
	}

	double Energy(Mesh& m){
		VectorXd PAx = aPA*m.red_x() + aPAx0;
		m.constTimeFPAx0(aFPAx0);
		double En = 0.5*(PAx - aFPAx0).squaredNorm();
		return En;
	}


	double Energy(Mesh& m, VectorXd& z, VectorXd& redw, VectorXd& redr, VectorXd& reds){
		VectorXd ms = m.sW()*reds;
		VectorXd mr = a_Wr*redr;
		VectorXd mw = a_Ww*redw;
		VectorXd PAx = aPAG*z + aPAx0;

		VectorXd FPAx0(PAx.size());
		for(int i=0; i<m.T().rows(); i++){
            Matrix3d ri = Map<Matrix3d>(mr.segment<9>(9*i).data()).transpose();
            Matrix3d r;
            Vector3d w;
            w<<mw(3*i+0),mw(3*i+1),mw(3*i+2);
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
            
            Matrix3d s;
            s<< ms[6*i + 0], ms[6*i + 3], ms[6*i + 4],
                ms[6*i + 3], ms[6*i + 1], ms[6*i + 5],
                ms[6*i + 4], ms[6*i + 5], ms[6*i + 2];

            Matrix3d rs = r*s;
            FPAx0.segment<3>(12*i+0) = rs*aPAx0.segment<3>(12*i+0);
            FPAx0.segment<3>(12*i+3) = rs*aPAx0.segment<3>(12*i+3);
            FPAx0.segment<3>(12*i+6) = rs*aPAx0.segment<3>(12*i+6);
            FPAx0.segment<3>(12*i+9) = rs*aPAx0.segment<3>(12*i+9);
        }
		return 0.5*(PAx - FPAx0).squaredNorm();
	}




	int Gradients(Mesh& m){
		
		// print("			+ Gradients");
		// print("		Ex");
		m.constTimeFPAx0(aFPAx0);
		aEx = dEdx(m);
		
		// print("		Er");
		aEr.setZero();
		Matrix3d Jx = cross_prod_mat(1,0,0);
		Matrix3d Jy = cross_prod_mat(0,1,0);
		Matrix3d Jz = cross_prod_mat(0,0,1);

		VectorXd PAg = aPAG*m.red_x() + aPAx0;
		VectorXd ms = m.sW()*m.red_s();
		VectorXd mr = a_Wr*m.red_r();

		for(int t=0; t<m.T().rows(); t++){
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];

			Matrix3d r0= Map<Matrix3d>(mr.segment<9>(9*t).data()).transpose();
			Matrix3d r1 = r0*Jx;
			Matrix3d r2 = r0*Jy;
			Matrix3d r3 = r0*Jz;

			Matrix3d p1 = -PAg.segment<3>(12*t+0)*(s*aPAx0.segment<3>(12*t+0)).transpose();
			Matrix3d p2 = -PAg.segment<3>(12*t+3)*(s*aPAx0.segment<3>(12*t+3)).transpose();
			Matrix3d p3 = -PAg.segment<3>(12*t+6)*(s*aPAx0.segment<3>(12*t+6)).transpose();
			Matrix3d p4 = -PAg.segment<3>(12*t+9)*(s*aPAx0.segment<3>(12*t+9)).transpose();

			double Er1 = p1.cwiseProduct(r1).sum() + p2.cwiseProduct(r1).sum() + p3.cwiseProduct(r1).sum() + p4.cwiseProduct(r1).sum();
			double Er2 = p1.cwiseProduct(r2).sum() + p2.cwiseProduct(r2).sum() + p3.cwiseProduct(r2).sum() + p4.cwiseProduct(r2).sum();
			double Er3 = p1.cwiseProduct(r3).sum() + p2.cwiseProduct(r3).sum() + p3.cwiseProduct(r3).sum() + p4.cwiseProduct(r3).sum();

			aEr_max[3*t+0] = Er1;
			aEr_max[3*t+1] = Er2;
			aEr_max[3*t+2] = Er3;

		}

		aEr = a_Ww.transpose()*aEr_max;
		
		// print("		Es");
		aEs_max.setZero();
		for(int t=0; t<m.T().rows(); t++){
			Matrix3d rt = Map<Matrix3d>(mr.segment<9>(9*t).data());
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];

			Matrix3d p1 = s*aPAx0.segment<3>(12*t+0)*aPAx0.segment<3>(12*t+0).transpose() - (rt*PAg.segment<3>(12*t+0))*aPAx0.segment<3>(12*t+0).transpose();
			Matrix3d p2 = s*aPAx0.segment<3>(12*t+3)*aPAx0.segment<3>(12*t+3).transpose() - (rt*PAg.segment<3>(12*t+3))*aPAx0.segment<3>(12*t+3).transpose();
			Matrix3d p3 = s*aPAx0.segment<3>(12*t+6)*aPAx0.segment<3>(12*t+6).transpose() - (rt*PAg.segment<3>(12*t+6))*aPAx0.segment<3>(12*t+6).transpose();
			Matrix3d p4 = s*aPAx0.segment<3>(12*t+9)*aPAx0.segment<3>(12*t+9).transpose() - (rt*PAg.segment<3>(12*t+9))*aPAx0.segment<3>(12*t+9).transpose();
			
			double Es1 = p1(0,0) + p2(0,0) + p3(0,0) + p4(0,0);
			double Es2 = p1(1,1) + p2(1,1) + p3(1,1) + p4(1,1);
			double Es3 = p1(2,2) + p2(2,2) + p3(2,2) + p4(2,2);
			double Es4 = p1(0,1) + p2(0,1) + p3(0,1) + p4(0,1)+ p1(1,0) + p2(1,0) + p3(1,0) + p4(1,0);
			double Es5 = p1(0,2) + p2(0,2) + p3(0,2) + p4(0,2)+ p1(2,0) + p2(2,0) + p3(2,0) + p4(2,0);
			double Es6 = p1(2,1) + p2(2,1) + p3(2,1) + p4(2,1)+ p1(1,2) + p2(1,2) + p3(1,2) + p4(1,2);
			aEs_max[6*t+0] = Es1;
			aEs_max[6*t+1] = Es2;
			aEs_max[6*t+2] = Es3;
			aEs_max[6*t+3] = Es4;
			aEs_max[6*t+4] = Es5;
			aEs_max[6*t+5] = Es6;

		}
		aEs = m.sW().transpose()*aEs_max;
		

		// print("			- Gradients");
		return 1;
	}

	VectorXd fastEs(Mesh& m){
		// print("		Es");
		VectorXd PAg = aPAG*m.red_x() + aPAx0;
		VectorXd ms = m.sW()*m.red_s();
		VectorXd mr = a_Wr*m.red_r();

		aEs_max.setZero();
		for(int t=0; t<m.T().rows(); t++){
			Matrix3d rt = Map<Matrix3d>(mr.segment<9>(9*t).data());
			Matrix3d ut = m.U().block<3,3>(3*t,0).transpose();
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];

			Matrix3d p1 = s*ut*aPAx0.segment<3>(12*t+0)*(ut*aPAx0.segment<3>(12*t+0)).transpose() - (ut*rt*PAg.segment<3>(12*t+0))*(ut*aPAx0.segment<3>(12*t+0)).transpose();
			Matrix3d p2 = s*ut*aPAx0.segment<3>(12*t+3)*(ut*aPAx0.segment<3>(12*t+3)).transpose() - (ut*rt*PAg.segment<3>(12*t+3))*(ut*aPAx0.segment<3>(12*t+3)).transpose();
			Matrix3d p3 = s*ut*aPAx0.segment<3>(12*t+6)*(ut*aPAx0.segment<3>(12*t+6)).transpose() - (ut*rt*PAg.segment<3>(12*t+6))*(ut*aPAx0.segment<3>(12*t+6)).transpose();
			Matrix3d p4 = s*ut*aPAx0.segment<3>(12*t+9)*(ut*aPAx0.segment<3>(12*t+9)).transpose() - (ut*rt*PAg.segment<3>(12*t+9))*(ut*aPAx0.segment<3>(12*t+9)).transpose();
			
			double Es1 = p1(0,0) + p2(0,0) + p3(0,0) + p4(0,0);
			double Es2 = p1(1,1) + p2(1,1) + p3(1,1) + p4(1,1);
			double Es3 = p1(2,2) + p2(2,2) + p3(2,2) + p4(2,2);
			double Es4 = p1(0,1) + p2(0,1) + p3(0,1) + p4(0,1)+ p1(1,0) + p2(1,0) + p3(1,0) + p4(1,0);
			double Es5 = p1(0,2) + p2(0,2) + p3(0,2) + p4(0,2)+ p1(2,0) + p2(2,0) + p3(2,0) + p4(2,0);
			double Es6 = p1(2,1) + p2(2,1) + p3(2,1) + p4(2,1)+ p1(1,2) + p2(1,2) + p3(1,2) + p4(1,2);
			aEs_max[6*t+0] = Es1;
			aEs_max[6*t+1] = Es2;
			aEs_max[6*t+2] = Es3;
			aEs_max[6*t+3] = Es4;
			aEs_max[6*t+4] = Es5;
			aEs_max[6*t+5] = Es6;

		}
		aEs = m.sW().transpose()*aEs_max;
		return aEs;
	}


	VectorXd dEdx(Mesh& m){
		VectorXd PAx = aPA*m.red_x() + aPAx0;
		m.constTimeFPAx0(aFPAx0);
		VectorXd res = (aPA).transpose()*(PAx - aFPAx0);
		return res;
	}

	void itT(Mesh& m){
		//TODO DENSIFY
		VectorXd AtPtFPAx0 = (aPAG).transpose()*aFPAx0;
		VectorXd AtPtPAx0 = (aPAG).transpose()*(aPAx0);
		VectorXd gb = AtPtFPAx0 - AtPtPAx0;

		VectorXd zer = VectorXd::Zero(m.JointY().rows());

		VectorXd gd (gb.size() + zer.size());
		gd<<gb, zer; 

		VectorXd result = aARAPKKTSolver.solve(gd);
		VectorXd gu = result.head(gb.size());
		m.red_x(gu);
 	}

	void itR(Mesh& m, VectorXd& USUtPAx0){
		
		VectorXd PAx = aPAG*m.red_x() + aPAx0;
		VectorXd& mr =m.red_r();
		
		std::map<int, std::vector<int>>& c_e_map = m.r_cluster_elem_map();
		for (int i=0; i<mr.size()/9; i++){
			std::vector<int> cluster_elem = c_e_map[i];
			for(int c=0; c<cluster_elem.size(); c++){
				aePAx[i].row(4*c+0) = PAx.segment<3>(12*cluster_elem[c]);
				aePAx[i].row(4*c+1) = PAx.segment<3>(12*cluster_elem[c]+3);
				aePAx[i].row(4*c+2) = PAx.segment<3>(12*cluster_elem[c]+6);
				aePAx[i].row(4*c+3) = PAx.segment<3>(12*cluster_elem[c]+9);

				aeUSUtPAx0[i].row(4*c+0) = USUtPAx0.segment<3>(12*cluster_elem[c]);
				aeUSUtPAx0[i].row(4*c+1) = USUtPAx0.segment<3>(12*cluster_elem[c]+3);
				aeUSUtPAx0[i].row(4*c+2) = USUtPAx0.segment<3>(12*cluster_elem[c]+6);
				aeUSUtPAx0[i].row(4*c+3) = USUtPAx0.segment<3>(12*cluster_elem[c]+9);
			}


			Matrix3d F = aePAx[i].transpose()*aeUSUtPAx0[i];
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

	bool minimize(Mesh& m){
		print("		+ ARAP minimize");
		VectorXd ms = m.sW()*m.red_s();
		VectorXd USUtPAx0 = VectorXd::Zero(12*m.T().rows());
		for(int t =0; t<m.T().rows(); t++){
			Matrix3d u = m.U().block<3,3>(3*t,0);
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];
			for(int j=0; j<4; j++){
				USUtPAx0.segment<3>(12*t+3*j) = u*s*u.transpose()*aPAx0.segment<3>(12*t+3*j);
			}
		}
		m.constTimeFPAx0(aFPAx0);


		double itRTimes = 0;
		double itTTimes = 0;

		double previous5ItE = Energy(m);
		double oldE = Energy(m);
		double newE;
		for(int i=1; i< 1000; i++){
			// atimer.start();
			itT(m);
			// atimer.stop();
			// itTTimes += atimer.getElapsedTimeInMicroSec();
			// atimer.start();
			itR(m, USUtPAx0);
			// atimer.stop();
			// itRTimes += atimer.getElapsedTimeInMicroSec();
			m.constTimeFPAx0(aFPAx0);

			newE = Energy(m);
			// cout<<i<<", "<<newE-oldE<<endl;
			// VectorXd newEx = dEdx(m);
			// if((newE - oldE)>1e-5 && i>1){
			// 	print("Reduced_Arap::minimize() error. ARAP should monotonically decrease.");
			// 	print(i);
			// 	print(oldE);
			// 	print(newE);
			// 	exit(0);
			// }
			
			if(fabs(newE - oldE)<5e-6){
				// m.constTimeFPAx0(aFPAx0);
				// std::cout<<"		ItTtime: "<<itTTimes<<", ItRtime: "<<itRTimes<<", Iterations: "<<i<<endl;
				return true;
			}
			
			oldE = newE;
		}
		// std::cout<<"		ItTtime: "<<itTTimes<<", ItRtime: "<<itRTimes<<", Iterations: "<<1000<<endl;
		
		// std::cout<<"		NotConvergedARAPDiffInEnergy: "<<Energy(m)-previous5ItE<<std::endl;
		// m.constTimeFPAx0(aFPAx0);
		return false;
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