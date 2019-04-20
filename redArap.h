#ifndef RED_ARAP
#define RED_ARAP

#include <igl/polar_svd.h>
#include "mesh.h"
#include<Eigen/LU>
#include <iostream>
#include <string>
#include <igl/Timer.h>


using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;
typedef Matrix<double, 12, 1> Vector12d;
typedef Matrix<double, 9, 1> Vector9d;


class Reduced_Arap
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
	Reduced_Arap(Mesh& m){
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
		setupFASTARAPTerms(m);
		setupFastPAx0DSTerm(m);
		setupFastEsTerms(m);
		setupFastEnergyTerms(m);

		print("Jacobian solve pre-processing");
		aJacKKT.block(0,0,aExx.rows(), aExx.cols()) = Exx();
		aJacKKT.block(aExx.rows()+aExr.cols(), 0, aCG.rows(), aCG.cols()) = aCG;
		aJacKKT.block(0, aExx.cols()+aExr.cols(), aCG.cols(), aCG.rows())= aCG.transpose();
	}

	void setupFastEnergyTerms(Mesh& m){
		aPAx0squaredNorm = aPAx0.transpose()*aPAx0;
		aPAx0tPAG = aPAx0.transpose()*aPAG;

			std::vector<Trip> SPAx0_trips;
			for(int t=0; t<m.T().rows(); t++){
				for(int j=0; j<4; j++){
					Vector3d PAx0 = aPAx0.segment<3>(12*t + 3*j);
					SPAx0_trips.push_back(Trip( 12*t+3*j+0, 6*t+0 , PAx0[0]));
					SPAx0_trips.push_back(Trip( 12*t+3*j+0, 6*t+3 , PAx0[1]));
					SPAx0_trips.push_back(Trip( 12*t+3*j+0, 6*t+4 , PAx0[2]));

					SPAx0_trips.push_back(Trip( 12*t+3*j+1, 6*t+3 , PAx0[0]));
					SPAx0_trips.push_back(Trip( 12*t+3*j+1, 6*t+1 , PAx0[1]));
					SPAx0_trips.push_back(Trip( 12*t+3*j+1, 6*t+5 , PAx0[2]));

					SPAx0_trips.push_back(Trip( 12*t+3*j+2, 6*t+4 , PAx0[0]));
					SPAx0_trips.push_back(Trip( 12*t+3*j+2, 6*t+5 , PAx0[1]));
					SPAx0_trips.push_back(Trip( 12*t+3*j+2, 6*t+2 , PAx0[2]));
					
				}
			}
			SparseMatrix<double> SMPAx0(12*m.T().rows(), 6*m.T().rows());
			SMPAx0.setFromTriplets(SPAx0_trips.begin(), SPAx0_trips.end());
		aRSPAx0tRSPAx0 = (SMPAx0*m.sW()).transpose()*(SMPAx0*m.sW());

	
			std::vector<Trip> PAx0tR_trips;
			for(int t=0; t<m.T().rows(); t++){
				for(int j=0; j<4; j++){
					Vector3d PAx0 = aPAx0.segment<3>(12*t + 3*j);
					PAx0tR_trips.push_back(Trip(12*t+3*j+0, 9*t+0, PAx0[0]));
					PAx0tR_trips.push_back(Trip(12*t+3*j+0, 9*t+3, PAx0[1]));
					PAx0tR_trips.push_back(Trip(12*t+3*j+0, 9*t+6, PAx0[2]));

					PAx0tR_trips.push_back(Trip(12*t+3*j+1, 9*t+1, PAx0[0]));
					PAx0tR_trips.push_back(Trip(12*t+3*j+1, 9*t+4, PAx0[1]));
					PAx0tR_trips.push_back(Trip(12*t+3*j+1, 9*t+7, PAx0[2]));

					PAx0tR_trips.push_back(Trip(12*t+3*j+2, 9*t+2, PAx0[0]));
					PAx0tR_trips.push_back(Trip(12*t+3*j+2, 9*t+5, PAx0[1]));
					PAx0tR_trips.push_back(Trip(12*t+3*j+2, 9*t+8, PAx0[2]));
				}
			}
			SparseMatrix<double> RMPAx0(12*m.T().rows(), 9*m.T().rows());
			RMPAx0.setFromTriplets(PAx0tR_trips.begin(), PAx0tR_trips.end());
		aPAx0tRSPAx0 = (RMPAx0*a_Wr).transpose()*(SMPAx0*m.sW());
	}

	void setupFASTARAPTerms(Mesh& m){
		print("setupFASTARAPTerms");

		VectorXd AtPtPAx0 = (aPAG).transpose()*(aPAx0);
		aFASTARAPDenseTerms.push_back(AtPtPAx0);
		std::vector<Trip> PAx0_trips;
		for(int t=0; t<m.T().rows(); t++){
			for(int j=0; j<4; j++){
				Vector3d PAx0 = aPAx0.segment<3>(12*t + 3*j);
				PAx0_trips.push_back(Trip( 12*t+3*j+0, 6*t+0 , PAx0[0]));
				PAx0_trips.push_back(Trip( 12*t+3*j+0, 6*t+3 , PAx0[1]));
				PAx0_trips.push_back(Trip( 12*t+3*j+0, 6*t+4 , PAx0[2]));

				PAx0_trips.push_back(Trip( 12*t+3*j+1, 6*t+1 , PAx0[1]));
				PAx0_trips.push_back(Trip( 12*t+3*j+1, 6*t+3 , PAx0[0]));
				PAx0_trips.push_back(Trip( 12*t+3*j+1, 6*t+5 , PAx0[2]));

				PAx0_trips.push_back(Trip( 12*t+3*j+2, 6*t+2 , PAx0[2]));
				PAx0_trips.push_back(Trip( 12*t+3*j+2, 6*t+4 , PAx0[0]));
				PAx0_trips.push_back(Trip( 12*t+3*j+2, 6*t+5 , PAx0[1]));
				
			}
		}
		SparseMatrix<double> MPAx0(12*m.T().rows(), 6*m.T().rows());
		MPAx0.setFromTriplets(PAx0_trips.begin(), PAx0_trips.end());
		MatrixXd MPAx0sW = MPAx0*m.sW();


		MatrixXd& sW = m.sW();
		for(int s=0; s<sW.cols(); s++){
			VectorXd p = MPAx0sW.col(s);
			std::vector<Trip> BigCube_s_trips;
			for(int t=0; t<m.T().rows(); t++){
				for(int jj=0; jj<4; jj++){
					BigCube_s_trips.push_back(Trip(12*t+3*jj+0, 9*t+0, p[12*t+3*jj+0]));
					BigCube_s_trips.push_back(Trip(12*t+3*jj+0, 9*t+1, p[12*t+3*jj+1]));
					BigCube_s_trips.push_back(Trip(12*t+3*jj+0, 9*t+2, p[12*t+3*jj+2]));

					BigCube_s_trips.push_back(Trip(12*t+3*jj+1, 9*t+3, p[12*t+3*jj+0]));
					BigCube_s_trips.push_back(Trip(12*t+3*jj+1, 9*t+4, p[12*t+3*jj+1]));
					BigCube_s_trips.push_back(Trip(12*t+3*jj+1, 9*t+5, p[12*t+3*jj+2]));

					BigCube_s_trips.push_back(Trip(12*t+3*jj+2, 9*t+6, p[12*t+3*jj+0]));
					BigCube_s_trips.push_back(Trip(12*t+3*jj+2, 9*t+7, p[12*t+3*jj+1]));
					BigCube_s_trips.push_back(Trip(12*t+3*jj+2, 9*t+8, p[12*t+3*jj+2]));
				}
			}


			SparseMatrix<double> BigCube_s(12*m.T().rows(), 9*m.T().rows());
			BigCube_s.setFromTriplets(BigCube_s_trips.begin(), BigCube_s_trips.end());

			MatrixXd ClusteredCube_s = (aPAG).transpose()*BigCube_s*a_Wr;
			aFASTARAPCubeGtAtPtRSPAx0.push_back(ClusteredCube_s);
		}


		for(int g=0; g<m.red_r().size()/9; g++){
			std::vector<Trip> try_trips1;
			std::vector<Trip> try_trips2;
			std::vector<Trip> try_trips3;
			SparseMatrix<double>& B = m.RotBLOCK()[g];
			for(int i=0; i< B.cols()/3; i++){
				try_trips1.push_back(Trip(i, 3*i+0, 1));
				try_trips2.push_back(Trip(i, 3*i+1, 1));
				try_trips3.push_back(Trip(i, 3*i+2, 1));

			}

			SparseMatrix<double> try1(B.cols()/3, B.cols());
			SparseMatrix<double> try2(B.cols()/3, B.cols());
			SparseMatrix<double> try3(B.cols()/3, B.cols());
			try1.setFromTriplets(try_trips1.begin(), try_trips1.end());
			try2.setFromTriplets(try_trips2.begin(), try_trips2.end());
			try3.setFromTriplets(try_trips3.begin(), try_trips3.end());

			MatrixXd BtPAG = B.transpose()*aPAG;
			MatrixXd BtPAx0 = B.transpose()*aPAx0;
			MatrixXd BtMPAx0sW = B.transpose()*MPAx0sW;

			MatrixXd BtPAG1 = try1*BtPAG;
			MatrixXd BtPAG2 = try2*BtPAG;
			MatrixXd BtPAG3 = try3*BtPAG;

			VectorXd BtPAx0_1 = try1*BtPAx0;
			VectorXd BtPAx0_2 = try2*BtPAx0;
			VectorXd BtPAx0_3 = try3*BtPAx0;

			MatrixXd BtMPAx0sW1 = try1*BtMPAx0sW;
			MatrixXd BtMPAx0sW2 = try2*BtMPAx0sW;
			MatrixXd BtMPAx0sW3 = try3*BtMPAx0sW;

			MatrixXd zs11 = BtPAG1.transpose()*BtMPAx0sW1;
			MatrixXd zs12 = BtPAG1.transpose()*BtMPAx0sW2;
			MatrixXd zs13 = BtPAG1.transpose()*BtMPAx0sW3;
			MatrixXd zs21 = BtPAG2.transpose()*BtMPAx0sW1;
			MatrixXd zs22 = BtPAG2.transpose()*BtMPAx0sW2;
			MatrixXd zs23 = BtPAG2.transpose()*BtMPAx0sW3;
			MatrixXd zs31 = BtPAG3.transpose()*BtMPAx0sW1;
			MatrixXd zs32 = BtPAG3.transpose()*BtMPAx0sW2;
			MatrixXd zs33 = BtPAG3.transpose()*BtMPAx0sW3;

			VectorXd ps11 = BtPAx0_1.transpose()*BtMPAx0sW1;
			VectorXd ps12 = BtPAx0_1.transpose()*BtMPAx0sW2;
			VectorXd ps13 = BtPAx0_1.transpose()*BtMPAx0sW3;
			VectorXd ps21 = BtPAx0_2.transpose()*BtMPAx0sW1;
			VectorXd ps22 = BtPAx0_2.transpose()*BtMPAx0sW2;
			VectorXd ps23 = BtPAx0_2.transpose()*BtMPAx0sW3;
			VectorXd ps31 = BtPAx0_3.transpose()*BtMPAx0sW1;
			VectorXd ps32 = BtPAx0_3.transpose()*BtMPAx0sW2;
			VectorXd ps33 = BtPAx0_3.transpose()*BtMPAx0sW3;

			aFASTARAPItRTerms1.push_back(zs11);
			aFASTARAPItRTerms1.push_back(zs12);
			aFASTARAPItRTerms1.push_back(zs13);
			aFASTARAPItRTerms1.push_back(zs21);
			aFASTARAPItRTerms1.push_back(zs22);
			aFASTARAPItRTerms1.push_back(zs23);
			aFASTARAPItRTerms1.push_back(zs31);
			aFASTARAPItRTerms1.push_back(zs32);
			aFASTARAPItRTerms1.push_back(zs33);

			aFASTARAPItRTerms2.push_back(ps11);
			aFASTARAPItRTerms2.push_back(ps12);
			aFASTARAPItRTerms2.push_back(ps13);
			aFASTARAPItRTerms2.push_back(ps21);
			aFASTARAPItRTerms2.push_back(ps22);
			aFASTARAPItRTerms2.push_back(ps23);
			aFASTARAPItRTerms2.push_back(ps31);
			aFASTARAPItRTerms2.push_back(ps32);
			aFASTARAPItRTerms2.push_back(ps33);

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

	void setupFastPAx0DSTerm(Mesh& m){
		aPAx0DS = MatrixXd::Zero(aPAx0.size(), m.red_s().size());
		for(int s=0; s<m.red_s().size()/6; s++){
			VectorXd sWx = m.sW().col(6*s+0);
			VectorXd sWy = m.sW().col(6*s+1);
			VectorXd sWz = m.sW().col(6*s+2);
			VectorXd sW01 = m.sW().col(6*s+3);
			VectorXd sW02 = m.sW().col(6*s+4);
			VectorXd sW12 = m.sW().col(6*s+5);

			VectorXd diag_x = VectorXd::Zero(12*m.T().rows());
			VectorXd diag_y = VectorXd::Zero(12*m.T().rows());
			VectorXd diag_z = VectorXd::Zero(12*m.T().rows());
			VectorXd diag_1 = VectorXd::Zero(12*m.T().rows());
			VectorXd diag_2 = VectorXd::Zero(12*m.T().rows());
			VectorXd diag_3 = VectorXd::Zero(12*m.T().rows());
			for(int i=0; i<m.T().rows(); i++){
				diag_x[12*i+0] = sWx[6*i];
				diag_x[12*i+3] = sWx[6*i];
				diag_x[12*i+6] = sWx[6*i];
				diag_x[12*i+9] = sWx[6*i];

				diag_y[12*i+0+1] = sWy[6*i+1];
				diag_y[12*i+3+1] = sWy[6*i+1];
				diag_y[12*i+6+1] = sWy[6*i+1];
				diag_y[12*i+9+1] = sWy[6*i+1];

				diag_z[12*i+0+2] = sWz[6*i+2];
				diag_z[12*i+3+2] = sWz[6*i+2];
				diag_z[12*i+6+2] = sWz[6*i+2];
				diag_z[12*i+9+2] = sWz[6*i+2];

				diag_1[12*i+0] = sW01[6*i+3];
				diag_1[12*i+3] = sW01[6*i+3];
				diag_1[12*i+6] = sW01[6*i+3];
				diag_1[12*i+9] = sW01[6*i+3];

				diag_2[12*i+0+1] = sW02[6*i+4];
				diag_2[12*i+3+1] = sW02[6*i+4];
				diag_2[12*i+6+1] = sW02[6*i+4];
				diag_2[12*i+9+1] = sW02[6*i+4];

				diag_3[12*i+0] = sW12[6*i+5];
				diag_3[12*i+3] = sW12[6*i+5];
				diag_3[12*i+6] = sW12[6*i+5];
				diag_3[12*i+9] = sW12[6*i+5];
			}

			aPAx0DS.col(6*s+0) += aPAx0.cwiseProduct(diag_x);
			aPAx0DS.col(6*s+1) += aPAx0.cwiseProduct(diag_y);
			aPAx0DS.col(6*s+2) += aPAx0.cwiseProduct(diag_z);
			
			aPAx0DS.col(6*s+3).tail(aPAx0.size()-1) += aPAx0.head(aPAx0.size()-1).cwiseProduct(diag_1.head(aPAx0.size()-1));
			aPAx0DS.col(6*s+3).head(aPAx0.size()-1) += aPAx0.tail(aPAx0.size()-1).cwiseProduct(diag_1.head(aPAx0.size()-1));

			aPAx0DS.col(6*s+4).tail(aPAx0.size()-2) += aPAx0.head(aPAx0.size()-2).cwiseProduct(diag_3.head(aPAx0.size()-2)); 
			aPAx0DS.col(6*s+4).head(aPAx0.size()-2) += aPAx0.tail(aPAx0.size()-2).cwiseProduct(diag_3.head(aPAx0.size()-2));

			aPAx0DS.col(6*s+5).tail(aPAx0.size()-1) += aPAx0.head(aPAx0.size()-1).cwiseProduct(diag_2.head(aPAx0.size()-1));
			aPAx0DS.col(6*s+5).head(aPAx0.size()-1) += aPAx0.tail(aPAx0.size()-1).cwiseProduct(diag_2.head(aPAx0.size()-1));
		}
	

		// aPAx0DS.resize(aPAx0.size(), m.red_s().size());
		// std::vector<Trip> temptrips;
		// for(int i=0; i<m.T().rows(); i++){
		// 	temptrips.push_back(Trip(12*i+0, 6*i+0, aPAx0[12*i+0]));
		// 	temptrips.push_back(Trip(12*i+3, 6*i+0, aPAx0[12*i+3]));
		// 	temptrips.push_back(Trip(12*i+6, 6*i+0, aPAx0[12*i+6]));
		// 	temptrips.push_back(Trip(12*i+9, 6*i+0, aPAx0[12*i+9]));

		// 	temptrips.push_back(Trip(12*i+0+1, 6*i+1, aPAx0[12*i+0+1]));
		// 	temptrips.push_back(Trip(12*i+3+1, 6*i+1, aPAx0[12*i+3+1]));
		// 	temptrips.push_back(Trip(12*i+6+1, 6*i+1, aPAx0[12*i+6+1]));
		// 	temptrips.push_back(Trip(12*i+9+1, 6*i+1, aPAx0[12*i+9+1]));

		// 	temptrips.push_back(Trip(12*i+0+2, 6*i+2, aPAx0[12*i+0+2]));
		// 	temptrips.push_back(Trip(12*i+3+2, 6*i+2, aPAx0[12*i+3+2]));
		// 	temptrips.push_back(Trip(12*i+6+2, 6*i+2, aPAx0[12*i+6+2]));
		// 	temptrips.push_back(Trip(12*i+9+2, 6*i+2, aPAx0[12*i+9+2]));

		// 	temptrips.push_back(Trip(12*i+0+0, 6*i+3, aPAx0[12*i+0+1]));
		// 	temptrips.push_back(Trip(12*i+0+1, 6*i+3, aPAx0[12*i+0+0]));
		// 	temptrips.push_back(Trip(12*i+3+0, 6*i+3, aPAx0[12*i+3+1]));
		// 	temptrips.push_back(Trip(12*i+3+1, 6*i+3, aPAx0[12*i+3+0]));
		// 	temptrips.push_back(Trip(12*i+6+0, 6*i+3, aPAx0[12*i+6+1]));
		// 	temptrips.push_back(Trip(12*i+6+1, 6*i+3, aPAx0[12*i+6+0]));
		// 	temptrips.push_back(Trip(12*i+9+0, 6*i+3, aPAx0[12*i+9+1]));
		// 	temptrips.push_back(Trip(12*i+9+1, 6*i+3, aPAx0[12*i+9+0]));

		// 	temptrips.push_back(Trip(12*i+0+0, 6*i+4, aPAx0[12*i+0+2]));
		// 	temptrips.push_back(Trip(12*i+0+2, 6*i+4, aPAx0[12*i+0+0]));
		// 	temptrips.push_back(Trip(12*i+3+0, 6*i+4, aPAx0[12*i+3+2]));
		// 	temptrips.push_back(Trip(12*i+3+2, 6*i+4, aPAx0[12*i+3+0]));
		// 	temptrips.push_back(Trip(12*i+6+0, 6*i+4, aPAx0[12*i+6+2]));
		// 	temptrips.push_back(Trip(12*i+6+2, 6*i+4, aPAx0[12*i+6+0]));
		// 	temptrips.push_back(Trip(12*i+9+0, 6*i+4, aPAx0[12*i+9+2]));
		// 	temptrips.push_back(Trip(12*i+9+2, 6*i+4, aPAx0[12*i+9+0]));

		// 	temptrips.push_back(Trip(12*i+0+1, 6*i+5, aPAx0[12*i+0+2]));
		// 	temptrips.push_back(Trip(12*i+0+2, 6*i+5, aPAx0[12*i+0+1]));
		// 	temptrips.push_back(Trip(12*i+3+1, 6*i+5, aPAx0[12*i+3+2]));
		// 	temptrips.push_back(Trip(12*i+3+2, 6*i+5, aPAx0[12*i+3+1]));
		// 	temptrips.push_back(Trip(12*i+6+1, 6*i+5, aPAx0[12*i+6+2]));
		// 	temptrips.push_back(Trip(12*i+6+2, 6*i+5, aPAx0[12*i+6+1]));
		// 	temptrips.push_back(Trip(12*i+9+1, 6*i+5, aPAx0[12*i+9+2]));
		// 	temptrips.push_back(Trip(12*i+9+2, 6*i+5, aPAx0[12*i+9+1]));
		// }
		// aPAx0DS.setFromTriplets(temptrips.begin(), temptrips.end());
	}

	void setupFastEsTerms(Mesh& m){
		std::vector<Trip> SPAx0_trips;
		for(int t=0; t<m.T().rows(); t++){
			for(int j=0; j<4; j++){
				Vector3d PAx0 = aPAx0.segment<3>(12*t + 3*j);
				SPAx0_trips.push_back(Trip( 12*t+3*j+0, 6*t+0 , PAx0[0]));
				SPAx0_trips.push_back(Trip( 12*t+3*j+0, 6*t+3 , PAx0[1]));
				SPAx0_trips.push_back(Trip( 12*t+3*j+0, 6*t+4 , PAx0[2]));

				SPAx0_trips.push_back(Trip( 12*t+3*j+1, 6*t+1 , PAx0[1]));
				SPAx0_trips.push_back(Trip( 12*t+3*j+1, 6*t+3 , PAx0[0]));
				SPAx0_trips.push_back(Trip( 12*t+3*j+1, 6*t+5 , PAx0[2]));

				SPAx0_trips.push_back(Trip( 12*t+3*j+2, 6*t+2 , PAx0[2]));
				SPAx0_trips.push_back(Trip( 12*t+3*j+2, 6*t+4 , PAx0[0]));
				SPAx0_trips.push_back(Trip( 12*t+3*j+2, 6*t+5 , PAx0[1]));
				
			}
		}
		SparseMatrix<double> SMPAx0(12*m.T().rows(), 6*m.T().rows());
		SMPAx0.setFromTriplets(SPAx0_trips.begin(), SPAx0_trips.end());
		MatrixXd SMPAx0sW = SMPAx0*m.sW();
		aFastEsTerm1 = SMPAx0sW.transpose()*aPAx0DS;

		std::vector<Trip> RtPAx0_trips;
		for(int t=0; t<m.T().rows(); t++){
			for(int j=0; j<4; j++){
				Vector3d PAx0 = aPAx0.segment<3>(12*t + 3*j);
				RtPAx0_trips.push_back(Trip( 12*t+3*j+0, 9*t+0 , PAx0[0]));
				RtPAx0_trips.push_back(Trip( 12*t+3*j+0, 9*t+3 , PAx0[1]));
				RtPAx0_trips.push_back(Trip( 12*t+3*j+0, 9*t+6 , PAx0[2]));

				RtPAx0_trips.push_back(Trip( 12*t+3*j+1, 9*t+1 , PAx0[0]));
				RtPAx0_trips.push_back(Trip( 12*t+3*j+1, 9*t+4 , PAx0[1]));
				RtPAx0_trips.push_back(Trip( 12*t+3*j+1, 9*t+7 , PAx0[2]));

				RtPAx0_trips.push_back(Trip( 12*t+3*j+2, 9*t+2 , PAx0[0]));
				RtPAx0_trips.push_back(Trip( 12*t+3*j+2, 9*t+5 , PAx0[1]));
				RtPAx0_trips.push_back(Trip( 12*t+3*j+2, 9*t+8 , PAx0[2]));
				
			}
		}
		SparseMatrix<double> RMPAx0(12*m.T().rows(), 9*m.T().rows());
		RMPAx0.setFromTriplets(RtPAx0_trips.begin(), RtPAx0_trips.end());
		SparseMatrix<double> RMPAx0Wr= RMPAx0*a_Wr;
		aFastEsTerm2 = RMPAx0Wr.transpose()*aPAx0DS;

		for(int g=0; g<m.red_x().size(); g++){
			std::vector<Trip> RtPAGzi_trips;
			for(int t=0; t<m.T().rows(); t++){
				for(int j=0; j<4; j++){
					Vector3d PAGi = aPAG.col(g).segment<3>(12*t + 3*j);
					RtPAGzi_trips.push_back(Trip( 12*t+3*j+0, 9*t+0 , PAGi[0]));
					RtPAGzi_trips.push_back(Trip( 12*t+3*j+0, 9*t+3 , PAGi[1]));
					RtPAGzi_trips.push_back(Trip( 12*t+3*j+0, 9*t+6 , PAGi[2]));

					RtPAGzi_trips.push_back(Trip( 12*t+3*j+1, 9*t+1 , PAGi[0]));
					RtPAGzi_trips.push_back(Trip( 12*t+3*j+1, 9*t+4 , PAGi[1]));
					RtPAGzi_trips.push_back(Trip( 12*t+3*j+1, 9*t+7 , PAGi[2]));

					RtPAGzi_trips.push_back(Trip( 12*t+3*j+2, 9*t+2 , PAGi[0]));
					RtPAGzi_trips.push_back(Trip( 12*t+3*j+2, 9*t+5 , PAGi[1]));
					RtPAGzi_trips.push_back(Trip( 12*t+3*j+2, 9*t+8 , PAGi[2]));
					
				}
			}
			SparseMatrix<double> RMPAGzi(12*m.T().rows(), 9*m.T().rows());
			RMPAGzi.setFromTriplets(RtPAGzi_trips.begin(), RtPAGzi_trips.end());
			SparseMatrix<double> RMPAGzi0Wr= RMPAGzi*a_Wr;
			MatrixXd EsTerm3 = RMPAGzi0Wr.transpose()*aPAx0DS;
			aFastEsTerm3s.push_back(EsTerm3);
		}
	}

	double Energy(Mesh& m){
		VectorXd GtAtPtFPAx0 = VectorXd::Zero(m.red_x().size());
		for(int s=0; s<m.red_s().size(); s++){
			GtAtPtFPAx0 += m.red_s()[s]*(aFASTARAPCubeGtAtPtRSPAx0[s]*m.red_r());
		}

		double t1 = 0.5*m.red_x().transpose()*aExx*m.red_x();
		double t2 = 0.5*aPAx0squaredNorm;
		double t3 = 0.5*m.red_s().transpose()*aRSPAx0tRSPAx0*m.red_s();
		double t4 = aPAx0tPAG.dot(m.red_x());
		double t5 = GtAtPtFPAx0.dot(m.red_x());
		double t6 = m.red_r().transpose()*aPAx0tRSPAx0*m.red_s();
		double En = t1+t2+t3+t4-t5-t6;
		
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

	VectorXd Jacobians(Mesh& m){
		// int h = Hessians(m);
		int gg = Gradients(m);
		return aEs;
		
		// //Dense
		// aJacKKT.setZero();
		// aJacConstrains.setZero();
		// //col1
		
		// aJacKKT.block(aExx.rows(), 0, aExr.cols(), aExr.rows()) = Exr().transpose();
		
		// //col2
		// aJacKKT.block(0,aExx.cols(),aExr.rows(), aExr.cols()) = Exr();
		// aJacKKT.block(aExr.rows(), aExx.cols(), aErr.rows(), aErr.cols()) = Err();
		// // // //col3
		
		// // //rhs
		// aJacConstrains.block(0,0, aExs.rows(), aExs.cols()) = Exs();
		// aJacConstrains.block(aExs.rows(), 0, aErs.rows(), aErs.cols()) = Ers();
		// // print("before LU");		
		// VectorXd ExEr(aEx.size()+aEr.size());
		// ExEr<<aEx,aEr;
		// VectorXd PtExEr = adjointP.transpose()*ExEr;
		// VectorXd g = aJacKKT.fullPivLu().solve(PtExEr);
		// aDEDs = aJacConstrains.transpose()*g + aEs;
		// std::ofstream ExxFile("Exx.mat");
		// if (ExxFile.is_open())
		// {
		// 	ExxFile << aExx;
		// }
		// ExxFile.close();


	
		return aDEDs;
	}

	int Hessians(Mesh& m){
		int w_size = m.red_w().size();
		int z_size = m.red_x().size();
		int s_size = m.red_s().size();
		int t_size = m.T().rows();

		// print("		+Hessians");
		aExr_max_trips.clear();
		aErr_max_trips.clear();
		aExs_max_trips.clear();
		aErs_max_trips.clear();
		// Exr_trip.reserve(3*t_size*z_size);
		// Err_trip.reserve(3*3*t_size);
		// Exs_trip.reserve();
		// Ers_trip.reserve();
		//Exx is constant

		Matrix3d Jx = cross_prod_mat(1,0,0);
		Matrix3d Jy = cross_prod_mat(0,1,0);
		Matrix3d Jz = cross_prod_mat(0,0,1);
		VectorXd ms = m.sW()*m.red_s();
		VectorXd mr = a_Wr*m.red_r();
		VectorXd PAg = aPAG*m.red_x() + aPAx0;

		// Exr
		print("		Exr");
		for(int t=0; t<m.T().rows(); t++){
			//Tet
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];
			
			Matrix3d r0= Map<Matrix3d>(mr.segment<9>(9*t).data()).transpose();

			Matrix3d r1 = r0*Jx;
			Matrix3d r2 = r0*Jy;
			Matrix3d r3 = r0*Jz;

			for(int e=0; e<4; e++){
				//Vert on tet
				for(int a =0; a<3; a++){
					Matrix3d p1 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+0)*(s*aPAx0.segment<3>(12*t+0)).transpose();
					Matrix3d p2 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+3)*(s*aPAx0.segment<3>(12*t+3)).transpose();
					Matrix3d p3 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+6)*(s*aPAx0.segment<3>(12*t+6)).transpose();
					Matrix3d p4 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+9)*(s*aPAx0.segment<3>(12*t+9)).transpose();
					

					double Exar1 = p1.cwiseProduct(r1).sum() + p2.cwiseProduct(r1).sum() + p3.cwiseProduct(r1).sum() + p4.cwiseProduct(r1).sum();
					double Exar2 = p1.cwiseProduct(r2).sum() + p2.cwiseProduct(r2).sum() + p3.cwiseProduct(r2).sum() + p4.cwiseProduct(r2).sum();
					double Exar3 = p1.cwiseProduct(r3).sum() + p2.cwiseProduct(r3).sum() + p3.cwiseProduct(r3).sum() + p4.cwiseProduct(r3).sum();

					aExr_max_trips.push_back(Trip(3*m.T().row(t)[e]+a,3*t+0, Exar1));
					aExr_max_trips.push_back(Trip(3*m.T().row(t)[e]+a,3*t+1, Exar2));
					aExr_max_trips.push_back(Trip(3*m.T().row(t)[e]+a,3*t+2, Exar3));
				}
			}
		}

		//Err
		print("		Err");
		for(int t=0; t<m.T().rows(); t++){
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];

			Matrix3d r0= Map<Matrix3d>(mr.segment<9>(9*t).data()).transpose();
			Matrix3d r1 = r0*0.5*(Jx*Jx + Jx*Jx);
			Matrix3d r2 = r0*0.5*(Jx*Jy + Jy*Jx);
			Matrix3d r3 = r0*0.5*(Jx*Jz + Jz*Jx);
			Matrix3d r5 = r0*0.5*(Jy*Jy + Jy*Jy);
			Matrix3d r6 = r0*0.5*(Jy*Jz + Jz*Jy);
			Matrix3d r9 = r0*0.5*(Jz*Jz + Jz*Jz);

			Matrix3d pr1 = -PAg.segment<3>(12*t+0)*(s*aPAx0.segment<3>(12*t+0)).transpose();
			Matrix3d pr2 = -PAg.segment<3>(12*t+3)*(s*aPAx0.segment<3>(12*t+3)).transpose();
			Matrix3d pr3 = -PAg.segment<3>(12*t+6)*(s*aPAx0.segment<3>(12*t+6)).transpose();
			Matrix3d pr4 = -PAg.segment<3>(12*t+9)*(s*aPAx0.segment<3>(12*t+9)).transpose();

			double v00 = pr1.cwiseProduct(r1).sum()+pr2.cwiseProduct(r1).sum()+pr3.cwiseProduct(r1).sum()+pr4.cwiseProduct(r1).sum();
			double v01 = pr1.cwiseProduct(r2).sum()+pr2.cwiseProduct(r2).sum()+pr3.cwiseProduct(r2).sum()+pr4.cwiseProduct(r2).sum();
			double v02 = pr1.cwiseProduct(r3).sum()+pr2.cwiseProduct(r3).sum()+pr3.cwiseProduct(r3).sum()+pr4.cwiseProduct(r3).sum();
			double v11 = pr1.cwiseProduct(r5).sum()+pr2.cwiseProduct(r5).sum()+pr3.cwiseProduct(r5).sum()+pr4.cwiseProduct(r5).sum();
			double v12 = pr1.cwiseProduct(r6).sum()+pr2.cwiseProduct(r6).sum()+pr3.cwiseProduct(r6).sum()+pr4.cwiseProduct(r6).sum();
			double v22 = pr1.cwiseProduct(r9).sum()+pr2.cwiseProduct(r9).sum()+pr3.cwiseProduct(r9).sum()+pr4.cwiseProduct(r9).sum();
			
			aErr_max_trips.push_back(Trip(3*t+0,3*t+0, v00));
			aErr_max_trips.push_back(Trip(3*t+0,3*t+1, v01));
			aErr_max_trips.push_back(Trip(3*t+0,3*t+2, v02));
			aErr_max_trips.push_back(Trip(3*t+1,3*t+1, v11));
			aErr_max_trips.push_back(Trip(3*t+1,3*t+2, v12));
			aErr_max_trips.push_back(Trip(3*t+2,3*t+2, v22));
			aErr_max_trips.push_back(Trip(3*t+1,3*t+0, v01));
			aErr_max_trips.push_back(Trip(3*t+2,3*t+0, v02));
			aErr_max_trips.push_back(Trip(3*t+2,3*t+1, v12));
		}
	
		//Exs
		print("		Exs");
		for(int t =0; t<m.T().rows(); t++){
			//Tet
			Matrix3d r = Map<Matrix3d>(mr.segment<9>(9*t).data()).transpose();
			for(int e=0; e<4; e++){
				//Vert on tet
				for(int a=0; a<3; a++){
					//x, y, or z axis
					Vector3d GtAtPtRU_row1 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+0).transpose()*r;
					Vector3d GtAtPtRU_row2 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+3).transpose()*r;
					Vector3d GtAtPtRU_row3 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+6).transpose()*r;
					Vector3d GtAtPtRU_row4 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+9).transpose()*r;
					Matrix3d p1 = GtAtPtRU_row1*aPAx0.segment<3>(12*t+0).transpose();
					Matrix3d p2 = GtAtPtRU_row2*aPAx0.segment<3>(12*t+3).transpose();
					Matrix3d p3 = GtAtPtRU_row3*aPAx0.segment<3>(12*t+6).transpose();
					Matrix3d p4 = GtAtPtRU_row4*aPAx0.segment<3>(12*t+9).transpose();

					double Exas1 = p1(0,0) + p2(0,0) + p3(0,0) + p4(0,0);
					double Exas2 = p1(1,1) + p2(1,1) + p3(1,1) + p4(1,1);
					double Exas3 = p1(2,2) + p2(2,2) + p3(2,2) + p4(2,2);
					double Exas4 = p1(0,1) + p2(0,1) + p3(0,1) + p4(0,1)+ p1(1,0) + p2(1,0) + p3(1,0) + p4(1,0);
					double Exas5 = p1(0,2) + p2(0,2) + p3(0,2) + p4(0,2)+ p1(2,0) + p2(2,0) + p3(2,0) + p4(2,0);
					double Exas6 = p1(2,1) + p2(2,1) + p3(2,1) + p4(2,1)+ p1(1,2) + p2(1,2) + p3(1,2) + p4(1,2);
					aExs_max_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+0, Exas1));
					aExs_max_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+1, Exas2));
					aExs_max_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+2, Exas3));
					aExs_max_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+3, Exas4));
					aExs_max_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+4, Exas5));
					aExs_max_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+5, Exas6));
				}
			}
		}

		//Ers
		print("		Ers");
		Matrix3d Id3 = Matrix3d::Identity();
		for(int t=0; t<m.T().rows(); t++){
			//Tet
			Matrix3d r0= Map<Matrix3d>(mr.segment<9>(9*t).data()).transpose();
			Matrix3d r1 = r0*Jx;
			Matrix3d r2 = r0*Jy;
			Matrix3d r3 = r0*Jz;

			Matrix<double, 12,3> innerContraction;
			for(int a =0; a<3; a++){
				Matrix3d pr1 = -PAg.segment<3>(12*t+0)*Id3.col(a).transpose();
				Matrix3d pr2 = -PAg.segment<3>(12*t+3)*Id3.col(a).transpose();
				Matrix3d pr3 = -PAg.segment<3>(12*t+6)*Id3.col(a).transpose();
				Matrix3d pr4 = -PAg.segment<3>(12*t+9)*Id3.col(a).transpose();

				Vector3d inner1(pr1.cwiseProduct(r1).sum(), pr1.cwiseProduct(r2).sum(), pr1.cwiseProduct(r3).sum());
				Vector3d inner2(pr2.cwiseProduct(r1).sum(), pr2.cwiseProduct(r2).sum(), pr2.cwiseProduct(r3).sum());
				Vector3d inner3(pr3.cwiseProduct(r1).sum(), pr3.cwiseProduct(r2).sum(), pr3.cwiseProduct(r3).sum());
				Vector3d inner4(pr4.cwiseProduct(r1).sum(), pr4.cwiseProduct(r2).sum(), pr4.cwiseProduct(r3).sum());

				innerContraction.row(3*0+a)=inner1;
				innerContraction.row(3*1+a)=inner2;
				innerContraction.row(3*2+a)=inner3;
				innerContraction.row(3*3+a)=inner4;
			}

			for(int a =0; a<3; a++){
				Matrix3d p1 = innerContraction.col(a).segment<3>(0)*aPAx0.segment<3>(12*t+0).transpose();
				Matrix3d p2 = innerContraction.col(a).segment<3>(3)*aPAx0.segment<3>(12*t+3).transpose();
				Matrix3d p3 = innerContraction.col(a).segment<3>(6)*aPAx0.segment<3>(12*t+6).transpose();
				Matrix3d p4 = innerContraction.col(a).segment<3>(9)*aPAx0.segment<3>(12*t+9).transpose();

				double Eras1 = p1(0,0) + p2(0,0) + p3(0,0) + p4(0,0);
				double Eras2 = p1(1,1) + p2(1,1) + p3(1,1) + p4(1,1);
				double Eras3 = p1(2,2) + p2(2,2) + p3(2,2) + p4(2,2);
				double Eras4 = p1(0,1) + p2(0,1) + p3(0,1) + p4(0,1)+ p1(1,0) + p2(1,0) + p3(1,0) + p4(1,0);
				double Eras5 = p1(0,2) + p2(0,2) + p3(0,2) + p4(0,2)+ p1(2,0) + p2(2,0) + p3(2,0) + p4(2,0);
				double Eras6 = p1(2,1) + p2(2,1) + p3(2,1) + p4(2,1)+ p1(1,2) + p2(1,2) + p3(1,2) + p4(1,2);
				
				aErs_max_trips.push_back(Trip(3*t+a, 6*t+0, Eras1));
				aErs_max_trips.push_back(Trip(3*t+a, 6*t+1, Eras2));
				aErs_max_trips.push_back(Trip(3*t+a, 6*t+2, Eras3));
				aErs_max_trips.push_back(Trip(3*t+a, 6*t+3, Eras4));
				aErs_max_trips.push_back(Trip(3*t+a, 6*t+4, Eras5));
				aErs_max_trips.push_back(Trip(3*t+a, 6*t+5, Eras6));

			}	
		}

		print("		1");
		aErr_max.setFromTriplets(aErr_max_trips.begin(), aErr_max_trips.end());
		print("		2");
		aErs_max.setFromTriplets(aErs_max_trips.begin(), aErs_max_trips.end());
		print("		3");
		aExs_max.setFromTriplets(aExs_max_trips.begin(), aExs_max_trips.end());
		print("		4");
		aExr_max.setFromTriplets(aExr_max_trips.begin(), aExr_max_trips.end());

		SparseMatrix<double> temp1 = a_Ww.transpose()*aErr_max*a_Ww;
		MatrixXd temp2 = m.G().transpose()*aExr_max*a_Ww;
		MatrixXd temp3 = a_Ww.transpose()*aErs_max*m.sW();
		MatrixXd temp4 = m.G().transpose()*aExs_max*m.sW();
		aErr = MatrixXd(temp1);
		aExr = MatrixXd(temp2);
		aErs = MatrixXd(temp3);
		aExs = MatrixXd(temp4);

		// aErr_trips = to_triplets(aErr);
		// aExr_trips = to_triplets(aExr);
		// aErs_trips = to_triplets(aErs);
		// aExs_trips = to_triplets(aExs);


		// print("		-Hessians");		
		return 1;
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
		aEs = aFastEsTerm1.transpose()*m.red_s() - aFastEsTerm2.transpose()*m.red_r();
		for(int g=0; g<m.red_x().size(); g++){
			aEs -= m.red_x()[g]*aFastEsTerm3s[g].transpose()*m.red_r();
		}
		return aEs;
	}


	VectorXd dEdx(Mesh& m){
		VectorXd GtAtPtFPAx0 = VectorXd::Zero(m.red_x().size());
		for(int s=0; s<m.red_s().size(); s++){
			GtAtPtFPAx0 += m.red_s()[s]*(aFASTARAPCubeGtAtPtRSPAx0[s]*m.red_r());
		}

		VectorXd res = aExx*m.red_x() + aFASTARAPDenseTerms[0] - GtAtPtFPAx0;

		return res;
	}

	void itT(Mesh& m){
		//TODO DENSIFY
		VectorXd GtAtPtFPAx0 = VectorXd::Zero(m.red_x().size());
		for(int s=0; s<m.red_s().size(); s++){
			GtAtPtFPAx0 += m.red_s()[s]*(aFASTARAPCubeGtAtPtRSPAx0[s]*m.red_r());
		}		
		VectorXd gb = GtAtPtFPAx0 - aFASTARAPDenseTerms[0];
		VectorXd zer = VectorXd::Zero(m.JointY().rows());

		VectorXd gd (gb.size() + zer.size());
		gd<<gb, zer; 
		VectorXd result = aARAPKKTSolver.solve(gd);
		VectorXd gu = result.head(gb.size());
		m.red_x(gu);
 	}

	void itR(Mesh& m){
		
		VectorXd& mr =m.red_r();
		// std::map<int, std::vector<int>>& c_e_map = m.r_cluster_elem_map();
		for (int i=0; i<mr.size()/9; i++){
			double r11 = m.red_x().transpose()*aFASTARAPItRTerms1[9*i+0]*m.red_s() + aFASTARAPItRTerms2[9*i+0].dot(m.red_s());
			double r12 = m.red_x().transpose()*aFASTARAPItRTerms1[9*i+1]*m.red_s() + aFASTARAPItRTerms2[9*i+1].dot(m.red_s());
			double r13 = m.red_x().transpose()*aFASTARAPItRTerms1[9*i+2]*m.red_s() + aFASTARAPItRTerms2[9*i+2].dot(m.red_s());
			double r21 = m.red_x().transpose()*aFASTARAPItRTerms1[9*i+3]*m.red_s() + aFASTARAPItRTerms2[9*i+3].dot(m.red_s());
			double r22 = m.red_x().transpose()*aFASTARAPItRTerms1[9*i+4]*m.red_s() + aFASTARAPItRTerms2[9*i+4].dot(m.red_s());
			double r23 = m.red_x().transpose()*aFASTARAPItRTerms1[9*i+5]*m.red_s() + aFASTARAPItRTerms2[9*i+5].dot(m.red_s());
			double r31 = m.red_x().transpose()*aFASTARAPItRTerms1[9*i+6]*m.red_s() + aFASTARAPItRTerms2[9*i+6].dot(m.red_s());
			double r32 = m.red_x().transpose()*aFASTARAPItRTerms1[9*i+7]*m.red_s() + aFASTARAPItRTerms2[9*i+7].dot(m.red_s());
			double r33 = m.red_x().transpose()*aFASTARAPItRTerms1[9*i+8]*m.red_s() + aFASTARAPItRTerms2[9*i+8].dot(m.red_s());

			Matrix3d F;
			F.row(0) = Vector3d(r11, r12, r13);
			F.row(1) = Vector3d(r21, r22, r23);
			F.row(2) = Vector3d(r31, r32, r33);
			

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
	

		double itRTimes = 0;
		double itTTimes = 0;

		double previous5ItE = Energy(m);
		double oldE = Energy(m);
		double newE;
		for(int i=1; i< 10000; i++){
			// atimer.start();
			itT(m);
			// atimer.stop();
			// itTTimes += atimer.getElapsedTimeInMicroSec();
			// atimer.start();
			itR(m);
			// atimer.stop();
			// itRTimes += atimer.getElapsedTimeInMicroSec();
			// m.constTimeFPAx0(aFPAx0);

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
			
			if(fabs(newE - oldE)<5e-8){
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