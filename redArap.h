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
typedef Matrix<double, 12, 1> Vector12d;
typedef Matrix<double, 9, 1> Vector9d;


class Reduced_Arap
{

protected:
	SparseLU<SparseMatrix<double>>  aARAPKKTSparseSolver;
	SparseLU<SparseMatrix<double>> ajacLU;

	FullPivLU<MatrixXd>  aARAPKKTSolver;
	VectorXd aPAx0, aEr, aEs, aEx, aDEDs, aFPAx0;
	SparseMatrix<double> aExx, aExs, aPAG, aCG;
	SparseMatrix<double> aJacKKTSparse, aJacConstrainsSparse;
	MatrixXd aExr, aErr, aErs, aJacKKT, aJacConstrains;

	std::vector<vector<Trip>> aDS, aDR, aDDR;
	std::vector<SparseMatrix<double>> aredDR, aredDDR;
	std::vector<Matrix3d> asingDR;

	std::vector<MatrixXd> aConstItRTerms;
	std::vector<VectorXd> aUSUtPAx_E;
	std::vector<SparseMatrix<double>> aConstErTerms12;
	std::vector<VectorXd> aConstErTerms3;
	std::vector<MatrixXd> aErsTermsPAx0U;
	std::vector<std::vector<MatrixXd>> aErsTermsPAGU;
	MatrixXd aPAx0DS;
	std::vector<Trip> aExx_trips, aExr_trips, aErr_trips, aExs_trips, aErs_trips, aC_trips;


	SparseMatrix<double> adjointP, a_Wr, a_Ww;

public:
	Reduced_Arap(Mesh& m){
		int r_size = m.red_w().size();
		int z_size = m.red_x().size();
		int s_size = m.red_s().size();
		int t_size = m.T().rows();

		aFPAx0.resize(12*m.T().rows());
		aFPAx0.setZero();

		//TODO aExx = (m.P()*m.A()*m.G()).transpose()*(m.P()*m.A()*m.G());
		aExx = (m.P()*m.A()).transpose()*(m.P()*m.A());
		aErr = MatrixXd::Zero(r_size, r_size);
		aExs.resize(z_size, s_size);
		aErs = MatrixXd::Zero(r_size, s_size);
		aExr = MatrixXd::Zero(z_size, r_size);
		aEr = VectorXd::Zero(r_size);
		aEs = VectorXd::Zero(s_size);

		aPAx0 = m.P()*m.A()*m.x0();
		aPAG = m.P()*m.A();//TODO comment this in when G is reduced *m.G();

		aCG = m.AB().transpose();

		print("arap 4");
		SparseMatrix<double> Exx = (m.P()*m.A()).transpose()*(m.P()*m.A());
		VectorXd PAx0 = m.P()*m.A()*m.x0();
		SparseMatrix<double> spKKTmat(Exx.rows()+aCG.rows(), Exx.rows()+aCG.rows());
		spKKTmat.setZero();
		std::vector<Trip> ExxTrips = to_triplets(Exx);
		aC_trips = to_triplets(aCG);
		std::vector<Trip> CtTrips = to_triplets(m.AB());
		for(int i=0; i<aC_trips.size(); i++){
			int row = aC_trips[i].row();
			int col = aC_trips[i].col();
			int val = aC_trips[i].value();
			ExxTrips.push_back(Trip(row+Exx.rows(), col, val));
			ExxTrips.push_back(Trip(col, row+Exx.cols(), val));
		}
		ExxTrips.insert(ExxTrips.end(),aC_trips.begin(), aC_trips.end());
		spKKTmat.setFromTriplets(ExxTrips.begin(), ExxTrips.end());
		aARAPKKTSparseSolver.analyzePattern(spKKTmat);
		aARAPKKTSparseSolver.factorize(spKKTmat);

		// MatrixXd KKTmat = MatrixXd::Zero(aExx.rows()+aCG.rows(), aExx.rows()+aCG.rows());
		// KKTmat.block(0,0, aExx.rows(), aExx.cols()) = aExx;
		// KKTmat.block(aExx.rows(), 0, aCG.rows(), aCG.cols()) = aCG;
		// KKTmat.block(0, aExx.cols(), aCG.cols(), aCG.rows()) = aCG.transpose();
		// aARAPKKTSolver.compute(KKTmat);

		print("arap 5");
		aJacKKTSparse.resize(z_size+r_size+aCG.rows(), z_size+r_size+aCG.rows());
		aJacConstrainsSparse.resize(z_size+r_size+aCG.rows() ,s_size);
		// aJacKKT.resize(z_size+r_size+aCG.rows(), z_size+r_size+aCG.rows());
		// aJacConstrains.resize(z_size+r_size+aCG.rows() ,s_size);



		print("rarap 6");
		setupAdjointP();

		// setupRedSparseDSds(m);//one time pre-processing
		setupRedSparseDRdr(m);
		setupRedSparseDDRdrdr(m);

		print("rarap 7");
		setupWrWw(m);
		setupFastErTerms(m);
		setupFastPAx0DSTerm(m);
		setupFastErsTerms(m);

		// setupFastItRTerms(m);
		// setupFastUSUtPAx0Terms(m);
	}

	void setupAdjointP(){
		adjointP.resize(aExx.rows()+aErr.rows(), aExx.rows()+aErr.rows()+aCG.rows());
		for(int i=0; i<aExx.rows()+aErr.rows(); i++){
			adjointP.coeffRef(i,i) = 1;
		}
	}

	void setupRedSparseDRdr(Mesh& m){
		aredDR.clear();
		asingDR.clear();

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
			
			asingDR.push_back(r1);
			asingDR.push_back(r2);
			asingDR.push_back(r3);

			SparseMatrix<double> block1 = Eigen::kroneckerProduct(Ident, r1);
			SparseMatrix<double> block2 = Eigen::kroneckerProduct(Ident, r2);
			SparseMatrix<double> block3 = Eigen::kroneckerProduct(Ident, r3);
			aredDR.push_back(block1);
			aredDR.push_back(block2);
			aredDR.push_back(block3);
		
		}
	}

	void setupRedSparseDDRdrdr(Mesh& m){
		aDDR.clear();
		aredDDR.clear();

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

			aredDDR.push_back(block1);
			aredDDR.push_back(block2);
			aredDDR.push_back(block3);
			aredDDR.push_back(block4);
			aredDDR.push_back(block5);
			aredDDR.push_back(block6);
			aredDDR.push_back(block7);
			aredDDR.push_back(block8);
			aredDDR.push_back(block9);
		}	
	}

	void setupRedSparseDSds(Mesh& m){
		for(int j=0; j<m.T().rows(); j++){
			// VectorXd sWx = m.sW().col(6*i+0); 
			vector<Trip> sx = {};
			vector<Trip> sy = {};
			vector<Trip> sz = {};
			vector<Trip> s01 = {};
			vector<Trip> s02 = {};
			vector<Trip> s12 = {};
			sx.push_back(Trip(12*j+0, 12*j+0, 1));
			sx.push_back(Trip(12*j+3, 12*j+3, 1));
			sx.push_back(Trip(12*j+6, 12*j+6, 1));
			sx.push_back(Trip(12*j+9, 12*j+9, 1));
			sy.push_back(Trip(12*j+0+1, 12*j+0+1, 1));
			sy.push_back(Trip(12*j+3+1, 12*j+3+1, 1));
			sy.push_back(Trip(12*j+6+1, 12*j+6+1, 1));
			sy.push_back(Trip(12*j+9+1, 12*j+9+1, 1));
			sz.push_back(Trip(12*j+0+2, 12*j+0+2, 1));
			sz.push_back(Trip(12*j+3+2, 12*j+3+2, 1));
			sz.push_back(Trip(12*j+6+2, 12*j+6+2, 1));
			sz.push_back(Trip(12*j+9+2, 12*j+9+2, 1));
			s01.push_back(Trip(12*j+0, 12*j+0+1, 1));
			s01.push_back(Trip(12*j+3, 12*j+3+1, 1));
			s01.push_back(Trip(12*j+6, 12*j+6+1, 1));
			s01.push_back(Trip(12*j+9, 12*j+9+1, 1));
			s01.push_back(Trip(12*j+0+1, 12*j+0, 1));
			s01.push_back(Trip(12*j+3+1, 12*j+3, 1));
			s01.push_back(Trip(12*j+6+1, 12*j+6, 1));
			s01.push_back(Trip(12*j+9+1, 12*j+9, 1));
			s02.push_back(Trip(12*j+0, 12*j+0+2, 1));
			s02.push_back(Trip(12*j+3, 12*j+3+2, 1));
			s02.push_back(Trip(12*j+6, 12*j+6+2, 1));
			s02.push_back(Trip(12*j+9, 12*j+9+2, 1));
			s02.push_back(Trip(12*j+0+2, 12*j+0, 1));
			s02.push_back(Trip(12*j+3+2, 12*j+3, 1));
			s02.push_back(Trip(12*j+6+2, 12*j+6, 1));
			s02.push_back(Trip(12*j+9+2, 12*j+9, 1));
			s12.push_back(Trip(12*j+0+1, 12*j+0+2, 1));
			s12.push_back(Trip(12*j+3+1, 12*j+3+2, 1));
			s12.push_back(Trip(12*j+6+1, 12*j+6+2, 1));
			s12.push_back(Trip(12*j+9+1, 12*j+9+2, 1));
			s12.push_back(Trip(12*j+0+2, 12*j+0+1, 1));
			s12.push_back(Trip(12*j+3+2, 12*j+3+1, 1));
			s12.push_back(Trip(12*j+6+2, 12*j+6+1, 1));
			s12.push_back(Trip(12*j+9+2, 12*j+9+1, 1));
		
			
			aDS.push_back(sx);
			aDS.push_back(sy);
			aDS.push_back(sz);
			aDS.push_back(s01);
			aDS.push_back(s02);
			aDS.push_back(s12);
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

	void setupFastItRTerms(Mesh& m){
		// vector<Trip> MUUtPAx0_trips;
		// for(int t=0; t<m.T().rows(); ++t){
		// 	Vector12d x = aUtPAx0.segment<12>(12*t);
		// 	Vector9d u = m.red_u().segment<9>(9*t);
		// 	for(int j=0; j<4; j++){
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+0, x[3*j+0]*u[0]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+1, x[3*j+1]*u[1]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+2, x[3*j+2]*u[2]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+3, x[3*j+0]*u[1]+x[3*j+1]*u[0]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+4, x[3*j+0]*u[2]+x[3*j+2]*u[0]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+5, x[3*j+1]*u[2]+x[3*j+2]*u[1]));

		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+0, x[3*j+0]*u[3]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+1, x[3*j+1]*u[4]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+2, x[3*j+2]*u[5]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+3, x[3*j+0]*u[4]+x[3*j+1]*u[3]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+4, x[3*j+0]*u[5]+x[3*j+2]*u[3]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+5, x[3*j+1]*u[5]+x[3*j+2]*u[4]));

		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+0, x[3*j+0]*u[6]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+1, x[3*j+1]*u[7]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+2, x[3*j+2]*u[8]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+3, x[3*j+0]*u[7]+x[3*j+1]*u[6]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+4, x[3*j+0]*u[8]+x[3*j+2]*u[6]));
		// 		MUUtPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+5, x[3*j+1]*u[8]+x[3*j+2]*u[7]));
		// 	}
		// }

		// SparseMatrix<double> MUUtPAx0(12*m.T().rows(), 6*m.T().rows());
		// MUUtPAx0.setFromTriplets(MUUtPAx0_trips.begin(), MUUtPAx0_trips.end());
		// MatrixXd MUUtPAx0sW = MUUtPAx0*m.sW();


		// for(int i=0; i<m.red_w().size()/3; i++){
		// 	SparseMatrix<double>& B = m.RotBLOCK()[i];
		// 	MatrixXd BMUUtPAx0sW = B.transpose()*MUUtPAx0sW;
		// 	aConstItRTerms.push_back(BMUUtPAx0sW);
		// }
	}

	void setupFastErTerms(Mesh& m){
		vector<Trip> MPAx0_trips;
		for(int t=0; t<m.T().rows(); ++t){
			Vector12d x = aPAx0.segment<12>(12*t);
			for(int j=0; j<4; j++){
				MPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+0, x[3*j+0]));
				MPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+1, 0));
				MPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+2, 0));
				MPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+3, x[3*j+1]));
				MPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+4, x[3*j+2]));
				MPAx0_trips.push_back(Trip(12*t+3*j+0, 6*t+5, 0));

				MPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+0, 0));
				MPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+1, x[3*j+1]));
				MPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+2, 0));
				MPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+3, x[3*j+0]));
				MPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+4, x[3*j+2]));
				MPAx0_trips.push_back(Trip(12*t+3*j+1, 6*t+5, 0));

				MPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+0, 0));
				MPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+1, 0));
				MPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+2, x[3*j+2]));
				MPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+3, 0));
				MPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+4, x[3*j+0]));
				MPAx0_trips.push_back(Trip(12*t+3*j+2, 6*t+5, x[3*j+1]));
			}
		}


		SparseMatrix<double> MPAx0(12*m.T().rows(), 6*m.T().rows());
		MPAx0.setFromTriplets(MPAx0_trips.begin(), MPAx0_trips.end());
		//TODO: MatrixXd MUUtPAx0sW = MUUtPAx0*m.sW();
		SparseMatrix<double> MPAx0sW = MPAx0;
	

		for(int r=0; r<m.red_w().size()/3; ++r){
			SparseMatrix<double>& B = m.RotBLOCK()[r];
			SparseMatrix<double> BMUtPAx0sW = B.transpose()*MPAx0sW;//TODO dense
			SparseMatrix<double> BPAG = -1*B.transpose()*aPAG;//TODO make dense
			VectorXd BPAx0 = -1*B.transpose()*aPAx0;
			aConstErTerms12.push_back(BMUtPAx0sW);
			aConstErTerms12.push_back(BPAG);
			aConstErTerms3.push_back(BPAx0);
		}
	}

	void setupFastExsTerms(Mesh& m){
		// std::vector<Trip> U_trips;
		// for(int i=0; i<m.T().rows(); i++){
		// 	Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
		// 	U_trips.push_back(Trip(9*i+0, 9*i+0, u(0,0)));
		// 	U_trips.push_back(Trip(9*i+0, 9*i+1, u(0,1)));
		// 	U_trips.push_back(Trip(9*i+0, 9*i+2, u(0,2)));
		// 	U_trips.push_back(Trip(9*i+1, 9*i+0, u(1,0)));
		// 	U_trips.push_back(Trip(9*i+1, 9*i+1, u(1,1)));
		// 	U_trips.push_back(Trip(9*i+1, 9*i+2, u(1,2)));
		// 	U_trips.push_back(Trip(9*i+2, 9*i+0, u(2,0)));
		// 	U_trips.push_back(Trip(9*i+2, 9*i+1, u(2,1)));
		// 	U_trips.push_back(Trip(9*i+2, 9*i+2, u(2,2)));
		// }
		// SparseMatrix<double> Umat(3*m.T().rows(), 3*m.T().rows());
		// Umat.setFromTriplets(U_trips.begin(), U_trips.end());
	}

	void setupFastErsTerms(Mesh& m){
		// std::map<int, std::vector<int>>& c_e_map = m.r_cluster_elem_map();
		// for(int i=0; i<m.red_w().size()/3; i++){
		// 	std::vector<int> cluster_elem = c_e_map[i];			
		// 	MatrixXd PAx0U = MatrixXd::Zero(12*cluster_elem.size(), 9);

		// 	SparseMatrix<double>& B = m.RotBLOCK()[i];
		// 	SparseMatrix<double> BPAG = -1*B.transpose()*aPAG;//TODO
		// 	VectorXd BPAx0 = -1*B.transpose()*aPAx0;//TODO
		// 	SparseMatrix<double> BUtPAx0DS = B.transpose()*aUtPAx0DS;

		// 	for(int e=0; e<cluster_elem.size(); e++){
		// 		Matrix3d u_e = Map<Matrix3d>(m.red_u().segment<9>(9*cluster_elem[e]).data()).transpose();
		// 		Matrix3d outer10 = BPAx0.segment<3>(12*e+0)*u_e.col(0).transpose();
		// 		Matrix3d outer20 = BPAx0.segment<3>(12*e+3)*u_e.col(0).transpose();
		// 		Matrix3d outer30 = BPAx0.segment<3>(12*e+6)*u_e.col(0).transpose();
		// 		Matrix3d outer40 = BPAx0.segment<3>(12*e+9)*u_e.col(0).transpose();

		// 		Matrix3d outer11 = BPAx0.segment<3>(12*e+0)*u_e.col(1).transpose();
		// 		Matrix3d outer21 = BPAx0.segment<3>(12*e+3)*u_e.col(1).transpose();
		// 		Matrix3d outer31 = BPAx0.segment<3>(12*e+6)*u_e.col(1).transpose();
		// 		Matrix3d outer41 = BPAx0.segment<3>(12*e+9)*u_e.col(1).transpose();

		// 		Matrix3d outer12 = BPAx0.segment<3>(12*e+0)*u_e.col(2).transpose();
		// 		Matrix3d outer22 = BPAx0.segment<3>(12*e+3)*u_e.col(2).transpose();
		// 		Matrix3d outer32 = BPAx0.segment<3>(12*e+6)*u_e.col(2).transpose();
		// 		Matrix3d outer42 = BPAx0.segment<3>(12*e+9)*u_e.col(2).transpose();



		// 		PAx0U.row(12*e+0) = Map<Vector9d>(outer10.transpose().data());
		// 		PAx0U.row(12*e+1) = Map<Vector9d>(outer11.transpose().data());
		// 		PAx0U.row(12*e+2) = Map<Vector9d>(outer12.transpose().data());

		// 		PAx0U.row(12*e+3) = Map<Vector9d>(outer20.transpose().data());
		// 		PAx0U.row(12*e+4) = Map<Vector9d>(outer21.transpose().data());
		// 		PAx0U.row(12*e+5) = Map<Vector9d>(outer22.transpose().data());

		// 		PAx0U.row(12*e+6) = Map<Vector9d>(outer30.transpose().data());
		// 		PAx0U.row(12*e+7) = Map<Vector9d>(outer31.transpose().data());
		// 		PAx0U.row(12*e+8) = Map<Vector9d>(outer32.transpose().data());

		// 		PAx0U.row(12*e+9) = Map<Vector9d>(outer40.transpose().data());
		// 		PAx0U.row(12*e+10) = Map<Vector9d>(outer41.transpose().data());
		// 		PAx0U.row(12*e+11) = Map<Vector9d>(outer42.transpose().data());				
		// 	}

		// 	std::vector<MatrixXd> UtPAx0DST_PAGzU;
		// 	for(int v=0; v<aPAG.cols(); v++){
		// 		MatrixXd PAGvzU = MatrixXd::Zero(12*cluster_elem.size(), 9);
		// 		for(int e=0; e<cluster_elem.size(); e++){
		// 			Matrix3d u_e = Map<Matrix3d>(m.red_u().segment<9>(9*cluster_elem[e]).data()).transpose();;
					
		// 			Matrix3d outer10 = BPAG.col(v).segment<3>(12*e+0)*u_e.col(0).transpose();
		// 			Matrix3d outer20 = BPAG.col(v).segment<3>(12*e+3)*u_e.col(0).transpose();
		// 			Matrix3d outer30 = BPAG.col(v).segment<3>(12*e+6)*u_e.col(0).transpose();
		// 			Matrix3d outer40 = BPAG.col(v).segment<3>(12*e+9)*u_e.col(0).transpose();

		// 			Matrix3d outer11 = BPAG.col(v).segment<3>(12*e+0)*u_e.col(1).transpose();
		// 			Matrix3d outer21 = BPAG.col(v).segment<3>(12*e+3)*u_e.col(1).transpose();
		// 			Matrix3d outer31 = BPAG.col(v).segment<3>(12*e+6)*u_e.col(1).transpose();
		// 			Matrix3d outer41 = BPAG.col(v).segment<3>(12*e+9)*u_e.col(1).transpose();

		// 			Matrix3d outer12 = BPAG.col(v).segment<3>(12*e+0)*u_e.col(2).transpose();
		// 			Matrix3d outer22 = BPAG.col(v).segment<3>(12*e+3)*u_e.col(2).transpose();
		// 			Matrix3d outer32 = BPAG.col(v).segment<3>(12*e+6)*u_e.col(2).transpose();
		// 			Matrix3d outer42 = BPAG.col(v).segment<3>(12*e+9)*u_e.col(2).transpose();


		// 			PAGvzU.row(12*e+0) = Map<Vector9d>(outer10.transpose().data());
		// 			PAGvzU.row(12*e+1) = Map<Vector9d>(outer11.transpose().data());
		// 			PAGvzU.row(12*e+2) = Map<Vector9d>(outer12.transpose().data());

		// 			PAGvzU.row(12*e+3) = Map<Vector9d>(outer20.transpose().data());
		// 			PAGvzU.row(12*e+4) = Map<Vector9d>(outer21.transpose().data());
		// 			PAGvzU.row(12*e+5) = Map<Vector9d>(outer22.transpose().data());

		// 			PAGvzU.row(12*e+6) = Map<Vector9d>(outer30.transpose().data());
		// 			PAGvzU.row(12*e+7) = Map<Vector9d>(outer31.transpose().data());
		// 			PAGvzU.row(12*e+8) = Map<Vector9d>(outer32.transpose().data());

		// 			PAGvzU.row(12*e+9) = Map<Vector9d>(outer40.transpose().data());
		// 			PAGvzU.row(12*e+10) = Map<Vector9d>(outer41.transpose().data());
		// 			PAGvzU.row(12*e+11) = Map<Vector9d>(outer42.transpose().data());	
		// 		}
		// 		UtPAx0DST_PAGzU.push_back(BUtPAx0DS.transpose()*PAGvzU);
		// 	}
		// 	aErsTermsPAx0U.push_back(BUtPAx0DS.transpose()*PAx0U);

			
		// 	aErsTermsPAGU.push_back(UtPAx0DST_PAGzU);
		// }
	}

	void setupFastPAx0DSTerm(Mesh& m){
		// aUtPAx0DS = MatrixXd::Zero(aUtPAx0.size(), m.red_s().size());
		// for(int s=0; s<m.red_s().size()/6; s++){
		// 	VectorXd sWx = m.sW().col(6*s+0);
		// 	VectorXd sWy = m.sW().col(6*s+1);
		// 	VectorXd sWz = m.sW().col(6*s+2);
		// 	VectorXd sW01 = m.sW().col(6*s+3);
		// 	VectorXd sW02 = m.sW().col(6*s+4);
		// 	VectorXd sW12 = m.sW().col(6*s+5);

		// 	VectorXd diag_x = VectorXd::Zero(12*m.T().rows());
		// 	VectorXd diag_y = VectorXd::Zero(12*m.T().rows());
		// 	VectorXd diag_z = VectorXd::Zero(12*m.T().rows());
		// 	VectorXd diag_1 = VectorXd::Zero(12*m.T().rows());
		// 	VectorXd diag_2 = VectorXd::Zero(12*m.T().rows());
		// 	VectorXd diag_3 = VectorXd::Zero(12*m.T().rows());
		// 	for(int i=0; i<m.T().rows(); i++){
		// 		diag_x[12*i+0] = sWx[6*i];
		// 		diag_x[12*i+3] = sWx[6*i];
		// 		diag_x[12*i+6] = sWx[6*i];
		// 		diag_x[12*i+9] = sWx[6*i];

		// 		diag_y[12*i+0+1] = sWy[6*i+1];
		// 		diag_y[12*i+3+1] = sWy[6*i+1];
		// 		diag_y[12*i+6+1] = sWy[6*i+1];
		// 		diag_y[12*i+9+1] = sWy[6*i+1];

		// 		diag_z[12*i+0+2] = sWz[6*i+2];
		// 		diag_z[12*i+3+2] = sWz[6*i+2];
		// 		diag_z[12*i+6+2] = sWz[6*i+2];
		// 		diag_z[12*i+9+2] = sWz[6*i+2];

		// 		diag_1[12*i+0] = sW01[6*i+3];
		// 		diag_1[12*i+3] = sW01[6*i+3];
		// 		diag_1[12*i+6] = sW01[6*i+3];
		// 		diag_1[12*i+9] = sW01[6*i+3];

		// 		diag_2[12*i+0+1] = sW02[6*i+4];
		// 		diag_2[12*i+3+1] = sW02[6*i+4];
		// 		diag_2[12*i+6+1] = sW02[6*i+4];
		// 		diag_2[12*i+9+1] = sW02[6*i+4];

		// 		diag_3[12*i+0] = sW12[6*i+5];
		// 		diag_3[12*i+3] = sW12[6*i+5];
		// 		diag_3[12*i+6] = sW12[6*i+5];
		// 		diag_3[12*i+9] = sW12[6*i+5];
		// 	}

		// 	aUtPAx0DS.col(6*s+0) += aUtPAx0.cwiseProduct(diag_x);
		// 	aUtPAx0DS.col(6*s+1) += aUtPAx0.cwiseProduct(diag_y);
		// 	aUtPAx0DS.col(6*s+2) += aUtPAx0.cwiseProduct(diag_z);
			
		// 	aUtPAx0DS.col(6*s+3).tail(aUtPAx0.size()-1) += aUtPAx0.head(aUtPAx0.size()-1).cwiseProduct(diag_1.head(aUtPAx0.size()-1));
		// 	aUtPAx0DS.col(6*s+3).head(aUtPAx0.size()-1) += aUtPAx0.tail(aUtPAx0.size()-1).cwiseProduct(diag_1.head(aUtPAx0.size()-1));

		// 	aUtPAx0DS.col(6*s+4).tail(aUtPAx0.size()-2) += aUtPAx0.head(aUtPAx0.size()-2).cwiseProduct(diag_3.head(aUtPAx0.size()-2)); 
		// 	aUtPAx0DS.col(6*s+4).head(aUtPAx0.size()-2) += aUtPAx0.tail(aUtPAx0.size()-2).cwiseProduct(diag_3.head(aUtPAx0.size()-2));

		// 	aUtPAx0DS.col(6*s+5).tail(aUtPAx0.size()-1) += aUtPAx0.head(aUtPAx0.size()-1).cwiseProduct(diag_2.head(aUtPAx0.size()-1));
		// 	aUtPAx0DS.col(6*s+5).head(aUtPAx0.size()-1) += aUtPAx0.tail(aUtPAx0.size()-1).cwiseProduct(diag_2.head(aUtPAx0.size()-1));
		// }
		// print(aUtPAx0DS);
		// exit(0);
		// aUtPAx0DS.resize(aUtPAx0.size(), m.red_s().size());
		// std::vector<Trip> temptrips;
		// for(int i=0; i<m.T().rows(); i++){
		// 	temptrips.push_back(Trip(12*i, ));
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());

		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());

		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());

		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());

		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());

		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// 	temptrips.push_back(Trip());
		// }
	}

	double Energy(Mesh& m){
		VectorXd PAx = aPAG*m.red_x() + aPAx0;
		double En= 0.5*(PAx - aFPAx0).squaredNorm();
		return En;
	}

	double Energy(Mesh& m, VectorXd& z, VectorXd& redw, VectorXd& redr, VectorXd& reds){
		//TODO VectorXd ms = m.sW()*reds;
		VectorXd ms = reds;
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
		// print("before LU");		
		VectorXd ExEr(aEx.size()+aEr.size());
		ExEr<<aEx,aEr;
		VectorXd PtExEr = adjointP.transpose()*ExEr;
		VectorXd g = aJacKKT.fullPivLu().solve(PtExEr);
		aDEDs = aJacConstrains.transpose()*g + aEs;
		// std::ofstream ExxFile("Exx.mat");
		// if (ExxFile.is_open())
		// {
		// 	ExxFile << aExx;
		// }
		// ExxFile.close();

		//Sparseify
		aJacKKTSparse.setZero();
		aJacConstrainsSparse.setZero();

		vector<Trip> jac_trips;
		vector<Trip> cons_trips;

		//col1
		for(int i=0; i<aExx_trips.size(); i++){
			int row = aExx_trips[i].row();
			int col = aExx_trips[i].col();
			double val = aExx_trips[i].value();
			jac_trips.push_back(Trip(row, col, val));
		}
		for(int i=0; i<aExr_trips.size(); i++){
			int row = aExr_trips[i].row();
			int col = aExr_trips[i].col()+aExx.rows();
			double val = aExr_trips[i].value();
			jac_trips.push_back(Trip(col, row, val));
		}
		for(int i=0; i<aC_trips.size(); i++){
			int row = aC_trips[i].row()+aExx.rows()+aExr.cols();
			int col = aC_trips[i].col();
			double val = aC_trips[i].value();
			jac_trips.push_back(Trip(row, col, val));
		}

		//col2
		for(int i=0; i<aExr_trips.size(); i++){
			int row = aExr_trips[i].row();
			int col = aExr_trips[i].col()+aExx.cols();
			double val = aExr_trips[i].value();
			jac_trips.push_back(Trip(row, col, val));
		}
		for(int i=0; i<aErr_trips.size(); i++){
			int row = aErr_trips[i].row()+aExr.rows();
			int col = aErr_trips[i].col()+aExx.cols();
			double val = aErr_trips[i].value();
			jac_trips.push_back(Trip(row, col, val));
		}
		//col3
		for(int i=0; i<aC_trips.size(); i++){
			int row = aC_trips[i].row()+aExx.cols()+aExr.cols();
			int col = aC_trips[i].col();
			double val = aC_trips[i].value();
			jac_trips.push_back(Trip(col, row, val));
		}
		aJacKKTSparse.setFromTriplets(jac_trips.begin(), jac_trips.end());

		//rhs col
		for(int i=0; i<aExs_trips.size(); i++){
			int row = aExs_trips[i].row();
			int col = aExs_trips[i].col();
			double val = aExs_trips[i].value();
			cons_trips.push_back(Trip(row, col, val));
		}
		for(int i=0; i<aErs_trips.size(); i++){
			int row = aErs_trips[i].row()+aExs.rows();
			int col = aErs_trips[i].col();
			double val = aErs_trips[i].value();
			cons_trips.push_back(Trip(row, col, val));
		}
		aJacConstrainsSparse.setFromTriplets(cons_trips.begin(), cons_trips.end());


		return aDEDs;
	}

	int Hessians(Mesh& m){
		// print("		+Hessians");
		setupRedSparseDRdr(m);
		setupRedSparseDDRdrdr(m);
		// Exr
		// print("		Exr");
		constTimeExr(m);

		//Err
		// print("		Err");
		constTimeErr(m);

		//Exs
		// print("		Exs");
		Exs(m);

		//Ers
		// print("		Ers");
		constTimeErs(m);

		print("		-Hessians");		
		return 1;
	}

	MatrixXd& constTimeExr(Mesh& m){
		for(int i=0; i<m.red_w().size()/3; i++){
			VectorXd BSPAx0 = aConstErTerms12[2*i+0]*m.red_s();
			aExr.col(3*i+0) = aConstErTerms12[2*i+1].transpose()*(aredDR[3*i+0]*BSPAx0);
			aExr.col(3*i+1) = aConstErTerms12[2*i+1].transpose()*(aredDR[3*i+1]*BSPAx0);
			aExr.col(3*i+2) = aConstErTerms12[2*i+1].transpose()*(aredDR[3*i+2]*BSPAx0);
		}
		return aExr;
	}

	MatrixXd& constTimeErr(Mesh& m){
		for(int i=0; i<aErr.rows()/3; i++){
			VectorXd BSPAx0 = aConstErTerms12[2*i+0]*m.red_s();
			VectorXd BPAGz = aConstErTerms12[2*i+1]*m.red_x();
			VectorXd BPAx0 = aConstErTerms3[i];
			auto v00 = aredDDR[9*i+0];
			auto v01 = aredDDR[9*i+1];
			auto v02 = aredDDR[9*i+2];
			auto v11 = aredDDR[9*i+4];
			auto v12 = aredDDR[9*i+5];
			auto v22 = aredDDR[9*i+8];
		
			aErr(3*i+0,3*i+0) = BSPAx0.dot(v00.transpose()*(BPAGz + BPAx0));
			aErr(3*i+1,3*i+1) = BSPAx0.dot(v11.transpose()*(BPAGz + BPAx0));
			aErr(3*i+2,3*i+2) = BSPAx0.dot(v22.transpose()*(BPAGz + BPAx0));
			aErr(3*i+0,3*i+1) = BSPAx0.dot(v01.transpose()*(BPAGz + BPAx0));
			aErr(3*i+0,3*i+2) = BSPAx0.dot(v02.transpose()*(BPAGz + BPAx0));
			aErr(3*i+1,3*i+2) = BSPAx0.dot(v12.transpose()*(BPAGz + BPAx0));
			
			
			aErr(3*i+1, 3*i+0) = aErr(3*i+0, 3*i+1);
			aErr(3*i+2, 3*i+0) = aErr(3*i+0, 3*i+2);
			aErr(3*i+2, 3*i+1) = aErr(3*i+1, 3*i+2);
		}
		return aErr;
	}

	MatrixXd& constTimeErs(Mesh& m){
		// print("new");
		// for(int i=0; i<m.red_w().size(); i++){
		// 	Vector9d r1vec = Map<Vector9d>((asingDR[i].transpose()).data());
		// 	aErs.row(i) = aErsTermsPAx0U[i/3]*r1vec;
		// 	for(int v=0; v<aPAG.cols(); v++){
		// 		aErs.row(i) += m.red_x()[v]*aErsTermsPAGU[i/3][v]*r1vec;
		// 	}
		// }

		// print(aErs);


		// print("old");
		// VectorXd PAg = aPAG*m.red_x() + aPAx0;
		// // VectorXd PAg = aPAx0;
		// MatrixXd TEMP1 = MatrixXd::Zero(12*m.T().rows(), aErs.rows());
		// for(int i=0; i<TEMP1.rows(); i++){
		// 	for(int j=0; j<TEMP1.cols(); j++){
		// 		auto v = aDR[j];
		// 		for(int k=0; k<v.size(); k++){
		// 			TEMP1(i,j) += v[k].value()*(-1*m.GU().coeff(v[k].col(), i)*PAg[v[k].row()]);
		// 		}
		// 	}
		// }

		// aErs.setZero();
		// for(int i=0; i<aErs.cols(); i++){
		// 	std::vector<Trip> v = aDS[i];
		// 	for(int j=0; j<TEMP1.cols(); j++){
		// 		for(int k=0; k<v.size(); k++){
		// 			aErs(j, i) += v[k].value()*(aUtPAx0[v[k].row()]*TEMP1(v[k].col(), j));
		// 		}
		// 	}
		// }
		// print(aErs);
		// exit(0);
		return aErs;
	}

	void Exs(Mesh& m){
		//Exs
		std::vector<Trip> aExs_trips;
		// print("		Exs");
		for(int t =0; t<m.T().rows(); t++){
			//Tet
			Matrix3d r = Map<Matrix3d>(m.red_r().segment<9>(9*m.r_elem_cluster_map()[t]).data()).transpose();
			for(int e=0; e<4; e++){
				//Vert on tet
				for(int a=0; a<3; a++){
					//x, y, or z axis
					Vector3d GtAtPtRU_row1 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+0).transpose()*r;
					Vector3d GtAtPtRU_row2 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+3).transpose()*r;
					Vector3d GtAtPtRU_row3 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+6).transpose()*r;
					Vector3d GtAtPtRU_row4 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+9).transpose()*r;
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
					aExs_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+0, Exas1));
					aExs_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+1, Exas2));
					aExs_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+2, Exas3));
					aExs_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+3, Exas4));
					aExs_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+4, Exas5));
					aExs_trips.push_back(Trip(3*m.T().row(t)[e]+a, 6*t+5, Exas6));
				}
			}
		}
		aExs.setFromTriplets(aExs_trips.begin(), aExs_trips.end());
	}

	int Gradients(Mesh& m){
		// print("			+ Gradients");

		m.constTimeFPAx0(aFPAx0);
		aEx = dEdx(m);
		
		// print("		Er");
		aEr = constTimeEr(m);
		
		// print("		Es");
		aEs = dEds(m);

		// print("			- Gradients");
		return 1;
	}

	VectorXd dEds(Mesh& m){
		// VectorXd ms = m.sW()*m.red_s();
		// VectorXd PAx = aPAG*m.red_x() + aPAx0;
		// VectorXd UtRtPAx1 = VectorXd::Zero(12*m.T().rows());
		// VectorXd SUtPAx01 = VectorXd::Zero(12*m.T().rows());
		// for(int t =0; t<m.T().rows(); t++){
		// 	Matrix3d rt = Map<Matrix3d>(m.red_r().segment<9>(9*m.r_elem_cluster_map()[t]).data());
		// 	Matrix3d ut = Map<Matrix3d>(m.red_u().segment<9>(9*t).data());
		// 	Matrix3d s;
		// 	s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
		// 		ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
		// 		ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];

		// 	for(int j=0; j<4; j++){
		// 		Vector3d p = PAx.segment<3>(12*t+3*j);
		// 		UtRtPAx1.segment<3>(12*t+3*j) = ut*rt*p;
		// 		SUtPAx01.segment<3>(12*t+3*j) = s*aUtPAx0.segment<3>(12*t+3*j);
		// 	}
		// }
	
		// aEs.setZero();
		// for(int i=0; i<aEs.size(); i++){
		// 	std::vector<Trip> v = aDS[i];
		// 	for(int k=0; k<aDS[i].size(); k++){
		// 		int t= v[k].row()/12;
		// 		aEs[i] -= (UtRtPAx1[v[k].row()])*aUtPAx0[v[k].col()]*v[k].value();
		// 		aEs[i] += SUtPAx01[v[k].row()]*aUtPAx0[v[k].col()]*v[k].value();
		// 	}
		// }
		VectorXd PAg = aPAG*m.red_x() + aPAx0;
		VectorXd& ms = m.red_s();
		aEs.setZero();
		for(int t=0; t<m.T().rows(); t++){
			Matrix3d rt = Map<Matrix3d>(m.red_r().segment<9>(9*m.r_elem_cluster_map()[t]).data());
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
			aEs[6*t+0] = Es1;
			aEs[6*t+1] = Es2;
			aEs[6*t+2] = Es3;
			aEs[6*t+3] = Es4;
			aEs[6*t+4] = Es5;
			aEs[6*t+5] = Es6;

		}
		return aEs;
	}

	VectorXd& constTimeEr(Mesh& m){
		for(int i=0; i<m.red_w().size()/3; i++){
			VectorXd BUSUtPAx0 = aConstErTerms12[2*i+0]*m.red_s();
			VectorXd BPAGz = aConstErTerms12[2*i+1]*m.red_x();
			VectorXd BPAx0 = aConstErTerms3[i];
			aEr[3*i+0] = BUSUtPAx0.dot(aredDR[3*i+0].transpose()*(BPAGz + BPAx0));
			aEr[3*i+1] = BUSUtPAx0.dot(aredDR[3*i+1].transpose()*(BPAGz + BPAx0));
			aEr[3*i+2] = BUSUtPAx0.dot(aredDR[3*i+2].transpose()*(BPAGz + BPAx0));
		}
		return aEr;
	}

	VectorXd dEdx(Mesh& m){
		VectorXd PAx = aPAG*m.red_x() + aPAx0;
		VectorXd res = (aPAG).transpose()*(PAx - aFPAx0);
		return res;
	}

	bool itT(Mesh& m){
		//TODO DENSIFY
		// VectorXd deltaABtx = m.AB().transpose()*m.dx();
		// VectorXd AtPtFPAx0 = (aPAG).transpose()*aFPAx0;
		// VectorXd AtPtPAx0 = (aPAG).transpose()*(aPAx0);
		// VectorXd gb = AtPtFPAx0 - AtPtPAx0;
		// VectorXd gd(gb.size()+deltaABtx.size());
		// gd<<gb,deltaABtx;
		// VectorXd result = aARAPKKTSolver.solve(gd);
		// VectorXd gu = result.head(gb.size());
		// m.red_x(gu);

		VectorXd deltaABtx = m.AB().transpose()*m.dx();
		VectorXd AtPtFPAx0 = (aPAG).transpose()*aFPAx0;
		VectorXd AtPtPAx0 = (aPAG).transpose()*(aPAx0);
		VectorXd gb = AtPtFPAx0 - AtPtPAx0;
		VectorXd gd(gb.size()+deltaABtx.size());
		gd<<gb,deltaABtx;
		VectorXd result = aARAPKKTSparseSolver.solve(gd);
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
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];
			for(int j=0; j<4; j++){
				USUtPAx0.segment<3>(12*t+3*j) = s*aPAx0.segment<3>(12*t+3*j);
			}
		}

		m.constTimeFPAx0(aFPAx0);

		double previous5ItE = Energy(m);
		double oldE = Energy(m);
		for(int i=1; i< 1000; i++){
			bool converged = itT(m);
			itR(m, USUtPAx0);
			m.constTimeFPAx0(aFPAx0);
			double newE = Energy(m);
			// cout<<i<<",";
			if((newE - oldE)>1e-8 && i>1){
				print("Reduced_Arap::minimize() error. ARAP should monotonically decrease.");
				print(i);
				print(oldE);
				print(newE);
				exit(0);
			}
			oldE = newE;
	
			if (i%5==0){
				if(fabs(newE - previous5ItE)<1e-10){
					if(i>1000){
						// print(m.red_s().transpose());
						// exit(0);
					}
					// std::cout<<"		- Red_ARAP minimize "<<i<<", "<<(newE - previous5ItE)<<std::endl;
					return i;
				}
				previous5ItE = newE;
			}
		
		}
		
		// std::cout<<"		- ARAP never converged "<<Energy(m)-previous5ItE<<std::endl;
		// exit(0);
		return 1000;
	}

	MatrixXd Exx(){ return aExx; }
	MatrixXd Exr(){ return aExr; }
	MatrixXd Exs(){ return MatrixXd(aExs); }
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