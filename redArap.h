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

	// FullPivLU<MatrixXd>  aARAPKKTSolver;
	VectorXd aPAx0, aEr, aEr_max, aEs, aEx, aDEDs, aFPAx0;
	SparseMatrix<double> aExx, aExs, aExr, aErr, aErs, aExr_max, aErr_max, aErs_max, aPAG, aCG;
	SparseMatrix<double> aJacKKTSparse, aJacConstrainsSparse;
	MatrixXd aJacKKT, aJacConstrains;


	SparseMatrix<double> aPAx0DS;
	std::vector<Trip> aExx_trips, aExr_max_trips, aExr_trips, aErr_trips, aErs_trips, aErr_max_trips, aExs_trips, aErs_max_trips, aC_trips;
	double jacLUanalyzed = false;

	SparseMatrix<double> adjointP, a_Wr, a_Ww;
	std::vector<MatrixXd> aePAx;
	std::vector<MatrixXd> aeUSUtPAx0;

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
		aExx_trips = to_triplets(aExx);

		print("arap 2");
		aExs.resize(z_size, s_size);
		aErr_max.resize(3*t_size, 3*t_size);
		aErs_max.resize(3*t_size, s_size);
		aExr_max.resize(z_size, 3*t_size);
		aEr_max.resize(3*t_size);
		aErr.resize(r_size, r_size);
		aErs.resize(r_size, s_size);
		aExr.resize(z_size, r_size);
		aEr.resize(r_size);
		aEs.resize(s_size);

		aPAx0 = m.P()*m.A()*m.x0();
		aPAG = m.P()*m.A();//TODO comment this in when G is reduced *m.G();

		aCG = m.AB().transpose();

		print("rarap 4");
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

		print("rarap 5");
		aJacKKTSparse.resize(z_size+r_size+aCG.rows(), z_size+r_size+aCG.rows());
		aJacConstrainsSparse.resize(z_size+r_size+aCG.rows() ,s_size);
		// aJacKKT.resize(z_size+r_size+aCG.rows(), z_size+r_size+aCG.rows());
		// aJacConstrains.resize(z_size+r_size+aCG.rows() ,s_size);



		print("rarap 6");
		setupAdjointP();

		print("pre-processing");
		setupWrWw(m);
		setupFastItR(m);

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

	void setupFastItR(Mesh& m){
		std::map<int, std::vector<int>>& c_e_map = m.r_cluster_elem_map();
		for (int i=0; i<m.red_w().size()/3; i++){
			std::vector<int> cluster_elem = c_e_map[i];
			aePAx.push_back(MatrixXd::Zero(4*cluster_elem.size(), 3));
			aeUSUtPAx0.push_back(MatrixXd::Zero(4*cluster_elem.size(), 3));
		}
	}

	void setupFastPAx0DSTerm(Mesh& m){
		// aPAx0DS = MatrixXd::Zero(aPAx0.size(), m.red_s().size());
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

		// 	aPAx0DS.col(6*s+0) += aPAx0.cwiseProduct(diag_x);
		// 	aPAx0DS.col(6*s+1) += aPAx0.cwiseProduct(diag_y);
		// 	aPAx0DS.col(6*s+2) += aPAx0.cwiseProduct(diag_z);
			
		// 	aPAx0DS.col(6*s+3).tail(aPAx0.size()-1) += aPAx0.head(aPAx0.size()-1).cwiseProduct(diag_1.head(aPAx0.size()-1));
		// 	aPAx0DS.col(6*s+3).head(aPAx0.size()-1) += aPAx0.tail(aPAx0.size()-1).cwiseProduct(diag_1.head(aPAx0.size()-1));

		// 	aPAx0DS.col(6*s+4).tail(aPAx0.size()-2) += aPAx0.head(aPAx0.size()-2).cwiseProduct(diag_3.head(aPAx0.size()-2)); 
		// 	aPAx0DS.col(6*s+4).head(aPAx0.size()-2) += aPAx0.tail(aPAx0.size()-2).cwiseProduct(diag_3.head(aPAx0.size()-2));

		// 	aPAx0DS.col(6*s+5).tail(aPAx0.size()-1) += aPAx0.head(aPAx0.size()-1).cwiseProduct(diag_2.head(aPAx0.size()-1));
		// 	aPAx0DS.col(6*s+5).head(aPAx0.size()-1) += aPAx0.tail(aPAx0.size()-1).cwiseProduct(diag_2.head(aPAx0.size()-1));
		// }
	

		aPAx0DS.resize(aPAx0.size(), m.red_s().size());
		std::vector<Trip> temptrips;
		for(int i=0; i<m.T().rows(); i++){
			temptrips.push_back(Trip(12*i+0, 6*i+0, aPAx0[12*i+0]));
			temptrips.push_back(Trip(12*i+3, 6*i+0, aPAx0[12*i+3]));
			temptrips.push_back(Trip(12*i+6, 6*i+0, aPAx0[12*i+6]));
			temptrips.push_back(Trip(12*i+9, 6*i+0, aPAx0[12*i+9]));

			temptrips.push_back(Trip(12*i+0+1, 6*i+1, aPAx0[12*i+0+1]));
			temptrips.push_back(Trip(12*i+3+1, 6*i+1, aPAx0[12*i+3+1]));
			temptrips.push_back(Trip(12*i+6+1, 6*i+1, aPAx0[12*i+6+1]));
			temptrips.push_back(Trip(12*i+9+1, 6*i+1, aPAx0[12*i+9+1]));

			temptrips.push_back(Trip(12*i+0+2, 6*i+2, aPAx0[12*i+0+2]));
			temptrips.push_back(Trip(12*i+3+2, 6*i+2, aPAx0[12*i+3+2]));
			temptrips.push_back(Trip(12*i+6+2, 6*i+2, aPAx0[12*i+6+2]));
			temptrips.push_back(Trip(12*i+9+2, 6*i+2, aPAx0[12*i+9+2]));

			temptrips.push_back(Trip(12*i+0+0, 6*i+3, aPAx0[12*i+0+1]));
			temptrips.push_back(Trip(12*i+0+1, 6*i+3, aPAx0[12*i+0+0]));
			temptrips.push_back(Trip(12*i+3+0, 6*i+3, aPAx0[12*i+3+1]));
			temptrips.push_back(Trip(12*i+3+1, 6*i+3, aPAx0[12*i+3+0]));
			temptrips.push_back(Trip(12*i+6+0, 6*i+3, aPAx0[12*i+6+1]));
			temptrips.push_back(Trip(12*i+6+1, 6*i+3, aPAx0[12*i+6+0]));
			temptrips.push_back(Trip(12*i+9+0, 6*i+3, aPAx0[12*i+9+1]));
			temptrips.push_back(Trip(12*i+9+1, 6*i+3, aPAx0[12*i+9+0]));

			temptrips.push_back(Trip(12*i+0+0, 6*i+4, aPAx0[12*i+0+2]));
			temptrips.push_back(Trip(12*i+0+2, 6*i+4, aPAx0[12*i+0+0]));
			temptrips.push_back(Trip(12*i+3+0, 6*i+4, aPAx0[12*i+3+2]));
			temptrips.push_back(Trip(12*i+3+2, 6*i+4, aPAx0[12*i+3+0]));
			temptrips.push_back(Trip(12*i+6+0, 6*i+4, aPAx0[12*i+6+2]));
			temptrips.push_back(Trip(12*i+6+2, 6*i+4, aPAx0[12*i+6+0]));
			temptrips.push_back(Trip(12*i+9+0, 6*i+4, aPAx0[12*i+9+2]));
			temptrips.push_back(Trip(12*i+9+2, 6*i+4, aPAx0[12*i+9+0]));

			temptrips.push_back(Trip(12*i+0+1, 6*i+5, aPAx0[12*i+0+2]));
			temptrips.push_back(Trip(12*i+0+2, 6*i+5, aPAx0[12*i+0+1]));
			temptrips.push_back(Trip(12*i+3+1, 6*i+5, aPAx0[12*i+3+2]));
			temptrips.push_back(Trip(12*i+3+2, 6*i+5, aPAx0[12*i+3+1]));
			temptrips.push_back(Trip(12*i+6+1, 6*i+5, aPAx0[12*i+6+2]));
			temptrips.push_back(Trip(12*i+6+2, 6*i+5, aPAx0[12*i+6+1]));
			temptrips.push_back(Trip(12*i+9+1, 6*i+5, aPAx0[12*i+9+2]));
			temptrips.push_back(Trip(12*i+9+2, 6*i+5, aPAx0[12*i+9+1]));
		}
		aPAx0DS.setFromTriplets(temptrips.begin(), temptrips.end());
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
		// //Dense
		// aJacKKT.setZero();
		// aJacConstrains.setZero();
		// //col1
		// aJacKKT.block(0,0,aExx.rows(), aExx.cols()) = Exx();
		// aJacKKT.block(aExx.rows(), 0, aExr.cols(), aExr.rows()) = Exr().transpose();
		// aJacKKT.block(aExx.rows()+aExr.cols(), 0, aCG.rows(), aCG.cols()) = aCG;
		// //col2
		// aJacKKT.block(0,aExx.cols(),aExr.rows(), aExr.cols()) = Exr();
		// aJacKKT.block(aExr.rows(), aExx.cols(), aErr.rows(), aErr.cols()) = Err();
		// // // //col3
		// aJacKKT.block(0, aExx.cols()+aExr.cols(), aCG.cols(), aCG.rows())= aCG.transpose();
		// // //rhs
		// aJacConstrains.block(0,0, aExs.rows(), aExs.cols()) = Exs();
		// aJacConstrains.block(aExs.rows(), 0, aErs.rows(), aErs.cols()) = Ers();
		// // print("before LU");		
		// VectorXd ExEr(aEx.size()+aEr.size());
		// ExEr<<aEx,aEr;
		// VectorXd PtExEr = adjointP.transpose()*ExEr;
		// VectorXd g = aJacKKT.fullPivLu().solve(PtExEr);
		// aDEDs = aJacConstrains.transpose()*g + aEs;
		// // std::ofstream ExxFile("Exx.mat");
		// // if (ExxFile.is_open())
		// // {
		// // 	ExxFile << aExx;
		// // }
		// // ExxFile.close();

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

		// print("before LU");
		if(!jacLUanalyzed){
			ajacLU.analyzePattern(aJacKKTSparse);	
			jacLUanalyzed =true;
		}
		ajacLU.factorize(aJacKKTSparse);
		if(ajacLU.info() == Eigen::NumericalIssue){
            cout<<"Possibly using a non- pos def matrix in the LLT method"<<endl;
            exit(0);
        }
		

		// print("after LU");
		// MatrixXd results = aJacKKT.fullPivLu().solve(aJacConstrains).topRows(aExs.rows()+aErs.rows());
		// SparseMatrix<double> results = ajacLU.solve(aJacConstrainsSparse);
		// SparseMatrix<double> allres = results.topRows(aExx.rows()+aErr.rows());
		// SparseMatrix<double> dgds = allres.topRows(aExx.rows());
		// SparseMatrix<double> drds = allres.bottomRows(aErr.rows());
		// aDEDs = dgds.transpose()*aEx + drds.transpose()*aEr + aEs;

		VectorXd ExEr(aEx.size()+aEr.size());
		ExEr<<aEx,aEr;
		VectorXd PtExEr = adjointP.transpose()*ExEr;
		VectorXd g = ajacLU.solve(PtExEr);
		aDEDs = aJacConstrainsSparse.transpose()*g + aEs;
	
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
		aExs_trips.clear();
		aErs_max_trips.clear();
		// Exr_trip.reserve(3*t_size*z_size);
		// Err_trip.reserve(3*3*t_size);
		// Exs_trip.reserve();
		// Ers_trip.reserve();
		//Exx is constant

		Matrix3d Jx = cross_prod_mat(1,0,0);
		Matrix3d Jy = cross_prod_mat(0,1,0);
		Matrix3d Jz = cross_prod_mat(0,0,1);
		VectorXd ms = m.red_s();
		VectorXd mr = a_Wr*m.red_r();
		VectorXd PAg = aPAG*m.red_x() + aPAx0;

		// Exr
		// print("		Exr");
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
					Matrix3d p1 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+0)*(s*aPAx0.segment<3>(12*t+0)).transpose();
					Matrix3d p2 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+3)*(s*aPAx0.segment<3>(12*t+3)).transpose();
					Matrix3d p3 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+6)*(s*aPAx0.segment<3>(12*t+6)).transpose();
					Matrix3d p4 = -1*aPAG.col(3*m.T().row(t)[e]+a).segment<3>(12*t+9)*(s*aPAx0.segment<3>(12*t+9)).transpose();
					

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
		// print("		Err");
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
		// print("		Exs");
		for(int t =0; t<m.T().rows(); t++){
			//Tet
			Matrix3d r = Map<Matrix3d>(mr.segment<9>(9*t).data()).transpose();
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

		//Ers
		// print("		Ers");
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

		aExr_max.setFromTriplets(aExr_max_trips.begin(), aExr_max_trips.end());
		aErr_max.setFromTriplets(aErr_max_trips.begin(), aErr_max_trips.end());
		aExs.setFromTriplets(aExs_trips.begin(), aExs_trips.end());
		aErs_max.setFromTriplets(aErs_max_trips.begin(), aErs_max_trips.end());
		
		aErr = a_Ww.transpose()*aErr_max*a_Ww;
		aExr = aExr_max*a_Ww;
		aErs = a_Ww.transpose()*aErs_max;
		aErr_trips = to_triplets(aErr);
		aExr_trips = to_triplets(aExr);
		aErs_trips = to_triplets(aErs);
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
		VectorXd ms = m.red_s();
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
		aEs.setZero();
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
			aEs[6*t+0] = Es1;
			aEs[6*t+1] = Es2;
			aEs[6*t+2] = Es3;
			aEs[6*t+3] = Es4;
			aEs[6*t+4] = Es5;
			aEs[6*t+5] = Es6;

		}
		

		// print("			- Gradients");
		return 1;

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
			cout<<i<<",";
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
					std::cout<<"		- Red_ARAP minimize "<<i<<", "<<(newE - previous5ItE)<<std::endl;
					return i;
				}
				previous5ItE = newE;
			}
		
		}
		
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