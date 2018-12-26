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
	SparseLU<SparseMatrix<double>>  aARAPKKTSparseSolver;
	SparseLU<SparseMatrix<double>> ajacLU;
	// FullPivLU<MatrixXd>  aARAPKKTSolver;
	VectorXd aPAx0, aUtPAx0, aEr, aEs, aEx, aDEDs, aFPAx0;
	SparseMatrix<double> aExx, aExr, aErr, aExs, aErs, aPA, aC;
	MatrixXd aJacKKT, aJacConstrains;
	SparseMatrix<double> aJacKKTSparse, aJacConstrainsSparse;
	std::vector<Trip> aExx_trips, aExr_trips, aErr_trips, aExs_trips, aErs_trips, aC_trips;
	double jacLUanalyzed = false;

	SparseMatrix<double> adjointP;

public:
	Arap(Mesh& m){
		int r_size = m.red_w().size();
		int z_size = m.red_x().size();
		int s_size = m.red_s().size();
		int t_size = m.T().rows();
		if(!(6*t_size==s_size && 3*t_size==r_size && z_size==3*m.V().rows())){
			print(t_size);
			print(m.V().rows());
			print(s_size);
			print(r_size);
			print(z_size);
			print("ARAP::Arap(Mesh& m) Problem is reduced. Use reduced ARAP");
			exit(0);
		}

		aFPAx0.resize(12*m.T().rows());
		print("arap 1");
		aExx = (m.P()*m.A()).transpose()*(m.P()*m.A());
		aExx_trips = to_triplets(aExx);
		
		print("arap 2");
		aErr.resize(r_size, r_size);
		aExs.resize(z_size, s_size);
		aErs.resize(r_size, s_size);
		aExr.resize(z_size, r_size);
		aEr.resize(r_size);
		aEs.resize(s_size);

		print("arap 3");
		aPAx0 = m.P()*m.A()*m.x0();
		aUtPAx0 = m.GU().transpose()*aPAx0;
		aPA = m.P()*m.A();
		aC = m.AB().transpose();
		
		print("arap 4");
		SparseMatrix<double> Exx = (m.P()*m.A()).transpose()*(m.P()*m.A());
		VectorXd PAx0 = m.P()*m.A()*m.x0();
		VectorXd UtPAx0 = m.GU().transpose()*PAx0;
		SparseMatrix<double> spKKTmat(Exx.rows()+aC.rows(), Exx.rows()+aC.rows());
		spKKTmat.setZero();
		std::vector<Trip> ExxTrips = to_triplets(Exx);
		aC_trips = to_triplets(aC);
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
		// aARAPKKTSolver.compute(MatrixXd(spKKTmat));
		
		print("arap 5");
		// aJacKKT.resize(z_size+r_size+aC.rows(), z_size+r_size+aC.rows());
		// aJacConstrains.resize(z_size+r_size+aC.rows() ,s_size);
		aJacKKTSparse.resize(z_size+r_size+aC.rows(), z_size+r_size+aC.rows());
		aJacConstrainsSparse.resize(z_size+r_size+aC.rows() ,s_size);
		
		print("arap 6");
		setupAdjointP();
	}

	void setupAdjointP(){
		adjointP.resize(aExx.rows()+aErr.rows(), aExx.rows()+aErr.rows()+aC.rows());
		for(int i=0; i<aExx.rows()+aErr.rows(); i++){
			adjointP.coeffRef(i,i) = 1;
		}
	}

	double Energy(Mesh& m){
		VectorXd PAx = aPA*m.red_x() + aPAx0;
		m.constTimeFPAx0(aFPAx0);
		double En= 0.5*(PAx - aFPAx0).squaredNorm();
		return En;
	}

	double Energy(Mesh& m, VectorXd& z, VectorXd& redw, VectorXd& redr, VectorXd& reds, VectorXd& redu){
		VectorXd PAx = aPA*z + aPAx0;
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
            s<< reds[6*i + 0], reds[6*i + 3], reds[6*i + 4],
                reds[6*i + 3], reds[6*i + 1], reds[6*i + 5],
                reds[6*i + 4], reds[6*i + 5], reds[6*i + 2];

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
		// aJacKKT.setZero();
		// aJacConstrains.setZero();
		// //col1
		// aJacKKT.block(0,0,aExx.rows(), aExx.cols()) = MatrixXd(Exx());
		// aJacKKT.block(aExx.rows(), 0, aExr.cols(), aExr.rows()) = MatrixXd(Exr()).transpose();
		// aJacKKT.block(aExx.rows()+aExr.cols(), 0, aC.rows(), aC.cols()) = MatrixXd(aC);
		// //col2
		// aJacKKT.block(0,aExx.cols(),aExr.rows(), aExr.cols()) = MatrixXd(Exr());
		// aJacKKT.block(aExr.rows(), aExx.cols(), aErr.rows(), aErr.cols()) = MatrixXd(Err());
		// // // //col3
		// aJacKKT.block(0, aExx.cols()+aExr.cols(), aC.cols(), aC.rows())= MatrixXd(aC).transpose();
		// // //rhs
		// aJacConstrains.block(0,0, aExs.rows(), aExs.cols()) = MatrixXd(Exs());
		// aJacConstrains.block(aExs.rows(), 0, aErs.rows(), aErs.cols()) = MatrixXd(Ers());
		// print(aJacKKT - MatrixXd(aJacKKTSparse));
		// print(aJacConstrains - MatrixXd(aJacConstrainsSparse));

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
		
		// std::ofstream ExxFile("Exx.mat");
		// if (ExxFile.is_open())
		// {
		// 	ExxFile << aExx;
		// }
		// ExxFile.close();

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
		aExr_trips.clear();
		aErr_trips.clear();
		aExs_trips.clear();
		aErs_trips.clear();
		// Exr_trip.reserve(3*t_size*z_size);
		// Err_trip.reserve(3*3*t_size);
		// Exs_trip.reserve();
		// Ers_trip.reserve();
		//Exx is constant

		Matrix3d Jx = cross_prod_mat(1,0,0);
		Matrix3d Jy = cross_prod_mat(0,1,0);
		Matrix3d Jz = cross_prod_mat(0,0,1);
		VectorXd ms = m.red_s();
		VectorXd PAg = aPA*m.red_x() + aPAx0;
		
		// Exr
		// print("		Exr");
		for(int t=0; t<m.T().rows(); t++){
			//Tet
			Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];
			
			Matrix3d r0= Map<Matrix3d>(m.red_r().segment<9>(9*t).data()).transpose();

			Matrix3d r1 = r0*Jx;
			Matrix3d r2 = r0*Jy;
			Matrix3d r3 = r0*Jz;

			for(int e=0; e<4; e++){
				//Vert on tet
				for(int a =0; a<3; a++){
					Matrix3d p1 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+0)*(u*s*aUtPAx0.segment<3>(12*t+0)).transpose();
					Matrix3d p2 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+3)*(u*s*aUtPAx0.segment<3>(12*t+3)).transpose();
					Matrix3d p3 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+6)*(u*s*aUtPAx0.segment<3>(12*t+6)).transpose();
					Matrix3d p4 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+9)*(u*s*aUtPAx0.segment<3>(12*t+9)).transpose();
					

					double Exar1 = p1.cwiseProduct(r1).sum() + p2.cwiseProduct(r1).sum() + p3.cwiseProduct(r1).sum() + p4.cwiseProduct(r1).sum();
					double Exar2 = p1.cwiseProduct(r2).sum() + p2.cwiseProduct(r2).sum() + p3.cwiseProduct(r2).sum() + p4.cwiseProduct(r2).sum();
					double Exar3 = p1.cwiseProduct(r3).sum() + p2.cwiseProduct(r3).sum() + p3.cwiseProduct(r3).sum() + p4.cwiseProduct(r3).sum();

					aExr_trips.push_back(Trip(3*m.T().row(t)[e]+a,3*t+0, Exar1));
					aExr_trips.push_back(Trip(3*m.T().row(t)[e]+a,3*t+1, Exar2));
					aExr_trips.push_back(Trip(3*m.T().row(t)[e]+a,3*t+2, Exar3));
				}
			}
		}

		//Err
		// print("		Err");
		for(int t=0; t<m.T().rows(); t++){
			Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];

			Matrix3d r0= Map<Matrix3d>(m.red_r().segment<9>(9*t).data()).transpose();
			Matrix3d r1 = r0*0.5*(Jx*Jx + Jx*Jx);
			Matrix3d r2 = r0*0.5*(Jx*Jy + Jy*Jx);
			Matrix3d r3 = r0*0.5*(Jx*Jz + Jz*Jx);
			Matrix3d r5 = r0*0.5*(Jy*Jy + Jy*Jy);
			Matrix3d r6 = r0*0.5*(Jy*Jz + Jz*Jy);
			Matrix3d r9 = r0*0.5*(Jz*Jz + Jz*Jz);

			
			Matrix3d pr1 = -PAg.segment<3>(12*t+0)*(u*s*aUtPAx0.segment<3>(12*t+0)).transpose();
			Matrix3d pr2 = -PAg.segment<3>(12*t+3)*(u*s*aUtPAx0.segment<3>(12*t+3)).transpose();
			Matrix3d pr3 = -PAg.segment<3>(12*t+6)*(u*s*aUtPAx0.segment<3>(12*t+6)).transpose();
			Matrix3d pr4 = -PAg.segment<3>(12*t+9)*(u*s*aUtPAx0.segment<3>(12*t+9)).transpose();

			double v00 = pr1.cwiseProduct(r1).sum()+pr2.cwiseProduct(r1).sum()+pr3.cwiseProduct(r1).sum()+pr4.cwiseProduct(r1).sum();
			double v01 = pr1.cwiseProduct(r2).sum()+pr2.cwiseProduct(r2).sum()+pr3.cwiseProduct(r2).sum()+pr4.cwiseProduct(r2).sum();
			double v02 = pr1.cwiseProduct(r3).sum()+pr2.cwiseProduct(r3).sum()+pr3.cwiseProduct(r3).sum()+pr4.cwiseProduct(r3).sum();
			double v11 = pr1.cwiseProduct(r5).sum()+pr2.cwiseProduct(r5).sum()+pr3.cwiseProduct(r5).sum()+pr4.cwiseProduct(r5).sum();
			double v12 = pr1.cwiseProduct(r6).sum()+pr2.cwiseProduct(r6).sum()+pr3.cwiseProduct(r6).sum()+pr4.cwiseProduct(r6).sum();
			double v22 = pr1.cwiseProduct(r9).sum()+pr2.cwiseProduct(r9).sum()+pr3.cwiseProduct(r9).sum()+pr4.cwiseProduct(r9).sum();
			
			aErr_trips.push_back(Trip(3*t+0,3*t+0, v00));
			aErr_trips.push_back(Trip(3*t+0,3*t+1, v01));
			aErr_trips.push_back(Trip(3*t+0,3*t+2, v02));
			aErr_trips.push_back(Trip(3*t+1,3*t+1, v11));
			aErr_trips.push_back(Trip(3*t+1,3*t+2, v12));
			aErr_trips.push_back(Trip(3*t+2,3*t+2, v22));
			aErr_trips.push_back(Trip(3*t+1,3*t+0, v01));
			aErr_trips.push_back(Trip(3*t+2,3*t+0, v02));
			aErr_trips.push_back(Trip(3*t+2,3*t+1, v12));
		}

		//Exs
		// print("		Exs");
		for(int t =0; t<m.T().rows(); t++){
			//Tet
			Matrix3d r = Map<Matrix3d>(m.red_r().segment<9>(9*m.r_elem_cluster_map()[t]).data()).transpose();
			Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
			for(int e=0; e<4; e++){
				//Vert on tet
				for(int a=0; a<3; a++){
					//x, y, or z axis
					Vector3d GtAtPtRU_row1 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+0).transpose()*r*u;
					Vector3d GtAtPtRU_row2 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+3).transpose()*r*u;
					Vector3d GtAtPtRU_row3 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+6).transpose()*r*u;
					Vector3d GtAtPtRU_row4 = -1*aPA.col(3*m.T().row(t)[e]+a).segment<3>(12*t+9).transpose()*r*u;
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
		for(int t=0; t<m.T().rows(); t++){
			//Tet
			Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
			Matrix3d r0= Map<Matrix3d>(m.red_r().segment<9>(9*t).data()).transpose();
			Matrix3d r1 = r0*Jx;
			Matrix3d r2 = r0*Jy;
			Matrix3d r3 = r0*Jz;

			Matrix<double, 12,3> innerContraction;
			for(int a =0; a<3; a++){
				Matrix3d pr1 = -PAg.segment<3>(12*t+0)*u.col(a).transpose();
				Matrix3d pr2 = -PAg.segment<3>(12*t+3)*u.col(a).transpose();
				Matrix3d pr3 = -PAg.segment<3>(12*t+6)*u.col(a).transpose();
				Matrix3d pr4 = -PAg.segment<3>(12*t+9)*u.col(a).transpose();

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
				Matrix3d p1 = innerContraction.col(a).segment<3>(0)*aUtPAx0.segment<3>(12*t+0).transpose();
				Matrix3d p2 = innerContraction.col(a).segment<3>(3)*aUtPAx0.segment<3>(12*t+3).transpose();
				Matrix3d p3 = innerContraction.col(a).segment<3>(6)*aUtPAx0.segment<3>(12*t+6).transpose();
				Matrix3d p4 = innerContraction.col(a).segment<3>(9)*aUtPAx0.segment<3>(12*t+9).transpose();

				double Eras1 = p1(0,0) + p2(0,0) + p3(0,0) + p4(0,0);
				double Eras2 = p1(1,1) + p2(1,1) + p3(1,1) + p4(1,1);
				double Eras3 = p1(2,2) + p2(2,2) + p3(2,2) + p4(2,2);
				double Eras4 = p1(0,1) + p2(0,1) + p3(0,1) + p4(0,1)+ p1(1,0) + p2(1,0) + p3(1,0) + p4(1,0);
				double Eras5 = p1(0,2) + p2(0,2) + p3(0,2) + p4(0,2)+ p1(2,0) + p2(2,0) + p3(2,0) + p4(2,0);
				double Eras6 = p1(2,1) + p2(2,1) + p3(2,1) + p4(2,1)+ p1(1,2) + p2(1,2) + p3(1,2) + p4(1,2);
				
				aErs_trips.push_back(Trip(3*t+a, 6*t+0, Eras1));
				aErs_trips.push_back(Trip(3*t+a, 6*t+1, Eras2));
				aErs_trips.push_back(Trip(3*t+a, 6*t+2, Eras3));
				aErs_trips.push_back(Trip(3*t+a, 6*t+3, Eras4));
				aErs_trips.push_back(Trip(3*t+a, 6*t+4, Eras5));
				aErs_trips.push_back(Trip(3*t+a, 6*t+5, Eras6));

			}	
		}

		// print("		-Hessians");		
		return 1;
	}

	int Gradients(Mesh& m){
		// print("			+ Gradients");
		// print("		Ex");
		aEx = dEdx(m);
		
		// print("		Er");
		aEr.setZero();
		Matrix3d Jx = cross_prod_mat(1,0,0);
		Matrix3d Jy = cross_prod_mat(0,1,0);
		Matrix3d Jz = cross_prod_mat(0,0,1);

		VectorXd PAg = aPA*m.red_x() + aPAx0;
		VectorXd ms = m.red_s();
		for(int t=0; t<m.T().rows(); t++){
			Matrix3d u = Map<Matrix3d>(m.red_u().segment<9>(9*t).data()).transpose();
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];

			Matrix3d r0= Map<Matrix3d>(m.red_r().segment<9>(9*t).data()).transpose();
			Matrix3d r1 = r0*Jx;
			Matrix3d r2 = r0*Jy;
			Matrix3d r3 = r0*Jz;

			Matrix3d p1 = -PAg.segment<3>(12*t+0)*(u*s*aUtPAx0.segment<3>(12*t+0)).transpose();
			Matrix3d p2 = -PAg.segment<3>(12*t+3)*(u*s*aUtPAx0.segment<3>(12*t+3)).transpose();
			Matrix3d p3 = -PAg.segment<3>(12*t+6)*(u*s*aUtPAx0.segment<3>(12*t+6)).transpose();
			Matrix3d p4 = -PAg.segment<3>(12*t+9)*(u*s*aUtPAx0.segment<3>(12*t+9)).transpose();

			double Er1 = p1.cwiseProduct(r1).sum() + p2.cwiseProduct(r1).sum() + p3.cwiseProduct(r1).sum() + p4.cwiseProduct(r1).sum();
			double Er2 = p1.cwiseProduct(r2).sum() + p2.cwiseProduct(r2).sum() + p3.cwiseProduct(r2).sum() + p4.cwiseProduct(r2).sum();
			double Er3 = p1.cwiseProduct(r3).sum() + p2.cwiseProduct(r3).sum() + p3.cwiseProduct(r3).sum() + p4.cwiseProduct(r3).sum();

			aEr[3*t+0] = Er1;
			aEr[3*t+1] = Er2;
			aEr[3*t+2] = Er3;

		}
		
		// print("		Es");
		aEs.setZero();
		for(int t=0; t<m.T().rows(); t++){
			Matrix3d rt = Map<Matrix3d>(m.red_r().segment<9>(9*m.r_elem_cluster_map()[t]).data());
			Matrix3d ut = Map<Matrix3d>(m.red_u().segment<9>(9*t).data());
			Matrix3d s;
			s<< ms[6*t + 0], ms[6*t + 3], ms[6*t + 4],
				ms[6*t + 3], ms[6*t + 1], ms[6*t + 5],
				ms[6*t + 4], ms[6*t + 5], ms[6*t + 2];

			Matrix3d p1 = s*aUtPAx0.segment<3>(12*t+0)*aUtPAx0.segment<3>(12*t+0).transpose() - (ut*rt*PAg.segment<3>(12*t+0))*aUtPAx0.segment<3>(12*t+0).transpose();
			Matrix3d p2 = s*aUtPAx0.segment<3>(12*t+3)*aUtPAx0.segment<3>(12*t+3).transpose() - (ut*rt*PAg.segment<3>(12*t+3))*aUtPAx0.segment<3>(12*t+3).transpose();
			Matrix3d p3 = s*aUtPAx0.segment<3>(12*t+6)*aUtPAx0.segment<3>(12*t+6).transpose() - (ut*rt*PAg.segment<3>(12*t+6))*aUtPAx0.segment<3>(12*t+6).transpose();
			Matrix3d p4 = s*aUtPAx0.segment<3>(12*t+9)*aUtPAx0.segment<3>(12*t+9).transpose() - (ut*rt*PAg.segment<3>(12*t+9))*aUtPAx0.segment<3>(12*t+9).transpose();
			
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
		VectorXd PAx = aPA*m.red_x() + aPAx0;
		m.constTimeFPAx0(aFPAx0);
		VectorXd res = (aPA).transpose()*(PAx - aFPAx0);
		return res;
	}

	bool itT(Mesh& m){
		m.constTimeFPAx0(aFPAx0);
		VectorXd deltaABtx = m.AB().transpose()*m.dx();
		VectorXd AtPtFPAx0 = (aPA).transpose()*aFPAx0;
		VectorXd AtPtPAx0 = (aPA).transpose()*(aPAx0);
		VectorXd gb = AtPtFPAx0 - AtPtPAx0;
		VectorXd gd(gb.size()+deltaABtx.size());
		gd<<gb,deltaABtx;
		VectorXd result = aARAPKKTSparseSolver.solve(gd);
		VectorXd gu = result.head(gb.size());
		m.red_x(gu);
		// VectorXd lambda = result.tail(deltaABtx.size());
		// print("first order optimality");
		// cout<<(aC*dEdx(m) + lambda).norm()<<", ";
		// print(lambda.transpose());
		// print("q");
		// print((m.AB().transpose()*gu).transpose());
		// print("q");
		// print(aPAx0.transpose());
		// print(aFPAx0.transpose());
		// print(deltaABtx.transpose());
		// print(AtPtPAx0.transpose());
		// print("q");
		// print(AtPtFPAx0.transpose());
		// print("q");
		// print(gu.transpose());
		// print("q");
		// print((m.AB().transpose()*gu).transpose());
		return false;

	}

	void itR(Mesh& m, VectorXd& USUtPAx0){
		VectorXd PAx = aPA*m.red_x() + aPAx0;
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

		double previous5ItE = Energy(m);
		double oldE = Energy(m);
		// VectorXd E0 = m.B().transpose()*dEdx(m);
		for(int i=1; i< 10000; i++){
			bool converged = itT(m);
			itR(m, USUtPAx0);
			double newE = Energy(m);
			if((newE - oldE)>1e-8){
				print("Arap::minimize() error. ARAP should monotonically decrease.");
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
				if(fabs(newE - previous5ItE)<1e-12){
					if(i>1000){
						// print(m.red_s().transpose());
						// exit(0);
					}
					// std::cout<<"		- ARAP minimize "<<i<<", "<<(newE - previous5ItE)<<std::endl;
					return;
				}
				previous5ItE = newE;
			}
			
			// VectorXd Ex = m.B().transpose()*dEdx(m);
			// if((Ex - E0).norm()<1e-8){
			// 	std::cout<<"		- ARAP minimize "<<i<<", "<<(Ex - E0).norm()<<std::endl;
			// 	return;
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
		// std::cout<<"		- ARAP never converged "<<Energy(m)-previous5ItE<<std::endl;
		// exit(0);
	}

	SparseMatrix<double> Exx(){ 
		return aExx; 
	}
	SparseMatrix<double> Exr(){ 
		aExr.setZero();
		aExr.setFromTriplets(aExr_trips.begin(), aExr_trips.end());
		return aExr; 
	}
	SparseMatrix<double> Exs(){
		aExs.setZero();
		aExs.setFromTriplets(aExs_trips.begin(), aExs_trips.end()); 
		return aExs; 
	}
	SparseMatrix<double> Ers(){
		aErs.setZero();
		aErs.setFromTriplets(aErs_trips.begin(), aErs_trips.end());
		return aErs; 
	}
	SparseMatrix<double> Err(){ 
		aErr.setZero();
		aErr.setFromTriplets(aErr_trips.begin(), aErr_trips.end());
		return aErr; 
	}
	VectorXd Er() { return aEr; }
	VectorXd Es() { return aEs; }
	VectorXd Ex() { return aEx; }

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