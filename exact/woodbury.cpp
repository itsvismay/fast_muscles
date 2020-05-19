#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <igl/Timer.h>
#include <iostream>
#include <igl/writeDMAT.h>
#include <Eigen/Cholesky>
#include <Eigen/QR>

#include "woodbury.h"

using namespace Eigen;

int exact::woodbury(VectorXd& lambda,
				VectorXd& Fvec,
				VectorXd& g,
				Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Hinv,
				Eigen::SparseMatrix<double, Eigen::RowMajor>& H,
				const SparseMatrix<double, Eigen::RowMajor>& PF,
				const VectorXd& d,
				const MatrixXd& Ai, 
				const MatrixXd& V,
				const SparseMatrix<double, Eigen::RowMajor>& Id9T,
				const MatrixXd& VAi,
				MatrixXd& HiV,
				MatrixXd& HiVAi,
				MatrixXd& C,
				MatrixXd& Phi,
				MatrixXd& L,
				const MatrixXd& IdL,
				MatrixXd& Q){

	igl::Timer timer;

	timer.start();
	HiV = Hinv.solve(V);
	HiVAi = HiV*Ai;
	C =  VAi.transpose()*HiVAi;
	Phi<<HiV, V;
	timer.stop();
	double time1 = timer.getElapsedTimeInMicroSec();

	timer.start();
	L = 1e-6*IdL;
	L.block(0,0, Ai.rows(), Ai.rows()) = MatrixXd::Zero(Ai.rows(), Ai.cols()); //TL
	L.block(0, Ai.cols(), Ai.rows(), Ai.cols()) = -Ai; //BL
	L.block(Ai.rows(), 0, Ai.rows(), Ai.cols()) = -Ai; //TR
	L.block(Ai.rows(), Ai.cols(), Ai.rows(), Ai.cols()) = C; //BR


	Q = L.inverse() + Phi.transpose()*H*Phi;
	LDLT<MatrixXd> Qinv;
	Qinv.compute(Q);
	timer.stop();
	double time2 = timer.getElapsedTimeInMicroSec();

	timer.start();
	VectorXd Hig = Hinv.solve(g);
		VectorXd JHig = Id9T*Hig;
		VectorXd d1 = V.transpose()*Hig;
		VectorXd d2 = Ai*d1;
		VectorXd d3 = V*d2;
		JHig -= d3;

	VectorXd JFvec = Id9T*Fvec;
		VectorXd d4 = V.transpose()*Fvec;
		VectorXd d5 = Ai*d4;
		VectorXd d6 = V*d5;
		JFvec -= d6;

	VectorXd rhs = -JHig - (d - JFvec);
	VectorXd w1 = Phi.transpose()*H*rhs;
	VectorXd w2 = Qinv.solve(w1);
	VectorXd w3 = H*Phi*w2;
	lambda = H*rhs - w3;
	timer.stop();
	double time3 = timer.getElapsedTimeInMicroSec();
	// std::cout<<"wood times: "<<time1<<", "<<time2<<", "<<time3<<std::endl;
	
	return 1;
}