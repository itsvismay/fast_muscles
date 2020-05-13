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

int exact::woodbury(const exact::Store& store,
						VectorXd& lambda,
						SparseMatrix<double, Eigen::RowMajor>& PF,
						VectorXd& Fvec,
						VectorXd& g,
						VectorXd& d,
						Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Hinv,
						Eigen::SparseMatrix<double, Eigen::RowMajor>& H,
						MatrixXd& Ai, 
						MatrixXd& Vtilde){

	SparseMatrix<double, Eigen::RowMajor> Id9T(9*store.T.rows(), 9*store.T.rows());
	Id9T.setIdentity();

	MatrixXd VAi = PF.transpose()*Vtilde*Ai;
	MatrixXd HiVAi = Hinv.solve(VAi);
	MatrixXd C =  VAi.transpose()*HiVAi;

	MatrixXd HiV = PF*Hinv.solve(PF.transpose()*Vtilde);
	MatrixXd Phi (HiV.rows(), HiV.cols() + Vtilde.cols());
	Phi<<HiV, Vtilde;

	MatrixXd L = 1e-6*MatrixXd::Identity(2*Ai.rows(), 2*Ai.cols());
	L.block(0,0, Ai.rows(), Ai.rows()) = MatrixXd::Zero(Ai.rows(), Ai.cols()); //TL
	L.block(0, Ai.cols(), Ai.rows(), Ai.cols()) = -Ai; //BL
	L.block(Ai.rows(), 0, Ai.rows(), Ai.cols()) = -Ai; //TR
	L.block(Ai.rows(), Ai.cols(), Ai.rows(), Ai.cols()) = C; //BR

	MatrixXd  Id = MatrixXd::Identity(L.rows(), L.cols());

	MatrixXd Q = L.inverse() + Phi.transpose()*H*Phi;
	LDLT<MatrixXd> Qinv;
	Qinv.compute(Q);

	VectorXd PHig = PF*Hinv.solve(g);
		VectorXd JHig = Id9T*PHig;
		VectorXd d1 = Vtilde.transpose()*PHig;
		VectorXd d2 = Ai*d1;
		VectorXd d3 = Vtilde*d2;
		JHig -= d3;
	// VectorXd JHig = store.J*Hig;

	VectorXd JFvec = Id9T*PF*Fvec;
		VectorXd d4 = Vtilde.transpose()*PF*Fvec;
		VectorXd d5 = Ai*d4;
		VectorXd d6 = Vtilde*d5;
		JFvec -= d6;
	// VectorXd JFvec = store.J*PF*Fvec;

	VectorXd rhs = -JHig - (d - JFvec);
	VectorXd w1 = Phi.transpose()*H*rhs;
	VectorXd w2 = Qinv.solve(w1);
	VectorXd w3 = H*Phi*w2;
	lambda = H*rhs - w3;

	// igl::writeDMAT("rhs.dmat", rhs);
	// igl::writeDMAT("w1.dmat", w1);
	// igl::writeDMAT("w2.dmat", w2);
	// igl::writeDMAT("w3.dmat", w3);
	// igl::writeDMAT("lambda.dmat", lambda);
	
	return 1;
}