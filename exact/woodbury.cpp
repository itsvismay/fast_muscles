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
						VectorXd& d,
						Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Hinv,
						Eigen::SparseMatrix<double, Eigen::RowMajor>& H,
						MatrixXd& Ai, 
						MatrixXd& Vtilde,
						MatrixXd& J){

	MatrixXd VAi = Vtilde*Ai;
	MatrixXd HiVAi = Hinv.solve(VAi);
	MatrixXd C =  VAi.transpose()*HiVAi;

	MatrixXd HiV = Hinv.solve(Vtilde);
	MatrixXd Phi (HiV.rows(), HiV.cols() + Vtilde.cols());
	Phi<<HiV, Vtilde;

	MatrixXd L = MatrixXd::Zero(2*Ai.rows(), 2*Ai.cols());
	L.block(0,0, Ai.rows(), Ai.rows()) = MatrixXd::Zero(Ai.rows(), Ai.cols()); //TL
	L.block(0, Ai.cols(), Ai.rows(), Ai.cols()) = -Ai; //BL
	L.block(Ai.rows(), 0, Ai.rows(), Ai.cols()) = -Ai; //TR
	L.block(Ai.rows(), Ai.cols(), Ai.rows(), Ai.cols()) = C; //BR

	MatrixXd  Id = MatrixXd::Identity(L.rows(), L.cols());

	MatrixXd Q = L.inverse() + Phi.transpose()*H*Phi + 1e-6*Id;
	LDLT<MatrixXd> Qinv;
	Qinv.compute(Q);

	// igl::writeDMAT("L.dmat", L);
	// igl::writeDMAT("phi.dmat", Phi);
	// igl::writeDMAT("Q.dmat", Q);
	// std::cout<<Q.rows()<<", "<<Q.cols()<<":"<< Q.norm()<<std::endl;
	// std::cout<<Q.col(199).transpose()<<std::endl;

	VectorXd rhs = -J*Hinv.solve(g) - (d - J*Fvec);
	VectorXd w1 = Phi.transpose()*H*rhs;
	VectorXd w2 = Qinv.solve(w1);
	VectorXd w3 = H*Phi*w2;
	lambda = H*rhs - w3;

	// std::cout<<"rhs: "<<rhs.norm()<<std::endl;
	// // std::cout<<rhs.transpose()<<std::endl;
	// // std::cout<<"w1: "<<w1.norm()<<std::endl;
	// // std::cout<<w1.transpose()<<std::endl;
	// // igl::writeDMAT("w1.dmat", w1);
	// // std::cout<<"w2: "<<w2.norm()<<std::endl;
	// // std::cout<<w2.transpose()<<std::endl;
	// // igl::writeDMAT("w2.dmat", w2);
	// // std::cout<<"w3: "<<w3.norm()<<std::endl;
	// // std::cout<<w3.transpose()<<std::endl;
	// std::cout<<"lambda: "<<lambda.norm()<<std::endl;
	// igl::writeDMAT("lambda.dmat", lambda);
	// std::cout<<lambda.transpose()<<std::endl;
	// exit(0);
	return 1;
}