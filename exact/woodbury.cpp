#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <igl/Timer.h>
#include <iostream>
#include <igl/writeDMAT.h>
#include <Eigen/Cholesky>
#include <Eigen/QR>

#include "woodbury.h"
#include "omp.h"
using namespace Eigen;

void exact::sparse_to_dense(MatrixXd& denseHess, MatrixXd& denseHessInv, SparseMatrix<double, Eigen::RowMajor>& H){
	//Fill denseHess with 9x9 block diags from H
	//TODO: this should be done in the hessians code. coeffRef is expensive
	//FIX AFTER THE DEADLINE

	#pragma omp parallel for
	for(int i=0; i<denseHess.rows()/9; i++){
		//loop through 9x9 block and fill denseH
		Matrix9d A;
		#pragma omp parallel for collapse(2)
		for(int j =0; j<9; j++){
			for(int k=0; k<9; k++){
				A(j, k) = H.coeffRef(9*i + j, 9*i +k);
			}
		}
		denseHess.block<9,9>(9*i, 0) = A;
		denseHessInv.block<9,9>(9*i, 0) = A.inverse();

	}
}

int exact::woodbury(VectorXd& lambda,
				VectorXd& Fvec,
				VectorXd& g,
				MatrixXd& denseHinv,
				MatrixXd& denseH,
				SparseMatrix<double, Eigen::RowMajor>& H,
				Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Hinv,
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
				MatrixXd& HPhi,
				MatrixXd& L,
				const MatrixXd& IdL,
				MatrixXd& Q){

	HiV.setZero();
	HiVAi.setZero();
	C.setZero();
	L.setZero();
	Q.setZero();
	Phi.setZero();

	// igl::Timer timer;

	// // #pragma omp parallel for
	// for(int i=0; i<HiV.rows()/9; i++){
	// 	HiV.block(9*i,0, 9, V.cols()) = denseHinv.block<9,9>(9*i, 0)*V.block(9*i, 0, 9, V.cols());
	// }

	// HiVAi = HiV*Ai;
	// C =  VAi.transpose()*HiVAi;
	// Phi.block(0,0,HiV.rows(), HiV.cols()) = HiV;

	// L.block(Ai.rows(), Ai.cols(), Ai.rows(), Ai.cols()) = C + 1e-6*IdL; //BR


	// Q.setZero();
	// //-------------------------------------
	// timer.start();
	// int modes = V.cols();
	// int num_elements = denseH.rows()/9;
	// int num_threads = Eigen::nbThreads( );
	// int block_size = 9;
	// int num_elements_per_thread = num_elements/num_threads;
	
	// int num_meta_blocks = 1;
	// int num_blocks = denseH.rows()/9;
	// #pragma omp parallel
 //    {
 //    	int thread_num = omp_get_thread_num();
 //        unsigned int start_id = thread_num*(num_blocks/num_threads);
 //        unsigned int start_id_elements = thread_num*num_elements_per_thread;
 //        unsigned int start_id_blocks = start_id_elements; 

 //        unsigned int actual_row_size = block_size;
 //        Eigen::MatrixXd Htmp = Phi.block(9*start_id_elements, 0, 9, 2*modes).transpose()
	// 				        	*denseH.block<9,9>(9*start_id_elements, 0)
	// 				        	*Phi.block(9*start_id_elements, 0, 9, 2*modes);
        
 //        for(unsigned int ii = start_id_elements+1; ii < start_id_elements+num_elements_per_thread;  ii+=1) {
 //            Htmp.noalias() += Phi.block(9*ii, 0, 9, 2*modes).transpose()
	// 		            	*denseH.block<9,9>(9*ii,0)
	// 		            	*(Phi.block(9*ii, 0, 9, 2*modes));
 //        }

 //        #pragma omp critical
 //        {
 //            Q.noalias() += Htmp;
 //        }
 //    }
 //    int last_index_counted = num_elements_per_thread*num_threads;
 //    for(int i=0; i<num_elements%num_threads; i+=1){
 //    	int ii = i + last_index_counted;
 //    	Q.noalias() += Phi.block(9*ii, 0, 9, 2*modes).transpose()
	// 		            	*denseH.block<9,9>(9*ii,0)
	// 		            	*(Phi.block(9*ii, 0, 9, 2*modes));
 //    }
 //    //-------------------------------------
	// timer.stop();
	// double time1 = timer.getElapsedTimeInMicroSec();

	// timer.start();
	// 	// MatrixXd Q2 = Phi.transpose()*H*Phi;
	// timer.stop();
	// // std::cout<<"error: "<<(Q - Q2).norm()<<std::endl;
	// // exit(0);

	// double time2 = timer.getElapsedTimeInMicroSec();

	// timer.start();
	// 	Q += L.inverse();
	// 	LDLT<MatrixXd> Qinv;
	// 	Qinv.compute(Q);
	// timer.stop();
	// double time3 = timer.getElapsedTimeInMicroSec();

	// VectorXd Hig = g;
	// // #pragma omp parallel for
	// for(int i=0; i<denseHinv.rows()/9; i++){
	// 	Hig.segment<9>(9*i) = denseHinv.block<9,9>(9*i,0)*g.segment<9>(9*i);
	// }
	// std::cout<<"	Hig: "<<Hig.norm()<<std::endl;
		
	// 	VectorXd d1 = V.transpose()*Hig;
	// 	VectorXd d2 = Ai*d1;
	// 	VectorXd d3 = V*d2;
	// 	VectorXd JHig = Hig - d3;

	// 	VectorXd d4 = V.transpose()*Fvec;
	// 	VectorXd d5 = Ai*d4;
	// 	VectorXd d6 = V*d5;
	// 	VectorXd JFvec = Fvec - d6;

	// VectorXd rhs = -JHig - (d - JFvec);
	// VectorXd w0 = H*rhs;
	// VectorXd w1 = Phi.transpose()*w0;
	// VectorXd w2 = Qinv.solve(w1);
	// VectorXd w3 = Phi*w2;
	// VectorXd w4 = H*w3;
	// lambda = H*rhs - w4;
	// // std::cout<<"wood times: "<<time1<<", "<<time2<<", "<<time3<<std::endl;


	HiV = Hinv.solve(V);
	HiVAi = HiV*Ai;
	C =  VAi.transpose()*HiVAi;
	Phi<<HiV, V;
	std::cout<<"	Phi: "<<Phi.norm()<<std::endl;

	L.block(0,0, Ai.rows(), Ai.rows()) = MatrixXd::Zero(Ai.rows(), Ai.cols()); //TL
	L.block(0, Ai.cols(), Ai.rows(), Ai.cols()) = -Ai; //BL
	L.block(Ai.rows(), 0, Ai.rows(), Ai.cols()) = -Ai; //TR
	L.block(Ai.rows(), Ai.cols(), Ai.rows(), Ai.cols()) = C; //BR
	L += 1e-6*MatrixXd::Identity(L.rows(), L.cols());

	Q = L.inverse() + Phi.transpose()*H*Phi + 1e-6*MatrixXd::Identity(L.rows(), L.cols());
	PartialPivLU<MatrixXd> Qinv;
	Qinv.compute(Q);
	
	VectorXd Hig = g;
		// #pragma omp parallel for
		for(int i=0; i<denseHinv.rows()/9; i++){
			Hig.segment<9>(9*i) = denseHinv.block<9,9>(9*i,0)*g.segment<9>(9*i);
		}
		VectorXd d1 = V.transpose()*Hig;
		VectorXd d2 = Ai*d1;
		VectorXd d3 = V*d2;
		VectorXd JHig = Hig - d3;

	std::cout<<"	Hig: "<<Hig.norm()<<std::endl;
		VectorXd d4 = V.transpose()*Fvec;
		VectorXd d5 = Ai*d4;
		VectorXd d6 = V*d5;
		VectorXd JFvec = Fvec - d6;

	VectorXd rhs = -JHig - (d - JFvec);
	VectorXd w0 = H*rhs;
	VectorXd w1 = Phi.transpose()*w0;
	VectorXd w2 = Qinv.solve(w1);
	VectorXd w3 = H*Phi*w2;
	lambda = H*rhs - w3;
	std::cout<<"	Q: "<<Q.norm()<<std::endl;
	std::cout<<"	JHig: "<<JHig.norm()<<std::endl;
	std::cout<<"	JFvec: "<<JFvec.norm()<<std::endl;
	std::cout<<"	rhs: "<<rhs.norm()<<std::endl;
	std::cout<<"	w0: "<<w0.norm()<<std::endl;
	std::cout<<"	w1: "<<w1.norm()<<std::endl;
	std::cout<<"	w2: "<<w2.norm()<<std::endl;
	std::cout<<"	w3: "<<w3.norm()<<std::endl;

	return 1;
}