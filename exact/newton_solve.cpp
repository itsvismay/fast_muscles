#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <igl/Timer.h>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <igl/writeDMAT.h>


#include "newton_solve.h"
#include "woodbury.h"
#include "muscle_energy_gradient.h"
#include "stablenh_energy_gradient.h"
#include "store.h"

using namespace Eigen;
using Store = exact::Store;


int exact::newton_solve(const Store& store, VectorXd& Fvec, VectorXd& d, MatrixXd& Ai, MatrixXd& Vtilde, MatrixXd& J, double activation){

	int MAX_ITERS = 1;
	double tol = 1e-3;

	VectorXd g = VectorXd::Zero(Fvec.size());
	VectorXd g_n = VectorXd::Zero(Fvec.size());
	VectorXd g_m = VectorXd::Zero(Fvec.size());
	VectorXd deltaF = VectorXd::Zero(Fvec.size());
	VectorXd lambda;

	SparseMatrix<double, Eigen::RowMajor> H(Fvec.size(), Fvec.size());
	SparseMatrix<double, Eigen::RowMajor> H_n(Fvec.size(), Fvec.size());
	SparseMatrix<double, Eigen::RowMajor> H_m(Fvec.size(), Fvec.size());

	Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> Hinv;

	SparseMatrix<double, Eigen::RowMajor> Id(Fvec.size(), Fvec.size());
	Id.setIdentity();


	for(int its = 0; its<MAX_ITERS; its++){
		exact::stablenh::hessian(store, Fvec, H_n);
		exact::muscle::hessian(store, Fvec, H_m);
		exact::stablenh::gradient(store, Fvec, g_n);
		exact::muscle::gradient(store, Fvec, g_m);


		VectorXd g = g_n + activation*g_m;
		std::cout<<"grad: "<<g.norm()<<std::endl;

		H = H_n + activation*H_m + 1e-6*Id;

		Hinv.compute(H);
		if(Hinv.info()!=Eigen::Success){
			std::cout<<"SOLVER FAILED iteration: "<<its<<std::endl;
			std::cout<<Hinv.info()<<std::endl;
			exit(0);
		}
		exact::woodbury(lambda, Fvec, g, d, Hinv, H, Ai, Vtilde, J);

		deltaF = -1*Hinv.solve(g + J.transpose()*lambda);
		Fvec += deltaF;

		if(deltaF != deltaF){
			std::cout<<"NANS"<<std::endl;
			exit(0);
		}
		if((g.norm()/g.size()) < tol){
			//convergence
			break;
		}
	}
	return 1;

}