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

int exact::newton_solve(const Store& store, 
						VectorXd& Fvec, 
						VectorXi& bone_or_muscle, 
						SparseMatrix<double, Eigen::RowMajor>& PF, 
						VectorXd& d, 
						MatrixXd& Ai, 
						MatrixXd& Vtilde, 
						double activation){

	int MAX_ITERS = 1;
	double tol = 1e-3;

	VectorXd g = VectorXd::Zero(Fvec.size());
	VectorXd g_n = VectorXd::Zero(Fvec.size());
	VectorXd g_m = VectorXd::Zero(Fvec.size());
	
	VectorXd lambda;

	SparseMatrix<double, Eigen::RowMajor> H(Fvec.size(), Fvec.size());
	SparseMatrix<double, Eigen::RowMajor> H_n(Fvec.size(), Fvec.size());
	SparseMatrix<double, Eigen::RowMajor> H_m(Fvec.size(), Fvec.size());

	Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> Hinv, BigHinv;

	SparseMatrix<double, Eigen::RowMajor> Id9T(9*store.T.rows(), 9*store.T.rows());
	Id9T.setIdentity();
	SparseMatrix<double, Eigen::RowMajor> Id(Fvec.size(), Fvec.size());
	Id.setIdentity();
	// std::cout<<Fvec.transpose()<<std::endl;

	for(int its = 0; its<MAX_ITERS; its++){
		exact::stablenh::hessian(store, Fvec, H_n);
		exact::muscle::hessian(store, Fvec, H_m);
		exact::stablenh::gradient(store, Fvec, g_n);
		exact::muscle::gradient(store, Fvec, g_m);

		double E1n = exact::stablenh::energy(store, Fvec);
		double E1m = activation*exact::muscle::energy(store, Fvec);
		double E1 = E1n + E1m;


		VectorXd g = g_n + activation*g_m;
		std::cout<<"E1: "<<E1<<","<<E1n<<","<<E1m<<std::endl;
		std::cout<<"	grad: "<<g.norm()<<std::endl;
		std::cout<<"	g_n: "<<g_n.norm()<<std::endl;
		std::cout<<"	g_m: "<<activation*g_m.norm()<<std::endl;
	
		H = H_n + activation*H_m + 1e-6*Id;;
		SparseMatrix<double, RowMajor> BigH = PF*H*PF.transpose();
		BigH += 1e-6*Id9T;

		BigHinv.compute(BigH);
		Hinv.compute(H);
		
		if(Hinv.info()!=Eigen::Success){
			std::cout<<"H SOLVER FAILED iteration: "<<its<<std::endl;
			std::cout<<Hinv.info()<<std::endl;
			exit(0);
		}
		if(BigHinv.info()!=Eigen::Success){
			std::cout<<"Big H SOLVER FAILED iteration: "<<its<<std::endl;
			std::cout<<Hinv.info()<<std::endl;
			exit(0);
		}

		exact::woodbury(store, lambda, PF, Fvec, g, d, Hinv, H, Ai, Vtilde);
		// std::cout<<"lambda:"<<lambda.size()<<std::endl;
		// std::cout<<"Fvec:"<<Fvec.size()<<std::endl;
		// std::cout<<"Vtilde:"<<Vtilde.rows()<<","<<Vtilde.cols()<<std::endl;
		// std::cout<<"Vtilde:"<<Vtilde.rows()<<","<<Vtilde.cols()<<std::endl;
		VectorXd Jlambda = Id9T*lambda;
		VectorXd d1 = Vtilde.transpose()*lambda;
		VectorXd d2 = Ai*d1;
		VectorXd d3 = Vtilde*d2;
		Jlambda -= d3;

		VectorXd deltaF = -1*Hinv.solve(g + PF.transpose()*Jlambda);

		// igl::writeDMAT("dF.dmat", deltaF);
		
		// //KKT 
		// VectorXd rhs(g.size()+d.size());
		// rhs<<-g, (d - store.J*PF*Fvec);
		// MatrixXd fH = MatrixXd(H);
		// MatrixXd JPF = store.J*PF;
		// MatrixXd KKT = MatrixXd::Zero(H.rows() + JPF.rows(), H.cols()+JPF.rows());
		// KKT.block(0,0, H.rows(), H.cols()) = fH;
		// KKT.block(H.rows(), 0, JPF.rows(), JPF.cols()) = JPF;
		// KKT.block(0, H.cols(), JPF.cols(), JPF.rows()) = JPF.transpose();
		// FullPivLU<MatrixXd> KKTinv(KKT);

		// VectorXd res = KKTinv.solve(rhs);
		// VectorXd deltaF = res.head(Fvec.size());
		

		Fvec += deltaF;

		
		double E2n = exact::stablenh::energy(store, Fvec);
		double E2m = activation*exact::muscle::energy(store, Fvec);
		double E2 = E2n + E2m;
		std::cout<<"E2: "<<E2<<","<<E2n<<","<<E2m<<std::endl;

		std::cout<<"delta E: "<<fabs(E2- E1)<<std::endl;

		if(deltaF != deltaF){
			std::cout<<"NANS"<<std::endl;
			exit(0);
		}
		if(fabs(E2- E1) < tol){
			//convergence
			break;
		}
	}
	return 1;

}