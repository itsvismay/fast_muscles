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
#include "linesearch.h"

using namespace Eigen;
using Store = exact::Store;

int exact::newton_solve(VectorXd& Fvec, 
						VectorXd& q,
						const double tol,
						const MatrixXi& T,
						const VectorXd& eY,
						const VectorXd& eP,
						const MatrixXd& Uvec,
						const VectorXd& rest_tet_vols,
						const VectorXi& bone_or_muscle, 
						const SparseMatrix<double, Eigen::RowMajor>& PF, 
						const VectorXd& d, 
						const MatrixXd& Ai, 
						const MatrixXd& Vtilde, 
						const double activation,
						const Eigen::SparseMatrix<double, Eigen::RowMajor>& Y, 
						const Eigen::SparseMatrix<double, Eigen::RowMajor>& B, 
						const VectorXd& c,
						const std::vector<Eigen::VectorXi>& bone_tets,
						const SparseMatrix<double, Eigen::RowMajor>& wId9T,
						const MatrixXd& wVAi,
						MatrixXd& wHiV,
						MatrixXd& wHiVAi,
						MatrixXd& wC,
						MatrixXd& wPhi,
						MatrixXd& wHPhi,
						MatrixXd& wL,
						const MatrixXd& wIdL,
						MatrixXd& wQ,
						const exact::Store& store){

	int MAX_ITERS = 1000;
	int tot_ls_its=0;
	VectorXd g = VectorXd::Zero(Fvec.size());
	VectorXd g_n = VectorXd::Zero(Fvec.size());
	VectorXd g_m = VectorXd::Zero(Fvec.size());
	VectorXd deltaF = VectorXd::Zero(Fvec.size());
	
	VectorXd lambda;

	SparseMatrix<double, Eigen::RowMajor> H(Fvec.size(), Fvec.size());
	SparseMatrix<double, Eigen::RowMajor> H_n(Fvec.size(), Fvec.size());
	SparseMatrix<double, Eigen::RowMajor> H_m(Fvec.size(), Fvec.size());

	SparseMatrix<double, Eigen::RowMajor> Id(Fvec.size(), Fvec.size());
	Id.setIdentity();
	// std::cout<<Fvec.transpose()<<std::endl;
	MatrixXd denseH = MatrixXd::Zero(Fvec.size(), 9);
	MatrixXd denseHi = MatrixXd::Zero(Fvec.size(), 9);

	exact::muscle::hessian(H_m, Fvec, T, rest_tet_vols, Uvec);
	for(int its = 0; its<MAX_ITERS; its++){
		exact::stablenh::gradient(g_n, Fvec, T, eY, eP, rest_tet_vols);
		exact::muscle::gradient(g_m, Fvec, T, rest_tet_vols, Uvec);
		exact::stablenh::hessian(H_n, Fvec, T, eY, eP, rest_tet_vols);

		double E1n = exact::stablenh::energy(Fvec, T, eY, eP, rest_tet_vols);
		double E1m = activation*exact::muscle::energy(Fvec, T, rest_tet_vols, Uvec);
		double E1 = E1n + E1m;

		g = g_n + activation*g_m;

		H = H_n + activation*H_m + 1e-6*Id;
		
		Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> Hinv;
		Hinv.compute(H);
			
		if(Hinv.info()!=Eigen::Success){
			std::cout<<"H SOLVER FAILED iteration: "<<std::endl;
			std::cout<<Hinv.info()<<std::endl;
			exit(0);
		}

		exact::sparse_to_dense(denseH, denseHi, H);
		std::cout<<"Inputs"<<std::endl;
		std::cout<<"	Fvec: "<<lambda.norm()<<std::endl;
		std::cout<<"	g: "<<g.norm()<<", "<<g_n.norm()<<", "<<g_m.norm()<<std::endl;
		exact::woodbury(lambda, Fvec, g, denseHi, denseH, H, Hinv, PF, d, Ai, Vtilde,
						wId9T, wVAi, wHiV, wHiVAi, wC, wPhi, wHPhi, wL, wIdL, wQ);
		

		VectorXd d1 = Vtilde.transpose()*lambda;
		VectorXd d2 = Ai*d1;
		VectorXd d3 = Vtilde*d2;
		VectorXd Jlambda = lambda - d3;

		VectorXd  rhs = g + Jlambda;
		igl::writeDMAT("/Users/vismay/recode/fast_muscles/data/simple_muscle_tendon/T335/rhs_nm.dmat", rhs);

		for(int i=0; i<denseHi.rows()/9; i++){
			deltaF.segment<9>(9*i) = -denseHi.block<9,9>(9*i,0)*rhs.segment<9>(9*i);
		}
		deltaF = -Hinv.solve(g+Jlambda);

		std::cout<<"Outputs"<<std::endl;
		std::cout<<"	lambda: "<<lambda.norm()<<std::endl;
		std::cout<<"	d3: "<<d3.norm()<<std::endl;
		std::cout<<"	Jlambda: "<<Jlambda.norm()<<std::endl;
		std::cout<<"	g: "<<g.norm()<<std::endl;
		std::cout<<"	deltaF: "<<deltaF.norm()<<std::endl;
		double alpha = exact::linesearch(tot_ls_its, Fvec, g, deltaF, activation, q, T, eY, eP, rest_tet_vols, Uvec, Y, B, PF, c, bone_tets, store);
		
		// //KKT 
		VectorXd rhs2(g.size()+d.size());
		rhs2<<-g, (d - store.J*Fvec);
		MatrixXd fH = MatrixXd(H);
		MatrixXd KKT = 1e-6*MatrixXd::Identity(H.rows() + store.J.rows(), H.cols() + store.J.rows());
		KKT.block(0,0, H.rows(), H.cols()) += fH;
		KKT.block(H.rows(), 0, store.J.rows(), store.J.cols()) += store.J;
		KKT.block(0, H.cols(), store.J.cols(), store.J.rows()) += store.J.transpose();
		igl::writeDMAT("/Users/vismay/recode/fast_muscles/data/simple_muscle_tendon/T335/Q.dmat", wQ);
		igl::writeDMAT("/Users/vismay/recode/fast_muscles/data/simple_muscle_tendon/T335/H.dmat", fH);
		igl::writeDMAT("/Users/vismay/recode/fast_muscles/data/simple_muscle_tendon/T335/KKT.dmat", KKT);
		igl::writeDMAT("/Users/vismay/recode/fast_muscles/data/simple_muscle_tendon/T335/rhs_kkt.dmat", rhs2);
		PartialPivLU<MatrixXd> KKTinv(KKT);
		VectorXd res = KKTinv.solve(rhs2);
		VectorXd deltaF2 = res.head(Fvec.size());
		igl::writeDMAT("/Users/vismay/recode/fast_muscles/data/simple_muscle_tendon/T335/res_kkt.dmat", res);
		std::cout<<"	JFvec: "<<(store.J*Fvec).norm()<<std::endl;
		std::cout<<"	lambda2: "<<res.tail(lambda.size()).norm()<<std::endl;
		std::cout<<"	deltaF2: "<<deltaF2.norm()<<std::endl;
		exit(0);
		Fvec += alpha*deltaF;
		

		
		double E2n = exact::stablenh::energy(Fvec, T, eY, eP, rest_tet_vols);
		double E2m = activation*exact::muscle::energy(Fvec, T, rest_tet_vols, Uvec);
		double E2 = E2n + E2m;
		std::cout<<std::endl;

		if(deltaF != deltaF){
			std::cout<<"NANS"<<std::endl;
			exit(0);
		}
		if(fabs(E2- E1) < tol){
			std::cout<<"NM converged: "<<its<<std::endl;
			break;
		}
	}
	return 1;

}