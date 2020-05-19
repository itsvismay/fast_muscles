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
						const Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& ACAP, 
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
						MatrixXd& wL,
						const MatrixXd& wIdL,
						MatrixXd& wQ){

	int MAX_ITERS = 100;
	double tol = 1e-3;
	int tot_ls_its=0;
	VectorXd g = VectorXd::Zero(Fvec.size());
	VectorXd g_n = VectorXd::Zero(Fvec.size());
	VectorXd g_m = VectorXd::Zero(Fvec.size());
	
	VectorXd lambda;

	SparseMatrix<double, Eigen::RowMajor> H(Fvec.size(), Fvec.size());
	SparseMatrix<double, Eigen::RowMajor> H_n(Fvec.size(), Fvec.size());
	SparseMatrix<double, Eigen::RowMajor> H_m(Fvec.size(), Fvec.size());

	Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> Hinv;

	SparseMatrix<double, Eigen::RowMajor> Id(Fvec.size(), Fvec.size());
	Id.setIdentity();
	// std::cout<<Fvec.transpose()<<std::endl;
	igl::Timer timer;
	for(int its = 0; its<MAX_ITERS; its++){
		timer.start();
		exact::stablenh::gradient(g_n, Fvec, T, eY, eP, rest_tet_vols);
		exact::stablenh::hessian(H_n, Fvec, T, eY, eP, rest_tet_vols);
		exact::muscle::gradient(g_m, Fvec, T, rest_tet_vols, Uvec);
		exact::muscle::hessian(H_m, Fvec, T, rest_tet_vols, Uvec);
		timer.stop();
		double time1 = timer.getElapsedTimeInMicroSec();

		timer.start();
		double E1n = exact::stablenh::energy(Fvec, T, eY, eP, rest_tet_vols);
		double E1m = activation*exact::muscle::energy(Fvec, T, rest_tet_vols, Uvec);
		double E1 = E1n + E1m;
		timer.stop();
		double time2 = timer.getElapsedTimeInMicroSec();

		g = g_n + activation*g_m;
		// std::cout<<"E1: "<<E1<<","<<E1n<<","<<E1m<<std::endl;
		// std::cout<<"	grad: "<<g.norm()<<std::endl;
		// std::cout<<"	g_n: "<<g_n.norm()<<std::endl;
		// std::cout<<"	g_m: "<<activation*g_m.norm()<<std::endl;
	
		H = H_n + activation*H_m + 1e-6*Id;;
		Hinv.compute(H);
		
		if(Hinv.info()!=Eigen::Success){
			std::cout<<"H SOLVER FAILED iteration: "<<its<<std::endl;
			std::cout<<Hinv.info()<<std::endl;
			exit(0);
		}

		timer.start();
		exact::woodbury(lambda, Fvec, g, Hinv, H, PF, d, Ai, Vtilde,
						wId9T, wVAi, wHiV, wHiVAi, wC, wPhi, wL, wIdL, wQ);
		timer.stop();
		double time3 = timer.getElapsedTimeInMicroSec();

		timer.start();
		VectorXd Jlambda = Id*lambda;
		VectorXd d1 = Vtilde.transpose()*lambda;
		VectorXd d2 = Ai*d1;
		VectorXd d3 = Vtilde*d2;
		Jlambda -= d3;

		VectorXd deltaF = -1*Hinv.solve(g + PF.transpose()*Jlambda);
		timer.stop();
		double time4 = timer.getElapsedTimeInMicroSec();

		timer.start();
		double alpha = exact::linesearch(tot_ls_its, Fvec, g, deltaF, activation, q, T, eY, eP, rest_tet_vols, Uvec, ACAP, Y, B, PF, c, bone_tets);
		timer.stop();
		double time5 = timer.getElapsedTimeInMicroSec();

		// std::cout<<"times: "<<time1<<", "<<time2<<", "<<time3<<", "<<time4<<", "<<time5<<std::endl;
		
		// igl::writeDMAT("dF.dmat", deltaF);
		
		// //KKT 
		// VectorXd rhs(g.size()+d.size());
		// rhs<<-g, (d - *PF*Fvec);
		// MatrixXd fH = MatrixXd(H);
		// MatrixXd JPF = *PF;
		// MatrixXd KKT = MatrixXd::Zero(H.rows() + JPF.rows(), H.cols()+JPF.rows());
		// KKT.block(0,0, H.rows(), H.cols()) = fH;
		// KKT.block(H.rows(), 0, JPF.rows(), JPF.cols()) = JPF;
		// KKT.block(0, H.cols(), JPF.cols(), JPF.rows()) = JPF.transpose();
		// FullPivLU<MatrixXd> KKTinv(KKT);

		// VectorXd res = KKTinv.solve(rhs);
		// VectorXd deltaF = res.head(Fvec.size());
		

		Fvec += alpha*deltaF;

		
		double E2n = exact::stablenh::energy(Fvec, T, eY, eP, rest_tet_vols);
		double E2m = activation*exact::muscle::energy(Fvec, T, rest_tet_vols, Uvec);
		double E2 = E2n + E2m;
		// std::cout<<"E2: "<<E2<<","<<E2n<<","<<E2m<<std::endl;

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