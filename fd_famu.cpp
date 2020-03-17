#include <igl/opengl/glfw/Viewer.h>
#include "famu/setup_store.h"
#include "famu/stablenh_energy_gradient.h"
#include "famu/muscle_energy_gradient.h"
#include "famu/acap_solve_energy_gradient.h"
#include "famu/newton_solver.h"


using namespace Eigen;
using namespace std;
using json = nlohmann::json;

using Store = famu::Store;
json j_input;

VectorXd NeohookeanGradient(Store& store, int num_indices){
	Eigen::VectorXd fake = Eigen::VectorXd::Zero(num_indices);
	double eps =1e-6;
	
	for(int i=0; i<fake.size(); i++){
		store.dFvec[i] += 0.5*eps;
		// setDF(store.dFvec, store.dF);
		double Eleft = famu::stablenh::energy(store, store.dFvec);
		store.dFvec[i] -= 0.5*eps;

		store.dFvec[i] -= 0.5*eps;
		// setDF(store.dFvec, store.dF);
		double Eright = famu::stablenh::energy(store, store.dFvec);
		store.dFvec[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	return fake;
}

MatrixXd NeohookeanHessian (Store& store, int num_indices){
	Eigen::MatrixXd fake = Eigen::MatrixXd::Zero(num_indices, num_indices);
	double eps =1e-1;
	VectorXd dFvec = store.dFvec;
	double E0 = famu::stablenh::energy(store, dFvec);
	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			dFvec[i] += eps;
			dFvec[j] += eps;
			double Eij = famu::stablenh::energy(store, dFvec);
			dFvec[i] -= eps;
			dFvec[j] -= eps;

			dFvec[i] += eps;
			dFvec[j] -= eps;
			double Ei_j = famu::stablenh::energy(store, dFvec);
			dFvec[i] -= eps;
			dFvec[j] += eps;

			dFvec[i] -= eps;
			dFvec[j] += eps;
			double E_ij = famu::stablenh::energy(store, dFvec);
			dFvec[i] += eps;
			dFvec[j] -= eps;

			dFvec[i] -= eps;
			dFvec[j] -= eps;
			double E_i_j = famu::stablenh::energy(store, dFvec);
			dFvec[i] += eps;
			dFvec[j] += eps;
			
			fake(i,j) = ((Eij - Ei_j - E_ij + E_i_j)/(4*eps*eps));
		}
	}
	return fake;
}

VectorXd MuscleGradient(Store& store, int num_indices){
	Eigen::VectorXd fake = Eigen::VectorXd::Zero(num_indices);
	double eps =1e-4;
	
	for(int i=0; i<fake.size(); i++){
		store.dFvec[i] += 0.5*eps;
		double Eleft = famu::muscle::energy(store, store.dFvec);
		store.dFvec[i] -= 0.5*eps;

		store.dFvec[i] -= 0.5*eps;
		double Eright = famu::muscle::energy(store, store.dFvec);
		store.dFvec[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	return fake;
}

MatrixXd MuscleHessian (Store& store, int num_indices){
	Eigen::MatrixXd fake = Eigen::MatrixXd::Zero(num_indices, num_indices);
	double eps =1e-1;
	VectorXd dFvec = store.dFvec;
	double E0 = famu::muscle::energy(store, dFvec);
	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			dFvec[i] += eps;
			dFvec[j] += eps;
			double Eij = famu::muscle::energy(store, dFvec);
			dFvec[i] -= eps;
			dFvec[j] -= eps;

			dFvec[i] += eps;
			dFvec[j] -= eps;
			double Ei_j = famu::muscle::energy(store, dFvec);
			dFvec[i] -= eps;
			dFvec[j] += eps;

			dFvec[i] -= eps;
			dFvec[j] += eps;
			double E_ij = famu::muscle::energy(store, dFvec);
			dFvec[i] += eps;
			dFvec[j] -= eps;

			dFvec[i] -= eps;
			dFvec[j] -= eps;
			double E_i_j = famu::muscle::energy(store, dFvec);
			dFvec[i] += eps;
			dFvec[j] += eps;
			
			fake(i,j) = ((Eij - Ei_j - E_ij + E_i_j)/(4*eps*eps));
		}
	}
	return fake;
}

VectorXd ACAPGradient(Store& store, int num_indices){
	Eigen::VectorXd fake = Eigen::VectorXd::Zero(num_indices);
	double eps =1e-4;
	
	for(int i=0; i<fake.size(); i++){
		store.dFvec[i] += 0.5*eps;
		double Eleft = famu::acap::fastEnergy(store, store.dFvec);
		store.dFvec[i] -= 0.5*eps;

		store.dFvec[i] -= 0.5*eps;
		double Eright = famu::acap::fastEnergy(store, store.dFvec);
		store.dFvec[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	return fake;
}

void woodburyCorrectnessTests(Store& store){
	//ACAP solve
 	famu::acap::solve(store, store.dFvec);

 	cout<<"gradients"<<endl;
 	//Get gradients
 	VectorXd muscle_grad, neo_grad, acap_grad;
	muscle_grad.resize(store.dFvec.size());
	neo_grad.resize(store.dFvec.size());
	acap_grad.resize(store.dFvec.size());
	famu::muscle::gradient(store, muscle_grad);
	famu::stablenh::gradient(store, neo_grad);
	famu::acap::fastGradient(store, acap_grad);
	VectorXd graddFvec = muscle_grad + neo_grad + acap_grad;

	VectorXd res1, res2, res3, res4;
 	// First, the unreduced code
 	{
 		famu::acap::setJacobian(store, true);
 		famu::muscle::fastHessian(store, store.muscleHess, store.denseMuscleHess);
 		famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, false);
 		famu::acap::fastHessian(store, store.acapHess, store.denseAcapHess, true);
 		Eigen::SparseMatrix<double> fullHess = store.muscleHess + store.neoHess + store.acapHess;
 		cout<<fullHess.rows()<<", "<<fullHess.cols()<<endl;
 		cout<<fullHess.nonZeros()<<endl;
 		store.NM_SPLU.compute(fullHess);
 		if(store.NM_SPLU.info() == Eigen::NumericalIssue)
 	    {
 	        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
 	    }
 		res1 = -store.NM_SPLU.solve(graddFvec);
 		cout<<res1.segment<50>(0).transpose()<<endl<<endl;
 	}


	//Second, unoptimized woodbury solve
	{
		famu::muscle::fastHessian(store, store.muscleHess, store.denseMuscleHess);
		famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, false);
		famu::acap::fastHessian(store, store.acapHess, store.denseAcapHess, false);
		Eigen::SparseMatrix<double> sparseHess = store.muscleHess + store.neoHess + store.acapHess;
		store.NM_SPLU.compute(sparseHess);
		VectorXd InvAg = store.NM_SPLU.solve(graddFvec);
		MatrixXd CDAB = store.InvC + store.WoodD*store.NM_SPLU.solve(store.WoodB);
		FullPivLU<MatrixXd>  WoodburyDenseSolve;
		WoodburyDenseSolve.compute(CDAB);
		VectorXd temp1 = store.WoodB*WoodburyDenseSolve.solve(store.WoodD*InvAg);;

		VectorXd InvAtemp1 = store.NM_SPLU.solve(temp1);
		res2 =  -InvAg + InvAtemp1;
		cout<<res2.segment<50>(0).transpose()<<endl<<endl;
	}


	// //Third, the optimized woodbury solve
	{
		famu::acap::fastHessian(store, store.acapHess, store.denseAcapHess, false);
		res3 = VectorXd::Zero(graddFvec.size());
		MatrixModesxModes X;
		SparseMatrix<double, Eigen::RowMajor> constHess(store.dFvec.size(), store.dFvec.size());
		constHess.setZero();
		constHess = store.neoHess + store.muscleHess + store.acapHess;// + store.ContactHess;
		constHess -= store.neoHess;
		MatrixXd denseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		MatrixXd constDenseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		famu::sparse_to_dense(store, constHess, denseHess);
		famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, true);
		denseHess += store.denseNeoHess;
		famu::fastWoodbury(store, graddFvec, X, denseHess, res3);
		cout<<res3.segment<50>(0).transpose()<<endl<<endl;
	}

	//Fourth, totally unoptimized woodbury solve
	{

		famu::muscle::fastHessian(store, store.muscleHess, store.denseMuscleHess);
		famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, false);
		famu::acap::fastHessian(store, store.acapHess, store.denseAcapHess, false);
		// Eigen::SparseMatrix<double> A = store.muscleHess + store.neoHess + store.acapHess;
		Eigen::SparseMatrix<double> sparseHess = store.muscleHess + store.neoHess + store.acapHess;
		MatrixXd BCD = store.WoodB*store.WoodC*store.WoodD;
		Eigen::SparseMatrix<double> ABCD = sparseHess + BCD.sparseView();
		store.NM_SPLU.compute(ABCD);
		if(store.NM_SPLU.info() == Eigen::NumericalIssue)
	    {
	        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
	    }
		res4 = -store.NM_SPLU.solve(graddFvec);
		cout<<res4.segment<50>(0).transpose()<<endl<<endl<<endl;
	}

	// igl::writeDMAT("ABCD.dmat", MatrixXd(ABCD),false);
	// igl::writeDMAT("b.dmat", graddFvec, false);
	// igl::writeDMAT("res2.dmat", res2, false);
	// igl::writeDMAT("res4.dmat", res4, false);
	// cout<<store.eigenvalues.transpose()<<endl;

	// cout<<"residuals from optimized woodbury: "<<(res4 - res2).norm()<<endl;
	// cout<<"Residuals: "<<(ABCD*res2 - graddFvec).norm()<<", "<<(ABCD*res4 - graddFvec).norm()<<endl;

}

void woodburyParallelizedTests(Store& store){

	VectorXd res1, res2, res3, res4;
	int num_threads = 1;
	// optimized woodbury solve with 1 core
	{	
		#ifdef __linux__
			num_threads = 1;
			omp_set_num_threads(num_threads);
			Eigen::initParallel();
			store.ACAP_KKT_SPLU.pardisoParameterArray()[2] = num_threads;
		#endif
		//ACAP solve
	 	famu::acap::solve(store, store.dFvec);
	 	//Get gradients
	 	VectorXd muscle_grad, neo_grad, acap_grad;
		muscle_grad.resize(store.dFvec.size());
		neo_grad.resize(store.dFvec.size());
		acap_grad.resize(store.dFvec.size());
		famu::muscle::gradient(store, muscle_grad);
		famu::stablenh::gradient(store, neo_grad);
		famu::acap::fastGradient(store, acap_grad);
		VectorXd graddFvec = muscle_grad + neo_grad + acap_grad;
		///////////////-----------------------------------------
		famu::acap::fastHessian(store, store.acapHess, store.denseAcapHess, false);
		res3 = VectorXd::Zero(graddFvec.size());
		MatrixModesxModes X;
		SparseMatrix<double, Eigen::RowMajor> constHess(store.dFvec.size(), store.dFvec.size());
		constHess.setZero();
		constHess = store.neoHess + store.muscleHess + store.acapHess;// + store.ContactHess;
		constHess -= store.neoHess;
		MatrixXd denseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		MatrixXd constDenseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		famu::sparse_to_dense(store, constHess, denseHess);
		famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, true);
		denseHess += store.denseNeoHess;
		store.timer.start();
		famu::fastWoodbury(store, graddFvec, X, denseHess, res3);
		store.timer.stop();
		double woodTime = store.timer.getElapsedTimeInMicroSec();
		cout<<"Threads: "<<Eigen::nbThreads()<<", Time:"<<woodTime<<endl;
		cout<<res3.segment<50>(0).transpose()<<endl<<endl;
	}

	// Optimized woodbury with 2 cores
	{
		#ifdef __linux__
			num_threads = 2;
			omp_set_num_threads(num_threads);
			Eigen::initParallel();
			store.ACAP_KKT_SPLU.pardisoParameterArray()[2] = num_threads;
		#endif
		//ACAP solve
	 	famu::acap::solve(store, store.dFvec);
	 	//Get gradients
	 	VectorXd muscle_grad, neo_grad, acap_grad;
		muscle_grad.resize(store.dFvec.size());
		neo_grad.resize(store.dFvec.size());
		acap_grad.resize(store.dFvec.size());
		famu::muscle::gradient(store, muscle_grad);
		famu::stablenh::gradient(store, neo_grad);
		famu::acap::fastGradient(store, acap_grad);
		VectorXd graddFvec = muscle_grad + neo_grad + acap_grad;
		///////////////-----------------------------------------
		famu::acap::fastHessian(store, store.acapHess, store.denseAcapHess, false);
		res3 = VectorXd::Zero(graddFvec.size());
		MatrixModesxModes X;
		SparseMatrix<double, Eigen::RowMajor> constHess(store.dFvec.size(), store.dFvec.size());
		constHess.setZero();
		constHess = store.neoHess + store.muscleHess + store.acapHess;// + store.ContactHess;
		constHess -= store.neoHess;
		MatrixXd denseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		MatrixXd constDenseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		famu::sparse_to_dense(store, constHess, denseHess);
		famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, true);
		denseHess += store.denseNeoHess;
		store.timer.start();
		famu::fastWoodbury(store, graddFvec, X, denseHess, res3);
		store.timer.stop();
		double woodTime = store.timer.getElapsedTimeInMicroSec();
		cout<<"Threads: "<<Eigen::nbThreads()<<", Time:"<<woodTime<<endl;
		cout<<res3.segment<50>(0).transpose()<<endl<<endl;
	}

	//Woodbury with 3 cores
	{
		#ifdef __linux__
			num_threads = 3;
			omp_set_num_threads(num_threads);
			Eigen::initParallel();
			store.ACAP_KKT_SPLU.pardisoParameterArray()[2] = num_threads;
		#endif
		//ACAP solve
	 	famu::acap::solve(store, store.dFvec);
	 	//Get gradients
	 	VectorXd muscle_grad, neo_grad, acap_grad;
		muscle_grad.resize(store.dFvec.size());
		neo_grad.resize(store.dFvec.size());
		acap_grad.resize(store.dFvec.size());
		famu::muscle::gradient(store, muscle_grad);
		famu::stablenh::gradient(store, neo_grad);
		famu::acap::fastGradient(store, acap_grad);
		VectorXd graddFvec = muscle_grad + neo_grad + acap_grad;
		///////////////-----------------------------------------
		famu::acap::fastHessian(store, store.acapHess, store.denseAcapHess, false);
		res3 = VectorXd::Zero(graddFvec.size());
		MatrixModesxModes X;
		SparseMatrix<double, Eigen::RowMajor> constHess(store.dFvec.size(), store.dFvec.size());
		constHess.setZero();
		constHess = store.neoHess + store.muscleHess + store.acapHess;// + store.ContactHess;
		constHess -= store.neoHess;
		MatrixXd denseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		MatrixXd constDenseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		famu::sparse_to_dense(store, constHess, denseHess);
		famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, true);
		denseHess += store.denseNeoHess;
		store.timer.start();
		famu::fastWoodbury(store, graddFvec, X, denseHess, res3);
		store.timer.stop();
		double woodTime = store.timer.getElapsedTimeInMicroSec();
		cout<<"Threads: "<<Eigen::nbThreads()<<", Time:"<<woodTime<<endl;
		cout<<res3.segment<50>(0).transpose()<<endl<<endl;
	}

	//Woodbury with 4 cores
	{
		#ifdef __linux__
			num_threads = 4;
			omp_set_num_threads(num_threads);
			Eigen::initParallel();
			store.ACAP_KKT_SPLU.pardisoParameterArray()[2] = num_threads;
		#endif
		//ACAP solve
	 	famu::acap::solve(store, store.dFvec);
	 	//Get gradients
	 	VectorXd muscle_grad, neo_grad, acap_grad;
		muscle_grad.resize(store.dFvec.size());
		neo_grad.resize(store.dFvec.size());
		acap_grad.resize(store.dFvec.size());
		famu::muscle::gradient(store, muscle_grad);
		famu::stablenh::gradient(store, neo_grad);
		famu::acap::fastGradient(store, acap_grad);
		VectorXd graddFvec = muscle_grad + neo_grad + acap_grad;
		///////////////-----------------------------------------
		famu::acap::fastHessian(store, store.acapHess, store.denseAcapHess, false);
		res3 = VectorXd::Zero(graddFvec.size());
		MatrixModesxModes X;
		SparseMatrix<double, Eigen::RowMajor> constHess(store.dFvec.size(), store.dFvec.size());
		constHess.setZero();
		constHess = store.neoHess + store.muscleHess + store.acapHess;// + store.ContactHess;
		constHess -= store.neoHess;
		MatrixXd denseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		MatrixXd constDenseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		famu::sparse_to_dense(store, constHess, denseHess);
		famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, true);
		denseHess += store.denseNeoHess;
		store.timer.start();
		famu::fastWoodbury(store, graddFvec, X, denseHess, res3);
		store.timer.stop();
		double woodTime = store.timer.getElapsedTimeInMicroSec();
		cout<<"Threads: "<<Eigen::nbThreads()<<", Time:"<<woodTime<<endl;
		cout<<res3.segment<50>(0).transpose()<<endl<<endl;
	}


}

int main(int argc, char *argv[])
{
	
  	std::cout<<"-----FINITE DIFFERENCE CHECKS-------"<<std::endl;
		int num_threads = 1;
		int num_indices = 30;
		std::ifstream input_file(argv[1]);


		#ifdef __linux__
		omp_set_num_threads(num_threads);
		Eigen::initParallel();
		#endif
		input_file >> j_input;

		std::cout<<"Threads: "<<Eigen::nbThreads( )<<std::endl;
		std::cout<<"InputMesh: "<<argv[1]<<std::endl;
		std::cout<<"Check up to index: "<<num_indices<<std::endl;

		famu::Store store;
		store.jinput = j_input;
		famu::setupStore(store);

		//store.dFvec[9+0] = 0.7071;
		//store.dFvec[9+1] = 0.7071;
		//store.dFvec[9+2] = 0;
		//store.dFvec[9+3] = -0.7071;
		//store.dFvec[9+4] = 0.7071;
		//store.dFvec[9+5] = 0;
		//store.dFvec[9+6] = 0;
		//store.dFvec[9+7] = 0;
		//store.dFvec[9+8] = 1;
		//timer.start();
		// famu::acap::solve(store, store.dFvec);        	
		// timer.stop();
		// cout<<"+++Microsecs per solve: "<<timer.getElapsedTimeInMicroSec()<<endl;

		
	// std::cout<<"-----FD Check Neohookean Gradient-----"<<std::endl;
	// 	VectorXd neo_grad = VectorXd::Zero(9*store.T.rows());
	// 	famu::stablenh::gradient(store, neo_grad);
	// 	VectorXd neo_grad_segment = neo_grad.head(num_indices);
	// 	VectorXd fake_neo_grad = NeohookeanGradient(store, num_indices);
	// 	// std::cout<<neo_grad_segment.transpose()<<endl;
	// 	// std::cout<<fake_neo_grad.transpose()<<endl;
	// 	std::cout<<"norm: "<<(neo_grad_segment - fake_neo_grad).squaredNorm()<<std::endl;


	// std::cout<<"---FD Check Neohookean Hessian"<<std::endl;
	// 	famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, true);
	// 	MatrixXd neoHess = MatrixXd::Zero(num_indices, num_indices);

	// 	for(int i=0; i<num_indices/9 +  1; i++){
	// 		if(num_indices-9*i <= 9){
	// 			int remainder = num_indices - 9*i;
	// 			neoHess.block(9*i, 9*i, remainder, remainder) = store.denseNeoHess.block(9*i, 0, remainder, remainder);
	// 			break;
	// 		}else{
	// 			neoHess.block<9,9>(9*i, 9*i) = store.denseNeoHess.block<9,9>(9*i, 0);
	// 		}
	// 	}
	// 	MatrixXd fdNeoHess = NeohookeanHessian(store, num_indices);
	// 	cout<<"norm: "<<(neoHess -fdNeoHess).norm()<<endl<<endl;
	// 	// cout<<(neoHess)<<endl<<endl;
	// 	// cout<<(fdNeoHess)<<endl;
		

	// std::cout<<"-----FD Check Muscle Gradient-----"<<std::endl;
	// 	VectorXd muscle = VectorXd::Zero(9*store.T.rows());
	// 	famu::muscle::gradient(store, muscle);
	// 	VectorXd muscle_segment = muscle.head(num_indices);
	// 	VectorXd fake_muscle = MuscleGradient(store, num_indices);
	// 	std::cout<<"norm: "<<(muscle_segment - fake_muscle).squaredNorm()<<std::endl;
	
	// std::cout<<"---FD Check Muscle Hessian"<<std::endl;
	// 	famu::muscle::fastHessian(store, store.muscleHess, store.denseMuscleHess);
	// 	MatrixXd muscle_hess = MatrixXd(store.muscleHess).block(0,0,num_indices, num_indices);
	// 	MatrixXd fd_muscle_hess = MuscleHessian(store, num_indices);
	// 	cout<<"norm: "<<(muscle_hess -fd_muscle_hess).norm()<<endl<<endl;
	// 	// cout<<(neoHess)<<endl<<endl;
	// 	// cout<<(fdNeoHess)<<endl;

	// std::cout<<"-----FD Check ACAP Gradient-----"<<std::endl;
	// 	VectorXd acap = VectorXd::Zero(9*store.T.rows());
	// 	famu::acap::fastGradient(store, acap);
	// 	VectorXd acap_segment = acap.head(num_indices);
	// 	VectorXd fake_acap = ACAPGradient(store, num_indices);
	// 	std::cout<<"norm: "<<(acap_segment - fake_acap).squaredNorm()<<std::endl;

 	std::cout<<"----Check Woodbury Solve----"<<std::endl;
 	woodburyCorrectnessTests(store);
 	// woodburyParallelizedTests(store);
}
