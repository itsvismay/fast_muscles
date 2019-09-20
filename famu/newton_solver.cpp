#include "newton_solver.h"

#include "store.h"
#include "acap_solve_energy_gradient.h"
#include "muscle_energy_gradient.h"
#include "stablenh_energy_gradient.h"
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <igl/Timer.h>
#include <igl/polar_dec.h>
#include <unsupported/Eigen/MatrixFunctions>


using Store = famu::Store;
using namespace Eigen;
using namespace std;

double famu::Energy(Store& store){
	double EM = famu::muscle::energy(store, store.dFvec);
	double ENH = famu::stablenh::energy(store, store.dFvec);
	double EACAP = famu::acap::energy(store, store.dFvec, store.boneDOFS);

	return EM + ENH + EACAP;
}

void famu::update_dofs(Store& store, VectorXd& new_dofs){
		for(int b =0; b < store.bone_tets.size(); b++){
			Matrix3d R0 = Map<Matrix3d>(store.dFvec.segment<9>(9*b).data()).transpose();
			double wX = new_dofs(3*b + 0);
			double wY = new_dofs(3*b + 1);
			double wZ = new_dofs(3*b + 2);
			Matrix3d cross;
	        cross<<0, -wZ, wY,
	                wZ, 0, -wX,
	                -wY, wX, 0;
	        Matrix3d Rot = cross.exp();
			Matrix3d R = R0*Rot;

			store.dFvec[9*b+0] = R(0,0);
      		store.dFvec[9*b+1] = R(0,1);
      		store.dFvec[9*b+2] = R(0,2);
      		store.dFvec[9*b+3] = R(1,0);
      		store.dFvec[9*b+4] = R(1,1);
      		store.dFvec[9*b+5] = R(1,2);
      		store.dFvec[9*b+6] = R(2,0);
      		store.dFvec[9*b+7] = R(2,1);
      		store.dFvec[9*b+8] = R(2,2);

      		new_dofs(3*b + 0) = 0;
			new_dofs(3*b + 1) = 0;
			new_dofs(3*b + 2) = 0;
		
		}
		store.dFvec.tail(new_dofs.size() - 3*store.bone_tets.size()) = new_dofs.tail(new_dofs.size() - 3*store.bone_tets.size());
}

double famu::line_search(int& tot_ls_its, Store& store, VectorXd& grad, VectorXd& drt, VectorXd& new_dofs){
	// Decreasing and increasing factors
	VectorXd x = new_dofs;
	VectorXd xp = x;
	double step = 50;
    const double dec = 0.5;
    const double inc = 2.1;
    int pmax_linesearch = 100;
    int plinesearch = 1;//1 for armijo, 2 for wolfe
    double pftol = 1e-4;
    double pwolfe = 0.9;
    double pmax_step = 1e8;
    double pmin_step = 1e-20;


    // Check the value of step
    if(step <= double(0))
        std::invalid_argument("'step' must be positive");

    update_dofs(store, x);
    famu::acap::solve(store, store.dFvec);
   	double fx = Energy(store);
    // Save the function value at the current x
    const double fx_init = fx;
    // Projection of gradient on the search direction
    const double dg_init = grad.dot(drt);
    // Make sure d points to a descent direction
    if(dg_init > 0)
        std::logic_error("the moving direction increases the objective function value");

    const double dg_test = pftol * dg_init;
    double width;


    int iter;
    for(iter = 0; iter < pmax_linesearch; iter++)
    {
        // x_{k+1} = x_k + step * d_k
        x.noalias() = xp + step * drt;
        update_dofs(store, x);

        // Evaluate this candidate
        famu::acap::solve(store, x);
       	fx = Energy(store);

        if(fx > fx_init + step * dg_test)
        {
            width = dec;
        } else {
            // Armijo condition is met
            if(plinesearch == 1)
                break;

            const double dg = grad.dot(drt);
            if(dg < pwolfe * dg_init)
            {
                width = inc;
            } else {
                // Regular Wolfe condition is met
                if(plinesearch == 2)
                    break;

                if(dg > -pwolfe * dg_init)
                {
                    width = dec;
                } else {
                    // Strong Wolfe condition is met
                    break;
                }
            }
        }

        if(iter >= pmax_linesearch)
            throw std::runtime_error("the line search routine reached the maximum number of iterations");

        if(step < pmin_step)
            throw std::runtime_error("the line search step became smaller than the minimum value allowed");

        if(step > pmax_step)
            throw std::runtime_error("the line search step became larger than the maximum value allowed");

        step *= width;
    }
    // cout<<"			ls iters: "<<iter<<endl;
    // cout<<"			step: "<<step<<endl;
    tot_ls_its += iter;
    return step;
}

void famu::sparse_to_dense(const Store& store, SparseMatrix<double, Eigen::RowMajor>& H, MatrixXd& denseHess){
	//Fill denseHess with 9x9 block diags from H
	//TODO: this should be done in the hessians code. coeffRef is expensive
	//FIX AFTER THE DEADLINE

	#pragma omp parallel for
	for(int i=0; i<store.dFvec.size()/9; i++){
		//loop through 9x9 block and fill denseH
		Matrix9d A;
		#pragma omp parallel for collapse(2)
		for(int j =0; j<9; j++){
			for(int k=0; k<9; k++){
				A(j, k) = H.coeffRef(9*i + j, 9*i +k);
			}
		}
		denseHess.block<9,9>(9*i, 0) = A;
	}
}

// void famu::fastWoodbury(Store& store, const VectorXd& g, MatrixModesxModes X, VectorXd& BInvXDy, MatrixXd& denseHess, VectorXd& drt){
// 	//Woodbury parallel approach 1 (with reduction)

// 	Matrix<double, NUM_MODES, 1> DAg = Matrix<double, NUM_MODES, 1>::Zero(); 
// 	Matrix<double, NUM_MODES, 1> InvXDAg;
// 	FullPivLU<MatrixModesxModes> WoodburyDenseSolve;

// 	double aa = store.jinput["alpha_arap"];
// 	X = store.InvC;
// 	#pragma omp parallel
// 	{
// 		MatrixModesxModes Xpriv = MatrixModesxModes::Zero();
// 		Matrix<double, NUM_MODES, 1> DAgpriv = Matrix<double, NUM_MODES, 1>::Zero(); 


// 		#pragma omp for
// 		for(int i=store.bone_tets.size(); i<store.dFvec.size()/9; i++){
// 			Matrix9d A = denseHess.block<9,9>(3*store.bone_tets.size() + 9*i, 0);
// 			LDLT<Matrix9d> InvA;
// 			InvA.compute(A);
// 			store.vecInvA[i] = InvA;

// 			Vector9d invAg = InvA.solve(g.segment<9>(9*i));
// 			drt.segment<9>(9*i) = invAg;

// 			Matrix9xModes B = store.dRdW_WoodB.block<9, NUM_MODES>(3*store.bone_tets.size() + 9*i, 0);
// 			Matrix9xModes Dt = store.dRdW_WoodD.block<NUM_MODES, 9>(0, 9*i+3*store.bone_tets.size()).transpose();
// 			Xpriv = Xpriv + -Dt.transpose()*InvA.solve(B);

// 			DAgpriv  = DAgpriv + -Dt.transpose()*invAg;
// 		}
// 		#pragma omp critical
// 		{
// 			X += Xpriv;
// 			DAg += DAgpriv;
// 		}
// 	}

// 	#pragma omp single
// 	{

// 		WoodburyDenseSolve.compute(X);
// 		InvXDAg = WoodburyDenseSolve.solve(DAg);

// 	}


// 	#pragma omp parallel for
// 	for(int i=store.bone_tets.size(); i<store.dFvec.size()/9; i++){
// 		Matrix9xModes B = WoodB.block<9, NUM_MODES>(9*i, 0);

// 		Vector9d InvAtemp1 = store.vecInvA[i].solve(B*InvXDAg);
// 		drt.segment<9>(9*i) -=  InvAtemp1;
// 	}

// 	drt *= -1;
// }

int famu::newton_static_solve(Store& store){
	int MAX_ITERS = store.jinput["NM_MAX_ITERS"];
	int dFvec_size = store.dFvec.size();
	int numDOFS = store.dFvec.size() - 6*store.bone_tets.size();

	VectorXd muscle_grad, neo_grad, acap_grad;
	muscle_grad.resize(dFvec_size);
	neo_grad.resize(dFvec_size);
	acap_grad.resize(dFvec_size);
	
	famu::acap::updatedRdW(store);
	//Rigid body hessians
	SparseMatrix<double, Eigen::RowMajor> bNH = store.dRdW0*store.neoHess*store.dRdW0.transpose();
	SparseMatrix<double, Eigen::RowMajor> bMH = store.dRdW0*store.muscleHess*store.dRdW0.transpose();
	SparseMatrix<double, Eigen::RowMajor> bAH = store.dRdW*store.acapHess*store.dRdW.transpose();
	
	SparseMatrix<double, Eigen::RowMajor> constHess = bNH + bMH + bAH;
	constHess -= bNH;

	// MatrixXd denseHess = MatrixXd::Zero(numDOFS,  9);
	// MatrixXd constDenseHess = MatrixXd::Zero(numDOFS,  9);
	// sparse_to_dense(store, constHess, constDenseHess);

	VectorXd delta_dofs = VectorXd::Zero(numDOFS);
	VectorXd dofs = VectorXd::Zero(numDOFS);
	dofs.tail(numDOFS - 3*store.bone_tets.size()) = store.dFvec.tail(numDOFS - 3*store.bone_tets.size());

	VectorXd graddofs = VectorXd::Zero(numDOFS);
	
	VectorXd BInvXDy = VectorXd::Zero(numDOFS);
	MatrixModesxModes X;
		
	igl::Timer timer, timer1;
	double woodtimes =0;
	double linetimes =0;
	int tot_ls_its = 0;
	int iter =1;
	timer1.start();
	for(iter=1; iter<MAX_ITERS; iter++){
		graddofs.setZero();
		double prevfx = Energy(store);
		famu::acap::solve(store, store.dFvec);

		famu::muscle::gradient(store, muscle_grad);
		famu::stablenh::gradient(store, neo_grad);
		famu::acap::fastGradient(store, acap_grad);

		graddofs = store.dRdW*acap_grad + store.dRdW0*(muscle_grad + neo_grad);

		cout<<"		dFvec: "<<store.dFvec.norm()<<endl;
		cout<<"		muscle grad: "<<muscle_grad.norm()<<endl;
		cout<<"		neo grad: "<<neo_grad.norm()<<endl;
		cout<<"		acap grad: "<<acap_grad.norm()<<endl;
		cout<<"		total grad: "<<graddofs.norm()<<endl<<endl;
		
		if(graddofs != graddofs){
			cout<<"Error: nans in grad"<<endl;
			exit(0);
		}

		
		famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, false);
		bNH = store.dRdW0*store.neoHess*store.dRdW0.transpose();
		if(true){ //TestNotWoodbury
			
			SparseMatrix<double, Eigen::RowMajor> hessFvec = bNH + constHess;
			
			store.NM_SPLU.factorize(hessFvec);
			if(store.NM_SPLU.info()!=Success){
				cout<<"SOLVER FAILED"<<endl;
				cout<<store.NM_SPLU.info()<<endl;
			}
			delta_dofs = -.1*store.NM_SPLU.solve(graddofs);
		}
		
	// 	}else{

	// 		// //Sparse Woodbury code
	// 		// hessFvec.setZero();
	// 		// hessFvec = store.neoHess + constHess;
	// 		// store.NM_SPLU.factorize(hessFvec);
	// 		// if(store.NM_SPLU.info()!=Success){
	// 		// 	cout<<"SOLVER FAILED"<<endl;
	// 		// 	cout<<store.NM_SPLU.info()<<endl;
	// 		// }
	// 		// VectorXd InvAg = store.NM_SPLU.solve(graddFvec);
	// 		// MatrixXd CDAB = store.InvC + store.WoodD*store.NM_SPLU.solve(store.WoodB);
	// 		// FullPivLU<MatrixXd>  WoodburyDenseSolve;
	// 		// WoodburyDenseSolve.compute(CDAB);
	// 		// VectorXd temp1 = store.WoodB*WoodburyDenseSolve.solve(store.WoodD*InvAg);;

	// 		// VectorXd InvAtemp1 = store.NM_SPLU.solve(temp1);
	// 		// test_drt =  -InvAg + InvAtemp1;

	// 		//Dense Woodbury code
	// 		denseHess = constDenseHess + store.denseNeoHess;
	// 		timer.start();
	// 		fastWoodbury(store, graddFvec, X, BInvXDy, denseHess, delta_dFvec);
	// 		timer.stop();
	// 		woodtimes += timer.getElapsedTimeInMicroSec();
	// 		// cout<<"		woodbury diff: "<<(delta_dFvec - test_drt).norm()<<endl;

	// 	}

		if(delta_dofs != delta_dofs){
			cout<<"Error: nans"<<endl;
			exit(0);
		}
		
		double alpha =1;
	// 	//line search
	// 	timer.start();
	// 	double alpha = line_search(tot_ls_its, store, graddFvec, delta_dFvec);
	// 	timer.stop();
	// 	linetimes += timer.getElapsedTimeInMicroSec();
		dofs += delta_dofs;

		if(fabs(alpha)<1e-9 ){
			break;
		}

		update_dofs(store, dofs);
		famu::acap::updatedRdW(store);

		double fx = Energy(store);
		

		if(graddofs.squaredNorm()/graddofs.size()<1e-2 || fabs(fx - prevfx)<1e-3){
			break;
		}
	}
	famu::acap::solve(store, store.dFvec);
	timer1.stop();
	double nmtime = timer1.getElapsedTimeInMicroSec();

	cout<<"-----------QS STEP INFO----------"<<endl;
	cout<<"V, T:"<<store.V.rows()<<", "<<store.T.rows()<<endl;
	cout<<"Threads: "<<Eigen::nbThreads()<<endl;
	cout<<"NM Iters: "<<iter<<endl;
	cout<<"Total NM time: "<<nmtime<<endl;
	cout<<"Total Hess time: "<<woodtimes<<endl;
	cout<<"Total LS time: "<<linetimes<<endl;
	cout<<"LS iters: "<<tot_ls_its<<endl;
	cout<<"Bone rotations"<<endl;
	for(int b =0; b<store.bone_tets.size(); b++){
		Matrix3d R = Map<Matrix3d>(store.dFvec.segment<9>(9*b).data()).transpose();
		Eigen::Matrix3d _r, _t;
		igl::polar_dec(R, _r, _t);	
		cout<<(R - _r)<<endl<<endl;
	}
	// cout<<"Energy: "<<acap_energy<<endl;
	// cout<<"Energy Time: "<<energy_time<<endl;
	// cout<<"ACAP time: "<<timer1.getElapsedTimeInMicroSec()<<endl;
	// cout<<"dFvec: "<<store.dFvec.transpose()<<endl;
	cout<<"--------------------------------"<<endl;
    return 0;
}