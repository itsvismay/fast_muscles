#ifndef NEWTON_SOLVER
#define NEWTON_SOLVER
#include "store.h"
#include <Eigen/UmfPackSupport>
#include <igl/polar_dec.h>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <igl/Timer.h>

using Store = famu::Store;
using namespace Eigen;
using namespace std;


namespace famu
{
	double Energy(Store& store, VectorXd& dFvec){
		double EM = famu::muscle::energy(store, dFvec);
		double ENH = famu::stablenh::energy(store, dFvec);
		double EACAP = famu::acap::fastEnergy(store, dFvec);

		return EM + ENH + EACAP;
	}

	void polar_dec(Store& store, VectorXd& dFvec){
		if(store.jinput["polar_dec"]){
			//project bones dF back to rotations
			if(store.jinput["reduced"]){
				for(int b =0; b < store.bone_tets.size(); b++){
					Eigen::Matrix3d _r, _t;
					Matrix3d dFb = Map<Matrix3d>(dFvec.segment<9>(9*b).data()).transpose();
					igl::polar_dec(dFb, _r, _t);

					dFvec[9*b+0] = _r(0,0);
		      		dFvec[9*b+1] = _r(0,1);
		      		dFvec[9*b+2] = _r(0,2);
		      		dFvec[9*b+3] = _r(1,0);
		      		dFvec[9*b+4] = _r(1,1);
		      		dFvec[9*b+5] = _r(1,2);
		      		dFvec[9*b+6] = _r(2,0);
		      		dFvec[9*b+7] = _r(2,1);
		      		dFvec[9*b+8] = _r(2,2);
				
				}

			}else{
				for(int t = 0; t < store.bone_tets.size(); t++){
					for(int i=0; i<store.bone_tets[t].size(); i++){
						int b =store.bone_tets[t][i];

						Eigen::Matrix3d _r, _t;
						Matrix3d dFb = Map<Matrix3d>(dFvec.segment<9>(9*b).data()).transpose();
						igl::polar_dec(dFb, _r, _t);

						dFvec[9*b+0] = _r(0,0);
			      		dFvec[9*b+1] = _r(0,1);
			      		dFvec[9*b+2] = _r(0,2);
			      		dFvec[9*b+3] = _r(1,0);
			      		dFvec[9*b+4] = _r(1,1);
			      		dFvec[9*b+5] = _r(1,2);
			      		dFvec[9*b+6] = _r(2,0);
			      		dFvec[9*b+7] = _r(2,1);
			      		dFvec[9*b+8] = _r(2,2);
					}
				}
			}
		}
	}

	double line_search(int& tot_ls_its, Store& store, VectorXd& grad, VectorXd& drt){
		// Decreasing and increasing factors
		VectorXd x = store.dFvec;
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

        polar_dec(store, x);
        famu::acap::solve(store, x);
       	double fx = Energy(store, x);
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
            polar_dec(store, x);

            // Evaluate this candidate
            famu::acap::solve(store, x);
           	fx = Energy(store, x);

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

	void sparse_to_dense(const Store& store, SparseMatrix<double, Eigen::RowMajor>& H, MatrixXd& denseHess){
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

	void fastWoodbury(Store& store, const VectorXd& g, MatrixModesxModes X, VectorXd& BInvXDy, MatrixXd& denseHess, VectorXd& drt){
		//Woodbury parallel approach 1 (with reduction)

		Matrix<double, NUM_MODES, 1> DAg = Matrix<double, NUM_MODES, 1>::Zero(); 
		Matrix<double, NUM_MODES, 1> InvXDAg;
		FullPivLU<MatrixModesxModes> WoodburyDenseSolve;

		X = store.InvC;
		#pragma omp parallel
		{
			MatrixModesxModes Xpriv = MatrixModesxModes::Zero();
			Matrix<double, NUM_MODES, 1> DAgpriv = Matrix<double, NUM_MODES, 1>::Zero(); 

			#pragma omp for
			for(int i=0; i<store.dFvec.size()/9; i++){
				Matrix9d A = denseHess.block<9,9>(9*i, 0);
				LDLT<Matrix9d> InvA;
				InvA.compute(A);
				store.vecInvA[i] = InvA;

				Vector9d invAg = InvA.solve(g.segment<9>(9*i));
				drt.segment<9>(9*i) = invAg;

				Matrix9xModes B = store.WoodB.block<9, NUM_MODES>(9*i, 0);
				Xpriv = Xpriv + -B.transpose()*InvA.solve(B);

				DAgpriv  = DAgpriv + -B.transpose()*invAg;
			}
			#pragma omp critical
			{
				X += Xpriv;
				DAg += DAgpriv;
			}
		}

		#pragma omp single
		{

			WoodburyDenseSolve.compute(X);
			InvXDAg = WoodburyDenseSolve.solve(DAg);

		}

		#pragma omp parallel for
		for(int i=0; i<store.dFvec.size()/9; i++){
			Matrix9xModes B = store.WoodB.block<9, NUM_MODES>(9*i, 0);

			Vector9d InvAtemp1 = store.vecInvA[i].solve(B*InvXDAg);
			drt.segment<9>(9*i) -=  InvAtemp1;
		}

		drt *= -1;
	}


	int newton_static_solve(Store& store){
		int MAX_ITERS = store.jinput["NM_MAX_ITERS"];
		VectorXd muscle_grad, neo_grad, acap_grad;
		muscle_grad.resize(store.dFvec.size());
		neo_grad.resize(store.dFvec.size());
		acap_grad.resize(store.dFvec.size());
		SparseMatrix<double, Eigen::RowMajor> hessFvec(store.dFvec.size(), store.dFvec.size());
		SparseMatrix<double, Eigen::RowMajor> constHess(store.dFvec.size(), store.dFvec.size());
		constHess.setZero();



		constHess = store.neoHess + store.muscleHess + store.acapHess;
		constHess -= store.neoHess;

		MatrixXd denseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		MatrixXd constDenseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		sparse_to_dense(store, constHess, constDenseHess);

		VectorXd delta_dFvec = VectorXd::Zero(store.dFvec.size());
		VectorXd test_drt = delta_dFvec;
		VectorXd graddFvec = VectorXd::Zero(store.dFvec.size());
		
		VectorXd BInvXDy = VectorXd::Zero(store.dFvec.size());
		MatrixModesxModes X;
			
		igl::Timer timer, timer1;
		double woodtimes =0;
		double linetimes =0;
		int tot_ls_its = 0;
		int iter =1;
		timer1.start();

		double E00 = famu::stablenh::energy(store, store.dFvec);
		cout<<"INITIAL: "<<E00<<endl;
		for(iter=1; iter<MAX_ITERS; iter++){
			graddFvec.setZero();
			double prevfx = Energy(store, store.dFvec);

			famu::acap::solve(store, store.dFvec);
			famu::muscle::gradient(store, muscle_grad);
			famu::stablenh::gradient(store, neo_grad);
			famu::acap::fastGradient(store, acap_grad);
			graddFvec = muscle_grad + neo_grad + acap_grad;

			// cout<<"		muscle grad: "<<muscle_grad.norm()<<endl;
			// cout<<"		neo grad: "<<neo_grad.norm()<<endl;
			// cout<<"		acap grad: "<<acap_grad.norm()<<endl;
			// cout<<"		total grad: "<<graddFvec.norm()<<endl;
			
			if(graddFvec != graddFvec){
				cout<<"Error: nans in grad"<<endl;
				exit(0);
			}

			
			famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess, store.jinput["woodbury"]);

			if(!store.jinput["woodbury"]){
				
				hessFvec.setZero();
				hessFvec = store.neoHess + constHess;
				store.NM_SPLU.factorize(hessFvec);
				if(store.NM_SPLU.info()!=Success){
					cout<<"SOLVER FAILED"<<endl;
					cout<<store.NM_SPLU.info()<<endl;
				}
				delta_dFvec = -1*store.NM_SPLU.solve(graddFvec);
			
			}else{

				// //Sparse Woodbury code
				// hessFvec.setZero();
				// hessFvec = store.neoHess + constHess;
				// store.NM_SPLU.factorize(hessFvec);
				// if(store.NM_SPLU.info()!=Success){
				// 	cout<<"SOLVER FAILED"<<endl;
				// 	cout<<store.NM_SPLU.info()<<endl;
				// }
				// VectorXd InvAg = store.NM_SPLU.solve(graddFvec);
				// MatrixXd CDAB = store.InvC + store.WoodD*store.NM_SPLU.solve(store.WoodB);
				// FullPivLU<MatrixXd>  WoodburyDenseSolve;
				// WoodburyDenseSolve.compute(CDAB);
				// VectorXd temp1 = store.WoodB*WoodburyDenseSolve.solve(store.WoodD*InvAg);;

				// VectorXd InvAtemp1 = store.NM_SPLU.solve(temp1);
				// test_drt =  -InvAg + InvAtemp1;

				//Dense Woodbury code
				denseHess = constDenseHess + store.denseNeoHess;
				timer.start();
				fastWoodbury(store, graddFvec, X, BInvXDy, denseHess, delta_dFvec);
				timer.stop();
				woodtimes += timer.getElapsedTimeInMicroSec();
				// cout<<"		woodbury diff: "<<(delta_dFvec - test_drt).norm()<<endl;

			}

			if(delta_dFvec != delta_dFvec){
				cout<<"Error: nans"<<endl;
				exit(0);
			}
			
			//line search
			timer.start();
			double alpha = line_search(tot_ls_its, store, graddFvec, delta_dFvec);
			timer.stop();
			linetimes += timer.getElapsedTimeInMicroSec();
			

			if(fabs(alpha)<1e-9 ){
				break;
			}

			store.dFvec += alpha*delta_dFvec;
			polar_dec(store, store.dFvec);
			double fx = Energy(store, store.dFvec);

			cout<<fx - E00<<"\n";

			if(graddFvec.squaredNorm()/graddFvec.size()<1e-4 || fabs(fx - prevfx)<1e-4){
				break;
			}
		}
		timer1.stop();
		double nmtime = timer1.getElapsedTimeInMicroSec();

		timer1.start();
		double acap_energy = famu::acap::fastEnergy(store, store.dFvec);
		timer1.stop();
		double energy_time = timer1.getElapsedTimeInMicroSec();

		timer1.start();
		famu::acap::solve(store, store.dFvec);
		timer1.stop();

		cout<<"-----------QS STEP INFO----------"<<endl;
		cout<<"V, T:"<<store.V.rows()<<", "<<store.T.rows()<<endl;
		cout<<"Threads: "<<Eigen::nbThreads()<<endl;
		cout<<"NM Iters: "<<iter<<endl;
		cout<<"Total NM time: "<<nmtime<<endl;
		cout<<"Total Hess time: "<<woodtimes<<endl;
		cout<<"Total LS time: "<<linetimes<<endl;
		cout<<"LS iters: "<<tot_ls_its<<endl;
		cout<<"Energy: "<<acap_energy<<endl;
		cout<<"Energy Time: "<<energy_time<<endl;
		cout<<"ACAP time: "<<timer1.getElapsedTimeInMicroSec()<<endl;
		cout<<"--------------------------------"<<endl;
        return iter;
	}
}
#endif