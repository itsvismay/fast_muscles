#ifndef NEWTON_SOLVER
#define NEWTON_SOLVER
#include "store.h"
#include <Eigen/UmfPackSupport>
#include <igl/polar_dec.h>
#include <Eigen/LU>

using Store = famu::Store;
using namespace Eigen;
using namespace std;

namespace famu
{
	double Energy(Store& store){

		double EM = famu::muscle::energy(store, store.dFvec);
		double ENH = famu::stablenh::energy(store, store.dFvec);
		double EACAP = famu::acap::fastEnergy(store, store.dFvec);

		return EM + ENH + EACAP;
	}

	void fdHess(Store& store, SparseMatrix<double>& fake){
		fake.setZero();
		VectorXd dFvec = store.dFvec;
		double eps = 1e-3;
		double E0 = Energy(store);
		for(int i=0; i<dFvec.size(); i++){
			for(int j=0; j<dFvec.size(); j++){
				dFvec[i] += eps;
				dFvec[j] += eps;
				double Eij = Energy(store);
				dFvec[i] -= eps;
				dFvec[j] -= eps;

				dFvec[i] += eps;
				double Ei = Energy(store);
				dFvec[i] -=eps;

				dFvec[j] += eps;
				double Ej = Energy(store);
				dFvec[j] -=eps;
				
				fake.coeffRef(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
			}
		}
	}

	void polar_dec(Store& store, VectorXd& dFvec){
		if(store.jinput["polar_dec"]){
					//project bones dF back to rotations
					if(store.jinput["reduced"]){
						for(int t =0; t < store.T.rows(); t++){
							if(store.bone_or_muscle[t] < store.bone_tets.size()){
								int b = store.bone_or_muscle[t];
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

	double line_search(Store& store, VectorXd& grad, VectorXd& drt){
		// Decreasing and increasing factors
		VectorXd x = store.dFvec;
		VectorXd xp = x;
		double step = 1;
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
        double EM = famu::muscle::energy(store, x);
		double ENH = famu::stablenh::energy(store, x);
		double EACAP = famu::acap::fastEnergy(store, x);
        double fx = EM + ENH + EACAP;//f(x, grad, k, iter);
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
            double EM = famu::muscle::energy(store, x);
			double ENH = famu::stablenh::energy(store, x);
			double EACAP = famu::acap::fastEnergy(store, x);
            fx = EM + ENH + EACAP;//f(x, grad, k, iter);

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
        cout<<"		ls iters: "<<iter<<endl;
        cout<<"		step: "<<step<<endl;
        return step;
	}

	int newton_static_solve(Store& store){
		int MAX_ITERS = store.jinput["NM_MAX_ITERS"];
		
		VectorXd muscle_grad, neo_grad, acap_grad;
		muscle_grad.resize(store.dFvec.size());
		neo_grad.resize(store.dFvec.size());
		acap_grad.resize(store.dFvec.size());
		SparseMatrix<double> hessFvec(store.dFvec.size(), store.dFvec.size());


		

		VectorXd graddFvec = VectorXd::Zero(store.dFvec.size());
		FullPivLU<MatrixXd>  WoodburyDenseSolve;
		
		int iter =0;
		for(iter=0; iter<MAX_ITERS; iter++){
			hessFvec.setZero();
			graddFvec.setZero();
			famu::acap::solve(store, store.dFvec);

			graddFvec.setZero();
			famu::muscle::gradient(store, muscle_grad);
			famu::stablenh::gradient(store, neo_grad);
			famu::acap::fastGradient(store, acap_grad);

			cout<<"muscle grad: "<<muscle_grad.norm()<<endl;
			cout<<"neo grad: "<<neo_grad.norm()<<endl;
			cout<<"acap grad: "<<acap_grad.norm()<<endl;
			graddFvec = muscle_grad + neo_grad + acap_grad;
			cout<<"tot grad: "<<graddFvec.norm()<<endl;
			
			if(graddFvec != graddFvec){
				cout<<"Error: nans in grad"<<endl;
				exit(0);
			}

			famu::stablenh::hessian(store, store.neoHess);
			VectorXd delta_dFvec;

			if(!store.jinput["woodbury"]){
				
				hessFvec = store.neoHess + store.muscleHess + store.acapHess;
				store.NM_SPLU.factorize(hessFvec);
				if(store.NM_SPLU.info()!=Success){
					cout<<"SOLVER FAILED"<<endl;
					cout<<store.NM_SPLU.info()<<endl;
				}
				cout<<"4"<<endl;
				delta_dFvec = -1*store.NM_SPLU.solve(graddFvec);
			
			}else{

				SparseMatrix<double> A = store.neoHess + store.muscleHess + store.x0tStDt_dF_dF_DSx0;
				store.NM_SPLU.factorize(A);
				if(store.NM_SPLU.info()!=Success){
					cout<<"SOLVER FAILED"<<endl;
					cout<<store.NM_SPLU.info()<<endl;
				}
				VectorXd InvAg = store.NM_SPLU.solve(graddFvec);

				MatrixXd CDAB = store.InvC + store.WoodD*store.NM_SPLU.solve(store.WoodB);
				WoodburyDenseSolve.compute(CDAB);
				VectorXd temp = WoodburyDenseSolve.solve(store.WoodD*InvAg);
				VectorXd temp1 = store.WoodB*temp;

				VectorXd InvAtemp1 = store.NM_SPLU.solve(temp1);
				delta_dFvec = -InvAg + InvAtemp1;

				// SparseMatrix<double> A = neoHess + muscleHess + store.x0tStDt_dF_dF_DSx0;
				// SparseMatrix<double> BCD = (store.WoodB*store.WoodC*store.WoodD).sparseView();
				// store.NM_SPLU.compute((A + BCD));
				// if(store.NM_SPLU.info()!=Success){
				// 	cout<<"SOLVER FAILED"<<endl;
				// 	cout<<store.NM_SPLU.info()<<endl;
				// }
				// cout<<"4"<<endl;
				// delta_dFvec = -1*store.NM_SPLU.solve(graddFvec);

			}

			// store.NM_SPLU.compute(acapHess);
			
			cout<<"5"<<endl;
			if(delta_dFvec != delta_dFvec){
				cout<<"Error: nans"<<endl;
				exit(0);
			}
			cout<<"6"<<endl;

			
			//line search
			double alpha = line_search(store, graddFvec, delta_dFvec);
			store.dFvec += alpha*delta_dFvec;
			polar_dec(store, store.dFvec);
			

			if(graddFvec.squaredNorm()/graddFvec.size()<1e-4){
				break;
			}
			std::cout<<std::endl;
		}

		
		// if(iter== MAX_ITERS){
  //           cout<<"ERROR: Newton max reached"<<endl;
  //           cout<<iter<<endl;
  //           exit(0);
  //       }
		// famu::acap::solve(store);
        return iter;
	}
}
#endif