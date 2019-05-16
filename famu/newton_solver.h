#ifndef NEWTON_SOLVER
#define NEWTON_SOLVER
#include "store.h"
#include <Eigen/UmfPackSupport>
#include <igl/polar_dec.h>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <igl/Timer.h>
#define NUM_MODES 48


using Store = famu::Store;
using namespace Eigen;
using namespace std;
typedef Matrix<double, 9, 1> Vector9d;
typedef Matrix<double, 9, 9> Matrix9d;
typedef Matrix<double, 9, NUM_MODES> Matrix9xModes;
typedef Matrix<double, NUM_MODES, NUM_MODES> MatrixModesxModes;

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

	double line_search(Store& store, VectorXd& grad, VectorXd& drt){
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

        //Linesearch optimization. (x here is the cont. mesh positions)
        //F = F0 + step*dF
        //Ax = BF + c = BF0 + step*BdF + c
        //x = InvA*B*F0 + step*InvA*B*dF + c
        //x = InvA * rhs1 + step* InvA*rhs2 + c
        //x* = const1 + step* const2 + const3
        // xp.setZero();
        // drt.setZero();
        // VectorXd rhs1 = VectorXd::Zero(store.acap_solve_rhs.size());
        // 	rhs1.head(store.x.size()) = store.YtStDt_dF_DSx0*xp;
        // 	rhs1.tail(store.lambda2.size()) = store.Bf*xp;
        // VectorXd rhs2 = VectorXd::Zero(store.acap_solve_rhs.size());
        // 	rhs2.head(store.x.size()) = store.YtStDt_dF_DSx0*drt;
        // 	rhs2.tail(store.lambda2.size()) = store.Bf*drt;
        // VectorXd const3 = VectorXd::Zero(store.acap_solve_rhs.size());
        // 	const3.head(store.x.size()) = -store.x0tStDtDSY;
        // 	const3.tail(store.lambda2.size()) = -store.BfI0;
        // VectorXd result = VectorXd::Zero(store.acap_solve_rhs.size());
        // VectorXd const1 = store.ACAP_KKT_SPLU.solve(rhs1);
        // VectorXd const2 = store.ACAP_KKT_SPLU.solve(rhs2);
        // cout<<const1.norm()<<endl;

        int iter;
        for(iter = 0; iter < pmax_linesearch; iter++)
        {
            // x_{k+1} = x_k + step * d_k
            x.noalias() = xp + step * drt;
            polar_dec(store, x);

            // Evaluate this candidate
            famu::acap::solve(store, x);
            // cout<<store.lambda2.transpose()<<endl;
            // result = const1 + step*const2 + const3;
            // store.x = result.head(store.x.size());
            // store.lambda2 = result.tail(store.lambda2.size());
            // cout<<store.lambda2.transpose()<<endl;
            // exit(0);
            //
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
        cout<<"			ls iters: "<<iter<<endl;
        cout<<"			step: "<<step<<endl;
        return step;
	}

	void fastWoodbury(const Store& store, SparseMatrix<double>& H, const VectorXd& g, MatrixModesxModes X, VectorXd& BInvXDy, MatrixXd& denseHess, VectorXd& drt){
		//Fill denseHess with 9x9 block diags from H
		//TODO: this should be done in the hessians code. coeffRef is expensive
		// #pragma omp par
		for(int i=0; i<store.dFvec.size()/9; i++){
			//loop through 9x9 block and fill denseH
			Matrix9d A;
			for(int j =0; j<9; j++){
				for(int k=0; k<9; k++){
					A(j, k) = H.coeffRef(9*i + j, 9*i +k);
				}
			}
			// Eigen::SelfAdjointEigenSolver<Matrix9d> es(A);
			// Matrix9d DiagEval = es.eigenvalues().real().asDiagonal();
	  //       Matrix9d Evec = es.eigenvectors().real();
			// JacobiSVD<MatrixXd> svd(A);
			// double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
	        
	  //       for (int i = 0; i < 9; ++i) {
	  //           if (es.eigenvalues()[i]<1e-6) {
	  //           	// std::cout<<"small eig"<<", ";
	  //               DiagEval(i,i) = 1e-3;
	  //           }
	  //       }
	  //       A = Evec * DiagEval * Evec.transpose();

			// FullPivLU<Matrix9d> InvA;
			// InvA.compute(A);
			// cout<<cond<<", "<<(A - A.transpose()).norm()<<", "<<es.eigenvalues().transpose()<<endl;
			LDLT<Matrix9d> InvA;
			InvA.compute(A);
			if(InvA.info()!=Success){
				cout<<"woops LDLT failed"<<endl;
			}

			denseHess.block<9,9>(9*i, 0) = A;
		}


		igl::Timer timer;		
		// #pragma omp declare reduction (merge : MatrixModesxModes : omp_out += omp_in)
		// #pragma omp parallel for reduction(merge: X)
		X = store.InvC;
		Matrix9xModes B;
		for(int i=0; i<store.dFvec.size()/9; i++){
			Matrix9d A = denseHess.block<9,9>(9*i, 0);
			LDLT<Matrix9d> InvA;
			InvA.compute(A);
			drt.segment<9>(9*i) = InvA.solve(g.segment<9>(9*i));;

			B = store.WoodB.block(9*i, 0, 9, store.G.cols());
			X += -B.transpose()*InvA.solve(B);
		}
		FullPivLU<MatrixModesxModes> WoodburyDenseSolve;
		WoodburyDenseSolve.compute(X);
		
		BInvXDy = store.WoodB*WoodburyDenseSolve.solve(store.WoodD*drt);
		for(int i=0; i<store.dFvec.size()/9; i++){
			Matrix9d A = denseHess.block<9,9>(9*i, 0);
			LDLT<Matrix9d> InvA;
			InvA.compute(A);

			Vector9d InvAtemp1 = InvA.solve(BInvXDy.segment<9>(9*i));
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

		SparseMatrix<double> hessFvec(store.dFvec.size(), store.dFvec.size());
		MatrixXd denseHess = MatrixXd::Zero(store.dFvec.size(),  9);
		VectorXd delta_dFvec = VectorXd::Zero(store.dFvec.size());
		VectorXd test_drt = delta_dFvec;
		VectorXd graddFvec = VectorXd::Zero(store.dFvec.size());
		
		VectorXd BInvXDy = VectorXd::Zero(store.dFvec.size());
		MatrixModesxModes X;
			
		igl::Timer timer;
		double woodtimes =0;
		double linetimes =0;
		int iter =1;
		for(iter=1; iter<MAX_ITERS; iter++){
			hessFvec.setZero();
			graddFvec.setZero();

			famu::acap::solve(store, store.dFvec);
			famu::muscle::gradient(store, muscle_grad);
			famu::stablenh::gradient(store, neo_grad);
			famu::acap::fastGradient(store, acap_grad);

			graddFvec = muscle_grad + neo_grad + acap_grad;
			cout<<"		muscle grad: "<<muscle_grad.norm()<<endl;
			cout<<"		neo grad: "<<neo_grad.norm()<<endl;
			cout<<"		acap grad: "<<acap_grad.norm()<<endl;
			cout<<"		total grad: "<<graddFvec.norm()<<endl;
			
			if(graddFvec != graddFvec){
				cout<<"Error: nans in grad"<<endl;
				exit(0);
			}

			famu::stablenh::hessian(store, store.neoHess, store.denseNeoHess);
			hessFvec = store.neoHess + store.muscleHess + store.acapHess;
			


			if(!store.jinput["woodbury"]){
				
				store.NM_SPLU.factorize(hessFvec);
				if(store.NM_SPLU.info()!=Success){
					cout<<"SOLVER FAILED"<<endl;
					cout<<store.NM_SPLU.info()<<endl;
				}
				delta_dFvec = -1*store.NM_SPLU.solve(graddFvec);
			
			}else{

				// //Sparse Woodbury code
				//store.NM_SPLU.factorize(hessFvec);
				//if(store.NM_SPLU.info()!=Success){
				//	cout<<"SOLVER FAILED"<<endl;
				//	cout<<store.NM_SPLU.info()<<endl;
				//}
				//VectorXd InvAg = store.NM_SPLU.solve(graddFvec);
				//MatrixXd CDAB = store.InvC + store.WoodD*store.NM_SPLU.solve(store.WoodB);
				//FullPivLU<MatrixXd>  WoodburyDenseSolve;
				//WoodburyDenseSolve.compute(CDAB);
				//VectorXd temp1 = store.WoodB*WoodburyDenseSolve.solve(store.WoodD*InvAg);;

				//VectorXd InvAtemp1 = store.NM_SPLU.solve(temp1);
				//test_drt =  -InvAg + InvAtemp1;

				//Dense Woodbury code
				timer.start();
				fastWoodbury(store, hessFvec, graddFvec, X, BInvXDy, denseHess, delta_dFvec);
				timer.stop();
				woodtimes += timer.getElapsedTimeInMicroSec();
				cout<<"		woodbury diff: "<<(delta_dFvec - test_drt).norm()<<endl;


				//Naive dense woodbury test
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

			if(delta_dFvec != delta_dFvec){
				cout<<"Error: nans"<<endl;
				exit(0);
			}
			
			//line search
			timer.start();
			double alpha = line_search(store, graddFvec, delta_dFvec);
			timer.stop();
			linetimes += timer.getElapsedTimeInMicroSec();
			
			if(fabs(alpha)<1e-9){
				break;
			}

			store.dFvec += alpha*delta_dFvec;
			polar_dec(store, store.dFvec);

			if(graddFvec.squaredNorm()/graddFvec.size()<1e-4){
				break;
			}
		}

		cout<<"Woodbury per NM iter: "<<woodtimes/iter<<endl;
		cout<<"Linesearch per NM iter: "<<linetimes/iter<<endl;


        return iter;
	}
}
#endif
