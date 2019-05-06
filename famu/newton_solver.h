#ifndef NEWTON_SOLVER
#define NEWTON_SOLVER
#include "store.h"
#include <Eigen/UmfPackSupport>
#include <igl/polar_dec.h>

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

	int newton_static_solve(Store& store){
		VectorXd muscle_grad, neo_grad, acap_grad;
		muscle_grad.resize(store.ProjectF.cols());
		neo_grad.resize(store.ProjectF.cols());
		acap_grad.resize(store.ProjectF.cols());

		SparseMatrix<double> muscleHess(store.ProjectF.cols(), store.ProjectF.cols());
		famu::muscle::fastHessian(store, muscleHess);
		SparseMatrix<double> neoHess(store.ProjectF.cols(), store.ProjectF.cols());
		famu::stablenh::hessian(store, neoHess);
		SparseMatrix<double> acapHess(store.ProjectF.cols(), store.ProjectF.cols());
		famu::acap::fastHessian(store, acapHess);

		//check with FD
		// MatrixXd fakeMuscleHess = famu::muscle::fd_hessian(store);
		// cout<<fakeMuscleHess.rows()<<", "<<fakeMuscleHess.cols()<<endl;
		// MatrixXd realMuscleHess = MatrixXd(muscleHess);
		// cout<<"check1: "<<endl;
		// cout<<fakeMuscleHess.block<10,10>(0,0)<<endl;
		// cout<<realMuscleHess.block<10,10>(0,0)<<endl;


		// MatrixXd fakeNeoHess = famu::stablenh::fd_hessian(store);
		// cout<<fakeNeoHess.rows()<<", "<<fakeNeoHess.cols()<<endl;
		// MatrixXd realNeoHess = MatrixXd(neoHess);
		// cout<<"check2: "<<endl;
		// cout<<fakeNeoHess.block<10,10>(0,0)<<endl;
		// cout<<realNeoHess.block<10,10>(0,0)<<endl;
		

		// MatrixXd fakeACAPHess = famu::acap::fd_hessian(store);
		// MatrixXd realACAPHess =  MatrixXd(acapHess);
		// cout<<fakeACAPHess.rows()<<", "<<fakeACAPHess.cols()<<endl;
		// cout<<"check3: "<<endl;
		// cout<<fakeACAPHess.block<10,10>(0,0)<<endl;
		// cout<<realACAPHess.block<10,10>(0,0)<<endl;

		int sum = 0;
		for(int b=0; b<store.bone_tets.size(); b++){
			sum += store.bone_tets[b].size();
		}

		int MAX_ITERS = 20;
		UmfPackLU<SparseMatrix<double>> SPLU;
		VectorXd graddFvec = VectorXd::Zero(store.dFvec.size());
		SparseMatrix<double> hessFvec = neoHess + acapHess + muscleHess;
		SPLU.analyzePattern(hessFvec);
		int iter =0;
		for(iter=0; iter<MAX_ITERS; iter++){
			hessFvec.setZero();
			graddFvec.setZero();
			famu::acap::solve(store);

			graddFvec.setZero();
			famu::muscle::gradient(store, muscle_grad);
			famu::stablenh::gradient(store, neo_grad);
			famu::acap::fastGradient(store, acap_grad);
			cout<<"muscle grad: "<<muscle_grad.norm()<<endl;
			cout<<"neo grad: "<<neo_grad.norm()<<endl;
			cout<<"acap grad: "<<acap_grad.norm()<<endl;
			graddFvec = muscle_grad + neo_grad + acap_grad;
			cout<<"tot grad: "<<graddFvec.norm()<<endl;


			famu::stablenh::hessian(store, neoHess);
			hessFvec = neoHess + acapHess + muscleHess;


			// SPLU.compute(acapHess);
			SPLU.factorize(hessFvec);
			VectorXd delta_dFvec = -0.25*SPLU.solve(graddFvec);

			if(delta_dFvec != delta_dFvec){
				cout<<"nans"<<endl;
				exit(0);
			}


			store.dFvec += delta_dFvec;
			//project bones dF back to rotations
			if(store.jinput["reduced"]){
				for(int t =0; t < store.T.rows(); t++){
					if(store.bone_or_muscle[t] < store.bone_tets.size()){
						int b = store.bone_or_muscle[t];
						Eigen::Matrix3d _r, _t;
						Matrix3d dFb = Map<Matrix3d>(store.dFvec.segment<9>(9*b).data()).transpose();
						igl::polar_dec(dFb, _r, _t);

						store.dFvec[9*b+0] = _r(0,0);
			      		store.dFvec[9*b+1] = _r(0,1);
			      		store.dFvec[9*b+2] = _r(0,2);
			      		store.dFvec[9*b+3] = _r(1,0);
			      		store.dFvec[9*b+4] = _r(1,1);
			      		store.dFvec[9*b+5] = _r(1,2);
			      		store.dFvec[9*b+6] = _r(2,0);
			      		store.dFvec[9*b+7] = _r(2,1);
			      		store.dFvec[9*b+8] = _r(2,2);
					}
				}

			}else{
				for(int t = 0; t < store.bone_tets.size(); t++){
					for(int i=0; i<store.bone_tets[t].size(); i++){
						int b =store.bone_tets[t][i];

						Eigen::Matrix3d _r, _t;
						Matrix3d dFb = Map<Matrix3d>(store.dFvec.segment<9>(9*b).data()).transpose();
						igl::polar_dec(dFb, _r, _t);

						store.dFvec[9*b+0] = _r(0,0);
			      		store.dFvec[9*b+1] = _r(0,1);
			      		store.dFvec[9*b+2] = _r(0,2);
			      		store.dFvec[9*b+3] = _r(1,0);
			      		store.dFvec[9*b+4] = _r(1,1);
			      		store.dFvec[9*b+5] = _r(1,2);
			      		store.dFvec[9*b+6] = _r(2,0);
			      		store.dFvec[9*b+7] = _r(2,1);
			      		store.dFvec[9*b+8] = _r(2,2);
					}
				}
			}

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