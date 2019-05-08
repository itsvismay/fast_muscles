#ifndef BFGS_SOLVER
#define BFGS_SOLVER

#include "store.h"
#include <igl/polar_dec.h>
using Store = famu::Store;
using namespace Eigen;
using namespace std;

namespace famu{

	class FullSolver
	{
	private:
		Store* store;
		bool mtest;
		VectorXd muscle_grad, neo_grad, acap_grad;


	public:

		 FullSolver(int n_, Store* istore, bool test = false){

		 	store = istore;
		 	mtest = test;
		 	muscle_grad.resize(store->dFvec.size());
		 	neo_grad.resize(store->dFvec.size());
		 	acap_grad.resize(store->dFvec.size());

		}

		void setDF(VectorXd& dFvec, SparseMatrix<double>& dF){
			dF.setZero();
			std::vector<Trip> dF_trips;
			for(int t=0; t<dFvec.size()/9; t++){
				for(int j=0; j<4; j++){
		            dF_trips.push_back(Trip(3*j+12*t + 0, 3*j+12*t + 0, dFvec(9*t+0)));
		            dF_trips.push_back(Trip(3*j+12*t + 0, 3*j+12*t + 1, dFvec(9*t+1)));
		            dF_trips.push_back(Trip(3*j+12*t + 0, 3*j+12*t + 2, dFvec(9*t+2)));
		            dF_trips.push_back(Trip(3*j+12*t + 1, 3*j+12*t + 0, dFvec(9*t+3)));
		            dF_trips.push_back(Trip(3*j+12*t + 1, 3*j+12*t + 1, dFvec(9*t+4)));
		            dF_trips.push_back(Trip(3*j+12*t + 1, 3*j+12*t + 2, dFvec(9*t+5)));
		            dF_trips.push_back(Trip(3*j+12*t + 2, 3*j+12*t + 0, dFvec(9*t+6)));
		            dF_trips.push_back(Trip(3*j+12*t + 2, 3*j+12*t + 1, dFvec(9*t+7)));
		            dF_trips.push_back(Trip(3*j+12*t + 2, 3*j+12*t + 2, dFvec(9*t+8)));
		        }
			}
			dF.setFromTriplets(dF_trips.begin(), dF_trips.end());
		}

		double Energy(Store& store){

			double EM = famu::muscle::energy(store, store.dFvec);
			double ENH = famu::stablenh::energy(store, store.dFvec);
			double EACAP = famu::acap::fastEnergy(store, store.dFvec);

			return EM + ENH + EACAP;
		}

		VectorXd fdGradEnergy(Store& store){
			VectorXd fake = VectorXd::Zero(store.dFvec.size());
			double eps = 0.00001;
			for(int i=0; i<store.dFvec.size(); i++){
				store.dFvec[i] += 0.5*eps;
				// setDF(store.dFvec, store.dF);
				double Eleft = Energy(store);
				store.dFvec[i] -= 0.5*eps;

				store.dFvec[i] -= 0.5*eps;
				// setDF(store.dFvec, store.dF);
				double Eright = Energy(store);
				store.dFvec[i] += 0.5*eps;
				fake[i] = (Eleft - Eright)/eps;
			}
			return fake;
		}

		double operator()(const VectorXd& dFvec, VectorXd& graddFvec, int k=0, int v=0)
		{
			store->dFvec = dFvec;
			// //project bones dF back to rotations
			// if(store->jinput["reduced"]){
			// 	for(int t =0; t < store->T.rows(); t++){
			// 		if(store->bone_or_muscle[t] < store->bone_tets.size()){
			// 			int b = store->bone_or_muscle[t];
			// 			Eigen::Matrix3d _r, _t;
			// 			Matrix3d dFb = Map<Matrix3d>(store->dFvec.segment<9>(9*b).data()).transpose();
			// 			igl::polar_dec(dFb, _r, _t);

			// 			store->dFvec[9*b+0] = _r(0,0);
			//       		store->dFvec[9*b+1] = _r(0,1);
			//       		store->dFvec[9*b+2] = _r(0,2);
			//       		store->dFvec[9*b+3] = _r(1,0);
			//       		store->dFvec[9*b+4] = _r(1,1);
			//       		store->dFvec[9*b+5] = _r(1,2);
			//       		store->dFvec[9*b+6] = _r(2,0);
			//       		store->dFvec[9*b+7] = _r(2,1);
			//       		store->dFvec[9*b+8] = _r(2,2);
			// 		}
			// 	}

			// }else{
			// 	for(int t = 0; t < store->bone_tets.size(); t++){
			// 		for(int i=0; i<store->bone_tets[t].size(); i++){
			// 			int b =store->bone_tets[t][i];

			// 			Eigen::Matrix3d _r, _t;
			// 			Matrix3d dFb = Map<Matrix3d>(store->dFvec.segment<9>(9*b).data()).transpose();
			// 			igl::polar_dec(dFb, _r, _t);

			// 			store->dFvec[9*b+0] = _r(0,0);
			//       		store->dFvec[9*b+1] = _r(0,1);
			//       		store->dFvec[9*b+2] = _r(0,2);
			//       		store->dFvec[9*b+3] = _r(1,0);
			//       		store->dFvec[9*b+4] = _r(1,1);
			//       		store->dFvec[9*b+5] = _r(1,2);
			//       		store->dFvec[9*b+6] = _r(2,0);
			//       		store->dFvec[9*b+7] = _r(2,1);
			//       		store->dFvec[9*b+8] = _r(2,2);
			// 		}
			// 	}
			// }

			famu::acap::solve(*store, store->dFvec);
			
			double EM = famu::muscle::energy(*store, store->dFvec);
			double ENH = famu::stablenh::energy(*store, store->dFvec);
			double EACAP = famu::acap::fastEnergy(*store, store->dFvec);
			double E = EM + ENH + EACAP;

			if(true){
				graddFvec.setZero();
				famu::muscle::gradient(*store, muscle_grad);
				famu::stablenh::gradient(*store, neo_grad);
				famu::acap::fastGradient(*store, acap_grad);
				graddFvec = muscle_grad + neo_grad + acap_grad;
				if(false){
					setDF(store->dFvec, store->dF);
					VectorXd fakeEMgrad = famu::muscle::fd_gradient(*store);// VectorXd::Zero(graddFvec.size());
					VectorXd fakeENHgrad = famu::stablenh::fd_gradient(*store); //VectorXd::Zero(graddFvec.size());
					VectorXd fakeACAPgrad = famu::acap::fd_gradient(*store);//VectorXd::Zero(graddFvec.size());
					
					VectorXd EMgrad = VectorXd::Zero(graddFvec.size());
					famu::muscle::gradient(*store, EMgrad);

					VectorXd ENHgrad = VectorXd::Zero(graddFvec.size());
					famu::stablenh::gradient(*store, ENHgrad);

					VectorXd EACAPgrad = VectorXd::Zero(graddFvec.size());
					famu::acap::fastGradient(*store, EACAPgrad);

					cout<<"fd M  check: "<<(fakeEMgrad - EMgrad).norm()<<endl;
					cout<<"fd NH check: "<<(fakeENHgrad - ENHgrad).norm()<<endl;
					cout<<"fd ACAP check: "<<(fakeACAPgrad - EACAPgrad).norm()<<endl;

					VectorXd grad = fdGradEnergy(*store);
					cout<<"fakevsfakesum: "<<(grad - fakeENHgrad - fakeEMgrad - fakeACAPgrad).norm()<<endl;
					cout<<"fakevsrealsum: "<<(grad - ENHgrad - EMgrad - EACAPgrad ).norm()<<endl;
				}
				// if(graddFvec.size() != grad.size()){
				// 	cout<<"whats wrong here"<<endl;
				// 	cout<<graddFvec.size()<<endl;
				// 	cout<<grad.size()<<endl;
				// 	cout<<store->dFvec.size()<<endl;
				// 	exit(0);
				// }
				

				cout<<"---BFGS Info---"<<endl;
				cout<<"	Total Grad N: "<<graddFvec.norm()<<endl;
			}
			cout<<"	ACAP Energy: "<<EACAP<<endl;
			cout<<"	ENH Energy: "<<ENH<<endl;
			cout<<"	EM Energy: "<<EM<<endl;
			return E;
		}
	};
}


#endif