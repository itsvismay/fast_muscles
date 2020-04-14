#ifndef BFGS_SOLVER
#define BFGS_SOLVER

#include "store.h"
#include "acap_solve_energy_gradient.h"
#include "muscle_energy_gradient.h"
#include "stablenh_energy_gradient.h"
#include <igl/polar_dec.h>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <igl/writeOBJ.h>
#include <igl/Timer.h>
#include "vertex_bc.h"
#include <igl/writeDMAT.h>
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
			VectorXd delta_dFvec = store->RemFixedBones*VectorXd::Zero(store->dFvec.size());
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

		void polar_dec(Store& store, VectorXd& dFvec){
			if(store.jinput["polar_dec"]){
				//project bones dF back to rotations
				if(store.jinput["reduced"]){
					for(int b =store.fix_bones.size(); b < store.bone_tets.size(); b++){
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

		double operator()(const VectorXd& dFvec, VectorXd& graddFvec, int k=0, int v=0)
		{
			store->dFvec = dFvec;

			polar_dec(*store, store->dFvec);

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
				graddFvec = store->RemFixedBones*(muscle_grad + neo_grad + acap_grad - store->ContactForce);
				
				
				

				cout<<"---BFGS Info---"<<endl;
				cout<<"	Total Grad N: "<<graddFvec.norm()<<endl;
			}
			cout<<"		ACAP Energy: "<<EACAP<<endl;
			cout<<"		ENH Energy: "<<ENH<<endl;
			cout<<"	EM Energy: "<<EM<<endl;
			return E;
		}
	};
}


#endif