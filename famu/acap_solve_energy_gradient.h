#ifndef ACAP_SOLVE_ENERGY_GRADIENT
#define ACAP_SOLVE_ENERGY_GRADIENT

#include "store.h"

using namespace famu;
using Store = famu::Store;

namespace famu
{
	namespace acap
	{
		double energy(Store& store){
			SparseMatrix<double> DS = store.D*store.S;
			double E1 =  0.5*(store.D*store.S*(store.x+store.x0) - store.dF*store.DSx0).squaredNorm();

			double E2 = 0.5*store.x.transpose()*store.StDtDS*store.x;
			double E3 = store.x0.transpose()*store.StDtDS*store.x;
			double E4 = 0.5*store.x0.transpose()*store.StDtDS*store.x0;
			double E5 = -store.x.transpose()*DS.transpose()*store.dF*DS*store.x0;
			double E6 = -store.x0.transpose()*DS.transpose()*store.dF*DS*store.x0;
			double E7 = 0.5*(store.dF*store.DSx0).transpose()*(store.dF*store.DSx0);
			double E8 = E2+E3+E4+E5+E6+E7;
			assert(fabs(E1 - E8)< 1e-6);
			return E1;
		}

		double fastEnergy(Store& store){

			double E2 = 0.5*store.x.transpose()*store.StDtDS*store.x;
			double E3 = store.x0.transpose()*store.StDtDS*store.x;
			double E4 = 0.5*store.x0.transpose()*store.StDtDS*store.x0;
			double E5 = -store.x.transpose()*store.StDt_dF_DSx0*store.dFvec;
			double E6 = -store.x0tStDt_dF_DSx0.dot(store.dFvec);
			double E7 = 0.5*store.dFvec.transpose()*store.x0tStDt_dF_dF_DSx0*store.dFvec;
			double E9 = E2+E3+E4+E5+E6+E7;
			return E9;
		}

		void fastGradient(Store& store, VectorXd& grad){
			grad += -store.x.transpose()*store.StDt_dF_DSx0;
			grad += -store.x0tStDt_dF_DSx0;
			grad += store.dFvec.transpose()*store.x0tStDt_dF_dF_DSx0;
		}

		VectorXd fd_gradient(Store& store){
			VectorXd fake = VectorXd::Zero(store.dFvec.size());
			double eps = 0.00001;
			for(int i=0; i<store.dFvec.size(); i++){
				store.dFvec[i] += 0.5*eps;
				// setDF(store.dFvec, store.dF);
				double Eleft = energy(store);
				store.dFvec[i] -= 0.5*eps;

				store.dFvec[i] -= 0.5*eps;
				// setDF(store.dFvec, store.dF);
				double Eright = energy(store);
				store.dFvec[i] += 0.5*eps;
				fake[i] = (Eleft - Eright)/eps;
			}
			return fake;
		}

		void solve(Store& store){
			VectorXd constrains = store.ConstrainProjection.transpose()*store.dx;
			VectorXd top = store.StDt_dF_DSx0*store.dFvec - store.StDtDS*store.x0;
			VectorXd KKT_right(top.size() + constrains.size());
			KKT_right<<top, constrains;

			VectorXd result = store.SPLU.solve(KKT_right);
			store.x = result.head(top.size());
		}
	}
}

#endif