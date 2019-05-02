#ifndef ACAP_SOLVE_ENERGY_GRADIENT
#define ACAP_SOLVE_ENERGY_GRADIENT

#include "store.h"
#include <igl/writeDMAT.h>

using namespace famu;
using namespace std;
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
			return store.alpha_arap*E1;
		}

		double fastEnergy(Store& store){
			double E1 = 0.5*store.x0tStDtDSx0;
			double E2 = store.x0tStDtDSY.dot(store.x);
			double E3 = 0.5*store.x.transpose()*store.YtStDtDSY*store.x;
			double E4 = -store.x0tStDt_dF_DSx0.dot(store.dFvec);
			double E5 = -store.x.transpose()*store.YtStDt_dF_DSx0*store.dFvec;
			double E6 = 0.5*store.dFvec.transpose()*store.x0tStDt_dF_dF_DSx0*store.dFvec;
		 
			double E9 = E1+E2+E3+E4+E5+E6;
			return store.alpha_arap*E9;
		}

		void fastGradient(Store& store, VectorXd& grad){
			grad += -store.alpha_arap*store.x0tStDt_dF_DSx0;
			grad += -store.alpha_arap*store.x.transpose()*store.YtStDt_dF_DSx0;
			grad += store.alpha_arap*store.dFvec.transpose()*store.x0tStDt_dF_dF_DSx0;
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

			VectorXd zer = VectorXd::Zero(store.JointConstraints.rows());
			// VectorXd constrains = store.ConstrainProjection.transpose()*store.dx;
			VectorXd top = store.YtStDt_dF_DSx0*store.dFvec - store.x0tStDtDSY;
			VectorXd KKT_right(top.size() + zer.size());
			KKT_right<<top, zer;
			// igl::writeDMAT("rhs.dmat", KKT_right);

			VectorXd result = store.SPLU.solve(KKT_right);
			store.x = result.head(top.size());
		
		}
	}
}

#endif