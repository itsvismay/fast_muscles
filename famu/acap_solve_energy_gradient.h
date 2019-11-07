#ifndef ACAP_SOLVE_ENERGY_GRADIENT
#define ACAP_SOLVE_ENERGY_GRADIENT

#include "store.h"
#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include<Eigen/LU>


using namespace famu;
using namespace std;
using Store = famu::Store;

namespace famu
{
	namespace acap
	{

		double energy(Store& store);

		double fastEnergy(Store& store, Eigen::VectorXd& dFvec);

		void fastGradient(Store& store, Eigen::VectorXd& grad);

		void fastHessian(Store& store, Eigen::SparseMatrix<double, Eigen::RowMajor>& hess, Eigen::MatrixXd& denseHess, bool include_dense=false);

		Eigen::VectorXd fd_gradient(Store& store);

		Eigen::MatrixXd fd_hessian(Store& store);

		void adjointMethodExternalForces(Store& store);

		void solve(Store& store, Eigen::VectorXd& dFvec);

		void setJacobian(Store& store, bool include_dense = false);

	}
}

#endif