#ifndef ACAP_SOLVE_ENERGY_GRADIENT
#define ACAP_SOLVE_ENERGY_GRADIENT

#include "store.h"
#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include<Eigen/LU>


using namespace famu;
using namespace std;
using Store = famu::Store;
using namespace Eigen;

namespace famu
{
	namespace acap
	{

		double energy(Store& store, VectorXd& dFvec, VectorXd& boneDOFS);

		double fastEnergy(Store& store, Eigen::VectorXd& dFvec);

		void fastGradient(Store& store, Eigen::VectorXd& grad);

		void fastHessian(Store& store, Eigen::SparseMatrix<double, Eigen::RowMajor>& hess, Eigen::MatrixXd& denseHess);

		Eigen::VectorXd fd_gradient(Store& store);

		Eigen::MatrixXd fd_hessian(Store& store);

		MatrixXd fd_dxdF(Store& store);

		void adjointMethodExternalForces(Store& store);

		void solve(Store& store, Eigen::VectorXd& dFvec);

		void setJacobian(Store& store);

	}
}

#endif