#ifndef STABLENH_ENERGY_GRAD
#define STABLENH_ENERGY_GRAD

#include "store.h"

using namespace famu;
using namespace Eigen;
using Store = famu::Store;

namespace famu
{
	namespace stablenh
	{
		double energy(Store& store, VectorXd& dFvec);

		void gradient(Store& store, VectorXd& grad);
		
		void hessian(Store& store, Eigen::SparseMatrix<double>& hess, Eigen::MatrixXd& denseHess);
		
		VectorXd fd_gradient(Store& store);

		MatrixXd fd_hessian(Store& store);
	}
}
#endif