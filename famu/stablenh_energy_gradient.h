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
		double energy(const Store& store, VectorXd& dFvec);

		void gradient(const Store& store, VectorXd& grad);
		
		void hessian(const Store& store, Eigen::SparseMatrix<double, Eigen::RowMajor>& hess, Eigen::MatrixXd& denseHess, bool dense=false);
		
		VectorXd fd_gradient(Store& store);

		MatrixXd fd_hessian(Store& store);
	}
}
#endif