#ifndef STABLENH_ENERGY_GRAD
#define STABLENH_ENERGY_GRAD

#include "store.h"

using namespace exact;
using namespace Eigen;
using Store = exact::Store;

namespace exact
{
	namespace stablenh
	{
		double energy(const Store& store, Eigen::VectorXd& Fvec, const Eigen::VectorXi& bone_or_muscle);
		double energy(const Store& store, Eigen::VectorXd& Fvec);

		void gradient(const Store& store, const Eigen::VectorXd& Fvec, VectorXd& grad, const Eigen::VectorXi& bone_or_muscle);
		void gradient(const Store& store, const Eigen::VectorXd& Fvec, VectorXd& grad);
		
		void hessian(const Store& store, const Eigen::VectorXd& Fvec, Eigen::SparseMatrix<double, Eigen::RowMajor>& hess, const Eigen::VectorXi& bone_or_muscle);
		void hessian(const Store& store, const Eigen::VectorXd& Fvec, Eigen::SparseMatrix<double, Eigen::RowMajor>& hess);

	}
}
#endif