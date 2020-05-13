#ifndef MUSCLE_ENERGY_GRAD
#define MUSCLE_ENERGY_GRAD

#include "store.h"

using Store = exact::Store;
using namespace Eigen;

namespace exact
{
	namespace muscle{

		double energy(const Store& store, VectorXd& Fvec, const Eigen::VectorXi& bone_or_muscle);
		double energy(const Store& store, VectorXd& Fvec);

		void gradient(const Store& store, const VectorXd& Fvec, VectorXd& grad, const Eigen::VectorXi& bone_or_muscle);
		void gradient(const Store& store, const VectorXd& Fvec, VectorXd& grad);

		void hessian(const Store& store, const VectorXd& Fvec, SparseMatrix<double, Eigen::RowMajor>& H, const Eigen::VectorXi& bone_or_muscle);
		void hessian(const Store& store, const VectorXd& Fvec, SparseMatrix<double, Eigen::RowMajor>& H);

		void set_muscle_mag(const Store& store, const int step);
	}
}

#endif