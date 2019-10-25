#ifndef MUSCLE_ENERGY_GRAD
#define MUSCLE_ENERGY_GRAD

#include "store.h"

using namespace famu;
using Store = famu::Store;
using namespace Eigen;

namespace famu
{
	namespace muscle{

		void setupFastMuscles(Store& store);

		double fastEnergy(Store& store, VectorXd& dFvec);

		void fastGradient(Store& store, VectorXd& grad);

		double energy(Store& store, VectorXd& dFvec);

		void gradient(Store& store, VectorXd& grad);

		void fastHessian(Store& store, SparseMatrix<double, Eigen::RowMajor>& hess, Eigen::MatrixXd& denseHess);

		VectorXd fd_gradient(Store& store);

		MatrixXd fd_hessian(Store& store);

		void set_muscle_mag(Store& store, int step);
	}
}

#endif