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
		
		VectorXd fd_gradient(Store& store);
	}
}
#endif