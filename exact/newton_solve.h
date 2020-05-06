#ifndef NEWTON_SOLVE
#define NEWTON_SOLVE
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>
#include "store.h"

using Store= exact::Store;
using namespace Eigen;

namespace exact
{

	int newton_solve(const Store& store, VectorXd& Fvec, VectorXd& d, MatrixXd& Ai, MatrixXd& Vtilde, MatrixXd& J, double activation);	

}
#endif