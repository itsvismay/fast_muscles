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
	int newton_solve(const Store& store, VectorXd& Fvec, VectorXi& bone_or_muscle, SparseMatrix<double, Eigen::RowMajor>& PF, VectorXd& d, MatrixXd& Ai, MatrixXd& Vtilde, double activation);	

}
#endif