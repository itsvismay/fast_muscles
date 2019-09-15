#ifndef NEWTON_SOLVER
#define NEWTON_SOLVER
#include "store.h"
#include <igl/polar_dec.h>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <igl/Timer.h>

using Store = famu::Store;
using namespace Eigen;
using namespace std;


namespace famu
{
	double Energy(Store& store, VectorXd& dFvec);

	void polar_dec(Store& store, VectorXd& dFvec);

	double line_search(int& tot_ls_its, Store& store, VectorXd& grad, VectorXd& drt);

	void sparse_to_dense(const Store& store, SparseMatrix<double, Eigen::RowMajor>& H, MatrixXd& denseHess);

	void fastWoodbury(Store& store, const VectorXd& g, MatrixModesxModes X, VectorXd& BInvXDy, MatrixXd& denseHess, VectorXd& drt);
	
	int newton_static_solve(Store& store);
}
#endif