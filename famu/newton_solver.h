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

	void update_dofs(Store& store, VectorXd& new_dofs, VectorXd& dFvec, bool linesearch=false);

	double line_search(int& tot_ls_its, Store& store, VectorXd& grad, VectorXd& drt, VectorXd& new_dofs);

	void sparse_to_dense(const Store& store, SparseMatrix<double, Eigen::RowMajor>& H, MatrixXd& denseHess);

	void fastWoodbury(Store& store, const VectorXd& g, MatrixModesxModes X, VectorXd& BInvXDy, MatrixXd& denseHess, VectorXd& drt);
	
	int newton_static_solve(Store& store);
}
#endif