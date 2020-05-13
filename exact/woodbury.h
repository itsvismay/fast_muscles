#ifndef WOODBURY_SOLVE
#define WOODBURY_SOLVE

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>

#include "store.h"

using namespace Eigen;

namespace exact
{

	int woodbury(const exact::Store& store,
				VectorXd& lambda,
				SparseMatrix<double, Eigen::RowMajor>& PF,
				VectorXd& Fvec,
				VectorXd& g,
				VectorXd& d,
				Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Hinv,
				Eigen::SparseMatrix<double, Eigen::RowMajor>& H,
				MatrixXd& Ai, 
				MatrixXd& Vtilde);	

}
#endif