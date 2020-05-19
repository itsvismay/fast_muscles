#ifndef WOODBURY_SOLVE
#define WOODBURY_SOLVE

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>

#include "store.h"

using namespace Eigen;

namespace exact
{

	int woodbury(VectorXd& lambda,
				VectorXd& Fvec,
				VectorXd& g,
				Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Hinv,
				Eigen::SparseMatrix<double, Eigen::RowMajor>& H,
				const SparseMatrix<double, Eigen::RowMajor>& PF,
				const VectorXd& d,
				const MatrixXd& Ai, 
				const MatrixXd& Vtilde);	

}
#endif