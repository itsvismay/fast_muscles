#ifndef WOODBURY_SOLVE
#define WOODBURY_SOLVE

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>
using namespace Eigen;

namespace exact
{

	int woodbury(VectorXd& lambda,
						VectorXd& Fvec,
						VectorXd& g,
						VectorXd& d,
						Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Hinv,
						Eigen::SparseMatrix<double, Eigen::RowMajor>& H,
						MatrixXd& Ai, 
						MatrixXd& Vtilde,
						MatrixXd& J);	

}
#endif