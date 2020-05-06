#ifndef ACAP_SOLVE
#define ACAP_SOLVE
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>

using namespace Eigen;

namespace exact
{

	int acap_solve(VectorXd&  x, Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Ha_inv, SparseMatrix<double, Eigen::RowMajor>& P, SparseMatrix<double, Eigen::RowMajor>& B, VectorXd& F, VectorXd&  c);	

}
#endif