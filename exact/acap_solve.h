#ifndef ACAP_SOLVE
#define ACAP_SOLVE
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SparseCholesky>


using namespace Eigen;

namespace exact
{	
	int acap_solve(VectorXd&  x, 
					const SparseMatrix<double, Eigen::RowMajor>& PF, 
					const Eigen::SparseLU<Eigen::SparseMatrix<double,Eigen::RowMajor>>& Ha_inv, 
					const SparseMatrix<double, Eigen::RowMajor>& P, 
					const SparseMatrix<double, Eigen::RowMajor>& B, 
					const VectorXd& F, 
					const VectorXd& c);	

}
#endif