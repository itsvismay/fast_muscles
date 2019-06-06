#ifndef DFMATRIX_VECTOR_SWAP
#define DFMATRIX_VECTOR_SWAP

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
typedef Eigen::Triplet<double> Trip;


namespace famu
{
	void dFMatrix_Vector_Swap(Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, Eigen::VectorXd& vec);
}

#endif