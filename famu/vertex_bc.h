#ifndef VERTEX_BOUNDARY_CONDITIONS 
#define VERTEX_BOUNDARY_CONDITIONS
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>

namespace famu
{
		void vertex_bc(std::vector<int>& mmov, 
			std::vector<int>& mfix, 
			Eigen::SparseMatrix<double, Eigen::RowMajor>& mFree, 
			Eigen::SparseMatrix<double, Eigen::RowMajor>& mConstrained,
			Eigen::MatrixXd& mV);
}

#endif