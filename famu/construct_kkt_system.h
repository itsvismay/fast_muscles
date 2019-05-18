#ifndef CONSTRUCT_KKT_SYSTEM 
#define CONSTRUCT_KKT_SYSTEM
#include <Eigen/Sparse>
#include <iostream>

namespace famu
{

	void construct_kkt_system_left(Eigen::SparseMatrix<double , Eigen::RowMajor>& H, Eigen::SparseMatrix<double, Eigen::RowMajor>& C, Eigen::SparseMatrix<double, Eigen::RowMajor>& KKT_Left, double constraint_stiffness = 0);

	void construct_kkt_system_right(Eigen::VectorXd& top, Eigen::VectorXd& bottom, Eigen::VectorXd& KKT_right);
}

#endif