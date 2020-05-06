#ifndef CONSTRUCT_KKT_SYSTEM 
#define CONSTRUCT_KKT_SYSTEM
#include <Eigen/Sparse>
#include <iostream>

namespace exact
{

	void construct_kkt_system_left(Eigen::SparseMatrix<double , Eigen::RowMajor>& out, Eigen::SparseMatrix<double, Eigen::RowMajor>& TL, Eigen::SparseMatrix<double, Eigen::RowMajor>& TR, Eigen::SparseMatrix<double, Eigen::RowMajor>& BL, double constraint_stiffness = 0);

	void construct_kkt_system_right(Eigen::VectorXd& top, Eigen::VectorXd& bottom, Eigen::VectorXd& KKT_right);
}

#endif