#ifndef CONSTRUCT_KKT_SYSTEM 
#define CONSTRUCT_KKT_SYSTEM
#include <Eigen/Sparse>
#include <iostream>

namespace famu
{

	void construct_kkt_system_left(Eigen::SparseMatrix<double>& H, Eigen::SparseMatrix<double>& C, Eigen::SparseMatrix<double>& KKT_Left);

	void construct_kkt_system_right(Eigen::VectorXd& top, Eigen::VectorXd& bottom, Eigen::VectorXd& KKT_right);
}

#endif