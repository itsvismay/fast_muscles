#ifndef PASS_LAMBDA
#define PASS_LAMBDA

#include <Eigen/Dense>

namespace exact{
	void passlambda(
		std::function<int(Eigen::VectorXd& d, Eigen::MatrixXd& V, Eigen::MatrixXd& Ai, Eigen::VectorXd& b)>& Jtimes
		);
}

#endif