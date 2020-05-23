#include "passlambda.h"
#include <iostream>
void exact::passlambda(std::function<int(Eigen::VectorXd& d, Eigen::MatrixXd& V, Eigen::MatrixXd& Ai, Eigen::VectorXd& b)>& functor) {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
}