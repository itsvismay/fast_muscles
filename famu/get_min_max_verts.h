#ifndef GET_MIN_MAX_VERTS
#define GET_MIN_MAX_VERTS

#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace famu
{
	std::vector<int> getMaxVerts(Eigen::MatrixXd& mV, int dim, double tolerance=1e-5);
	std::vector<int> getMinVerts(Eigen::MatrixXd& mV, int dim, double tolerance=1e-5);
	
}

#endif