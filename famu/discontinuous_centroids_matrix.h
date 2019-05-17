#ifndef DISCONTINUOUS_CENTROIDS_MATRIX
#define DISCONTINUOUS_CENTROIDS_MATRIX

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>

typedef Eigen::Triplet<double> Trip;
using namespace Eigen;

namespace famu
{
	void discontinuous_centroids_matrix(SparseMatrix<double>& mC, MatrixXi& mT);
}

#endif