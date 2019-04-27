#ifndef DISCONTINUOUS_EDGE_VECTORS
#define DISCONTINUOUS_EDGE_VECTORS

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>

typedef Eigen::Triplet<double> Trip;
using namespace Eigen;

namespace famu
{
	void discontinuous_edge_vectors(Eigen::SparseMatrix<double>& mP, Eigen::SparseMatrix<double>& m_P, Eigen::MatrixXi mT);
}

#endif