#ifndef DISCONTINUOUS_EDGE_VECTORS
#define DISCONTINUOUS_EDGE_VECTORS

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include "store.h"

typedef Eigen::Triplet<double> Trip;
using namespace Eigen;

namespace famu
{
	void discontinuous_edge_vectors(famu::Store& store, Eigen::SparseMatrix<double, Eigen::RowMajor>& mP, Eigen::SparseMatrix<double, Eigen::RowMajor>& m_P, Eigen::MatrixXi mT, std::vector<Eigen::VectorXi>& muscle_tets);
}

#endif