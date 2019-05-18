#ifndef CONT_TO_DISCONT_TETS
#define CONT_TO_DISCONT_TETS

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::Triplet<double> Trip;
namespace famu
{
	void cont_to_discont_tets(Eigen::SparseMatrix<double, Eigen::RowMajor>& mA, Eigen::MatrixXi& mT, Eigen::MatrixXd& mV);
}

#endif