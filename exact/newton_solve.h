#ifndef NEWTON_SOLVE
#define NEWTON_SOLVE
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>
#include "store.h"

using Store= exact::Store;
using namespace Eigen;

namespace exact
{
	int newton_solve(VectorXd& Fvec, 
						VectorXd& q,
						const MatrixXi& T,
						const VectorXd& eY,
						const VectorXd& eP,
						const MatrixXd& Uvec,
						const VectorXd& rest_tet_vols,
						const VectorXi& bone_or_muscle, 
						const SparseMatrix<double, Eigen::RowMajor>& PF, 
						const VectorXd& d, 
						const MatrixXd& Ai, 
						const MatrixXd& Vtilde, 
						const double activation,
						const Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& ACAP, 
						const Eigen::SparseMatrix<double, Eigen::RowMajor>& Y, 
						const Eigen::SparseMatrix<double, Eigen::RowMajor>& B, 
						const VectorXd& c,
						const std::vector<Eigen::VectorXi>& bone_tets);	

}
#endif