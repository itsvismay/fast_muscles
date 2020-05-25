#ifndef NEWTON_SOLVE
#define NEWTON_SOLVE
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SparseCholesky>
#include "store.h"

using Store= exact::Store;
using namespace Eigen;

namespace exact
{	
	int newton_solve(VectorXd& Fvec, 
						VectorXd& q,
						const double tol,
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
						const Eigen::SparseMatrix<double, Eigen::RowMajor>& Y, 
						const Eigen::SparseMatrix<double, Eigen::RowMajor>& B, 
						const VectorXd& c,
						const std::vector<Eigen::VectorXi>& bone_tets,
						const SparseMatrix<double, Eigen::RowMajor>& wId9T,
						const MatrixXd& wVAi,
						MatrixXd& wHiV,
						MatrixXd& wHiVAi,
						MatrixXd& wC,
						MatrixXd& wPhi,
						MatrixXd& wHPhi,
						MatrixXd& wL,
						const MatrixXd& wIdL,
						MatrixXd& wQ,
						const exact::Store& store);	

}
#endif