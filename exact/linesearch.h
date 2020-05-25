#ifndef LINESEARCH
#define LINESEARCH
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SparseCholesky>
#include "store.h"
using namespace Eigen;

namespace exact
{	
	void polar_dec(VectorXd& x, const std::vector<Eigen::VectorXi>& bone_tets);

	double linesearch(	int& tot_ls_its, 
						VectorXd& Fvec, 
						const VectorXd& grad, 
						const VectorXd& drt, 
						double activation,
						VectorXd& q,
						const MatrixXi& T, 
                        const VectorXd& eY, 
                        const VectorXd& eP, 
                        const VectorXd& rest_tet_vols, 
                        const MatrixXd& Uvec,
						const Eigen::SparseMatrix<double, Eigen::RowMajor>& Y, 
						const Eigen::SparseMatrix<double, Eigen::RowMajor>& B,
						const SparseMatrix<double, Eigen::RowMajor>& PF,
						const VectorXd& c,
						const std::vector<Eigen::VectorXi>& bone_tets,
						const exact::Store& store);	

}
#endif