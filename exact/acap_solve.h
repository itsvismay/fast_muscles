#ifndef ACAP_SOLVE
#define ACAP_SOLVE
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SparseCholesky>
#ifdef __linux__
#include <Eigen/PardisoSupport>
#endif

using namespace Eigen;

namespace exact
{	
	template<typename T>
	int acap_solve(VectorXd&  x, 
					const SparseMatrix<double, Eigen::RowMajor>& PF, 
					const T& Ha_inv, 
					const SparseMatrix<double, Eigen::RowMajor>& P, 
					const SparseMatrix<double, Eigen::RowMajor>& B, 
					const VectorXd& F, 
					const VectorXd& c){
		x = P*Ha_inv.solve(P.transpose()*B.transpose()*(PF*F - B*c))  + c;
	}


}
#endif
