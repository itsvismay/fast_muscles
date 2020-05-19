#include "acap_solve.h"

using namespace  Eigen;

int exact::acap_solve(VectorXd&  x, 
					const SparseMatrix<double, Eigen::RowMajor>& PF, 
					const Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Ha_inv, 
					const SparseMatrix<double, Eigen::RowMajor>& P, 
					const SparseMatrix<double, Eigen::RowMajor>& B, 
					const VectorXd& F, 
					const VectorXd& c){

	x = P*Ha_inv.solve(P.transpose()*B.transpose()*(PF*F - B*c))  + c;
	
}