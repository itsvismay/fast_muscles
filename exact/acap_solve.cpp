#include "acap_solve.h"

using namespace  Eigen;

int exact::acap_solve(VectorXd&  x, Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Ha_inv, SparseMatrix<double, Eigen::RowMajor>& P, SparseMatrix<double, Eigen::RowMajor>& B, VectorXd& F, VectorXd&  c){

	x = P.transpose()*Ha_inv.solve(P*B.transpose()*(F - B*c))  + c;
	
}