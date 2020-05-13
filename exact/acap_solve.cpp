#include "acap_solve.h"

using namespace  Eigen;

int exact::acap_solve(VectorXd&  x, 
						SparseMatrix<double, Eigen::RowMajor>& PF, 
						Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>>& Ha_inv, 
						SparseMatrix<double, Eigen::RowMajor>& P, 
						SparseMatrix<double, Eigen::RowMajor>& B, 
						VectorXd& F, 
						VectorXd& c){

	x = P*Ha_inv.solve(P.transpose()*B.transpose()*(PF*F - B*c))  + c;
	
}