#ifndef SETUP_HESSIAN_MODES 
#define SETUP_HESSIAN_MODES

#include <MatOp/SparseGenMatProd.h>
#include <MatOp/SparseCholesky.h>
#include <SymGEigsSolver.h>
#include <SymEigsSolver.h>

using namespace Eigen;
namespace exact
{
	void setup_hessian_modes(SparseMatrix<double>& A, MatrixXd& mG, VectorXd& eigenvalues, int nummodes=48);
}

#endif