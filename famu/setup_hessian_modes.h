#ifndef SETUP_HESSIAN_MODES 
#define SETUP_HESSIAN_MODES

#include "store.h"
#include <MatOp/SparseGenMatProd.h>
#include <MatOp/SparseCholesky.h>
#include <SymGEigsSolver.h>
#include <SymEigsSolver.h>

using namespace Eigen;
using Store = famu::Store;
namespace famu
{
	void setup_hessian_modes(Store& store, SparseMatrix<double>& A, MatrixXd& mG);
}

#endif