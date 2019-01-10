#ifndef SETUPMODES 
#define SETUPMODES

#include "to_triplets.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <MatOp/SparseGenMatProd.h>
#include <MatOp/SparseCholesky.h>
#include <SymGEigsSolver.h>
#include <GenEigsSolver.h>

using namespace Eigen;
void setup_modes(int nummodes, bool reduced, SparseMatrix<double>& mP, SparseMatrix<double>& mA, SparseMatrix<double> mConstrained, SparseMatrix<double> mUnconstrained, MatrixXd& mV, VectorXd& mmass_diag, MatrixXd& mG);

#endif