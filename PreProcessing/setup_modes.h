#ifndef SETUPMODES 
#define SETUPMODES

#include "to_triplets.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <json.hpp>


#include <MatOp/SparseGenMatProd.h>
#include <MatOp/SparseCholesky.h>
#include <SymGEigsSolver.h>
#include <GenEigsSolver.h>

using namespace Eigen;
using json = nlohmann::json;

void setup_modes(json& j_input, int nummodes, bool reduced, SparseMatrix<double>& mP, SparseMatrix<double>& mA, SparseMatrix<double> mConstrained, SparseMatrix<double> mFree, SparseMatrix<double> mY, MatrixXd& mV, const MatrixXi& mT, VectorXd& mmass_diag, MatrixXd& mG);

#endif