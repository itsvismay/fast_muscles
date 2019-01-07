#ifndef SETUPSKINNINGHANDLES
#define SETUPSKINNINGHANDLES
#include "kmeans_clustering.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <igl/boundary_conditions.h>
#include <igl/lbs_matrix.h>
#include <igl/bbw.h>

using namespace Eigen;

void setup_skinning_handles(int nsh, bool reduced, const MatrixXi& mT, const MatrixXd& mV, std::vector<VectorXi>& ibones, VectorXi& imuscle, 
	SparseMatrix<double>& mC, SparseMatrix<double>& mA, MatrixXd& mG, VectorXd& mx0, 
	VectorXi& ms_handles_ind, VectorXd& mred_s, MatrixXd& msW);

#endif