#ifndef KMEANS_CLUSTERING
#define KMEANS_CLUSTERING

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "ocv_kmeans_wrapper.h"
using namespace Eigen;

void kmeans_clustering(VectorXi& idx, int clusters,  std::vector<VectorXi>& ibones, std::vector<VectorXi>& imuscle, MatrixXd& mG, SparseMatrix<double>& mC, SparseMatrix<double>& mA, VectorXd& mx0);
#endif