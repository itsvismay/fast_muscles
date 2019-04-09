#ifndef SETUPROTCLUST
#define SETUPROTCLUS

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include "kmeans_clustering.h"

using namespace Eigen;
void setup_rotation_cluster(int nrc, 
	bool reduced, 
	const MatrixXi& mT, 
	const MatrixXd& mV, 
	std::vector<VectorXi>& ibones, 
	std::vector<VectorXi>& imuscle,
	VectorXd& mred_x, 
	VectorXd& mred_r, 
	VectorXd& mred_w,
	SparseMatrix<double>& mC, 
	SparseMatrix<double>& mA, 
	MatrixXd& mG, 
	VectorXd& mx0, 
	std::vector<SparseMatrix<double>>& mRotationBLOCK, 
	std::map<int, std::vector<int>>& mr_cluster_elem_map, 
	VectorXi& mr_elem_cluster_map,
	VectorXd& relStiff);
#endif