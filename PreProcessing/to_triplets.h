#ifndef TO_TRIPS
#define TO_TRIPS

#include <Eigen/Dense>
#include <Eigen/Sparse>
std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> & M);
#endif