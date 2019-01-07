#include "to_triplets.h"

std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> & M){
    std::vector<Eigen::Triplet<double>> v;
    for(int i = 0; i < M.outerSize(); i++)
        for(typename Eigen::SparseMatrix<double>::InnerIterator it(M,i); it; ++it)
            v.emplace_back(it.row(),it.col(),it.value());
    return v;
}