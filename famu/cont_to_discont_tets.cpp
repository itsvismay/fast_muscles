#include "cont_to_discont_tets.h"

void famu::cont_to_discont_tets(Eigen::SparseMatrix<double>& mA, Eigen::MatrixXi& mT, Eigen::MatrixXd& mV){
	mA.resize(12*mT.rows(), 3*mV.rows());
    std::vector<Trip> triplets;
    triplets.reserve(12*mT.rows());

    for(int i=0; i<mT.rows(); i++){
        Eigen::Vector4i inds = mT.row(i);

        for(int j=0; j<4; j++){
            int v = inds[j];
            triplets.push_back(Trip(12*i+3*j+0, 3*v+0, 1));
            triplets.push_back(Trip(12*i+3*j+1, 3*v+1, 1));
            triplets.push_back(Trip(12*i+3*j+2, 3*v+2, 1));
        }
    }
    mA.setFromTriplets(triplets.begin(), triplets.end());
    
}