#include "discontinuous_centroids_matrix.h"

#include <vector>

void famu::discontinuous_centroids_matrix(Eigen::SparseMatrix<double, Eigen::RowMajor>& mC, Eigen::MatrixXi& mT){
        mC.resize(12*mT.rows(), 12*mT.rows());

        std::vector<Trip> triplets;
        triplets.reserve(3*16*mT.rows());

        for(int i=0; i<mT.rows(); i++){
            for(int j=0; j<3; j++){
                triplets.push_back(Trip(12*i+0+j, 12*i+0+j, 1.0/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+3+j, 1.0/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+6+j, 1.0/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+9+j, 1.0/4));

                triplets.push_back(Trip(12*i+3+j, 12*i+0+j, 1/4.0));
                triplets.push_back(Trip(12*i+3+j, 12*i+3+j, 1/4.0));
                triplets.push_back(Trip(12*i+3+j, 12*i+6+j, 1/4.0));
                triplets.push_back(Trip(12*i+3+j, 12*i+9+j, 1/4.0));

                triplets.push_back(Trip(12*i+6+j, 12*i+0+j, 1/4.0));
                triplets.push_back(Trip(12*i+6+j, 12*i+3+j, 1/4.0));
                triplets.push_back(Trip(12*i+6+j, 12*i+6+j, 1/4.0));
                triplets.push_back(Trip(12*i+6+j, 12*i+9+j, 1/4.0));

                triplets.push_back(Trip(12*i+9+j, 12*i+0+j, 1/4.0));
                triplets.push_back(Trip(12*i+9+j, 12*i+3+j, 1/4.0));
                triplets.push_back(Trip(12*i+9+j, 12*i+6+j, 1/4.0));
                triplets.push_back(Trip(12*i+9+j, 12*i+9+j, 1/4.0));
            }
        }   

        mC.setFromTriplets(triplets.begin(), triplets.end());
}