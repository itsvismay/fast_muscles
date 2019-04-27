#include "discontinuous_edge_vectors.h"

using namespace Eigen;
using namespace std;

void famu::discontinuous_edge_vectors(Eigen::SparseMatrix<double>& mP, Eigen::SparseMatrix<double>& m_P, Eigen::MatrixXi mT){
	mP.resize(12*mT.rows(), 12*mT.rows());
        Matrix4d p;
        p<< 3, -1, -1, -1,
            -1, 3, -1, -1,
            -1, -1, 3, -1,
            -1, -1, -1, 3;

        vector<Trip> triplets;
        triplets.reserve(3*16*mT.rows());
        VectorXd arap_weights = VectorXd::Ones(mT.rows());
        // for(int m=0; m<mmuscles.size(); m++){
        //     for(int i=0; i<mmuscles[m].size(); i++){
        //         double weight = 1;
        //         int t = mmuscles[m][i];
        //         double vol = tet_volume(t);
        //         weight = 1.0/vol/10;
        //         if(m_relativeStiffness[t]>10){
        //             weight *= 10;
        //         }
        //         arap_weights[t] = weight;
        //    }

        // }

        for(int i=0; i<mT.rows(); i++){

            for(int j=0; j<3; j++){
                triplets.push_back(Trip(12*i+0+j, 12*i+0+j, arap_weights[i]*p(0,0)/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+3+j, arap_weights[i]*p(0,1)/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+6+j, arap_weights[i]*p(0,2)/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+9+j, arap_weights[i]*p(0,3)/4));

                triplets.push_back(Trip(12*i+3+j, 12*i+0+j, arap_weights[i]*p(1,0)/4));
                triplets.push_back(Trip(12*i+3+j, 12*i+3+j, arap_weights[i]*p(1,1)/4));
                triplets.push_back(Trip(12*i+3+j, 12*i+6+j, arap_weights[i]*p(1,2)/4));
                triplets.push_back(Trip(12*i+3+j, 12*i+9+j, arap_weights[i]*p(1,3)/4));

                triplets.push_back(Trip(12*i+6+j, 12*i+0+j, arap_weights[i]*p(2,0)/4));
                triplets.push_back(Trip(12*i+6+j, 12*i+3+j, arap_weights[i]*p(2,1)/4));
                triplets.push_back(Trip(12*i+6+j, 12*i+6+j, arap_weights[i]*p(2,2)/4));
                triplets.push_back(Trip(12*i+6+j, 12*i+9+j, arap_weights[i]*p(2,3)/4));

                triplets.push_back(Trip(12*i+9+j, 12*i+0+j, arap_weights[i]*p(3,0)/4));
                triplets.push_back(Trip(12*i+9+j, 12*i+3+j, arap_weights[i]*p(3,1)/4));
                triplets.push_back(Trip(12*i+9+j, 12*i+6+j, arap_weights[i]*p(3,2)/4));
                triplets.push_back(Trip(12*i+9+j, 12*i+9+j, arap_weights[i]*p(3,3)/4));
            }
        }   
        mP.setFromTriplets(triplets.begin(), triplets.end());



        m_P.resize(12*mT.rows(), 12*mT.rows());
        vector<Trip> triplets_;
        triplets_.reserve(3*16*mT.rows());
        for(int i=0; i<mT.rows(); i++){
            for(int j=0; j<3; j++){
                triplets_.push_back(Trip(12*i+0+j, 12*i+0+j, p(0,0)/4));
                triplets_.push_back(Trip(12*i+0+j, 12*i+3+j, p(0,1)/4));
                triplets_.push_back(Trip(12*i+0+j, 12*i+6+j, p(0,2)/4));
                triplets_.push_back(Trip(12*i+0+j, 12*i+9+j, p(0,3)/4));

                triplets_.push_back(Trip(12*i+3+j, 12*i+0+j, p(1,0)/4));
                triplets_.push_back(Trip(12*i+3+j, 12*i+3+j, p(1,1)/4));
                triplets_.push_back(Trip(12*i+3+j, 12*i+6+j, p(1,2)/4));
                triplets_.push_back(Trip(12*i+3+j, 12*i+9+j, p(1,3)/4));

                triplets_.push_back(Trip(12*i+6+j, 12*i+0+j, p(2,0)/4));
                triplets_.push_back(Trip(12*i+6+j, 12*i+3+j, p(2,1)/4));
                triplets_.push_back(Trip(12*i+6+j, 12*i+6+j, p(2,2)/4));
                triplets_.push_back(Trip(12*i+6+j, 12*i+9+j, p(2,3)/4));

                triplets_.push_back(Trip(12*i+9+j, 12*i+0+j, p(3,0)/4));
                triplets_.push_back(Trip(12*i+9+j, 12*i+3+j, p(3,1)/4));
                triplets_.push_back(Trip(12*i+9+j, 12*i+6+j, p(3,2)/4));
                triplets_.push_back(Trip(12*i+9+j, 12*i+9+j, p(3,3)/4));
            }
        }   
        m_P.setFromTriplets(triplets_.begin(), triplets_.end());
}