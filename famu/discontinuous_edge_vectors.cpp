#include "discontinuous_edge_vectors.h"
#include "store.h"
using namespace Eigen;
using namespace std;
using Store=famu::Store;

double get_volume(Vector3d p1, Vector3d p2, Vector3d p3, Vector3d p4){
    Matrix3d Dm;
    Dm.col(0) = p1 - p4;
    Dm.col(1) = p2 - p4;
    Dm.col(2) = p3 - p4;
    double density = 1000;
    double m_undeformedVol = (1.0/6)*fabs(Dm.determinant());
    return m_undeformedVol;
}

void famu::discontinuous_edge_vectors(Store& store, Eigen::SparseMatrix<double>& mP, Eigen::SparseMatrix<double>& m_P, Eigen::MatrixXi mT, std::vector<Eigen::VectorXi>& muscle_tets){
    mP.resize(12*mT.rows(), 12*mT.rows());
        Matrix4d p;
        p<< 3, -1, -1, -1,
            -1, 3, -1, -1,
            -1, -1, 3, -1,
            -1, -1, -1, 3;
        // MatrixXd P = Eigen::KroneckerProduct(p, Matrix3d::Identity());

        vector<Trip> triplets;
        triplets.reserve(3*16*mT.rows());
        VectorXd arap_weights = VectorXd::Ones(mT.rows());
       
        for(int m=0; m<muscle_tets.size(); m++){
            for(int i=0; i<muscle_tets[m].size(); i++){
                int t = muscle_tets[m][i];
                // double vol = tet_volume(t);
                double weight = 1;
                if(store.relativeStiffness[t]>1){
                    weight = 100;
                }else{
                    weight =1;
                }
                arap_weights[t] *= weight;
                
           }
        }

        VectorXd Sx0 = store.S*store.x0;

        for(int i=0; i<mT.rows(); i++){
            // VectorXd edges = P*Sx0.segment<12>(12*i);

            for(int j=0; j<3; j++){
                // double d = edges.segment<3>()
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


        arap_weights = VectorXd::Ones(store.T.rows());
        // for(int i =0; i<store.T.rows(); i++){
        //      double undef_vol = get_volume(
        //             store.V.row(store.T.row(i)[0]), 
        //             store.V.row(store.T.row(i)[1]), 
        //             store.V.row(store.T.row(i)[2]), 
        //             store.V.row(store.T.row(i)[3]));
        //      arap_weights[i] = undef_vol;
        // }

        for(int m=0; m<muscle_tets.size(); m++){
            for(int i=0; i<muscle_tets[m].size(); i++){
                int t = muscle_tets[m][i];
                // double vol = tet_volume(t);
                double weight = 1;
                if(store.relativeStiffness[t]>1){
                    weight = 2;
                }else{
                    weight =1;
                }
                arap_weights[t] *= weight;
                
           }
        }



        m_P.resize(12*mT.rows(), 12*mT.rows());
        vector<Trip> triplets_;
        triplets_.reserve(3*16*mT.rows());
        for(int i=0; i<mT.rows(); i++){
            for(int j=0; j<3; j++){
                triplets_.push_back(Trip(12*i+0+j, 12*i+0+j, arap_weights[i]*p(0,0)/4));
                triplets_.push_back(Trip(12*i+0+j, 12*i+3+j, arap_weights[i]*p(0,1)/4));
                triplets_.push_back(Trip(12*i+0+j, 12*i+6+j, arap_weights[i]*p(0,2)/4));
                triplets_.push_back(Trip(12*i+0+j, 12*i+9+j, arap_weights[i]*p(0,3)/4));

                triplets_.push_back(Trip(12*i+3+j, 12*i+0+j, arap_weights[i]*p(1,0)/4));
                triplets_.push_back(Trip(12*i+3+j, 12*i+3+j, arap_weights[i]*p(1,1)/4));
                triplets_.push_back(Trip(12*i+3+j, 12*i+6+j, arap_weights[i]*p(1,2)/4));
                triplets_.push_back(Trip(12*i+3+j, 12*i+9+j, arap_weights[i]*p(1,3)/4));

                triplets_.push_back(Trip(12*i+6+j, 12*i+0+j, arap_weights[i]*p(2,0)/4));
                triplets_.push_back(Trip(12*i+6+j, 12*i+3+j, arap_weights[i]*p(2,1)/4));
                triplets_.push_back(Trip(12*i+6+j, 12*i+6+j, arap_weights[i]*p(2,2)/4));
                triplets_.push_back(Trip(12*i+6+j, 12*i+9+j, arap_weights[i]*p(2,3)/4));

                triplets_.push_back(Trip(12*i+9+j, 12*i+0+j, arap_weights[i]*p(3,0)/4));
                triplets_.push_back(Trip(12*i+9+j, 12*i+3+j, arap_weights[i]*p(3,1)/4));
                triplets_.push_back(Trip(12*i+9+j, 12*i+6+j, arap_weights[i]*p(3,2)/4));
                triplets_.push_back(Trip(12*i+9+j, 12*i+9+j, arap_weights[i]*p(3,3)/4));
            }
        }   
        m_P.setFromTriplets(triplets_.begin(), triplets_.end());
}