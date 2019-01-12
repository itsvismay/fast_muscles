#include "kmeans_clustering.h"
#include "ocv_kmeans_wrapper.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

using namespace Eigen;
void kmeans_clustering(VectorXi& idx, int clusters, std::vector<VectorXi>& ibones, std::vector<VectorXi>& imuscle, MatrixXd& mG, SparseMatrix<double>& mC, SparseMatrix<double>& mA, VectorXd& mx0){
        MatrixXd Centroids;
        idx.resize(mC.rows()/12);
        idx.setZero();

        for(int b=0; b<ibones.size(); b++){
            for(int i=0; i<ibones[b].size(); i++){
                idx[ibones[b][i]] = clusters - 1 - b;
            }
        }
        clusters = clusters - ibones.size();


        std::cout<<"     kmeans1 un"<<std::endl;
        VectorXd CAx0 = mC*mA*mx0;
        int clusters_per_muscle = clusters/imuscle.size();

        for(int m=0; m<imuscle.size(); m++){
            std::cout<<"     kmeans2 un"<<std::endl;
            VectorXi labels;
            MatrixXd Data = MatrixXd::Zero(imuscle[m].size(), 3);
            for(int i=0; i<Data.rows(); i++){
                Data.row(i) = RowVector3d(CAx0[12*imuscle[m][i]+0],CAx0[12*imuscle[m][i]+1],CAx0[12*imuscle[m][i]+2]);
            }

            std::cout<<"     kmeans3 un"<<std::endl;
            if(m==imuscle.size()-1){
                //deal with remainder clusters
                ocv_kmeans(Data, clusters, 1000, Centroids, labels);
            }else{
                ocv_kmeans(Data, clusters_per_muscle, 1000, Centroids, labels);
            }

            std::cout<<"     kmeans4 un"<<std::endl;
            for(int q=0; q<imuscle[m].size(); q++){
                idx[imuscle[m][q]] = clusters_per_muscle*m + labels[q];
            }
            clusters = clusters - clusters_per_muscle;

        }
        assert(clusters==0);
        return;
    }