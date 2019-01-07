#include "kmeans_clustering.h"
#include "ocv_kmeans_wrapper.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

using namespace Eigen;
void kmeans_clustering(VectorXi& idx, int clusters, std::vector<VectorXi>& ibones, VectorXi& imuscle, MatrixXd& mG, SparseMatrix<double>& mC, SparseMatrix<double>& mA, VectorXd& mx0){
        MatrixXd Centroids;
        idx.resize(mC.rows()/12);
        idx.setZero();

        for(int b=0; b<ibones.size(); b++){
            for(int i=0; i<ibones[b].size(); i++){
                idx[ibones[b][i]] = clusters - 1 - b;
            }
        }
        clusters = clusters - ibones.size();

        VectorXi labels;
        if(mG.cols()!=0){
            std::cout<<"     kmeans1 reduced"<<std::endl;
            MatrixXd G = mG.array().colwise() + mx0.array();
            MatrixXd CAG = mC*mA*G;
            std::cout<<"     kmeans2 reduced"<<std::endl;
            MatrixXd Data = MatrixXd::Zero(imuscle.size(), 3*G.cols());
            for(int i=0; i<Data.rows(); i++){
                RowVectorXd r1 = CAG.row(12*imuscle[i]);
                RowVectorXd r2 = CAG.row(12*imuscle[i]+1);
                RowVectorXd r3 = CAG.row(12*imuscle[i]+2);
                RowVectorXd point(3*G.cols());
                point<<r1,r2,r3;
                Data.row(i) = point;
            }
            std::cout<<"     kmeans3 reduced"<<std::endl;
            ocv_kmeans(Data, clusters, 1000, Centroids, labels);
            std::cout<<"     kmeans4 reduced"<<std::endl;
        }else{
            std::cout<<"     kmeans1 un"<<std::endl;
            VectorXd CAx0 = mC*mA*mx0;
            std::cout<<"     kmeans2 un"<<std::endl;
            MatrixXd Data = MatrixXd::Zero(imuscle.size(), 3);
            for(int i=0; i<Data.rows(); i++){
                Data.row(i) = RowVector3d(CAx0[12*imuscle[i]+0],CAx0[12*imuscle[i]+1],CAx0[12*imuscle[i]+2]);
            }
            std::cout<<"     kmeans3 un"<<std::endl;
            ocv_kmeans(Data, clusters, 1000, Centroids, labels);
            std::cout<<"     kmeans4 un"<<std::endl;
        }
        std::cout<<"     kmeans5   "<<std::endl;

        for(int m=0; m<imuscle.size(); m++){
            idx[imuscle[m]] = labels[m];
        }
        return;
    }