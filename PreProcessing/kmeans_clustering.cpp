#include "kmeans_clustering.h"
#include "ocv_kmeans_wrapper.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <igl/list_to_matrix.h>

using namespace Eigen;

int kmeans_muscles_and_tendons(VectorXi& idx, int clusters_per_tendon, int clusters, int m, int clusters_per_muscle, std::vector<VectorXi>& imuscle, VectorXd& CAx0, VectorXd& relStiff){
    std::cout<<"     kmeans1.5 deal with tendons"<<std::endl;
    std::vector<int> tendon_elements_list;
    std::vector<int> muscle_elements_list;
    for(int i=0; i<imuscle[m].size(); i++){
        if(relStiff[imuscle[m][i]]>10){
            tendon_elements_list.push_back(imuscle[m][i]);
        }else{
            muscle_elements_list.push_back(imuscle[m][i]);
        }
    }
  
    VectorXi muscle_els, tendon_els;
    igl::list_to_matrix(tendon_elements_list, tendon_els);
    igl::list_to_matrix(muscle_elements_list, muscle_els);

    //--------------DEAL WITH TENDONS
    if(tendon_els.size()>0){
        MatrixXd TCentroids;
        VectorXi Tlabels;
        MatrixXd TData = MatrixXd::Zero(tendon_els.size(), 3);
        for(int i=0; i<TData.rows(); i++){
            TData.row(i) = RowVector3d(CAx0[12*tendon_els[i]+0],CAx0[12*tendon_els[i]+1],CAx0[12*tendon_els[i]+2]);
        }

        if(m==tendon_els.size()-1){
            //deal with remainder clusters
            ocv_kmeans(TData, clusters_per_tendon, 1000, TCentroids, Tlabels);
        }else{
            ocv_kmeans(TData, clusters_per_tendon, 1000, TCentroids, Tlabels);
        }

        std::cout<<"     kmeans4 create element_cluster_map"<<std::endl;
        for(int q=0; q<tendon_els.size(); q++){
            idx[tendon_els[q]] = clusters_per_muscle*m + Tlabels[q];
        }
        clusters = clusters - clusters_per_tendon;
        clusters_per_muscle -= clusters_per_tendon;
        //-------------------------------
    }

    std::cout<<"     kmeans2 un"<<std::endl;
    MatrixXd Centroids;
    VectorXi labels;
    MatrixXd Data = MatrixXd::Zero(muscle_els.size(), 3);
    
    for(int i=0; i<Data.rows(); i++){
        Data.row(i) = RowVector3d(CAx0[12*muscle_els[i]+0],CAx0[12*muscle_els[i]+1],CAx0[12*muscle_els[i]+2]);
    }

    std::cout<<"     kmeans3 do clustering"<<std::endl;
    std::cout<<clusters<<std::endl;
    if(m==muscle_els.size()-1){
        //deal with remainder clusters
        ocv_kmeans(Data, clusters, 1000, Centroids, labels);
    }else{
        ocv_kmeans(Data, clusters_per_muscle, 1000, Centroids, labels);
    }

    std::cout<<"     kmeans4 create element_cluster_map"<<std::endl;
    for(int q=0; q<muscle_els.size(); q++){
        idx[muscle_els[q]] = (clusters_per_muscle)*m + labels[q]+2;
    }
    clusters = clusters - clusters_per_muscle;
    std::cout<<clusters<<std::endl;
    return clusters;
}

void kmeans_clustering(VectorXi& idx, int clusters, int clusters_per_tendon, std::vector<VectorXi>& ibones, std::vector<VectorXi>& imuscle, MatrixXd& mG, SparseMatrix<double>& mC, SparseMatrix<double>& mA, VectorXd& mx0, VectorXd& relStiff){
        idx.resize(mC.rows()/12);
        idx.setZero();
        std::cout<<"    kmeans0 un "<<std::endl;

        std::cout<<clusters<<std::endl;
        for(int b=0; b<ibones.size(); b++){
            for(int i=0; i<ibones[b].size(); i++){
                idx[ibones[b][i]] = clusters - 1 - b;
            }
        }
        clusters = clusters - ibones.size();


        std::cout<<"     kmeans1 un"<<std::endl;
        std::cout<<clusters<<std::endl;
        VectorXd CAx0 = mC*mA*mx0;

        int clusters_per_muscle = clusters/imuscle.size();

        for(int m=0; m<imuscle.size(); m++){
            clusters = kmeans_muscles_and_tendons(idx, clusters_per_tendon, clusters, m, clusters_per_muscle, imuscle, CAx0, relStiff);
        }

        std::cout<<"    kmeans5 un"<<std::endl;

        assert(clusters==0);
        return;
    }