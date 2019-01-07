#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include "setup_rotation_cluster.h"
#include "kmeans_clustering.h"

using namespace Eigen;
using namespace std;
typedef Eigen::Triplet<double> Trip;


void setup_rotation_cluster(int nrc, bool reduced, const MatrixXi& mT, const MatrixXd& mV, std::vector<VectorXi>& ibones, VectorXi& imuscle, VectorXd& mred_x, VectorXd& mred_r, VectorXd& mred_w,
	SparseMatrix<double>& mC, SparseMatrix<double>& mA, MatrixXd& mG, VectorXd& mx0, 
	std::vector<SparseMatrix<double>>& mRotationBLOCK, std::map<int, std::vector<int>>& mr_cluster_elem_map, VectorXi& mr_elem_cluster_map){
    std::cout<<"+ Rotation Clusters"<<std::endl;
    if(nrc==0){
        //unreduced
        nrc = mT.rows();
    }

    mr_elem_cluster_map.resize(mT.rows());
    if(nrc==mT.rows() && reduced==false){
        //unreduced
        for(int i=0; i<mT.rows(); i++){
            mr_elem_cluster_map[i] = i;
        }   
    }else{

        if(3*mV.rows()==mred_x.size() && reduced==false){
            std::cout<<"Continuous mesh is unreduced. Kmeans won't work."<<std::endl;
            exit(0);
        }else{
            if(nrc==mT.rows()){
                for(int i=0; i<mT.rows(); i++){
                   mr_elem_cluster_map[i] = i;
                }  
            }else{
                kmeans_clustering(mr_elem_cluster_map, nrc, ibones, imuscle, mG, mC, mA, mx0);
            }
        }

    }

    for(int i=0; i<mT.rows(); i++){
        mr_cluster_elem_map[mr_elem_cluster_map[i]].push_back(i);
    }

    mred_r.resize(9*nrc);
    for(int i=0; i<nrc; i++){
        mred_r[9*i+0] = 1;
        mred_r[9*i+1] = 0;
        mred_r[9*i+2] = 0;
        mred_r[9*i+3] = 0;
        mred_r[9*i+4] = 1;
        mred_r[9*i+5] = 0;
        mred_r[9*i+6] = 0;
        mred_r[9*i+7] = 0;
        mred_r[9*i+8] = 1;
    }
    mred_w.resize(3*nrc);
    mred_w.setZero();

    if(reduced){
        for(int c=0; c<nrc; c++){
            std::vector<int> notfix = mr_cluster_elem_map[c];
            // SparseMatrix<double> bo(mT.rows(), notfix.size());
            // bo.setZero();
            std::vector<Trip> bo_trip;
            bo_trip.reserve(notfix.size());

            int i = 0;
            int f = 0;
            for(int j =0; j<notfix.size(); j++){
                if (i==notfix[f]){
                    bo_trip.push_back(Trip(i, j, 1));
                    f++;
                    i++;
                    continue;
                }
                j--;
                i++;
            }

            
            std::vector<Trip> b_trip;
            b_trip.reserve(bo_trip.size());  
            for(int k =0; k<bo_trip.size(); k++){
                b_trip.push_back(Trip(12*bo_trip[k].row()+0, 12*bo_trip[k].col()+0, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+1, 12*bo_trip[k].col()+1, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+2, 12*bo_trip[k].col()+2, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+3, 12*bo_trip[k].col()+3, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+4, 12*bo_trip[k].col()+4, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+5, 12*bo_trip[k].col()+5, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+6, 12*bo_trip[k].col()+6, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+7, 12*bo_trip[k].col()+7, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+8, 12*bo_trip[k].col()+8, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+9, 12*bo_trip[k].col()+9, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+10, 12*bo_trip[k].col()+10, bo_trip[k].value()));
                b_trip.push_back(Trip(12*bo_trip[k].row()+11, 12*bo_trip[k].col()+11, bo_trip[k].value()));
            }

            SparseMatrix<double> b(12*mT.rows(), 12*notfix.size());
            b.setFromTriplets(b_trip.begin(), b_trip.end());
            mRotationBLOCK.push_back(b);
        }
    }
    std::cout<<"- Rotation Clusters"<<std::endl;
}