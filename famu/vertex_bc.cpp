#include "vertex_bc.h"
#include <iostream>
#include <set>

using namespace Eigen;

void famu::vertex_bc(std::vector<int>& mmov, std::vector<int>& mfix, Eigen::SparseMatrix<double, Eigen::RowMajor>& mFree, Eigen::SparseMatrix<double, Eigen::RowMajor>& mConstrained, Eigen::MatrixXd& mV){
    std::vector<int> lfix = mfix;
	if(mmov.size()>0){
        lfix.insert(lfix.end(), mmov.begin(), mmov.end());
    }
    std::sort (lfix.begin(), lfix.end());

    std::vector<int> notfix;
    mFree.resize(3*mV.rows(), 3*mV.rows() - 3*lfix.size());
    mFree.setZero();

    int i = 0;
    int f = 0;
    for(int j =0; j<mFree.cols()/3; j++){
        if (i==lfix[f]){
            f++;
            i++;
            j--;

            continue;
        }   
        notfix.push_back(i);
        mFree.coeffRef(3*i+0, 3*j+0) = 1;
        mFree.coeffRef(3*i+1, 3*j+1) = 1;
        mFree.coeffRef(3*i+2, 3*j+2) = 1; 
        
        i++;
    }

    mConstrained.resize(3*mV.rows(), 3*mV.rows() - 3*notfix.size());
    mConstrained.setZero();

    i = 0;
    f = 0;
    for(int j =0; j<mConstrained.cols()/3; j++){
        if (i==notfix[f]){
            f++;
            i++;
            j--;

            continue;
        }   
        mConstrained.coeffRef(3*i+0, 3*j+0) = 1;
        mConstrained.coeffRef(3*i+1, 3*j+1) = 1;
        mConstrained.coeffRef(3*i+2, 3*j+2) = 1; 
        
        i++;
    }
}

void famu::penalty_spring_bc(std::vector<std::pair<int,int>>& springs, Eigen::SparseMatrix<double, Eigen::RowMajor>& mP, Eigen::MatrixXd& mV){

    mP.resize(3*springs.size(), 3*mV.rows());
    mP.setZero();

    for(int i=0; i<springs.size(); i++){
        int node1 = springs[i].first;
        int node2 = springs[i].second;
        mP.coeffRef(3*i+0, 3*node1+0) = 1;
        mP.coeffRef(3*i+1, 3*node1+1) = 1;
        mP.coeffRef(3*i+2, 3*node1+2) = 1;

        mP.coeffRef(3*i+0, 3*node2+0) = -1;
        mP.coeffRef(3*i+1, 3*node2+1) = -1;
        mP.coeffRef(3*i+2, 3*node2+2) = -1;
    }

}

std::vector<int> famu::getMaxVerts_Axis_Tolerance(MatrixXi& mT, MatrixXd& mV, int dim, double tolerance, Eigen::VectorXi& muscle){
    double maxX = mV(mT.row(muscle[0])[0], dim);
    for(int i=0; i<muscle.size(); i++){
        for(int j=0; j<mT.row(muscle[i]).size(); ++j){
            int ii=mT.row(muscle[i])[j];
            if( mV(ii, dim)>maxX){
                maxX = mV(ii, dim);
            }
        }
    }


    std::set<int> maxV;
    for(int i=0; i<muscle.size(); i++){

        for(unsigned int j=0; j<mT.row(muscle[i]).size(); ++j) {
            int ii= mT.row(muscle[i])[j];

            if(fabs(mV(ii,dim) - maxX) < tolerance) {
                maxV.insert(ii);
            }
        }
    }
    std::vector<int> ret;
    ret.assign(maxV.begin(), maxV.end());
    return ret;
}

std::vector<int> famu::getMinVerts_Axis_Tolerance(MatrixXi& mT, MatrixXd& mV, int dim, double tolerance, Eigen::VectorXi& muscle){
    double maxX = mV(mT.row(muscle[0])[0], dim);
    for(int i=0; i<muscle.size(); i++){
        for(int j=0; j<mT.row(muscle[i]).size(); ++j){
            int ii=mT.row(muscle[i])[j];
            if( mV(ii, dim)<maxX){
                maxX = mV(ii, dim);
            }
        }
    }

    std::set<int> maxV;
    for(int i=0; i<muscle.size(); i++){
        for(unsigned int j=0; j<mT.row(muscle[i]).size(); ++j) {
            int ii= mT.row(muscle[i])[j];
            
            if(fabs(mV(ii,dim) - maxX) < tolerance) {
                maxV.insert(ii);
            }
        }
    }
    std::vector<int> ret;
    ret.assign(maxV.begin(), maxV.end());
    return ret;
}

std::vector<int> famu::getMidVerts_Axis_Tolerance(MatrixXd& mV, int dim, double tolerance, bool left){
    auto midX = mV.col(dim).minCoeff() + (mV.col(dim).maxCoeff() - mV.col(dim).minCoeff())/2;
    std::vector<int> maxV;
    for(unsigned int ii=0; ii<mV.rows(); ++ii) {

        if(fabs(mV(ii,dim) - midX) < tolerance && left==((mV(ii, dim)- midX)<0)) {
            maxV.push_back(ii);
        }
    }
    return maxV;
}

void famu::make_closest_point_springs(Eigen::MatrixXi& mT, 
                                    Eigen::MatrixXd& mV, 
                                    Eigen::VectorXi& muscle,
                                    std::vector<int>& points, 
                                    std::vector<std::pair<int,int>>& springs){

    //for each point, find 3 closest points on muscle
    for(int j=0; j<points.size(); j++){
        Eigen::RowVector3d point = mV.row(points[j]);
        
        int close1= mT.row(muscle[0])[0], 
            close2= mT.row(muscle[0])[1], 
            close3= mT.row(muscle[0])[2];

        double dist1 = (point - mV.row(mT.row(muscle[0])[0])).norm(),
                dist2 = (point - mV.row(mT.row(muscle[0])[1])).norm(), 
                dist3 = (point - mV.row(mT.row(muscle[0])[2])).norm();

        for(int i=0; i<muscle.size(); i++){
            int t = muscle[i];
            
            for(int k=0; k<4; k++){
                int ind = mT.row(t)[k];
                double dist = (point - mV.row(ind)).norm();

                if(dist <= dist1){
                    dist1 = dist;
                    close1 = ind;
                }else if(dist <= dist2){
                    dist2 = dist;
                    close2 = ind;
                }else if(dist <= dist3){
                    dist3 = dist;
                    close3 = ind;
                }
            }
        }

        springs.push_back(std::make_pair(points[j], close1));
        springs.push_back(std::make_pair(points[j], close2));
        springs.push_back(std::make_pair(points[j], close3));
    }

}