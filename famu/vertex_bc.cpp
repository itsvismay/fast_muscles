#include "vertex_bc.h"

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