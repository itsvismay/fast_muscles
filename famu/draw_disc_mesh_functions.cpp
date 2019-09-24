#include "draw_disc_mesh_functions.h"
#include "dfmatrix_vector_swap.h"

void famu::setDiscontinuousMeshT(Eigen::MatrixXi& mT, Eigen::MatrixXi& discT){
    discT.resize(mT.rows(), 4);
    for(int i=0; i<mT.rows(); i++){
        discT(i, 0) = 4*i+0; 
        discT(i, 1) = 4*i+1; 
        discT(i, 2) = 4*i+2; 
        discT(i, 3) = 4*i+3;
    }
}

void famu::discontinuousV(Store& store){
    //discV.resize(4*mT.rows(), 3);
    Eigen::VectorXd DAx = store._D*store.S*store.x0;
    Eigen::VectorXd Ax = store.S*(store.Y*store.x+store.x0);
    Eigen::SparseMatrix<double, Eigen::RowMajor> M;
    famu::dFMatrix_Vector_Swap(M, DAx);
    Eigen::VectorXd CAx = store.C*store.S*(store.Y*store.x+store.x0);
    

    std::cout<<"M: "<<M.rows()<<" "<<M.cols()<<"\n";
    std::cout<<"PF: "<<store.ProjectF.rows()<<" "<<store.ProjectF.cols()<<"\n";
    std::cout<<"dFVect: "<<store.dFvec.rows()<<" "<<store.dFvec.cols()<<"\n";
    std::cout<<"CAx: "<<CAx.rows()<<" "<<CAx.cols()<<"\n";
    
    Eigen::VectorXd newx = M*store.ProjectF*store.dFvec;
    
    for(int t =0; t<store.T.rows(); t++){
        store.discV(4*t+3, 0) = Ax[12*t+9];
        store.discV(4*t+3, 1) = Ax[12*t+10];
        store.discV(4*t+3, 2) = Ax[12*t+11];
        store.discV(4*t+0, 0) = newx[9*t+0]+Ax[12*t+9];
        store.discV(4*t+0, 1) = newx[9*t+1]+Ax[12*t+10];
        store.discV(4*t+0, 2) = newx[9*t+2]+Ax[12*t+11];
        store.discV(4*t+1, 0) = newx[9*t+3]+Ax[12*t+9];
        store.discV(4*t+1, 1) = newx[9*t+4]+Ax[12*t+10];
        store.discV(4*t+1, 2) = newx[9*t+5]+Ax[12*t+11];
        store.discV(4*t+2, 0) = newx[9*t+6]+Ax[12*t+9];
        store.discV(4*t+2, 1) = newx[9*t+7]+Ax[12*t+10];
        store.discV(4*t+2, 2) = newx[9*t+8]+Ax[12*t+11];
    }
}

// void famu::discontinuousV(Store& store){
//     //discV.resize(4*mT.rows(), 3);
//     Eigen::VectorXd DAx = store._D*store.S*(store.Y*store.x+store.x0);
//     Eigen::SparseMatrix<double, Eigen::RowMajor> M;
//     famu::dFMatrix_Vector_Swap(M, DAx);
//     Eigen::VectorXd CAx = store.C*store.S*(store.Y*store.x+store.x0);
//     Eigen::VectorXd newx = M*store.ProjectF*store.dFvec+ CAx;

// 	for(int t =0; t<store.T.rows(); t++){
//         store.discV(4*t+0, 0) = newx[12*t+0];
//         store.discV(4*t+0, 1) = newx[12*t+1];
//         store.discV(4*t+0, 2) = newx[12*t+2];
//         store.discV(4*t+1, 0) = newx[12*t+3];
//         store.discV(4*t+1, 1) = newx[12*t+4];
//         store.discV(4*t+1, 2) = newx[12*t+5];
//         store.discV(4*t+2, 0) = newx[12*t+6];
//         store.discV(4*t+2, 1) = newx[12*t+7];
//         store.discV(4*t+2, 2) = newx[12*t+8];
//         store.discV(4*t+3, 0) = newx[12*t+9];
//         store.discV(4*t+3, 1) = newx[12*t+10];
//         store.discV(4*t+3, 2) = newx[12*t+11];
//     }
// }