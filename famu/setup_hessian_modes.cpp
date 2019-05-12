#include "setup_hessian_modes.h"
#include <iostream>
#include <igl/writeDMAT.h>
#include <unsupported/Eigen/KroneckerProduct>
#include <json.hpp>

#include <igl/cotmatrix.h>

using namespace Eigen;
using namespace std;

// double get_volume(Vector3d p1, Vector3d p2, Vector3d p3, Vector3d p4){
//     Matrix3d Dm;
//     Dm.col(0) = p1 - p4;
//     Dm.col(1) = p2 - p4;
//     Dm.col(2) = p3 - p4;
//     double density = 1000;
//     double m_undeformedVol = (1.0/6)*fabs(Dm.determinant());
//     return m_undeformedVol;
// }

// void setVertexWiseMassDiag(Store& store, VectorXd& mmass_diag){
//     mmass_diag.resize(3*store.V.rows());
//     mmass_diag.setZero();

//     for(int i=0; i<store.T.rows(); i++){
//         double undef_vol = get_volume(
//                 store.V.row(store.T.row(i)[0]), 
//                 store.V.row(store.T.row(i)[1]), 
//                 store.V.row(store.T.row(i)[2]), 
//                 store.V.row(store.T.row(i)[3]));

//         mmass_diag(3*store.T.row(i)[0]+0) += undef_vol/4.0;
//         mmass_diag(3*store.T.row(i)[0]+1) += undef_vol/4.0;
//         mmass_diag(3*store.T.row(i)[0]+2) += undef_vol/4.0;

//         mmass_diag(3*store.T.row(i)[1]+0) += undef_vol/4.0;
//         mmass_diag(3*store.T.row(i)[1]+1) += undef_vol/4.0;
//         mmass_diag(3*store.T.row(i)[1]+2) += undef_vol/4.0;

//         mmass_diag(3*store.T.row(i)[2]+0) += undef_vol/4.0;
//         mmass_diag(3*store.T.row(i)[2]+1) += undef_vol/4.0;
//         mmass_diag(3*store.T.row(i)[2]+2) += undef_vol/4.0;

//         mmass_diag(3*store.T.row(i)[3]+0) += undef_vol/4.0;
//         mmass_diag(3*store.T.row(i)[3]+1) += undef_vol/4.0;
//         mmass_diag(3*store.T.row(i)[3]+2) += undef_vol/4.0;
//     }
// }

void famu::setup_hessian_modes(Store& store, SparseMatrix<double>& A, MatrixXd& mG){
        int nummodes = store.jinput["number_modes"];
        nummodes = std::min(nummodes, store.S.cols());

        cout<<"+EIG SOLVE"<<endl;

        cout<<"     eig1"<<endl;
        //Spectra seems to freak out if you use row storage, this copy just ensures everything is setup the way the solver likes
        Spectra::SparseSymMatProd<double>Aop(A);
        Spectra::SymEigsSolver<double, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<double> > geigs(&Aop, nummodes, std::min(5*nummodes, A.rows()) );
        geigs.init();
 
        cout<<"     eig2"<<endl;
        int nconv = geigs.compute();
        if(geigs.info() != Spectra::SUCCESSFUL)
        {
            cout<<"EIG SOLVE FAILED: "<<endl<<geigs.info()<<endl;
            exit(0);
        }

        cout<<"     eig3"<<endl;
        VectorXd eigsCorrected = geigs.eigenvalues();
        MatrixXd evsCorrected = geigs.eigenvectors();
        cout<<eigsCorrected.transpose()<<endl;

        cout<<"     eig4"<<endl;
        // eigenvalues.head(eigenvalues.size() - 3));
        mG = store.NullJ*evsCorrected;
        store.eigenvalues = eigsCorrected;
        //std::string outputfile = store.jinput["output"];
        // igl::writeDMAT(outputfile+"/"+to_string((int)store.jinput["number_modes"])+"modes.dmat", evsCorrected);
        cout<<"-EIG SOLVE"<<endl;
        return;     
}