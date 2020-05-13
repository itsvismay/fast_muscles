#include "setup_hessian_modes.h"
#include <iostream>
#include <igl/writeDMAT.h>
#include <unsupported/Eigen/KroneckerProduct>
#include <json.hpp>

#include <igl/cotmatrix.h>

using namespace Eigen;
using namespace std;

void exact::setup_hessian_modes(SparseMatrix<double>& A, MatrixXd& mG, VectorXd& eigenvalues, int nummodes){
        nummodes = std::min(nummodes, (int)A.cols());

        cout<<"+EIG SOLVE"<<endl;

        cout<<"     eig1"<<endl;
        //Spectra seems to freak out if you use row storage, this copy just ensures everything is setup the way the solver likes
        Spectra::SparseSymMatProd<double>Aop(A);
        Spectra::SymEigsSolver<double, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<double> > geigs(&Aop, nummodes, std::min(5*nummodes, (int)A.rows()) );
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
        mG = evsCorrected;
        eigenvalues = eigsCorrected;
        cout<<"-EIG SOLVE"<<endl;
        return;     
}