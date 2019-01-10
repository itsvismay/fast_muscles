#include "setup_modes.h"
#include "to_triplets.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <igl/writeDMAT.h>

#include <MatOp/SparseSymMatProd.h>
#include <MatOp/SparseCholesky.h>
#include <SymGEigsSolver.h>
#include <GenEigsSolver.h>

using namespace Eigen;
using namespace std;
typedef Eigen::Triplet<double> Trip;

void setup_modes(int nummodes, bool reduced, SparseMatrix<double>& mP, SparseMatrix<double>& mA, SparseMatrix<double> mConstrained, SparseMatrix<double> mUnconstrained, SparseMatrix<double> mY, MatrixXd& mV, VectorXd& mmass_diag, MatrixXd& mG){
        if(nummodes==0 && reduced==false){
            //Unreduced just dont use G
            return;
        }
        if(nummodes==0){
            //reduced, but no modes
            cout<<"reduced, but no modes?"<<endl;
            mG = MatrixXd::Identity(3*mV.rows(), 3*mV.rows());
            return;
        }

        cout<<"+EIG SOLVE"<<endl;
        SparseMatrix<double> K = (mP*mA).transpose()*mP*mA;
        SparseMatrix<double> M(3*mV.rows(), 3*mV.rows());
        for(int i=0; i<mmass_diag.size(); i++){
            M.coeffRef(i,i) = mmass_diag[i];
        }


        cout<<"     eig1"<<endl;
        //Spectra seems to freak out if you use row storage, this copy just ensures everything is setup the way the solver likes
        Eigen::SparseMatrix<double> A = mY.transpose()*mUnconstrained*mUnconstrained.transpose()*K*mUnconstrained*mUnconstrained.transpose()*mY;
        Eigen::SparseMatrix<double> B = mY.transpose()*mUnconstrained*mUnconstrained.transpose()*M*mUnconstrained*mUnconstrained.transpose()*mY;

        double shift = 1e-6;
        Eigen::SparseMatrix<double> K1 = A + shift*B;
        Eigen::SparseMatrix<double> M1 = B;
        
        Spectra::SparseSymMatProd<double>Aop(M1);
        Spectra::SparseCholesky<double> Bop(K1);

        Spectra::SymGEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY>geigs(&Aop, &Bop, nummodes, std::min(5*nummodes, A.rows()));
        geigs.init();
        cout<<"     eig2"<<endl;
        int nconv = geigs.compute();
        cout<<"     eig3"<<endl;
 
        VectorXd eigsCorrected;
        MatrixXd evsCorrected; //magnitude of eigenvectors can be wrong in this formulation
        eigsCorrected.resize(geigs.eigenvalues().rows());
        evsCorrected.resize(geigs.eigenvectors().rows(), geigs.eigenvectors().cols());
        if(geigs.info() == Spectra::SUCCESSFUL)
        {
            for(unsigned int ii=0; ii<geigs.eigenvalues().rows(); ++ii) {
                eigsCorrected[ii] = -(static_cast<double>(1)/(geigs.eigenvalues()[ii]) + shift);
                evsCorrected.col(ii) = geigs.eigenvectors().col(ii).normalized();
            }

            igl::writeDMAT(to_string(nummodes)+"simplejoint.dmat", evsCorrected);
        }
        else
        {
            cout<<"EIG SOLVE FAILED: "<<endl<<geigs.info()<<endl;
            exit(0);
        }

        cout<<"     eig4"<<endl;
        // eigenvalues.head(eigenvalues.size() - 3));
        MatrixXd eV = mUnconstrained*mUnconstrained.transpose()*mY*evsCorrected;
        mG = eV;
        return;
        cout<<"-EIG SOLVE"<<endl;

        //############handle modes KKT solve#####
        cout<<"+ModesForHandles"<<endl;
        SparseMatrix<double> C = mConstrained.transpose();
        SparseMatrix<double> HandleModesKKTmat(K.rows()+C.rows(), K.rows()+C.rows());
        HandleModesKKTmat.setZero();
        std::vector<Trip> KTrips = to_triplets(K);
        std::vector<Trip> CTrips = to_triplets(C);
        cout<<"     eig5"<<endl;
        for(int i=0; i<CTrips.size(); i++){
            int row = CTrips[i].row();
            int col = CTrips[i].col();
            int val = CTrips[i].value();
            KTrips.push_back(Trip(row+K.rows(), col, val));
            KTrips.push_back(Trip(col, row+K.cols(), val));
        }
        KTrips.insert(KTrips.end(),CTrips.begin(), CTrips.end());
        HandleModesKKTmat.setFromTriplets(KTrips.begin(), KTrips.end());

        cout<<"     eig6"<<endl;
        SparseMatrix<double>eHconstrains(K.rows()+C.rows(), C.rows());
        eHconstrains.setZero();
        std::vector<Trip> eHTrips;
        for(int i=0; i<C.rows(); i++){
            eHTrips.push_back(Trip(i+K.rows(), i, 1));
        }
        eHconstrains.setFromTriplets(eHTrips.begin(), eHTrips.end());
        
        cout<<"     eig7"<<endl;
        SparseLU<SparseMatrix<double>> solver;
        solver.compute(HandleModesKKTmat);
        SparseMatrix<double> eHsparse = solver.solve(eHconstrains);
        MatrixXd eH = MatrixXd(eHsparse).topRows(K.rows());
        cout<<"-ModesForHandles"<<endl;

        //###############QR get orth basis of Modes, eH#######
        MatrixXd eHeV(eH.rows(), eH.cols()+eV.cols());
        eHeV<<eV,eH;
        igl::writeDMAT("TOQR.dmat", eHeV);
        HouseholderQR<MatrixXd> QR(eHeV);
        cout<<"     eig8"<<endl;
        MatrixXd thinQ = MatrixXd::Identity(eHeV.rows(), eHeV.cols());
        //SET Q TO G
        mG = QR.householderQ()*thinQ; 
        return;
        cout<<"     eig9"<<endl;       
}