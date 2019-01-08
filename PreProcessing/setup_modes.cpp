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

void setup_modes(int nummodes, bool reduced, SparseMatrix<double>& mP, SparseMatrix<double>& mA, SparseMatrix<double> mConstrained, MatrixXd& mV, VectorXd& mmass_diag, MatrixXd& mG){
        if(nummodes==0 && reduced==false){
            //Unreduced just dont use G
            return;
        }
        if(nummodes==0){
            //reduced, but no modes
            cout<<"reduced, but no modes?"<<endl;
            //mG = MatrixXd::Identity(3*mV.rows(), 3*mV.rows());
            return;
        }

        cout<<"+EIG SOLVE"<<endl;
        SparseMatrix<double> K = (mP*mA).transpose()*mP*mA;
        SparseMatrix<double> M(3*mV.rows(), 3*mV.rows());
        for(int i=0; i<mmass_diag.size(); i++){
            M.coeffRef(i,i) = mmass_diag[i];
        }
        cout<<"     eig1"<<endl;
        Eigen::SparseMatrix<double> K1 = K;
        Eigen::SparseMatrix<double> M1 = M;
        Spectra::SparseSymMatProd<double>Aop(K1);
        Spectra::SparseCholesky<double> Bop(M1);
        Spectra::SymGEigsSolver<double, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY>geigs(&Aop, &Bop, nummodes, 5*nummodes);
        geigs.init();
        cout<<"     eig2"<<endl;
        int nconv = geigs.compute();
        cout<<"     eig3"<<endl;
        VectorXd eigenvalues;
        MatrixXd eigenvectors;
        if(geigs.info() == Spectra::SUCCESSFUL)
        {
            eigenvalues = geigs.eigenvalues();
            eigenvectors = geigs.eigenvectors();
            MatrixXd Kdense = MatrixXd(K);
            igl::writeDMAT( "Kmat.dmat", Kdense);
            igl::writeDMAT( "Mmat.dmat", MatrixXd(M));
            igl::writeDMAT( "readEigs.dmat", eigenvalues);
            igl::writeDMAT( "readVecs.dmat", eigenvectors);
        }
        else
        {
            cout<<"EIG SOLVE FAILED: "<<endl<<geigs.info()<<endl;
            exit(0);
        }
        cout<<"     eig4"<<endl;
        // eigenvalues.head(eigenvalues.size() - 3));
        MatrixXd eV = eigenvectors.leftCols(eigenvalues.size() -3);
        mG = eigenvectors;//.leftCols(eigenvalues.size() -3);
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
        SparseLU<SparseMatrix<double>> solver;
        solver.compute(HandleModesKKTmat);
        SparseMatrix<double> eHsparse = solver.solve(eHconstrains);
        cout<<"     eig7"<<endl;
        MatrixXd eH = MatrixXd(eHsparse).topRows(K.rows());
        cout<<"-ModesForHandles"<<endl;
        //###############QR get orth basis of Modes, eH#######
        MatrixXd eHeV(eH.rows(), eH.cols()+eV.cols());
        eHeV<<eH,eV;
        HouseholderQR<MatrixXd> QR(eHeV);
        cout<<"     eig8"<<endl;
        MatrixXd thinQ = MatrixXd::Identity(eHeV.rows(), eHeV.cols());
        //SET Q TO G
        mG = QR.householderQ()*thinQ; 
        cout<<"     eig9"<<endl;       
}