#ifndef MESH 
#define MESH

#include <Eigen/CholmodSupport>
#include <unsupported/Eigen/KroneckerProduct>


using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;
typedef Matrix<double, 12, 1> Vector12d;

class Mesh
{

protected:
    MatrixXd mV;
    MatrixXi mT;

    //Used in the sim
    SparseMatrix<double> mMass, mFree, mConstrain, 
                        GF, GR, GS, GU, mP;
    SparseMatrix<int> mA, mC;

    VectorXi mfix, mmov, melemType;
    VectorXd contx, mx, mx0, ms, melemYoungs, melemPoissons;
    //end

    

public:
    Mesh(){}

    Mesh(MatrixXi& iT, MatrixXd& iV, VectorXi& ifix, VectorXi& imov){
        mV = iV;
        mT = iT;
        mfix = ifix;
        mmov = imov;

        double youngs = 600000;
        double poissons = 0.45;
        double mu = youngs/(2+ 2*poissons);
        double lambda = youngs*poissons/((1+poissons)*(1-2*poissons));
        
        mx0.resize(mV.cols()*mV.rows());
        mx.resize(mV.cols()*mV.rows());

        mA.resize(12*mT.rows(), 3*mV.rows());
        mP.resize(12*mT.rows(), 12*mT.rows());

        #pragma omp parallel for
        for(int i=0; i<mV.rows(); i++){
            mx0[3*i+0] = mV(i,0); mx0[3*i+1] = mV(i,1); mx0[3*i+2] = mV(i,2);   
        }
        mx.setZero();


        ms.resize(6*mT.rows());
        melemYoungs.resize(12*mT.rows());
        melemPoissons.resize(12*mT.rows());
        #pragma omp parallel for
        for(int i=0; i<mT.rows(); i++){
            ms[i+0] = 1;
            ms[i+1] = 1;
            ms[i+2] = 1;
            ms[i+3] = 0;
            ms[i+4] = 0;
            ms[i+5] = 0;
        }

        setP();
    }  
        
    void setP(){
        Matrix4d p;
        p<< 3, -1, -1, -1,
            -1, 3, -1, -1,
            -1, -1, 3, -1,
            -1, -1, -1, 3;

        MatrixXd subP = Eigen::kroneckerProduct(p/4.0, MatrixXd::Identity(3,3));
        SparseMatrix<double> Id(mT.rows(), mT.rows());
        Id.setIdentity();
        mP = Eigen::kroneckerProduct(Id, subP);
        std::cout<<mP<<std::endl;
        std::cout<<mP.nonZeros()<<std::endl;
    }

    void setA(){
        vector<Trip> triplets;
        triplets.reserve(12*mT.rows());

        for(int i=0; i<mT.rows(); i++){
            VectorXi inds = mT.row(i);

            for(int j=0; j<4; j++){
                int v = inds[j];
                triplets.push_back(Trip(12*i+3*j+0, 3*v+0, 1));
                triplets.push_back(Trip(12*i+3*j+1, 3*v+1, 1));
                triplets.push_back(Trip(12*i+3*j+2, 3*v+2, 1));
            }
        }
        mA.setFromTriplets(triplets.begin(), triplets.end());
    }

    void setVertexWiseMassMatrix(){

    }

    inline MatrixXd& V(){ return mV; }
    
    inline MatrixXi& T(){ return mT; }

    inline VectorXd& x(){ 
        contx = mx+mx0;
        return contx;
    }

    inline SparseMatrix<double>& M(){ return mMass;}

};

#endif