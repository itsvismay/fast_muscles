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
                        GF, GR, GS, GU, mP, mC;
    SparseMatrix<int> mA;

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
        setA();
        setC();
        setMassMatrix();
        setVertexWiseMassDiag();
    } 

    void setC(){
        mC.resize(12*mT.rows(), 12*mT.rows());
        Matrix4d inner;
        inner.setOnes();
        SparseMatrix<double> Id3(3, 3);
        Id3.setIdentity();
        SparseMatrix<double> subC = Eigen::kroneckerProduct(inner/4.0, Id3);
        std::cout<<subC<<std::endl;
        SparseMatrix<double> Id(mT.rows(), mT.rows());
        Id.setIdentity();
        mC = Eigen::kroneckerProduct(Id, subC);
    }
 
    void setP(){
        mP.resize(12*mT.rows(), 12*mT.rows());
        Matrix4d p;
        p<< 3, -1, -1, -1,
            -1, 3, -1, -1,
            -1, -1, 3, -1,
            -1, -1, -1, 3;
        SparseMatrix<double> Id3(3, 3);
        Id3.setIdentity();
        SparseMatrix<double> subP = Eigen::kroneckerProduct(p/4.0, Id3);
        SparseMatrix<double> Id(mT.rows(), mT.rows());
        Id.setIdentity();
        mP = Eigen::kroneckerProduct(Id, subP);
    }

    void setA(){
        mA.resize(12*mT.rows(), 3*mV.rows());
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

    void setMassMatrix(){
        mMass.resize(12*mT.rows(), 12*mT.rows());
        vector<Trip> triplets;
        triplets.reserve(2*12*mT.rows());

        for(int i=0; i<mT.rows(); i++){
            double undef_vol = get_volume(
                    mV.row(mT.row(i)[0]), 
                    mV.row(mT.row(i)[1]), 
                    mV.row(mT.row(i)[2]), 
                    mV.row(mT.row(i)[3]));
            
            triplets.push_back(Trip(12*i + 0 , 12*i + 0 , undef_vol/4.0));
            triplets.push_back(Trip(12*i + 1 , 12*i + 1 , undef_vol/4.0));
            triplets.push_back(Trip(12*i + 2 , 12*i + 2 , undef_vol/4.0));

            triplets.push_back(Trip(12*i + 3 , 12*i + 3 , undef_vol/4.0));
            triplets.push_back(Trip(12*i + 4 , 12*i + 4 , undef_vol/4.0));
            triplets.push_back(Trip(12*i + 5 , 12*i + 5 , undef_vol/4.0));

            triplets.push_back(Trip(12*i + 6 , 12*i + 6 , undef_vol/4.0));
            triplets.push_back(Trip(12*i + 7 , 12*i + 7 , undef_vol/4.0));
            triplets.push_back(Trip(12*i + 8 , 12*i + 8 , undef_vol/4.0));

            triplets.push_back(Trip(12*i + 9 , 12*i + 9 , undef_vol/4.0));
            triplets.push_back(Trip(12*i + 10, 12*i + 10, undef_vol/4.0));
            triplets.push_back(Trip(12*i + 11, 12*i + 11, undef_vol/4.0));
        }
        // mMass.resize(12*mT.rows(), 12*mT.row());
        mMass.setFromTriplets(triplets.begin(), triplets.end());
    }

    void setVertexWiseMassDiag(){
        VectorXd mass_diag(3*mV.rows());
        mass_diag.setZero();

        for(int i=0; i<mT.rows(); i++){
            double undef_vol = get_volume(
                    mV.row(mT.row(i)[0]), 
                    mV.row(mT.row(i)[1]), 
                    mV.row(mT.row(i)[2]), 
                    mV.row(mT.row(i)[3]));

            mass_diag(3*mT.row(i)[0]+0) += undef_vol/4.0;
            mass_diag(3*mT.row(i)[0]+1) += undef_vol/4.0;
            mass_diag(3*mT.row(i)[0]+2) += undef_vol/4.0;

            mass_diag(3*mT.row(i)[1]+0) += undef_vol/4.0;
            mass_diag(3*mT.row(i)[1]+1) += undef_vol/4.0;
            mass_diag(3*mT.row(i)[1]+2) += undef_vol/4.0;

            mass_diag(3*mT.row(i)[2]+0) += undef_vol/4.0;
            mass_diag(3*mT.row(i)[2]+1) += undef_vol/4.0;
            mass_diag(3*mT.row(i)[2]+2) += undef_vol/4.0;

            mass_diag(3*mT.row(i)[3]+0) += undef_vol/4.0;
            mass_diag(3*mT.row(i)[3]+1) += undef_vol/4.0;
            mass_diag(3*mT.row(i)[3]+2) += undef_vol/4.0;
        }
    }

    inline double get_volume(Vector3d p1, Vector3d p2, Vector3d p3, Vector3d p4){
        Matrix3d Dm;
        Dm.col(0) = p1 - p4;
        Dm.col(1) = p2 - p4;
        Dm.col(2) = p3 - p4;
        double density = 1000;
        double m_undeformedVol = (1.0/6)*fabs(Dm.determinant());
        return m_undeformedVol;
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