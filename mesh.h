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
    MatrixXd mV, discV;
    MatrixXi mT, discT;

    //Used in the sim
    SparseMatrix<double> mMass, mGF, mGR, mGS, mGU, mP, mC;
    SparseMatrix<double> mA, mFree, mConstrained;

    VectorXi melemType;
    VectorXd contx, discx, mx, mx0, ms, melemYoungs, melemPoissons, mu, mr;
    MatrixXd mR;
    std::vector<int> mfix, mmov;
    //end

    

public:
    Mesh(){}

    Mesh(MatrixXi& iT, MatrixXd& iV, std::vector<int>& ifix, std::vector<int>& imov){
        mV = iV;
        mT = iT;
        mfix = ifix;
        mmov = imov;

        double youngs = 600000;
        double poissons = 0.45;
        // double mu = youngs/(2+ 2*poissons);
        // double lambda = youngs*poissons/((1+poissons)*(1-2*poissons));
        
        mx0.resize(mV.cols()*mV.rows());
        mx.resize(mV.cols()*mV.rows());

        #pragma omp parallel for
        for(int i=0; i<mV.rows(); i++){
            mx0[3*i+0] = mV(i,0); mx0[3*i+1] = mV(i,1); mx0[3*i+2] = mV(i,2);   
        }
        mx.setZero();



        setP();
        setA();
        setC();
        setMassMatrix();
        setVertexWiseMassDiag();
        setFreedConstrainedMatrices();
        
        discT.resize(mT.rows(), 4);
        discV.resize(4*mT.rows(), 3);
        ms.resize(6*mT.rows());
        mu.resize(4*mT.rows());
        mr.resize(4*mT.rows());
        melemYoungs.resize(mT.rows());
        melemPoissons.resize(mT.rows());
        #pragma omp parallel for
        for(int i=0; i<mT.rows(); i++){
            ms[6*i+0] = 1; ms[6*i+1] = 1; ms[6*i+2] = 1; ms[6*i+3] = 0; ms[6*i+4] = 0; ms[6*i+5] = 0;
            melemYoungs[i] = youngs;
            melemPoissons[i] = poissons;
            // mu[i+0] = 0; mu[i+1] = 1; mu[i+2] = 0; mu[i+3] = 0;
            // mr[i+0] = 1; mr[i+1] = 0; mr[i+2] = 0; mr[i+3] = 0;
            discT(i, 0) = 4*i+0; discT(i, 1) = 4*i+1; discT(i, 2) = 4*i+2; discT(i, 3) = 4*i+3;
        }
        mGR.resize(12*mT.rows(), 12*mT.rows());
        mGR.setIdentity();
        mR.resize(3*mT.rows(), 3);
        for(int i=0; i<mT.rows(); i++){
            mR.block<3,3>(3*i, 0) = MatrixXd::Identity(3,3);
        }
        mGU.resize(12*mT.rows(), 12*mT.rows());
        mGS.resize(12*mT.rows(), 12*mT.rows());
        mGS.setIdentity();
        setGlobalF(true, true, true);
    }

    void setC(){
        mC.resize(12*mT.rows(), 12*mT.rows());
        Matrix4d inner;
        inner.setOnes();
        SparseMatrix<double> Id3(3, 3);
        Id3.setIdentity();
        SparseMatrix<double> subC = Eigen::kroneckerProduct(inner/4.0, Id3);
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

    void setFreedConstrainedMatrices(){
        if(mmov.size()>0){
            mfix.insert(mfix.end(), mmov.begin(), mmov.end());
        }
        std::sort (mfix.begin(), mfix.end());

        vector<int> notfix;
        mFree.resize(3*mV.rows(), 3*mV.rows() - 3*mfix.size());
        mFree.setZero();

        int i = 0;
        int f = 0;
        for(int j =0; j<mFree.cols()/3; j++){
            if (i==mfix[f]){
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
            notfix.push_back(i);
            mConstrained.coeffRef(3*i+0, 3*j+0) = 1;
            mConstrained.coeffRef(3*i+1, 3*j+1) = 1;
            mConstrained.coeffRef(3*i+2, 3*j+2) = 1; 
            
            i++;
        }
    }

    void setGlobalF(bool updateR, bool updateS, bool updateU){
        if(updateU){
            //TODO: update to be parametrized by input mU
            mGU.setIdentity();
            // for(int t = 0; t<mT.rows(); t++){
            //     GU.block<3,3>() = MatrixXd::Identity(3,3);
            // }
        }

        if(updateR){
            //TODO: update to be parametrized by input mR
            for(int t = 0; t<mT.rows(); t++){
                for(int j=0; j<4; j++){
                    mGR.coeffRef(3*j+12*t + 0, 3*j+12*t + 0) = mR(3*t + 0, 0);
                    mGR.coeffRef(3*j+12*t + 0, 3*j+12*t + 1) = mR(3*t + 0, 1);
                    mGR.coeffRef(3*j+12*t + 0, 3*j+12*t + 2) = mR(3*t + 0, 2);
                    mGR.coeffRef(3*j+12*t + 1, 3*j+12*t + 0) = mR(3*t + 1, 0);
                    mGR.coeffRef(3*j+12*t + 1, 3*j+12*t + 1) = mR(3*t + 1, 1);
                    mGR.coeffRef(3*j+12*t + 1, 3*j+12*t + 2) = mR(3*t + 1, 2);
                    mGR.coeffRef(3*j+12*t + 2, 3*j+12*t + 0) = mR(3*t + 2, 0);
                    mGR.coeffRef(3*j+12*t + 2, 3*j+12*t + 1) = mR(3*t + 2, 1);
                    mGR.coeffRef(3*j+12*t + 2, 3*j+12*t + 2) = mR(3*t + 2, 2);
                }
            }
        }

        if(updateS){
            for(int t = 0; t<mT.rows(); t++){
                for(int j = 0; j<4; j++){
                    mGS.coeffRef(3*j+12*t + 0, 3*j+12*t + 0) = ms[6*t + 0];
                    mGS.coeffRef(3*j+12*t + 0, 3*j+12*t + 1) = ms[6*t + 3];
                    mGS.coeffRef(3*j+12*t + 0, 3*j+12*t + 2) = ms[6*t + 4];
                    mGS.coeffRef(3*j+12*t + 1, 3*j+12*t + 0) = ms[6*t + 3];
                    mGS.coeffRef(3*j+12*t + 1, 3*j+12*t + 1) = ms[6*t + 1];
                    mGS.coeffRef(3*j+12*t + 1, 3*j+12*t + 2) = ms[6*t + 5];
                    mGS.coeffRef(3*j+12*t + 2, 3*j+12*t + 0) = ms[6*t + 4];
                    mGS.coeffRef(3*j+12*t + 2, 3*j+12*t + 1) = ms[6*t + 5];
                    mGS.coeffRef(3*j+12*t + 2, 3*j+12*t + 2) = ms[6*t + 2];
                }
            }
        }

        mGF = mGR*mGU*mGS*mGU.transpose();
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
    inline SparseMatrix<double>& GR(){ return mGR; }
    inline SparseMatrix<double>& GS(){ return mGS; }
    inline SparseMatrix<double>& GU(){ return mGU; }

    inline SparseMatrix<double>& P(){ return mP; }
    inline SparseMatrix<double>& A(){ return mA; }
    inline SparseMatrix<double>& B(){ return mFree; }
    inline SparseMatrix<double>& AB(){ return mConstrained; }
    inline MatrixXd& Rclusters(){ return mR; }
    inline VectorXd& x0(){ return mx0; }
    VectorXd& x(){ 
        contx = mx+mx0;
        return contx;
    }

    inline VectorXd& dx(){ return mx;}

    void dx(VectorXd& ix){ 
        mx = ix;
        return;
    }

    inline VectorXd& s(){
        return ms;
    }

    VectorXd& xbar(){
        discx = mGF*mP*mA*mx0;
        return discx;
    }

    MatrixXd continuousV(){
        Eigen::Map<Eigen::MatrixXd> newV(x().data(), mV.cols(), mV.rows());
        return newV.transpose();
    }

    inline MatrixXi& discontinuousT(){ return discT; }

    MatrixXd& discontinuousV(){
        VectorXd dx = xbar();
        VectorXd CAx = mC*mA*x();
        VectorXd newx = dx + CAx;

        #pragma omp parallel for
        for(int t =0; t<mT.rows(); t++){
            discV(4*t+0, 0) = newx[12*t+0];
            discV(4*t+0, 1) = newx[12*t+1];
            discV(4*t+0, 2) = newx[12*t+2];
            discV(4*t+1, 0) = newx[12*t+3];
            discV(4*t+1, 1) = newx[12*t+4];
            discV(4*t+1, 2) = newx[12*t+5];
            discV(4*t+2, 0) = newx[12*t+6];
            discV(4*t+2, 1) = newx[12*t+7];
            discV(4*t+2, 2) = newx[12*t+8];
            discV(4*t+3, 0) = newx[12*t+9];
            discV(4*t+3, 1) = newx[12*t+10];
            discV(4*t+3, 2) = newx[12*t+11];
        }
        return discV;
    }

    inline SparseMatrix<double>& M(){ return mMass; }

    template<class T>
    inline void print(T a){ std::cout<<a<<std::endl; }

};

#endif