#ifndef MESH 
#define MESH

#include <Eigen/CholmodSupport>
#include <unsupported/Eigen/KroneckerProduct>
#include <MatOp/SparseGenMatProd.h>
#include <MatOp/SparseCholesky.h>
#include <SymGEigsSolver.h>
#include <GenEigsSolver.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


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

    VectorXi melemType, r_elem_cluster_map;
    VectorXd contx, discx, mx, mx0, ms, melemYoungs, melemPoissons, mu, mr;
    MatrixXd mR, mG, msW;
    VectorXd mass_diag;

    std::vector<int> mfix, mmov;
    std::map<int, std::vector<int>> r_cluster_elem_map;
    std::vector<SparseMatrix<int>> RotationBLOCK;
    //end

    

public:
    Mesh(){}

    Mesh(MatrixXi& iT, MatrixXd& iV, std::vector<int>& ifix, std::vector<int>& imov){
        mV = iV;
        mT = iT;
        mfix = ifix;
        mmov = imov;

        double youngs = 60000;
        double poissons = 0.450;

        // double mu = youngs/(2+ 2*poissons);
        // double lambda = youngs*poissons/((1+poissons)*(1-2*poissons));
        
        mx0.resize(mV.cols()*mV.rows());
        mx.resize(mV.cols()*mV.rows());

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
        setElemWiseYoungsPoissons(youngs, poissons);
        setDiscontinuousMeshT();
        
        setupModes(10);
        setupRotationClusters(mT.rows());
        setupSkinningHandles(mT.rows());
        
        
        mu.resize(4*mT.rows());
        

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

    void setupModes(int nummodes){
        if(nummodes==0){
            //For now, no modes, just use G = Id
            mG = MatrixXd::Identity(3*mV.rows(), 3*mV.rows());
        }

        print("+EIG SOLVE");
        SparseMatrix<double> K = (mP*mA).transpose()*mP*mA;
        Spectra::SparseGenMatProd<double> Aop(K);
        SparseMatrix<double> M(3*mV.rows(), 3*mV.rows());
        for(int i=0; i<mass_diag.size(); i++){
            M.coeffRef(i,i) = mass_diag[i];
        }

        Spectra::SparseCholesky<double> Bop(M);
        Spectra::SymGEigsSolver<double, Spectra::SMALLEST_MAGN, Spectra::SparseGenMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY>geigs(&Aop, &Bop, nummodes, M.rows());
        geigs.init();
        int nconv = geigs.compute();
        VectorXd eigenvalues;
        MatrixXd eigenvectors;
        if(geigs.info() == Spectra::SUCCESSFUL)
        {
            eigenvalues = geigs.eigenvalues();
            eigenvectors = geigs.eigenvectors();
        }
        else
        {
            cout<<"EIG SOLVE FAILED: "<<endl<<geigs.info()<<endl;
            exit(0);
        }
        // eigenvalues.head(eigenvalues.size() - 3));
        MatrixXd eV = eigenvectors.leftCols(eigenvalues.size() -3);
        print("-EIG SOLVE");

        //############handle modes KKT solve#####
        print("+ModesForHandles");
        SparseMatrix<double> C = mConstrained.transpose();
        SparseMatrix<double> HandleModesKKTmat(K.rows()+C.rows(), K.rows()+C.rows());
        HandleModesKKTmat.setZero();
        std::vector<Trip> KTrips = to_triplets(K);
        std::vector<Trip> CTrips = to_triplets(C);
        for(int i=0; i<CTrips.size(); i++){
            int row = CTrips[i].row();
            int col = CTrips[i].col();
            int val = CTrips[i].value();
            KTrips.push_back(Trip(row+K.rows(), col, val));
            KTrips.push_back(Trip(col, row+K.cols(), val));
        }
        KTrips.insert(KTrips.end(),CTrips.begin(), CTrips.end());
        HandleModesKKTmat.setFromTriplets(KTrips.begin(), KTrips.end());
        
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
        MatrixXd eH = MatrixXd(eHsparse).topRows(K.rows());
        print("-ModesForHandles");
        
        //###############QR get orth basis of Modes, eH#######
        MatrixXd eHeV(eH.rows(), eH.cols()+eV.cols());
        eHeV<<eH,eV;
        HouseholderQR<MatrixXd> QR(eHeV);
        MatrixXd thinQ = MatrixXd::Identity(eHeV.rows(), eHeV.cols());
        //SET Q TO G
        mG = QR.householderQ()*thinQ;        

    }

    void setupRotationClusters(int nrc){
        r_elem_cluster_map.resize(mT.rows());
            // for(int i=0; i<mT.rows() ;i++){
            //     //Delete once rotation clusters works
            //     r_elem_cluster_map[i] = i;
            // }
        kmeans_rotation_clustering(r_elem_cluster_map, nrc); //output from kmeans_rotation_clustering
        
        for(int i=0; i<mT.rows(); i++){
            r_cluster_elem_map[r_elem_cluster_map[i]].push_back(i);
        }

        mr.resize(9*nrc);
        for(int i=0; i<nrc; i++){
            mr[9*i+0] = 1;
            mr[9*i+1] = 0;
            mr[9*i+2] = 0;
            mr[9*i+3] = 0;
            mr[9*i+4] = 1;
            mr[9*i+5] = 0;
            mr[9*i+6] = 0;
            mr[9*i+7] = 0;
            mr[9*i+8] = 1;
        }


        for(int c=0; c<nrc; c++){
            vector<int> notfix = r_cluster_elem_map[c];
            SparseMatrix<int> bo(mT.rows(), notfix.size());
            bo.setZero();

            int i = 0;
            int f = 0;
            for(int j =0; j<bo.cols(); j++){
                if (i==notfix[f]){
                    bo.coeffRef(i, j) = 1;
                    f++;
                    i++;
                    continue;
                }
                j--;
                i++;
            }

            SparseMatrix<int> Id12(12,12);
            Id12.setIdentity();
            SparseMatrix<int> b = Eigen::kroneckerProduct(bo, Id12);
            // for(int q=0; q<notfix.size();q++)
            //     print(notfix[q]);
            // print(b);
            RotationBLOCK.push_back(b);
        }
    }

    void setupSkinningHandles(int nsh){
        ms.resize(6*nsh);

        for(int i=0; i<nsh; i++){
            ms[6*i+0] = 1; ms[6*i+1] = 1; ms[6*i+2] = 1; ms[6*i+3] = 0; ms[6*i+4] = 0; ms[6*i+5] = 0;
        }

        //use BBW skinning, but for now, set by hand
        MatrixXd tW = MatrixXd::Identity(mT.rows(), nsh);
        msW = Eigen::kroneckerProduct(tW, MatrixXd::Identity(6,6));
    }

    void kmeans_rotation_clustering(VectorXi& idx, int clusters){
        MatrixXd G = mG.colwise() + mx0;
        // std::cout<<mC.rows()<<", "<<mC.cols()<<std::endl;
        // std::cout<<mA.rows()<<", "<<mA.cols()<<std::endl;
        // std::cout<<G.rows()<<", "<<G.cols()<<std::endl;
        // std::cout<<mT.rows()<<std::endl;
        Matrix<double, Dynamic, Dynamic, RowMajor> CAG = mC*mA*G;

        MatrixXd Data = MatrixXd::Zero(mT.rows(), 3*G.cols());
        for(int i=0; i<mT.rows(); i++){
            RowVectorXd r1 = CAG.row(12*i);
            RowVectorXd r2 = CAG.row(12*i+1);
            RowVectorXd r3 = CAG.row(12*i+2);
            RowVectorXd point(3*G.cols());
            point<<r1,r2,r3;
            Data.row(i) = point;
        }
        MatrixXd Centroids;
        kmeans(Data, clusters, 100, Centroids, idx);
        print(idx.transpose());
        print(idx.size());
        exit(0);
    }

    void kmeans(const Eigen::MatrixXd& F, //data. Every column is a feature
                const int num_labels, // number of clusters
                const int num_iter, // number of iterations
                Eigen::MatrixXd& D, // dictionary of clusters (every column is a cluster)
                Eigen::VectorXi& labels) // map D to F.
    {
        assert(sizeof(float) == 4);
        cv::Mat cv_F(F.rows(), F.cols(), CV_32F);
        for (int i = 0; i < F.rows(); ++i)
            for (int j = 0; j < F.cols(); ++j)
                cv_F.at<float>(i,j) = F(i,j);

        cv::Mat cv_labels;
        cv::Mat cv_centers;
        cv::TermCriteria criteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001);
        cv::kmeans(cv_F, num_labels, cv_labels, criteria, num_iter, cv::KMEANS_RANDOM_CENTERS, cv_centers);

        int num_points = F.rows();
        int num_features = F.cols();
    
        labels.resize(num_points);
        for (int i=0; i<labels.rows(); ++i){
            labels(i) = cv_labels.at<int>(i,0);
        }
    }

    void setElemWiseYoungsPoissons(double youngs, double poissons){
        melemYoungs.resize(mT.rows());
        melemPoissons.resize(mT.rows());
        for(int i=0; i<mT.rows(); i++){
            melemYoungs[i] = youngs; melemPoissons[i] = poissons;
        }
    }

    void setDiscontinuousMeshT(){
        discT.resize(mT.rows(), 4);
        discV.resize(4*mT.rows(), 3);
        for(int i=0; i<mT.rows(); i++){
            discT(i, 0) = 4*i+0; discT(i, 1) = 4*i+1; discT(i, 2) = 4*i+2; discT(i, 3) = 4*i+3;
        }
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
        mass_diag.resize(3*mV.rows());
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

    std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> & M){
        std::vector<Eigen::Triplet<double>> v;
        for(int i = 0; i < M.outerSize(); i++)
            for(typename Eigen::SparseMatrix<double>::InnerIterator it(M,i); it; ++it)
                v.emplace_back(it.row(),it.col(),it.value());
        return v;
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
    inline VectorXd& eYoungs(){ return melemYoungs; }
    inline VectorXd& ePoissons(){ return melemPoissons; }
    inline VectorXd& x0(){ return mx0; }
    inline VectorXd& s(){return ms;}
    
    VectorXd& x(){ 
        contx = mx+mx0;
        return contx;
    }

    inline VectorXd& dx(){ return mx;}

    void dx(VectorXd& ix){ 
        mx = ix;
        return;
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