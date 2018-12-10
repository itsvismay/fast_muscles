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
#include <igl/boundary_conditions.h>
#include <igl/lbs_matrix.h>
#include <igl/bbw.h>
#include <json.hpp>
#include <unsupported/Eigen/MatrixFunctions>




using namespace Eigen;
using namespace std;
using json = nlohmann::json;

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

    VectorXi melemType, mr_elem_cluster_map, ms_handles_ind;
    VectorXd mcontx, mdiscx, mx, mx0, mred_s, melemYoungs, melemPoissons, mu, mred_r, mred_x, mred_w;
    MatrixXd mR, mG, msW;
    VectorXd mmass_diag;

    std::vector<int> mfix, mmov;
    std::map<int, std::vector<int>> mr_cluster_elem_map;
    std::vector<SparseMatrix<double>> mRotationBLOCK;
    //end

    

public:
    Mesh(){}

    Mesh(MatrixXi& iT, MatrixXd& iV, std::vector<int>& ifix, std::vector<int>& imov, json& j_input){
        mV = iV;
        mT = iT;
        mfix = ifix;
        mmov = imov;

        double youngs = j_input["youngs"];
        double poissons = j_input["poissons"];
        int num_modes = j_input["number_modes"];
        int nsh = j_input["number_skinning_handles"];
        int nrc = j_input["number_rot_clusters"];

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

        setupModes(num_modes);
        setupRotationClusters(nrc);
        setupSkinningHandles(nsh);
        
        mu.resize(4*mT.rows());
        mred_x.resize(mG.cols());
        mred_x.setZero();
        

        mGR.resize(12*mT.rows(), 12*mT.rows());
        mGU.resize(12*mT.rows(), 12*mT.rows());
        mGS.resize(12*mT.rows(), 12*mT.rows());
        setGlobalF(true, true, true);
    }

    void setupModes(int nummodes){
        if(nummodes==0){
            //For now, no modes, just use G = Id
            mG = MatrixXd::Identity(3*mV.rows(), 3*mV.rows());
            return;
        }

        print("+EIG SOLVE");
        SparseMatrix<double> K = (mP*mA).transpose()*mP*mA;
        Spectra::SparseGenMatProd<double> Aop(K);
        SparseMatrix<double> M(3*mV.rows(), 3*mV.rows());
        for(int i=0; i<mmass_diag.size(); i++){
            M.coeffRef(i,i) = mmass_diag[i];
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
        print("+ Rotation Clusters");
        if(nrc==0){
            //unreduced
            nrc = mT.rows();
        }

        mr_elem_cluster_map.resize(mT.rows());
        if(nrc==mT.rows()){
            //unreduced
            for(int i=0; i<mT.rows(); i++){
                mr_elem_cluster_map[i] = i;
            }   
        }else{
            kmeans_rotation_clustering(mr_elem_cluster_map, nrc); //output from kmeans_rotation_clustering
        }

        for(int i=0; i<mT.rows(); i++){
            mr_cluster_elem_map[mr_elem_cluster_map[i]].push_back(i);
        }

        mred_r.resize(9*nrc);
        for(int i=0; i<nrc; i++){
            mred_r[9*i+0] = 1;
            mred_r[9*i+1] = 0;
            mred_r[9*i+2] = 0;
            mred_r[9*i+3] = 0;
            mred_r[9*i+4] = 1;
            mred_r[9*i+5] = 0;
            mred_r[9*i+6] = 0;
            mred_r[9*i+7] = 0;
            mred_r[9*i+8] = 1;
        }
        mred_w.resize(3*nrc);
        mred_w.setZero();


        for(int c=0; c<nrc; c++){
            vector<int> notfix = mr_cluster_elem_map[c];
            SparseMatrix<double> bo(mT.rows(), notfix.size());
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

            SparseMatrix<double> Id12(12,12);
            Id12.setIdentity();
            SparseMatrix<double> b = Eigen::kroneckerProduct(bo, Id12);
            mRotationBLOCK.push_back(b);
        }
        print("- Rotation Clusters");
    }

    void setupSkinningHandles(int nsh){
        print("+ Skinning Handles");
        if(nsh==0){
            nsh = mT.rows();
        }

        mred_s.resize(6*nsh);
        for(int i=0; i<nsh; i++){
            mred_s[6*i+0] = 1; 
            mred_s[6*i+1] = 1; 
            mred_s[6*i+2] = 1; 
            mred_s[6*i+3] = 0; 
            mred_s[6*i+4] = 0; 
            mred_s[6*i+5] = 0;
        }

        if(nsh==mT.rows()){
            //use BBW skinning, but for now, set by hand
            MatrixXd tW = MatrixXd::Identity(mT.rows(), nsh);
            msW = Eigen::kroneckerProduct(tW, MatrixXd::Identity(6,6));
            print("- Skinning Handles");
            return;
        }
        VectorXi skinning_elem_cluster_map;
        std::map<int, std::vector<int>> skinning_cluster_elem_map;
        kmeans_rotation_clustering(skinning_elem_cluster_map, nsh);
        for(int i=0; i<mT.rows(); i++){
            skinning_cluster_elem_map[mr_elem_cluster_map[i]].push_back(i);
        }
        ms_handles_ind.resize(nsh);
        VectorXd CAx0 = mC*mA*mx0;
        for(int k=0; k<nsh; k++){
            vector<int> els = skinning_cluster_elem_map[k];
            VectorXd centx = VectorXd::Zero(els.size());
            VectorXd centy = VectorXd::Zero(els.size());
            VectorXd centz = VectorXd::Zero(els.size());
            Vector3d avg_cent;

            for(int i=0; i<els.size(); i++){
                centx[i] = CAx0[12*els[i]];
                centy[i] = CAx0[12*els[i]+1];
                centz[i] = CAx0[12*els[i]+2];
            }
            avg_cent<<centx.sum()/centx.size(), centy.sum()/centy.size(),centz.sum()/centz.size();
            int minind = els[0];
            double mindist = (avg_cent - Vector3d(centx[0],centy[0],centz[0])).norm();
            for(int i=1; i<els.size(); i++){
                double dist = (avg_cent - Vector3d(centx[i], centy[i], centz[i])).norm();
                if(dist<mindist){
                    mindist = dist;
                    minind = els[i];
                }
            }
            ms_handles_ind[k] = minind;
        }
        msW = bbw_strain_skinning_matrix(ms_handles_ind);
        print("- Skinning Handles");
    }

    MatrixXd bbw_strain_skinning_matrix(VectorXi& handles){
        std::set<int> unique_vertex_handles;
        std::set<int>::iterator it;
        for(int i=0; i<handles.size(); i++){
            unique_vertex_handles.insert(mT(handles[i], 0));
            unique_vertex_handles.insert(mT(handles[i], 1));
            unique_vertex_handles.insert(mT(handles[i], 2));
            unique_vertex_handles.insert(mT(handles[i], 3));
        }

        int i=0;
        it = unique_vertex_handles.end();
        VectorXi map_verts_to_unique_verts = VectorXi::Zero(*(--it)+1).array() -1;
        for (it=unique_vertex_handles.begin(); it!=unique_vertex_handles.end(); ++it){
            map_verts_to_unique_verts[*it] = i;
            i++;
        }

        MatrixXi vert_to_tet = MatrixXi::Zero(handles.size(), 4);
        i=0;
        for(i=0; i<handles.size(); i++){
            vert_to_tet.row(i)[0] = map_verts_to_unique_verts[mT.row(handles[i])[0]];
            vert_to_tet.row(i)[1] = map_verts_to_unique_verts[mT.row(handles[i])[1]];
            vert_to_tet.row(i)[2] = map_verts_to_unique_verts[mT.row(handles[i])[2]];
            vert_to_tet.row(i)[3] = map_verts_to_unique_verts[mT.row(handles[i])[3]];
        }
        
        MatrixXd C = MatrixXd::Zero(unique_vertex_handles.size(), 3);
        VectorXi P = VectorXi::Zero(unique_vertex_handles.size());
        i=0;
        for (it=unique_vertex_handles.begin(); it!=unique_vertex_handles.end(); ++it){
            C.row(i) = mV.row(*it);
            P(i) = i;
            i++;
        }

        // List of boundary indices (aka fixed value indices into VV)
        VectorXi b;
        // List of boundary conditions of each weight function
        MatrixXd bc;
        igl::boundary_conditions(mV, mT, C, P, MatrixXi(), MatrixXi(), b, bc);
        // compute BBW weights matrix
        igl::BBWData bbw_data;
        // only a few iterations for sake of demo
        bbw_data.active_set_params.max_iter = 8;
        bbw_data.verbosity = 2;
        
        MatrixXd W, M;
        if(!igl::bbw(mV, mT, b, bc, bbw_data, W))
        {
            print("EXIT: Error here");
            exit(0);
            return MatrixXd();
        }

        // Normalize weights to sum to one
        igl::normalize_row_sums(W,W);
        // precompute linear blend skinning matrix
        igl::lbs_matrix(mV,W,M);

        MatrixXd tW = MatrixXd::Zero(mT.rows(), handles.size());
        for(int t =0; t<mT.rows(); t++){
            VectorXi e = mT.row(t);
            for(int h=0; h<handles.size(); h++){
                if(t==handles[h]){
                    tW.row(t) *= 0;
                    tW(t,h) = 1;
                    break;
                }
                double p0 = 0;
                double p1 = 0;
                double p2 = 0;
                double p3 = 0;
                for(int j=0; j<vert_to_tet.cols(); ++j){
                    p0 += W(e[0], vert_to_tet(h, j));
                    p1 += W(e[1], vert_to_tet(h, j));
                    p2 += W(e[2], vert_to_tet(h, j));
                    p3 += W(e[3], vert_to_tet(h, j));
                }
                tW(t, h) = (p0+p1+p2+p3)/4;  
            }
        }
        igl::normalize_row_sums(tW, tW);

        MatrixXd Id6 = MatrixXd::Identity(6, 6);
        return Eigen::kroneckerProduct(tW, Id6);
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
        kmeans(Data, clusters, 1000, Centroids, idx);
    }

    void kmeans(const Eigen::MatrixXd& F, const int num_labels, const int num_iter, Eigen::MatrixXd& D, Eigen::VectorXi& labels){ 
        // const Eigen::MatrixXd& F, //data. Every column is a feature
        // const int num_labels, // number of clusters
        // const int num_iter, // number of iterations
        // Eigen::MatrixXd& D, // dictionary of clusters (every column is a cluster)
        // Eigen::VectorXi& labels){ // map D to F.
    
        assert(sizeof(float) == 4);
        cv::Mat cv_F(F.rows(), F.cols(), CV_32F);
        for (int i = 0; i < F.rows(); ++i)
            for (int j = 0; j < F.cols(); ++j)
                cv_F.at<float>(i,j) = F(i,j);

        cv::Mat cv_labels;
        cv::Mat cv_centers;
        cv::TermCriteria criteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001);
        cv::kmeans(cv_F, num_labels, cv_labels, criteria, num_iter, cv::KMEANS_PP_CENTERS, cv_centers);

        int num_points = F.rows();
        int num_features = F.cols();
        // D.resize(num_features, num_labels);

        // for (int i=0; i<cv_centers.rows; ++i)
        //     for (int j=0; j<cv_centers.cols; ++j)
        //         D(j,i) = cv_centers.at<float>(i,j);

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
        mmass_diag.resize(3*mV.rows());
        mmass_diag.setZero();

        for(int i=0; i<mT.rows(); i++){
            double undef_vol = get_volume(
                    mV.row(mT.row(i)[0]), 
                    mV.row(mT.row(i)[1]), 
                    mV.row(mT.row(i)[2]), 
                    mV.row(mT.row(i)[3]));

            mmass_diag(3*mT.row(i)[0]+0) += undef_vol/4.0;
            mmass_diag(3*mT.row(i)[0]+1) += undef_vol/4.0;
            mmass_diag(3*mT.row(i)[0]+2) += undef_vol/4.0;

            mmass_diag(3*mT.row(i)[1]+0) += undef_vol/4.0;
            mmass_diag(3*mT.row(i)[1]+1) += undef_vol/4.0;
            mmass_diag(3*mT.row(i)[1]+2) += undef_vol/4.0;

            mmass_diag(3*mT.row(i)[2]+0) += undef_vol/4.0;
            mmass_diag(3*mT.row(i)[2]+1) += undef_vol/4.0;
            mmass_diag(3*mT.row(i)[2]+2) += undef_vol/4.0;

            mmass_diag(3*mT.row(i)[3]+0) += undef_vol/4.0;
            mmass_diag(3*mT.row(i)[3]+1) += undef_vol/4.0;
            mmass_diag(3*mT.row(i)[3]+2) += undef_vol/4.0;
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
            //     for(int j=0; j<4; j++){
            //         mGU.coeffRef(3*j+12*t + 0, 3*j+12*t + 0) = 0;
            //         mGU.coeffRef(3*j+12*t + 0, 3*j+12*t + 1) = -1;
            //         mGU.coeffRef(3*j+12*t + 0, 3*j+12*t + 2) = 0;
            //         mGU.coeffRef(3*j+12*t + 1, 3*j+12*t + 0) = 1;
            //         mGU.coeffRef(3*j+12*t + 1, 3*j+12*t + 1) = 0;
            //         mGU.coeffRef(3*j+12*t + 1, 3*j+12*t + 2) = 0;
            //         mGU.coeffRef(3*j+12*t + 2, 3*j+12*t + 0) = 0;
            //         mGU.coeffRef(3*j+12*t + 2, 3*j+12*t + 1) = 0;
            //         mGU.coeffRef(3*j+12*t + 2, 3*j+12*t + 2) = 1;
            //     }
            // }
            
        }

        if(updateR){
            //iterate through rotation clusters
            Matrix3d ri = Matrix3d::Zero();
            Matrix3d r;

            for (int t=0; t<mred_r.size()/9; t++){
                //dont use the RotBLOCK matrix like I did in python. 
                //Insert manually into elements within
                //the rotation cluster
                ri(0,0) = mred_r[9*t+0];
                ri(0,1) = mred_r[9*t+1];
                ri(0,2) = mred_r[9*t+2];
                ri(1,0) = mred_r[9*t+3];
                ri(1,1) = mred_r[9*t+4];
                ri(1,2) = mred_r[9*t+5];
                ri(2,0) = mred_r[9*t+6];
                ri(2,1) = mred_r[9*t+7];
                ri(2,2) = mred_r[9*t+8];

                // % Rodrigues formula %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                Vector3d w;
                w<<mred_w(3*t+0),mred_w(3*t+1),mred_w(3*t+2);
                double wlen = w.norm();
                if (wlen>1e-9){
                    double wX = w(0);
                    double wY = w(1);
                    double wZ = w(2);
                    Matrix3d cross;
                    cross<<0, -wZ, wY,
                            wZ, 0, -wX,
                            -wY, wX, 0;
                    Matrix3d Rot = cross.exp();
                    r = ri*Rot;
                }else{
                    r = ri;
                }
                // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                for(int c=0; c<mr_cluster_elem_map[t].size(); c++){
                    for(int j=0; j<4; j++){
                        mGR.coeffRef(3*j+12*mr_cluster_elem_map[t][c] + 0, 3*j+12*mr_cluster_elem_map[t][c] + 0) = r(0 ,0);
                        mGR.coeffRef(3*j+12*mr_cluster_elem_map[t][c] + 0, 3*j+12*mr_cluster_elem_map[t][c] + 1) = r(0 ,1);
                        mGR.coeffRef(3*j+12*mr_cluster_elem_map[t][c] + 0, 3*j+12*mr_cluster_elem_map[t][c] + 2) = r(0 ,2);
                        mGR.coeffRef(3*j+12*mr_cluster_elem_map[t][c] + 1, 3*j+12*mr_cluster_elem_map[t][c] + 0) = r(1 ,0);
                        mGR.coeffRef(3*j+12*mr_cluster_elem_map[t][c] + 1, 3*j+12*mr_cluster_elem_map[t][c] + 1) = r(1 ,1);
                        mGR.coeffRef(3*j+12*mr_cluster_elem_map[t][c] + 1, 3*j+12*mr_cluster_elem_map[t][c] + 2) = r(1 ,2);
                        mGR.coeffRef(3*j+12*mr_cluster_elem_map[t][c] + 2, 3*j+12*mr_cluster_elem_map[t][c] + 0) = r(2 ,0);
                        mGR.coeffRef(3*j+12*mr_cluster_elem_map[t][c] + 2, 3*j+12*mr_cluster_elem_map[t][c] + 1) = r(2 ,1);
                        mGR.coeffRef(3*j+12*mr_cluster_elem_map[t][c] + 2, 3*j+12*mr_cluster_elem_map[t][c] + 2) = r(2 ,2);
                    }
                }
            }
        }

        if(updateS){
            VectorXd ms = msW*mred_s;
            //iterate through skinning handles
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
        // print(mGS);
        mGF = mGR*mGU*mGS*mGU.transpose();
    }

    std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> & M){
        std::vector<Eigen::Triplet<double>> v;
        for(int i = 0; i < M.outerSize(); i++)
            for(typename Eigen::SparseMatrix<double>::InnerIterator it(M,i); it; ++it)
                v.emplace_back(it.row(),it.col(),it.value());
        return v;
    }

    double get_volume(Vector3d p1, Vector3d p2, Vector3d p3, Vector3d p4){
        Matrix3d Dm;
        Dm.col(0) = p1 - p4;
        Dm.col(1) = p2 - p4;
        Dm.col(2) = p3 - p4;
        double density = 1000;
        double m_undeformedVol = (1.0/6)*fabs(Dm.determinant());
        return m_undeformedVol;
    }

    MatrixXd& V(){ return mV; }
    MatrixXi& T(){ return mT; }
    SparseMatrix<double>& GR(){ return mGR; }
    SparseMatrix<double>& GS(){ return mGS; }
    SparseMatrix<double>& GU(){ return mGU; }
    SparseMatrix<double>& GF(){ return mGF; }

    SparseMatrix<double>& P(){ return mP; }
    SparseMatrix<double>& A(){ return mA; }
    MatrixXd& G(){ return mG; }

    SparseMatrix<double>& B(){ return mFree; }
    SparseMatrix<double>& AB(){ return mConstrained; }
    VectorXd& red_r(){ return mred_r; }
    VectorXd& red_w(){ return mred_w; }
    VectorXd& eYoungs(){ return melemYoungs; }
    VectorXd& ePoissons(){ return melemPoissons; }
    VectorXd& x0(){ return mx0; }
    VectorXd& red_s(){return mred_s; }
    std::map<int, std::vector<int>>& r_cluster_elem_map(){ return mr_cluster_elem_map; }
    MatrixXd& sW(){ return msW; }
    std::vector<SparseMatrix<double>>& RotBLOCK(){ return mRotationBLOCK; }

    VectorXd& dx(){ return mx;}
    VectorXd& red_x(){ return mred_x; }
    void red_x(VectorXd& ix){ 
        mred_x = ix;
        return; 
    }


    void dx(VectorXd& ix){ 
        mx = ix;
        return;
    }


    VectorXd& xbar(){
        mdiscx = mGF*mP*mA*mx0;
        return mdiscx;
    }

    MatrixXd continuousV(){
        VectorXd x = mG*mred_x + mx0;
        Eigen::Map<Eigen::MatrixXd> newV(x.data(), mV.cols(), mV.rows());
        return newV.transpose();
    }

    MatrixXi& discontinuousT(){ return discT; }

    MatrixXd& discontinuousV(){
        VectorXd x = mG*mred_x + mx0;
        VectorXd dx = xbar();
        VectorXd CAx = mC*mA*x;
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

    SparseMatrix<double>& M(){ return mMass; }

    template<class T>
    void print(T a){ std::cout<<a<<std::endl; }

};

#endif