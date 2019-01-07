#include "setup_skinning_handles.h"
#include "kmeans_clustering.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <igl/boundary_conditions.h>
#include <igl/lbs_matrix.h>
#include <igl/bbw.h>
#include <igl/normalize_row_sums.h>
#include <unsupported/Eigen/KroneckerProduct>

using namespace Eigen;
MatrixXd bbw_strain_skinning_matrix(VectorXi& handles, const MatrixXd& mV, const MatrixXi& mT){
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
        std::cout<<"EXIT: Error here"<<std::endl;
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

void setup_skinning_handles(int nsh, bool reduced, const MatrixXi& mT, const MatrixXd& mV, std::vector<VectorXi>& ibones, VectorXi& imuscle,
	SparseMatrix<double>& mC, SparseMatrix<double>& mA, MatrixXd& mG, VectorXd& mx0, 
	VectorXi& ms_handles_ind, VectorXd& mred_s, MatrixXd& msW){
    std::cout<<"+ Skinning Handles"<<std::endl;
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
        std::cout<<"- Unreduced Skinning Handles"<<std::endl;
        std::cout<<"Code should be sparseified"<<std::endl;
        std::cout<<"not setting sW matrix"<<std::endl;
        return;
    }

    VectorXi skinning_elem_cluster_map;
    std::map<int, std::vector<int>> skinning_cluster_elem_map;
    if(nsh==mT.rows()){
        skinning_elem_cluster_map.resize(mT.rows());
        for(int i=0; i<mT.rows(); i++){
            skinning_elem_cluster_map[i] = i;
        }
    }else{
        kmeans_clustering(skinning_elem_cluster_map, nsh, ibones, imuscle, mG, mC, mA, mx0);
    }
    for(int i=0; i<mT.rows(); i++){
        skinning_cluster_elem_map[skinning_elem_cluster_map[i]].push_back(i);
    }

    ms_handles_ind.resize(nsh);
    VectorXd CAx0 = mC*mA*mx0;
    for(int k=0; k<nsh; k++){
        std::vector<int> els = skinning_cluster_elem_map[k];
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
    msW = bbw_strain_skinning_matrix(ms_handles_ind, mV, mT);
    std::cout<<"- Skinning Handles"<<std::endl;
}

