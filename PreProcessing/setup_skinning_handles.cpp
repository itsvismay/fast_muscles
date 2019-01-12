#include "setup_skinning_handles.h"
#include "kmeans_clustering.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <igl/boundary_conditions.h>
#include <igl/lbs_matrix.h>
#include <igl/bbw.h>
#include <unsupported/Eigen/KroneckerProduct>
#include <igl/slice_into.h>
#include <igl/remove_unreferenced.h>

using namespace Eigen;
using namespace std;
MatrixXd bbw_strain_skinning_matrix(VectorXi& handles, const MatrixXd& mV, MatrixXi& mT){
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
    cout<<"---------0--------"<<endl;
    igl::boundary_conditions(mV, mT, C, P, MatrixXi(), MatrixXi(), b, bc);
    // compute BBW weights matrix
    igl::BBWData bbw_data;
    // only a few iterations for sake of demo
    bbw_data.active_set_params.max_iter = 8;
    bbw_data.verbosity = 2;
    
    MatrixXd W, M;
    cout<<"---------1--------"<<endl;
    if(!igl::bbw(mV, mT, b, bc, bbw_data, W))
    {
        std::cout<<"EXIT: Error here"<<std::endl;
        exit(0);
        return MatrixXd();
    }
    cout<<"---------2--------"<<endl;

    // Normalize weights to sum to one
    igl::normalize_row_sums(W,W);
    cout<<"---------3--------"<<endl;

    // precompute linear blend skinning matrix
    igl::lbs_matrix(mV,W,M);
    cout<<"---------4--------"<<endl;

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
    cout<<"---------5--------"<<endl;
    igl::normalize_row_sums(tW, tW);

    cout<<"---------6--------"<<endl;
    return tW;
}

MatrixXd setup_skinning_helper(int nsh_on_component, bool reduced, MatrixXi& mT, MatrixXd& mV, VectorXi& component,
    SparseMatrix<double>& mC, SparseMatrix<double>& mA, MatrixXd& mG, VectorXd& mx0){
    std::vector<VectorXi> empty_bones = {};

    VectorXi skinning_elem_cluster_map;
    std::map<int, std::vector<int>> skinning_cluster_elem_map;
  
    if(nsh_on_component==component.size()){
        skinning_elem_cluster_map.resize(mT.rows());
        for(int i=0; i<mT.rows(); i++){
            skinning_elem_cluster_map[i] = i;
        }
    }else if(nsh_on_component==1){
        skinning_elem_cluster_map.resize(mT.rows());
        for(int i=0; i<mT.rows(); i++){
            skinning_elem_cluster_map[i] = 0;
        }
    }else{
        std::vector<VectorXi> comp_vector = {component};
        kmeans_clustering(skinning_elem_cluster_map, nsh_on_component, empty_bones, comp_vector, mG, mC, mA, mx0);
    }

     for(int i=0; i<mT.rows(); i++){
        skinning_cluster_elem_map[skinning_elem_cluster_map[i]].push_back(i);
    }

    VectorXi handles_ind = VectorXi::Zero(nsh_on_component);
    VectorXd CAx0 = mC*mA*mx0;
    for(int k=0; k<nsh_on_component; k++){
        std::vector<int> els = skinning_cluster_elem_map[k];
        VectorXd centx = VectorXd::Zero(els.size());
        VectorXd centy = VectorXd::Zero(els.size());
        VectorXd centz = VectorXd::Zero(els.size());
        Vector3d avg_cent;

        for(int i=0; i<els.size(); i++){
            centx[i] = CAx0[12*component[els[i]]];
            centy[i] = CAx0[12*component[els[i]]+1];
            centz[i] = CAx0[12*component[els[i]]+2];
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
        handles_ind[k] = minind;
    }

    return bbw_strain_skinning_matrix(handles_ind, mV, mT);

}

void setup_skinning_handles(int nsh, bool reduced, const MatrixXi& mT, const MatrixXd& mV, std::vector<VectorXi>& ibones, std::vector<VectorXi>& imuscle,
	SparseMatrix<double>& mC, SparseMatrix<double>& mA, MatrixXd& mG, VectorXd& mx0, VectorXd& mred_s, MatrixXd& msW){

    std::cout<<"+ Skinning Handles"<<std::endl;
    if(nsh==0){
        nsh = ibones.size() + imuscle.size();
    }else if(nsh<(ibones.size()+imuscle.size())){//hard coded to 1 muscle for now
        std::cout<<"Too few skinning handles, too many components"<<std::endl;
        exit(0);
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

    if(!reduced){
        std::cout<<"- Unreduced Skinning Handles"<<std::endl;
        std::cout<<"Code should be sparseified"<<std::endl;
        std::cout<<"not setting sW matrix"<<std::endl;
        return;
    }

    //blocked construction of full skinning weights matrix
    MatrixXd sW = MatrixXd::Zero(mT.rows(), nsh);
    sW.setZero();
    cout<<"----------Bone COMPONENTS------------"<<endl;
    int insert_index = 0;
    for(int b=0; b<ibones.size(); b++){
        MatrixXi subT(ibones[b].size(), 4);
        MatrixXi componentT;
        MatrixXd componentV;
        VectorXi J;
        for(int i=0; i<ibones[b].size() ; i++){
            subT.row(i) = mT.row(ibones[b][i]);
        }
        igl::remove_unreferenced(mV, subT, componentV, componentT, J);
        MatrixXd sWi = setup_skinning_helper(1, reduced, componentT, componentV, ibones[b], mC, mA, mG, mx0);
        MatrixXd sWslice = MatrixXd::Zero(mT.rows(), sWi.cols());
        igl::slice_into(sWi , ibones[b], 1, sWslice);
        sW.block(0,insert_index, mT.rows(), sWi.cols()) = sWslice;
        nsh = nsh - 1; //bone skinning handle has been made, so decrease nsh by 1
        insert_index += 1;
    }
    cout<<"----------MUSCLE COMPONENTS------------"<<endl;
    int number_handles_per_muscle = nsh/imuscle.size();
    for(int m=0; m<imuscle.size(); m++){ //through muscle vector
        MatrixXi componentT;
        MatrixXd componentV;
        MatrixXi subT(imuscle[m].size(), 4);
        VectorXi J;
        for(int i=0; i<imuscle[m].size() ; i++){
            subT.row(i) = mT.row(imuscle[m][i]);
        }
        igl::remove_unreferenced(mV, subT, componentV, componentT, J);

        MatrixXd sWi;
        if(m==imuscle.size()-1){
            //Deal with remainder handles
            sWi = setup_skinning_helper(nsh, reduced, componentT, componentV, imuscle[m], mC, mA, mG, mx0);
        }else{
            sWi = setup_skinning_helper(number_handles_per_muscle, reduced, componentT, componentV, imuscle[m], mC, mA, mG, mx0);
        }

        MatrixXd sWslice = MatrixXd::Zero(mT.rows(), sWi.cols());
        igl::slice_into(sWi , imuscle[m], 1, sWslice);
        sW.block(0,insert_index, mT.rows(), sWi.cols()) = sWslice;
        insert_index += number_handles_per_muscle;
        nsh =  nsh - number_handles_per_muscle;
    }
    assert(nsh==0);

    MatrixXd Id6 = MatrixXd::Identity(6, 6);
    msW =  Eigen::kroneckerProduct(sW, Id6);
 
    std::cout<<"- Skinning Handles"<<std::endl;
}

