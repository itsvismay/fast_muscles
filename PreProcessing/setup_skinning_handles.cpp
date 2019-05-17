#include "setup_skinning_handles.h"
#include "kmeans_clustering.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <igl/boundary_conditions.h>
#include <igl/lbs_matrix.h>
#include <igl/bbw.h>
#include <igl/harmonic.h>
#include <igl/biharmonic_coordinates.h>
#include <unsupported/Eigen/KroneckerProduct>
#include <igl/slice_into.h>
#include <igl/remove_unreferenced.h>
#include <igl/normalize_row_sums.h>
#include <igl/writeOBJ.h>
#include <igl/list_to_matrix.h>

using namespace Eigen;
using namespace std;
MatrixXd bbw_strain_skinning_matrix(VectorXi& handles, const MatrixXd& mV, MatrixXi& mT){
    std::set<int> unique_vertex_handles;
    std::set<int>::iterator it;
    //from the tet handle indexes, get the unique verts that can act as handles
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
    VectorXi HandleIndexes = VectorXi::Zero(unique_vertex_handles.size());
    i=0;
    for (it=unique_vertex_handles.begin(); it!=unique_vertex_handles.end(); ++it){
        HandleIndexes[i] =  *it;
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
    bbw_data.active_set_params.max_iter = 100;
    bbw_data.verbosity = 2;
    
    MatrixXd W, M;
    cout<<"---------1--------"<<endl;
    if(handles.size()==1){
        return MatrixXd::Ones(mT.rows(), handles.size());
    }

    // if(!igl::bbw(mV, mT, b, bc, bbw_data, W))
    // {
    //     std::cout<<"EXIT: Error here"<<std::endl;
    //     exit(0);
    //     return MatrixXd();
    // }
    // if(!igl::harmonic(mV, mT, b, bc, 1, W))
    // {
    //     std::cout<<"EXIT: Error here"<<std::endl;
    //     exit(0);
    //     return MatrixXd();
    // }

    std::vector<std::vector<int> > S;
    igl::matrix_to_list(HandleIndexes,S);
 
    cout<<"Computing weights for "<<HandleIndexes.size()<<" handles at "<<mV.rows()<<" vertices..."<<endl;
    // Technically k should equal 3 for smooth interpolation in 3d, but 2 is
    // faster and looks OK
    const int k = 3;
    igl::biharmonic_coordinates(mV,mT,S,k,W);






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

Vector3d tet_center(MatrixXi& T, MatrixXd& V, int ind){
    Vector3d v1 = V.row(T.row(ind)[0]);
    Vector3d v2 = V.row(T.row(ind)[1]);
    Vector3d v3 = V.row(T.row(ind)[2]);
    Vector3d v4 = V.row(T.row(ind)[3]);

    return (v1 + v2 + v3 + v4)/4;
}

MatrixXd setup_skinning_helper(int indx, 
    int nsh_on_component, 
    MatrixXi& mT, 
    MatrixXd& mV, 
    SparseMatrix<double>& mC, 
    SparseMatrix<double>& mA, 
    VectorXd& mx0, 
    std::map<int, std::vector<int>>& ms_handle_elem_map){

    //---------BBW Skinning Handles ---------
    VectorXi handles_ind = VectorXi::Zero(nsh_on_component);
    VectorXd CAx0 = mC*mA*mx0;


    for(int k=0; k<nsh_on_component; k++){
        std::vector<int> els = ms_handle_elem_map[indx-k];
        double centx = 0;//= VectorXd::Zero(els.size());
        double centy = 0;//= VectorXd::Zero(els.size());
        double centz = 0;//= VectorXd::Zero(els.size());
        Vector3d avg_cent;
        for(int i=0; i<els.size(); i++){
            centx += CAx0[12*els[i]];
            centy += CAx0[12*els[i]+1];
            centz += CAx0[12*els[i]+2];
        }
        
        avg_cent<<centx/els.size(), centy/els.size(), centz/els.size();
        int minind = 0;
        double mindist = (avg_cent - tet_center(mT, mV, 0)).norm();
        for(int i=1; i<mT.rows(); i++){
            double dist = (avg_cent - tet_center(mT, mV, i)).norm();
            if(dist<mindist){
                mindist = dist;
                minind = i;
            }
        }
      
        handles_ind[k] = minind;
    }
    return bbw_strain_skinning_matrix(handles_ind, mV, mT);
    

}

void setup_skinning_handles(int nsh, bool reduced, const MatrixXi& mT, const MatrixXd& mV, std::vector<VectorXi>& ibones, std::vector<VectorXi>& imuscle,
    SparseMatrix<double>& mC, SparseMatrix<double>& mA, MatrixXd& mG, VectorXd& mx0, VectorXd& mred_s, MatrixXd& msW, std::map<int, std::vector<int>>& ms_handle_elem_map, VectorXd& relStiff){
    
    std::cout<<"+ Skinning Handles"<<std::endl;
    int handles_per_tendon=2;
    VectorXi skinning_elem_cluster_map;
    std::map<int, std::vector<int>> skinning_cluster_elem_map;

    std::cout<<nsh<<std::endl;
    if(nsh==0){
        nsh = mT.rows();
    } 

    if(nsh<(ibones.size()+imuscle.size())){
        std::cout<<"Too few skinning handles, too many components"<<std::endl;
        exit(0);
    }

    

    //-----------------------------------------------------
    if(nsh==mT.rows() && reduced==false){
        //unreduced
        std::cout<<"here"<<std::endl;
        skinning_elem_cluster_map.resize(mT.rows());
        skinning_elem_cluster_map.setZero();
        for(int i=0; i<mT.rows(); i++){
            skinning_elem_cluster_map[i] = i;
        }
    }else{
        kmeans_clustering(skinning_elem_cluster_map, nsh, handles_per_tendon, ibones, imuscle, mG, mC, mA, mx0, relStiff);
    }

    for(int i=0; i<mT.rows(); i++){
        ms_handle_elem_map[skinning_elem_cluster_map[i]].push_back(i);
    }
    //------------------------------------------------------

    if(reduced==false){
        mred_s.resize(6*mT.rows());
        for(int i=0; i<mT.rows(); i++){
            mred_s[6*i+0] = 1; 
            mred_s[6*i+1] = 1; 
            mred_s[6*i+2] = 1; 
            mred_s[6*i+3] = 0; 
            mred_s[6*i+4] = 0; 
            mred_s[6*i+5] = 0;
        }
        std::cout<<"- Skinning Handles"<<std::endl;
        return;
    
    }
    if(nsh==mT.rows()){
        std::cout<<"Too many skinning handles. "<<std::endl;
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

    MatrixXd Id6 = MatrixXd::Identity(6, 6);
    MatrixXd sW = MatrixXd::Zero(mT.rows(), nsh);
    sW.setZero();
    std::cout<<"nsh: "<<nsh<<std::endl;
    std::cout<<ms_handle_elem_map.size()<<std::endl;
    // //Constant weights of 1 for each handle region
    // for(int h=0; h<ms_handle_elem_map.size(); h++){
    //   std::vector<int> mshem =  ms_handle_elem_map[ms_handle_elem_map.size() - h - 1];
    //   MatrixXd ones_vec = VectorXd::Ones(mshem.size());
    //   MatrixXd sWslice = MatrixXd::Zero(mT.rows(), 1);
    //   VectorXi handles_elem_vec;
    //   igl::list_to_matrix(mshem, handles_elem_vec);
    //   igl::slice_into(ones_vec , handles_elem_vec, 1, sWslice);
    //   sW.block(0, h, mT.rows(), 1) = sWslice;
    // }
    // msW =  Eigen::kroneckerProduct(sW, Id6);
 
    // return;


    //blocked construction of full skinning weights matrix
    int maxnsh = nsh;
    cout<<"----------Bone HANDLES------------"<<maxnsh<<"--"<<ms_handle_elem_map.size()<<endl;
    int insert_index = 0;
    for(int b=0; b<ibones.size(); b++){
        MatrixXi subT(ms_handle_elem_map[maxnsh - 1 - insert_index].size(), 4);
        MatrixXi componentT;
        MatrixXd componentV;
        VectorXi J;

        for(int i=0; i<ms_handle_elem_map[maxnsh -1 - insert_index].size() ; i++){
            subT.row(i) = mT.row(ms_handle_elem_map[maxnsh -1 - insert_index][i]);
        }

        igl::remove_unreferenced(mV, subT, componentV, componentT, J);

        std::string nam = "bone"+to_string(b)+".obj";

        igl::writeOBJ(nam, componentV, componentT);

        MatrixXd sWi = setup_skinning_helper(maxnsh-1- insert_index , 1, componentT, componentV, mC, mA, mx0, ms_handle_elem_map);
        
        MatrixXd sWslice = MatrixXd::Zero(mT.rows(), sWi.cols());
        igl::slice_into(sWi , ibones[b], 1, sWslice);
        sW.block(0,insert_index, mT.rows(), sWi.cols()) = sWslice;
        nsh = nsh - 1; //bone skinning handle has been made, so decrease nsh by 1
        insert_index += 1;
    }

    cout<<"----------MUSCLE HANDLES----"<<nsh<<"---"<<insert_index<<"-----"<<endl;
    int number_handles_per_muscle = nsh/imuscle.size();
    for(int m=0; m<imuscle.size(); m++){ //through muscle vector
        // std::vector<int> tendon_elements_list;
        // std::vector<int> muscle_elements_list;
        // for(int i=0; i<imuscle[m].size(); i++){
        //     if(relStiff[imuscle[m][i]]>10){
        //         tendon_elements_list.push_back(imuscle[m][i]);
        //     }else{
        //         muscle_elements_list.push_back(imuscle[m][i]);
        //     }
        // }
      
        VectorXi muscle_els, tendon_els;
        muscle_els = imuscle[m];
        // igl::list_to_matrix(tendon_elements_list, tendon_els);
        // igl::list_to_matrix(muscle_elements_list, muscle_els);

        // //Tendon skinning handles---------------------------
        // if(tendon_els.size()>0){
            
        //     MatrixXi TcomponentT;
        //     MatrixXd TcomponentV;
        //     MatrixXi TsubT(tendon_els.size(), 4);
        //     VectorXi TJ;
        //     for(int i=0; i<tendon_els.size() ; i++){
        //         TsubT.row(i) = mT.row(tendon_els[i]);
        //     }
        //     igl::remove_unreferenced(mV, TsubT, TcomponentV, TcomponentT, TJ);
        //     std::string Tnam = "tendon"+to_string(m)+".obj";
        //     igl::writeOBJ(Tnam, TcomponentV, TcomponentT);
        //     MatrixXd TsWi;
        //     if(m==imuscle.size() -1){
        //         TsWi = setup_skinning_helper(maxnsh -1 - insert_index - nsh + handles_per_tendon, handles_per_tendon, TcomponentT, TcomponentV, mC, mA, mx0, ms_handle_elem_map);
        //     }else{
        //         TsWi = setup_skinning_helper(maxnsh -1 - insert_index - number_handles_per_muscle + handles_per_tendon, handles_per_tendon, TcomponentT, TcomponentV, mC, mA, mx0, ms_handle_elem_map);
        //     }
        //     MatrixXd TsWslice = MatrixXd::Zero(mT.rows(), TsWi.cols());
            
        //     igl::slice_into(TsWi , tendon_els, 1, TsWslice);
        //     sW.block(0,insert_index, mT.rows(), TsWi.cols()) = TsWslice;
        // }
        handles_per_tendon=0;

        //Muscle skinning handles---------------------------
        MatrixXi componentT;
        MatrixXd componentV;
        MatrixXi subT(muscle_els.size(), 4);
        VectorXi J;
        for(int i=0; i<muscle_els.size() ; i++){
            subT.row(i) = mT.row(muscle_els[i]);
        }
        igl::remove_unreferenced(mV, subT, componentV, componentT, J);
        std::string nam = "muscle"+to_string(m)+".obj";
        igl::writeOBJ(nam, componentV, componentT);
        MatrixXd sWi;
        if(m==imuscle.size()-1){
            //Deal with remainder handles
            sWi = setup_skinning_helper(maxnsh - 1 - insert_index, nsh - handles_per_tendon, componentT, componentV, mC, mA, mx0, ms_handle_elem_map);
        }else{
            sWi = setup_skinning_helper(maxnsh - 1 - insert_index, number_handles_per_muscle - handles_per_tendon, componentT, componentV, mC, mA, mx0, ms_handle_elem_map );
        }

        MatrixXd sWslice = MatrixXd::Zero(mT.rows(), sWi.cols());
        
        igl::slice_into(sWi , muscle_els, 1, sWslice);
        sW.block(0,insert_index+handles_per_tendon, mT.rows(), sWi.cols()) = sWslice;
        insert_index += number_handles_per_muscle;
        nsh =  nsh - number_handles_per_muscle;
    }
    assert(nsh==0);

    // MatrixXd Id6 = MatrixXd::Identity(6, 6);
    msW =  Eigen::kroneckerProduct(sW, Id6);
 
    std::cout<<"- Skinning Handles"<<std::endl;
}

