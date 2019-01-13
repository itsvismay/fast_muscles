#ifndef MESH 
#define MESH

#include <unsupported/Eigen/KroneckerProduct>
#include <json.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <igl/readDMAT.h>
#include "PreProcessing/to_triplets.h"
#include "PreProcessing/setup_modes.h"
#include "PreProcessing/setup_rotation_cluster.h"
#include "PreProcessing/setup_skinning_handles.h"

// #include <GaussIncludes.h>
// #include <FEMIncludes.h>
// #include <UtilitiesEigen.h>
// //Any extra things I need such as constraints
// #include <ConstraintFixedPoint.h>
// #include <TimeStepperLinearModes.h>

// using namespace Gauss;
// using namespace FEM;
// using namespace ParticleSystem; //For Force Spring






using namespace Eigen;
using namespace std;
using json = nlohmann::json;

typedef Eigen::Triplet<double> Trip;
typedef Matrix<double, 12, 1> Vector12d;

// typedef PhysicalSystemFEM<double, LinearTet> FEMLinearTets;
// typedef World<double, std::tuple<FEMLinearTets *>, std::tuple<ForceSpringFEMParticle<double> *>, std::tuple<ConstraintFixedPoint<double> *> > MyWorld;


class Mesh
{

protected:
    MatrixXd mV, discV;
    MatrixXi mT, discT;

    //Used in the sim
    SparseMatrix<double> mMass, mGF, mGR, mGS, mGU, mP, mC, m_P, mPA, mWr, mWw;
    SparseMatrix<double> mA, mFree, mConstrained, mN, mAN, mY;

    VectorXi melemType, mr_elem_cluster_map, ms_handles_ind;
    VectorXd mcontx, mx, mx0, mred_s, melemYoungs, 
    melemPoissons, mred_r, mred_x, mred_w, mPAx0, mFPAx;
    MatrixXd mR, mG, msW, mUvecs;
    VectorXd mmass_diag, mbones;

    std::vector<int> mfix, mmov;
    std::map<int, std::vector<int>> mr_cluster_elem_map;
    std::vector<SparseMatrix<double>> mRotationBLOCK;
    bool reduced;
    std::vector<VectorXi> mmuscles;
    //end

    

public:
    Mesh(){}

    Mesh(MatrixXi& iT, MatrixXd& iV, 
        std::vector<int>& ifix_bones, 
        std::vector<int>& imov, 
        std::vector<VectorXi>& ibones, 
        std::vector<VectorXi>& imuscle,
        MatrixXd& iUvecs, json& j_input, 
        std::vector<int> fd_fix = {})
    {
        mV = iV;
        mT = iT;
        mmov = imov;
        mmuscles = imuscle;

        double youngs = j_input["youngs"];
        double poissons = j_input["poissons"];
        int num_modes = j_input["number_modes"];
        int nsh = j_input["number_skinning_handles"];
        int nrc = j_input["number_rot_clusters"];
        reduced = j_input["reduced"];

        //###########Re-index ifix_bones (fixed bone indexes) to be the top indices mapping into ibones
        for(int ii=0; ii<ifix_bones.size(); ii++){
            std::swap(ibones[ifix_bones[ii]], ibones[ii]);
            ifix_bones[ii] = ii;
        }
        //###########


        cout<<"if it fails here, make sure indexing is within bounds"<<endl;
        std::set<int> fix_verts_set;
        for(int ii=0; ii<ifix_bones.size(); ii++){
            cout<<ifix_bones[ii]<<endl;
            fix_verts_set.insert(mT.row(ibones[ifix_bones[ii]][0])[0]);
            fix_verts_set.insert(mT.row(ibones[ifix_bones[ii]][0])[1]);
            fix_verts_set.insert(mT.row(ibones[ifix_bones[ii]][0])[2]);
            fix_verts_set.insert(mT.row(ibones[ifix_bones[ii]][0])[3]);
        }

        mfix.assign(fix_verts_set.begin(), fix_verts_set.end());
        std::sort (mfix.begin(), mfix.end());

        if(fd_fix.size() != 0){
            mfix = fd_fix;
        }



        
        print("step 1");
        mx0.resize(mV.cols()*mV.rows());
        mx.resize(mV.cols()*mV.rows());

        for(int i=0; i<mV.rows(); i++){
            mx0[3*i+0] = mV(i,0); mx0[3*i+1] = mV(i,1); mx0[3*i+2] = mV(i,2);   
        }
        mx.setZero();

        mbones.resize(mT.rows());
        mbones.setZero();
        for(int b=0; b<ibones.size(); b++){
            for(int i=0; i<ibones[b].size(); i++){
                mbones[ibones[b][i]] = 1;
            }
        }
        for(int m=0; m<imuscle.size(); m++){
            for(int i=0; i<imuscle[m].size(); i++){
                mbones[imuscle[m][i]] -=1;
            }
        }

        print("step 2");
        setP();
        print("step 3");
        setA();
        print("step 4");
        setC();
        print("step 5");
        setFreedConstrainedMatrices();
        print("step 6");
        setVertexWiseMassDiag();

        print("step 7");
        setHandleModesMatrix(ibones, imuscle, ifix_bones);
        
        print("step 8");
        setElemWiseYoungsPoissons(youngs, poissons);
        print("step 9");
        setDiscontinuousMeshT();

        print("step 10");
        setup_rotation_cluster(nrc, reduced, mT, mV, ibones, imuscle, mred_x, mred_r, mred_w, mC, mA, mG, mx0, mRotationBLOCK, mr_cluster_elem_map, mr_elem_cluster_map);

        print("step 11");
        setup_skinning_handles(nsh, reduced, mT, mV, ibones, imuscle, mC, mA, mG, mx0, mred_s, msW);
        
        print("step 12");
        if(num_modes == 1){
            MatrixXd temp1;
            igl::readDMAT("325simplejoint.dmat", temp1);
            mG = mY*temp1;
        }else{

            // MyWorld world;
            // FEMLinearTets *test = new FEMLinearTets(mV,mT);
            // world.addSystem(test);
            // world.finalize();
            // //build mass and stiffness matrices
            // AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > mass;
            // AssemblerParallel<double, AssemblerEigenSparseMatrix<double> > stiffness;
            // getMassMatrix(mass, world);
            // getStiffnessMatrix(stiffness, world);

            setup_modes(num_modes, reduced, mP, mA, mConstrained, mFree, mY, mV, mT, mmass_diag, mG);
            
        }

        print("step 13");
        mUvecs.resize(mT.rows(), 3);
        mUvecs.setZero();
        for(int i=0; i<imuscle.size(); i++){
            for(int j=0; j<imuscle[i].size(); j++){
                mUvecs.row(imuscle[i][j]) = iUvecs.row(imuscle[i][j]);
            }
        }

        if(mG.cols()==0){
            mred_x.resize(3*mV.rows());
        }else{
            mred_x.resize(mG.cols());
        }
        mred_x.setZero();

        mGR.resize(12*mT.rows(), 12*mT.rows());
        mGU.resize(12*mT.rows(), 12*mT.rows());
        mGS.resize(12*mT.rows(), 12*mT.rows());

        print("step 14");
        setGlobalF(false, false, true);
        print("step 15");
        setN(ibones);
        print("step 16");
        mPA = mP*mA;
        mPAx0 = mPA*mx0;
        mFPAx.resize(mPAx0.size());
        setupWrWw();

    }


    void setElemWiseYoungsPoissons(double youngs, double poissons){
        melemYoungs.resize(mT.rows());
        melemPoissons.resize(mT.rows());
        for(int i=0; i<mT.rows(); i++){
            if(mbones[i]>0.5){
                //if bone, give it higher youngs
                melemYoungs[i] = youngs*100;

            }else{
                melemYoungs[i] = youngs; 
            }
            melemPoissons[i] = poissons;
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

        vector<Trip> triplets;
        triplets.reserve(3*16*mT.rows());

        for(int i=0; i<mT.rows(); i++){
            for(int j=0; j<3; j++){
                triplets.push_back(Trip(12*i+0+j, 12*i+0+j, 1.0/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+3+j, 1.0/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+6+j, 1.0/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+9+j, 1.0/4));

                triplets.push_back(Trip(12*i+3+j, 12*i+0+j, 1/4.0));
                triplets.push_back(Trip(12*i+3+j, 12*i+3+j, 1/4.0));
                triplets.push_back(Trip(12*i+3+j, 12*i+6+j, 1/4.0));
                triplets.push_back(Trip(12*i+3+j, 12*i+9+j, 1/4.0));

                triplets.push_back(Trip(12*i+6+j, 12*i+0+j, 1/4.0));
                triplets.push_back(Trip(12*i+6+j, 12*i+3+j, 1/4.0));
                triplets.push_back(Trip(12*i+6+j, 12*i+6+j, 1/4.0));
                triplets.push_back(Trip(12*i+6+j, 12*i+9+j, 1/4.0));

                triplets.push_back(Trip(12*i+9+j, 12*i+0+j, 1/4.0));
                triplets.push_back(Trip(12*i+9+j, 12*i+3+j, 1/4.0));
                triplets.push_back(Trip(12*i+9+j, 12*i+6+j, 1/4.0));
                triplets.push_back(Trip(12*i+9+j, 12*i+9+j, 1/4.0));
            }
        }   

        mC.setFromTriplets(triplets.begin(), triplets.end());
    }

    void setHandleModesMatrix(std::vector<VectorXi>& ibones, std::vector<VectorXi>& imuscle, std::vector<int> fixedbones){
        VectorXd vert_free_or_not = mFree*mFree.transpose()*VectorXd::Ones(3*mV.rows());// check if vert is fixed
        //TODO fixed bones, used this vector (Y*Y'Ones(3*V.rows())) to create the mFree matrix and mConstrained matrix
        
        //ibones should have fixed bones at the front, rest at the back

        VectorXd bone_or_muscle = VectorXd::Zero(3*mV.rows()); //0 for muscle, 1,2 for each bone
        bone_or_muscle.array() += -1;
        for(int i=ibones.size() -1; i>=0; i--){ //go through in reverse so joints are part of the fixed bone
            for(int j=0; j<ibones[i].size(); j++){
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[0])[0] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[0])[1] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[0])[2] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[1])[0] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[1])[1] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[1])[2] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[2])[0] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[2])[1] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[2])[2] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[3])[0] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[3])[1] = i+1;
                bone_or_muscle.segment<3>(3*mT.row(ibones[i][j])[3])[2] = i+1; 
            }
        }

        int muscleVerts = 0;
        for(int i=0; i<bone_or_muscle.size()/3; i++){
            if(bone_or_muscle[3*i]<-1e-8){
                muscleVerts += 1;
            }
        }
    
        
        std::vector<Trip> mY_trips = {};
        //set top |bones|*12 dofs to be the bones
        //set the rest to be muscles, indexed correctly 
        int muscle_index = 0;
        for(int i=0; i<bone_or_muscle.size()/3; i++){
            if(bone_or_muscle[3*i]<-1e-8){
                //its a muscle, figure out which one, and index it correctly
                mY_trips.push_back(Trip(3*i+0, 12*(ibones.size()-fixedbones.size())+3*muscle_index+0, 1.0));
                mY_trips.push_back(Trip(3*i+1, 12*(ibones.size()-fixedbones.size())+3*muscle_index+1, 1.0));
                mY_trips.push_back(Trip(3*i+2, 12*(ibones.size()-fixedbones.size())+3*muscle_index+2, 1.0));
                muscle_index +=1;
            }
        }


        for(int i=0; i<bone_or_muscle.size()/3; i++){
                if(bone_or_muscle[3*i]>fixedbones.size()){
                    mY_trips.push_back(Trip(3*i+0, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+0, mV.row(i)[0]));
                    mY_trips.push_back(Trip(3*i+1, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+1, mV.row(i)[0]));
                    mY_trips.push_back(Trip(3*i+2, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+2, mV.row(i)[0]));

                    mY_trips.push_back(Trip(3*i+0, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+3, mV.row(i)[1]));
                    mY_trips.push_back(Trip(3*i+1, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+4, mV.row(i)[1]));
                    mY_trips.push_back(Trip(3*i+2, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+5, mV.row(i)[1]));

                    mY_trips.push_back(Trip(3*i+0, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+6, mV.row(i)[2]));
                    mY_trips.push_back(Trip(3*i+1, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+7, mV.row(i)[2]));
                    mY_trips.push_back(Trip(3*i+2, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+8, mV.row(i)[2]));

                    mY_trips.push_back(Trip(3*i+0, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+9,  1.0));
                    mY_trips.push_back(Trip(3*i+1, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+10, 1.0));
                    mY_trips.push_back(Trip(3*i+2, (int)(12*(bone_or_muscle[3*i]-1 - fixedbones.size()))+11, 1.0));
            }
        }

        mY.resize(3*mV.rows(), 3*muscleVerts + 12*(ibones.size() - fixedbones.size()));
        mY.setFromTriplets(mY_trips.begin(), mY_trips.end());

    }

    void setN(std::vector<VectorXi> bones){
        //TODO make this work for reduced skinning
        int nsh_bones = bones.size();
        mN.resize(mred_s.size(), mred_s.size() - 6*nsh_bones); //#nsh x nsh for muscles (project out bones handles)
        mN.setZero();

        int j=0;
        for(int i=0; i<mN.rows()/6; i++){
            if(i<bones.size()){
                continue;
            }
            mN.coeffRef(6*i+0, 6*j+0) = 1;
            mN.coeffRef(6*i+1, 6*j+1) = 1;
            mN.coeffRef(6*i+2, 6*j+2) = 1;
            mN.coeffRef(6*i+3, 6*j+3) = 1;
            mN.coeffRef(6*i+4, 6*j+4) = 1;
            mN.coeffRef(6*i+5, 6*j+5) = 1;
            j++;
        }

        mAN.resize(mred_s.size(), 6*nsh_bones); //#nsh x nsh for muscles (project out muscle handles)
        mAN.setZero();

        j=0;
        for(int i=0; i<mAN.rows()/6; i++){
            if(i>=bones.size()){
                continue;
            }
            mAN.coeffRef(6*i+0, 6*j+0) = 1;
            mAN.coeffRef(6*i+1, 6*j+1) = 1;
            mAN.coeffRef(6*i+2, 6*j+2) = 1;
            mAN.coeffRef(6*i+3, 6*j+3) = 1;
            mAN.coeffRef(6*i+4, 6*j+4) = 1;
            mAN.coeffRef(6*i+5, 6*j+5) = 1;
            j++;
        }
    }
 
    void setP(){
        mP.resize(12*mT.rows(), 12*mT.rows());
        Matrix4d p;
        p<< 3, -1, -1, -1,
            -1, 3, -1, -1,
            -1, -1, 3, -1,
            -1, -1, -1, 3;

        vector<Trip> triplets;
        triplets.reserve(3*16*mT.rows());

        for(int i=0; i<mT.rows(); i++){
            double weight = 1;
            if(mbones[i]==1){
                weight = 1;
            }
            for(int j=0; j<3; j++){
                triplets.push_back(Trip(12*i+0+j, 12*i+0+j, weight*p(0,0)/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+3+j, weight*p(0,1)/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+6+j, weight*p(0,2)/4));
                triplets.push_back(Trip(12*i+0+j, 12*i+9+j, weight*p(0,3)/4));

                triplets.push_back(Trip(12*i+3+j, 12*i+0+j, weight*p(1,0)/4));
                triplets.push_back(Trip(12*i+3+j, 12*i+3+j, weight*p(1,1)/4));
                triplets.push_back(Trip(12*i+3+j, 12*i+6+j, weight*p(1,2)/4));
                triplets.push_back(Trip(12*i+3+j, 12*i+9+j, weight*p(1,3)/4));

                triplets.push_back(Trip(12*i+6+j, 12*i+0+j, weight*p(2,0)/4));
                triplets.push_back(Trip(12*i+6+j, 12*i+3+j, weight*p(2,1)/4));
                triplets.push_back(Trip(12*i+6+j, 12*i+6+j, weight*p(2,2)/4));
                triplets.push_back(Trip(12*i+6+j, 12*i+9+j, weight*p(2,3)/4));

                triplets.push_back(Trip(12*i+9+j, 12*i+0+j, weight*p(3,0)/4));
                triplets.push_back(Trip(12*i+9+j, 12*i+3+j, weight*p(3,1)/4));
                triplets.push_back(Trip(12*i+9+j, 12*i+6+j, weight*p(3,2)/4));
                triplets.push_back(Trip(12*i+9+j, 12*i+9+j, weight*p(3,3)/4));
            }
        }   
        mP.setFromTriplets(triplets.begin(), triplets.end());



        m_P.resize(12*mT.rows(), 12*mT.rows());
        vector<Trip> triplets_;
        triplets_.reserve(3*16*mT.rows());
        for(int i=0; i<mT.rows(); i++){
            for(int j=0; j<3; j++){
                triplets_.push_back(Trip(12*i+0+j, 12*i+0+j, p(0,0)/4));
                triplets_.push_back(Trip(12*i+0+j, 12*i+3+j, p(0,1)/4));
                triplets_.push_back(Trip(12*i+0+j, 12*i+6+j, p(0,2)/4));
                triplets_.push_back(Trip(12*i+0+j, 12*i+9+j, p(0,3)/4));

                triplets_.push_back(Trip(12*i+3+j, 12*i+0+j, p(1,0)/4));
                triplets_.push_back(Trip(12*i+3+j, 12*i+3+j, p(1,1)/4));
                triplets_.push_back(Trip(12*i+3+j, 12*i+6+j, p(1,2)/4));
                triplets_.push_back(Trip(12*i+3+j, 12*i+9+j, p(1,3)/4));

                triplets_.push_back(Trip(12*i+6+j, 12*i+0+j, p(2,0)/4));
                triplets_.push_back(Trip(12*i+6+j, 12*i+3+j, p(2,1)/4));
                triplets_.push_back(Trip(12*i+6+j, 12*i+6+j, p(2,2)/4));
                triplets_.push_back(Trip(12*i+6+j, 12*i+9+j, p(2,3)/4));

                triplets_.push_back(Trip(12*i+9+j, 12*i+0+j, p(3,0)/4));
                triplets_.push_back(Trip(12*i+9+j, 12*i+3+j, p(3,1)/4));
                triplets_.push_back(Trip(12*i+9+j, 12*i+6+j, p(3,2)/4));
                triplets_.push_back(Trip(12*i+9+j, 12*i+9+j, p(3,3)/4));
            }
        }   
        m_P.setFromTriplets(triplets_.begin(), triplets_.end());
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

    void setupWrWw(){
        std::vector<Trip> wr_trips;
        std::vector<Trip> ww_trips;

        std::map<int, std::vector<int>>& c_e_map = mr_cluster_elem_map;
        for (int i=0; i<mred_r.size()/9; i++){
            std::vector<int> cluster_elem = c_e_map[i];
            for(int e=0; e<cluster_elem.size(); e++){
                wr_trips.push_back(Trip(9*cluster_elem[e]+0, 9*i+0, 1));
                wr_trips.push_back(Trip(9*cluster_elem[e]+1, 9*i+1, 1));
                wr_trips.push_back(Trip(9*cluster_elem[e]+2, 9*i+2, 1));
                wr_trips.push_back(Trip(9*cluster_elem[e]+3, 9*i+3, 1));
                wr_trips.push_back(Trip(9*cluster_elem[e]+4, 9*i+4, 1));
                wr_trips.push_back(Trip(9*cluster_elem[e]+5, 9*i+5, 1));
                wr_trips.push_back(Trip(9*cluster_elem[e]+6, 9*i+6, 1));
                wr_trips.push_back(Trip(9*cluster_elem[e]+7, 9*i+7, 1));
                wr_trips.push_back(Trip(9*cluster_elem[e]+8, 9*i+8, 1));
                
                ww_trips.push_back(Trip(3*cluster_elem[e]+0, 3*i+0, 1));
                ww_trips.push_back(Trip(3*cluster_elem[e]+1, 3*i+1, 1));
                ww_trips.push_back(Trip(3*cluster_elem[e]+2, 3*i+2, 1));
            
            }

        }

        mWr.resize( 9*mT.rows(), mred_r.size());
        mWr.setFromTriplets(wr_trips.begin(), wr_trips.end());

        mWw.resize( 3*mT.rows(), mred_w.size());
        mWw.setFromTriplets(ww_trips.begin(), ww_trips.end());
    }

    void constTimeFPAx0(VectorXd& iFPAx0){
        iFPAx0.setZero();
        if(reduced==true){
            //TODO make this more efficient
            VectorXd ms;
            if(6*mT.rows()==mred_s.size()){
                ms = mred_s;
            }else{
                ms = msW*mred_s;
            }
            VectorXd mr;
            if(9*mT.rows()==mred_r.size()){
                mr = mred_r;
            }else{
                mr = mWr*mred_r;
            }

            for(int i=0; i<mT.rows(); i++){
                Matrix3d r = Map<Matrix3d>(mred_r.segment<9>(9*mr_elem_cluster_map[i]).data()).transpose();
                Matrix3d s;
                s<< ms[6*i + 0], ms[6*i + 3], ms[6*i + 4],
                    ms[6*i + 3], ms[6*i + 1], ms[6*i + 5],
                    ms[6*i + 4], ms[6*i + 5], ms[6*i + 2];

                Matrix3d rs = r*s;
                iFPAx0.segment<3>(12*i+0) = rs*mPAx0.segment<3>(12*i+0);
                iFPAx0.segment<3>(12*i+3) = rs*mPAx0.segment<3>(12*i+3);
                iFPAx0.segment<3>(12*i+6) = rs*mPAx0.segment<3>(12*i+6);
                iFPAx0.segment<3>(12*i+9) = rs*mPAx0.segment<3>(12*i+9);
            }


        }else{
            for(int i=0; i<mT.rows(); i++){
                Matrix3d r = Map<Matrix3d>(mred_r.segment<9>(9*mr_elem_cluster_map[i]).data()).transpose();
                Matrix3d s;
                s<< mred_s[6*i + 0], mred_s[6*i + 3], mred_s[6*i + 4],
                    mred_s[6*i + 3], mred_s[6*i + 1], mred_s[6*i + 5],
                    mred_s[6*i + 4], mred_s[6*i + 5], mred_s[6*i + 2];

                Matrix3d rs = r*s;
                iFPAx0.segment<3>(12*i+0) = rs*mPAx0.segment<3>(12*i+0);
                iFPAx0.segment<3>(12*i+3) = rs*mPAx0.segment<3>(12*i+3);
                iFPAx0.segment<3>(12*i+6) = rs*mPAx0.segment<3>(12*i+6);
                iFPAx0.segment<3>(12*i+9) = rs*mPAx0.segment<3>(12*i+9);
            }
        }
    }

    void setGlobalF(bool updateR, bool updateS, bool updateU){
        if(updateU){
            mGU.setIdentity();
            
            Vector3d a = Vector3d::UnitX();
            for(int t = 0; t<mT.rows(); t++){
                //TODO: update to be parametrized by input mU
                Vector3d b = mUvecs.row(t);
                if(b.norm()==0){
                    b = Vector3d::UnitY();
                    mUvecs.row(t) = b;
                }
            }
        }

        // if(updateR){
        //     mGR.setZero();
        //     //iterate through rotation clusters
        //     Matrix3d ri = Matrix3d::Zero();
        //     Matrix3d r;
        //     vector<Trip> gr_trips;
        //     gr_trips.reserve(9*4*mT.rows());

        //     for (int t=0; t<mred_r.size()/9; t++){
        //         //dont use the RotBLOCK matrix like I did in python. 
        //         //Insert manually into elements within
        //         //the rotation cluster
        //         ri(0,0) = mred_r[9*t+0];
        //         ri(0,1) = mred_r[9*t+1];
        //         ri(0,2) = mred_r[9*t+2];
        //         ri(1,0) = mred_r[9*t+3];
        //         ri(1,1) = mred_r[9*t+4];
        //         ri(1,2) = mred_r[9*t+5];
        //         ri(2,0) = mred_r[9*t+6];
        //         ri(2,1) = mred_r[9*t+7];
        //         ri(2,2) = mred_r[9*t+8];

        //         // % Rodrigues formula %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        //         Vector3d w;
        //         w<<mred_w(3*t+0),mred_w(3*t+1),mred_w(3*t+2);
        //         double wlen = w.norm();
        //         if (wlen>1e-9){
        //             double wX = w(0);
        //             double wY = w(1);
        //             double wZ = w(2);
        //             Matrix3d cross;
        //             cross<<0, -wZ, wY,
        //                     wZ, 0, -wX,
        //                     -wY, wX, 0;
        //             Matrix3d Rot = cross.exp();
        //             r = ri*Rot;
        //         }else{
        //             r = ri;
        //         }
        //         // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        //         for(int c=0; c<mr_cluster_elem_map[t].size(); c++){
        //             for(int j=0; j<4; j++){
        //                 gr_trips.push_back(Trip(3*j+12*mr_cluster_elem_map[t][c] + 0, 3*j+12*mr_cluster_elem_map[t][c] + 0, r(0 ,0)));
        //                 gr_trips.push_back(Trip(3*j+12*mr_cluster_elem_map[t][c] + 0, 3*j+12*mr_cluster_elem_map[t][c] + 1, r(0 ,1)));
        //                 gr_trips.push_back(Trip(3*j+12*mr_cluster_elem_map[t][c] + 0, 3*j+12*mr_cluster_elem_map[t][c] + 2, r(0 ,2)));
        //                 gr_trips.push_back(Trip(3*j+12*mr_cluster_elem_map[t][c] + 1, 3*j+12*mr_cluster_elem_map[t][c] + 0, r(1 ,0)));
        //                 gr_trips.push_back(Trip(3*j+12*mr_cluster_elem_map[t][c] + 1, 3*j+12*mr_cluster_elem_map[t][c] + 1, r(1 ,1)));
        //                 gr_trips.push_back(Trip(3*j+12*mr_cluster_elem_map[t][c] + 1, 3*j+12*mr_cluster_elem_map[t][c] + 2, r(1 ,2)));
        //                 gr_trips.push_back(Trip(3*j+12*mr_cluster_elem_map[t][c] + 2, 3*j+12*mr_cluster_elem_map[t][c] + 0, r(2 ,0)));
        //                 gr_trips.push_back(Trip(3*j+12*mr_cluster_elem_map[t][c] + 2, 3*j+12*mr_cluster_elem_map[t][c] + 1, r(2 ,1)));
        //                 gr_trips.push_back(Trip(3*j+12*mr_cluster_elem_map[t][c] + 2, 3*j+12*mr_cluster_elem_map[t][c] + 2, r(2 ,2)));
        //             }
        //         }
        //     }
        //     mGR.setFromTriplets(gr_trips.begin(), gr_trips.end());
        // }

        // if(updateS){
        //     mGS.setZero();
        //     vector<Trip> gs_trips;
        //     gs_trips.reserve(9*4*mT.rows());

        //     VectorXd ms;
        //     if(6*mT.rows()==mred_s.size()){
        //         ms = mred_s;
        //     }else{
        //         ms = msW*mred_s;
        //     }

        //     //iterate through skinning handles
        //     for(int t = 0; t<mT.rows(); t++){
        //         for(int j = 0; j<4; j++){
        //             gs_trips.push_back(Trip(3*j+12*t + 0, 3*j+12*t + 0, ms[6*t + 0]));
        //             gs_trips.push_back(Trip(3*j+12*t + 0, 3*j+12*t + 1, ms[6*t + 3]));
        //             gs_trips.push_back(Trip(3*j+12*t + 0, 3*j+12*t + 2, ms[6*t + 4]));
        //             gs_trips.push_back(Trip(3*j+12*t + 1, 3*j+12*t + 0, ms[6*t + 3]));
        //             gs_trips.push_back(Trip(3*j+12*t + 1, 3*j+12*t + 1, ms[6*t + 1]));
        //             gs_trips.push_back(Trip(3*j+12*t + 1, 3*j+12*t + 2, ms[6*t + 5]));
        //             gs_trips.push_back(Trip(3*j+12*t + 2, 3*j+12*t + 0, ms[6*t + 4]));
        //             gs_trips.push_back(Trip(3*j+12*t + 2, 3*j+12*t + 1, ms[6*t + 5]));
        //             gs_trips.push_back(Trip(3*j+12*t + 2, 3*j+12*t + 2, ms[6*t + 2]));
        //         }
        //     }
        //     mGS.setFromTriplets(gs_trips.begin(), gs_trips.end());
        // }

        // mGF = mGR*mGU*mGS*mGU.transpose();
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
    // SparseMatrix<double>& GR(){ return mGR; }
    // SparseMatrix<double>& GS(){ return mGS; }
    // SparseMatrix<double>& GU(){ return mGU; }
    // SparseMatrix<double>& GF(){ return mGF; }

    SparseMatrix<double>& P(){ return mP; }
    SparseMatrix<double>& A(){ return mA; }
    SparseMatrix<double>& N(){ return mN; }
    SparseMatrix<double>& AN(){ return mAN; }
    SparseMatrix<double>& Y(){ return mY; }
    MatrixXd& G(){ return mG; }
    MatrixXd& Uvecs(){ return mUvecs; }

    SparseMatrix<double>& B(){ return mFree; }
    SparseMatrix<double>& AB(){ return mConstrained; }
    VectorXd& red_r(){ return mred_r; }
    VectorXd& red_w(){ return mred_w; }
    VectorXd& eYoungs(){ return melemYoungs; }
    VectorXd& ePoissons(){ return melemPoissons; }
    VectorXd& x0(){ return mx0; }
    VectorXd& red_s(){return mred_s; }
    std::vector<int>& fixed_verts(){ return mfix; }
    std::map<int, std::vector<int>>& r_cluster_elem_map(){ return mr_cluster_elem_map; }
    VectorXi& r_elem_cluster_map(){ return mr_elem_cluster_map; }
    VectorXd& bones(){ return mbones; }
    std::vector<VectorXi> muscle_vecs() { return mmuscles; }
    MatrixXd& sW(){ 
        if(mred_s.size()== 6*mT.rows() && reduced==false){
            print("skinning is unreduced, don't call this function");
            exit(0);
        }
        return msW; 
    }
    std::vector<SparseMatrix<double>>& RotBLOCK(){ return mRotationBLOCK; }

    VectorXd& dx(){ return mx;}
    VectorXd& red_x(){ return mred_x; }
    void red_x(VectorXd& ix){ 
        for(int i=0; i<ix.size(); i++){
            mred_x[i] = ix[i];
        }
        return; 
    }


    void dx(VectorXd& ix){ 
        mx = ix;
        return;
    }


    MatrixXd continuousV(){
        VectorXd x;
        if(3*mV.rows()==mred_x.size()){
            x = mred_x + mx0;
        }else{
            x = mG*mred_x + mx0;
        }

        Eigen::Map<Eigen::MatrixXd> newV(x.data(), mV.cols(), mV.rows());
        return newV.transpose();
    }

    MatrixXi& discontinuousT(){ return discT; }

    MatrixXd& discontinuousV(){
        VectorXd x;
        VectorXd ms;
        if(3*mV.rows()==mred_x.size()){
            x = mred_x+mx0;
        }else{
            print("reduced G");
            x = mG*mred_x + mx0;
        }
        if(6*mT.rows() == mred_s.size()){
            ms = mred_s;
        }else{
            ms = msW*mred_s;
        }

        VectorXd PAx = m_P*mA*x;
        mFPAx.setZero();
        for(int i=0; i<mT.rows(); i++){
            Matrix3d r = Map<Matrix3d>(mred_r.segment<9>(9*mr_elem_cluster_map[i]).data()).transpose();
            Matrix3d s;
            s<< ms[6*i + 0], ms[6*i + 3], ms[6*i + 4],
                ms[6*i + 3], ms[6*i + 1], ms[6*i + 5],
                ms[6*i + 4], ms[6*i + 5], ms[6*i + 2];

            Matrix3d rs = r*s;
            mFPAx.segment<3>(12*i+0) = rs*PAx.segment<3>(12*i+0);
            mFPAx.segment<3>(12*i+3) = rs*PAx.segment<3>(12*i+3);
            mFPAx.segment<3>(12*i+6) = rs*PAx.segment<3>(12*i+6);
            mFPAx.segment<3>(12*i+9) = rs*PAx.segment<3>(12*i+9);
        }

        VectorXd CAx = mC*mA*x;
        VectorXd newx = mFPAx + CAx;

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