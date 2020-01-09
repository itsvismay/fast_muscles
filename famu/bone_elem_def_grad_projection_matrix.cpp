#include "bone_elem_def_grad_projection_matrix.h"
#include <iostream>
using Store=famu::Store;
using namespace Eigen;
using namespace std;

void famu::bone_def_grad_projection_matrix(Store& store, Eigen::SparseMatrix<double, Eigen::RowMajor>& mN, Eigen::SparseMatrix<double, Eigen::RowMajor>& mAN){
	int num_bones = store.bone_tets.size();
	std::vector<Trip> mNProjectBonesElemsToOneDef_trips,mNRemoveFixedScriptedBones_trips, mAN_trips;

    Eigen::SparseMatrix<double, Eigen::RowMajor> NProjectBonesElemsToOneDef, NRemoveFixedBones;
	
    for(int i=0; i<store.bone_or_muscle.size(); i++){
		mNProjectBonesElemsToOneDef_trips.push_back(Trip(9*i+0, 9*store.bone_or_muscle[i]+0, 1.0));
		mNProjectBonesElemsToOneDef_trips.push_back(Trip(9*i+1, 9*store.bone_or_muscle[i]+1, 1.0));
		mNProjectBonesElemsToOneDef_trips.push_back(Trip(9*i+2, 9*store.bone_or_muscle[i]+2, 1.0));

		mNProjectBonesElemsToOneDef_trips.push_back(Trip(9*i+3, 9*store.bone_or_muscle[i]+3, 1.0));
		mNProjectBonesElemsToOneDef_trips.push_back(Trip(9*i+4, 9*store.bone_or_muscle[i]+4, 1.0));
		mNProjectBonesElemsToOneDef_trips.push_back(Trip(9*i+5, 9*store.bone_or_muscle[i]+5, 1.0));

		mNProjectBonesElemsToOneDef_trips.push_back(Trip(9*i+6, 9*store.bone_or_muscle[i]+6, 1.0));
		mNProjectBonesElemsToOneDef_trips.push_back(Trip(9*i+7, 9*store.bone_or_muscle[i]+7, 1.0));
		mNProjectBonesElemsToOneDef_trips.push_back(Trip(9*i+8, 9*store.bone_or_muscle[i]+8, 1.0));
		
    }
    int maxInd = store.bone_or_muscle.maxCoeff();
    NProjectBonesElemsToOneDef.resize(9*store.T.rows(), 9*(maxInd+1));
    NProjectBonesElemsToOneDef.setFromTriplets(mNProjectBonesElemsToOneDef_trips.begin(), mNProjectBonesElemsToOneDef_trips.end());

    for(int i=9*(store.fix_bones.size()+store.script_bones.size()); i<NProjectBonesElemsToOneDef.cols(); i++){
        mNRemoveFixedScriptedBones_trips.push_back(Trip(i - 9*(store.fix_bones.size()+store.script_bones.size()), i, 1.0));
    }
    NRemoveFixedBones.resize(9*(maxInd+1) -  9*(store.fix_bones.size()+store.script_bones.size()), 9*(maxInd+1));
    NRemoveFixedBones.setFromTriplets(mNRemoveFixedScriptedBones_trips.begin(), mNRemoveFixedScriptedBones_trips.end());

    mN = NProjectBonesElemsToOneDef;
    mAN = NRemoveFixedBones; //FIXED AND SCRIPTED

    // for(int i=0; i<store.bone_tets.size(); i++){
    //     mAN_trips.push_back(Trip(9*i+0, 9*i+0, 1.0));
    //     mAN_trips.push_back(Trip(9*i+1, 9*i+1, 1.0));
    //     mAN_trips.push_back(Trip(9*i+2, 9*i+2, 1.0));
    //     mAN_trips.push_back(Trip(9*i+3, 9*i+3, 1.0));
    //     mAN_trips.push_back(Trip(9*i+4, 9*i+4, 1.0));
    //     mAN_trips.push_back(Trip(9*i+5, 9*i+5, 1.0));
    //     mAN_trips.push_back(Trip(9*i+6, 9*i+6, 1.0));
    //     mAN_trips.push_back(Trip(9*i+7, 9*i+7, 1.0));
    //     mAN_trips.push_back(Trip(9*i+8, 9*i+8, 1.0));
    // }

    // mAN.resize(9*(maxInd+1), 9*store.bone_tets.size());
    // mAN.setFromTriplets(mAN_trips.begin(), mAN_trips.end());

    // int muscle_index = 0;
    // for(int i=0; i<store.bone_or_muscle.size(); i++){
    //     if(store.bone_or_muscle[i]<0.5){
    //         mAN_trips.push_back(Trip(9*num_bones + 9*muscle_index+0, 9*i+0, 1.0));
    //         mAN_trips.push_back(Trip(9*num_bones + 9*muscle_index+1, 9*i+1, 1.0));
    //         mAN_trips.push_back(Trip(9*num_bones + 9*muscle_index+2, 9*i+2, 1.0));

    //         mAN_trips.push_back(Trip(9*num_bones + 9*muscle_index+3, 9*i+3, 1.0));
    //         mAN_trips.push_back(Trip(9*num_bones + 9*muscle_index+4, 9*i+4, 1.0));
    //         mAN_trips.push_back(Trip(9*num_bones + 9*muscle_index+5, 9*i+5, 1.0));

    //         mAN_trips.push_back(Trip(9*num_bones + 9*muscle_index+6, 9*i+6, 1.0));
    //         mAN_trips.push_back(Trip(9*num_bones + 9*muscle_index+7, 9*i+7, 1.0));
    //         mAN_trips.push_back(Trip(9*num_bones + 9*muscle_index+8, 9*i+8, 1.0));
            
    //         muscle_index += 1;
    //     }else{
    //         mAN_trips.push_back(Trip((int) 9*(store.bone_or_muscle[i] - 1) + 0, 9*i + 0,  1.0));
    //         mAN_trips.push_back(Trip((int) 9*(store.bone_or_muscle[i] - 1) + 1, 9*i + 1,  1.0));
    //         mAN_trips.push_back(Trip((int) 9*(store.bone_or_muscle[i] - 1) + 2, 9*i + 2,  1.0));

    //         mAN_trips.push_back(Trip((int) 9*(store.bone_or_muscle[i] - 1) + 3, 9*i + 3,  1.0));
    //         mAN_trips.push_back(Trip((int) 9*(store.bone_or_muscle[i] - 1) + 4, 9*i + 4,  1.0));
    //         mAN_trips.push_back(Trip((int) 9*(store.bone_or_muscle[i] - 1) + 5, 9*i + 5,  1.0));

    //         mAN_trips.push_back(Trip((int) 9*(store.bone_or_muscle[i] - 1) + 6, 9*i + 6,  1.0));
    //         mAN_trips.push_back(Trip((int) 9*(store.bone_or_muscle[i] - 1) + 7, 9*i + 7,  1.0));
    //         mAN_trips.push_back(Trip((int) 9*(store.bone_or_muscle[i] - 1) + 8, 9*i + 8,  1.0));
    //     }
    // }

    // mAN.resize(9*(num_bones + muscle_index), 9*store.T.rows());
    // mAN.setFromTriplets(mAN_trips.begin(), mAN_trips.end());

}

// void famu::bone_continuous_constraints(Store& store, SparseMatrix<double>& B){

    // B.resize( 12*store.T.rows());
    // B.setFromTriplets(b_trips.begin(), b_trips.end())
// }