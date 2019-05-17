#ifndef FIXED_BONES_PROJECTION_MATRIX
#define FIXED_BONES_PROJECTION_MATRIX
#include "store.h"

using Store=famu::Store;
using namespace Eigen;
namespace famu
{
	void fixed_bones_projection_matrix(Store& store, SparseMatrix<double>& mY){
            VectorXd vert_free_or_not = store.ConstrainProjection*store.ConstrainProjection.transpose()*VectorXd::Ones(3*store.V.rows());// check if vert is fixed
            //TODO fixed bones, used this vector (Y*Y'Ones(3*V.rows())) to create the mFree matrix and mConstrained matrix
            
            //ibones should have fixed bones at the front, rest at the back
            VectorXd bone_or_muscle = VectorXd::Zero(3*store.V.rows()); //0 for muscle, 1,2 for each bone
            bone_or_muscle.array() += -1;
            for(int i=store.bone_tets.size() -1; i>=0; i--){ //go through in reverse so joints are part of the fixed bone
                for(int j=0; j<store.bone_tets[i].size(); j++){
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[0])[0] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[0])[1] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[0])[2] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[1])[0] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[1])[1] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[1])[2] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[2])[0] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[2])[1] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[2])[2] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[3])[0] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[3])[1] = i+1;
                    bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[3])[2] = i+1; 
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
                    mY_trips.push_back(Trip(3*i+0, 12*(store.bone_tets.size()-store.fix_bones.size())+3*muscle_index+0, 1.0));
                    mY_trips.push_back(Trip(3*i+1, 12*(store.bone_tets.size()-store.fix_bones.size())+3*muscle_index+1, 1.0));
                    mY_trips.push_back(Trip(3*i+2, 12*(store.bone_tets.size()-store.fix_bones.size())+3*muscle_index+2, 1.0));
                    muscle_index +=1;
                }
            }


            for(int i=0; i<bone_or_muscle.size()/3; i++){
                    if(bone_or_muscle[3*i]>store.fix_bones.size()){
                        mY_trips.push_back(Trip(3*i+0, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+0, store.V.row(i)[0]));
                        mY_trips.push_back(Trip(3*i+1, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+1, store.V.row(i)[0]));
                        mY_trips.push_back(Trip(3*i+2, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+2, store.V.row(i)[0]));
                        mY_trips.push_back(Trip(3*i+0, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+3, store.V.row(i)[1]));
                        mY_trips.push_back(Trip(3*i+1, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+4, store.V.row(i)[1]));
                        mY_trips.push_back(Trip(3*i+2, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+5, store.V.row(i)[1]));
                        mY_trips.push_back(Trip(3*i+0, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+6, store.V.row(i)[2]));
                        mY_trips.push_back(Trip(3*i+1, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+7, store.V.row(i)[2]));
                        mY_trips.push_back(Trip(3*i+2, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+8, store.V.row(i)[2]));
                        mY_trips.push_back(Trip(3*i+0, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+9,  1.0));
                        mY_trips.push_back(Trip(3*i+1, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+10, 1.0));
                        mY_trips.push_back(Trip(3*i+2, (int)(12*(bone_or_muscle[3*i]-1 - store.fix_bones.size()))+11, 1.0));
                }
            }

            mY.resize(3*store.V.rows(), 3*muscleVerts + 12*(store.bone_tets.size() - store.fix_bones.size()));
            mY.setFromTriplets(mY_trips.begin(), mY_trips.end());
    }

    void bone_acap_deformation_constraints(Store& store, SparseMatrix<double>& mBx, SparseMatrix<double>& mBf){
        // std::vector<Trip> mb_trips;

        // for(int b=0; b<store.bone_tets.size(); b++){
        //     int t = store.bone_tets[b][0];
        //     mb_trips.push_back(Trip(12*t + 0, 12*b + 0, 1.0));
        //     mb_trips.push_back(Trip(12*t + 1, 12*b + 1, 1.0));
        //     mb_trips.push_back(Trip(12*t + 2, 12*b + 2, 1.0));

        //     mb_trips.push_back(Trip(12*t + 3, 12*b + 3, 1.0));
        //     mb_trips.push_back(Trip(12*t + 4, 12*b + 4, 1.0));
        //     mb_trips.push_back(Trip(12*t + 5, 12*b + 5, 1.0));

        //     mb_trips.push_back(Trip(12*t + 6, 12*b + 6, 1.0));
        //     mb_trips.push_back(Trip(12*t + 7, 12*b + 7, 1.0));
        //     mb_trips.push_back(Trip(12*t + 8, 12*b + 8, 1.0));

        //     mb_trips.push_back(Trip(12*t + 9, 12*b + 9, 1.0));
        //     mb_trips.push_back(Trip(12*t +10, 12*b +10, 1.0));
        //     mb_trips.push_back(Trip(12*t +11, 12*b +11, 1.0));
        
        // }
        // mB.resize(12*store.T.rows(), 12*store.bone_tets.size());
        // mB.setFromTriplets(mb_trips.begin(), mb_trips.end());
    
        std::vector<Trip> x_trips, f_trips;

        for(int i=0; i<(store.bone_tets.size() - store.fix_bones.size()); i++){
            x_trips.push_back(Trip(9*i+0, 12*i+0 , 1.0));
            x_trips.push_back(Trip(9*i+1, 12*i+3 , 1.0));
            x_trips.push_back(Trip(9*i+2, 12*i+6 , 1.0));
            x_trips.push_back(Trip(9*i+3, 12*i+1 , 1.0));
            x_trips.push_back(Trip(9*i+4, 12*i+4 , 1.0));
            x_trips.push_back(Trip(9*i+5, 12*i+7 , 1.0));
            x_trips.push_back(Trip(9*i+6, 12*i+2 , 1.0));
            x_trips.push_back(Trip(9*i+7, 12*i+5 , 1.0));
            x_trips.push_back(Trip(9*i+8, 12*i+8 , 1.0));
        }

        mBx.resize(9*(store.bone_tets.size() - store.fix_bones.size()), store.Y.cols());
        mBx.setFromTriplets(x_trips.begin(), x_trips.end());

        for(int i=0; i<(store.bone_tets.size() - store.fix_bones.size()); i++){
            f_trips.push_back(Trip( 9*i+0, 9*(i +store.fix_bones.size())+0, 1.0));
            f_trips.push_back(Trip( 9*i+1, 9*(i +store.fix_bones.size())+1, 1.0));
            f_trips.push_back(Trip( 9*i+2, 9*(i +store.fix_bones.size())+2, 1.0));
            f_trips.push_back(Trip( 9*i+3, 9*(i +store.fix_bones.size())+3, 1.0));
            f_trips.push_back(Trip( 9*i+4, 9*(i +store.fix_bones.size())+4, 1.0));
            f_trips.push_back(Trip( 9*i+5, 9*(i +store.fix_bones.size())+5, 1.0));
            f_trips.push_back(Trip( 9*i+6, 9*(i +store.fix_bones.size())+6, 1.0));
            f_trips.push_back(Trip( 9*i+7, 9*(i +store.fix_bones.size())+7, 1.0));
            f_trips.push_back(Trip( 9*i+8, 9*(i +store.fix_bones.size())+8, 1.0));
        }

        mBf.resize(9*(store.bone_tets.size() - store.fix_bones.size()), store.ProjectF.cols());
        mBf.setFromTriplets(f_trips.begin(), f_trips.end());

    }

}
#endif