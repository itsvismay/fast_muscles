#ifndef JOINT_CONSTRAINT_MATRIX
#define JOINT_CONSTRAINT_MATRIX
#include "store.h"

using Store=famu::Store;
using namespace Eigen;
namespace famu
{
	void joint_constraint_matrix(Store& store, SparseMatrix<double, Eigen::RowMajor>& jointsY){
        int hingejoints = 0;
        int socketjoints = 0;

        std::vector<Trip> joint_trips;
        for(int i=0; i<store.joint_bones_verts.size(); i++){ 
            MatrixXd joint = store.joint_bones_verts[i].second;
            int bone1ind = store.bone_name_index_map[store.joint_bones_verts[i].first[0]];
            int bone2ind = store.bone_name_index_map[store.joint_bones_verts[i].first[1]];
            cout<<"bones: "<<bone1ind<<", "<<bone2ind<<", "<<joint.rows()<<", "<<store.fix_bones.size()<<endl;
            if(joint.rows()>1){
                RowVector3d p1 = joint.row(0);
                RowVector3d p2 = joint.row(1);

                if(bone1ind>=store.fix_bones.size()){

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +0, (int)12*(bone1ind-store.fix_bones.size())+0, p1[0]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +1, (int)12*(bone1ind-store.fix_bones.size())+1, p1[0]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +2, (int)12*(bone1ind-store.fix_bones.size())+2, p1[0]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +0, (int)12*(bone1ind-store.fix_bones.size())+3, p1[1]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +1, (int)12*(bone1ind-store.fix_bones.size())+4, p1[1]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +2, (int)12*(bone1ind-store.fix_bones.size())+5, p1[1]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +0, (int)12*(bone1ind-store.fix_bones.size())+6, p1[2]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +1, (int)12*(bone1ind-store.fix_bones.size())+7, p1[2]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +2, (int)12*(bone1ind-store.fix_bones.size())+8, p1[2]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +0, (int)12*(bone1ind-store.fix_bones.size())+9, 1));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +1, (int)12*(bone1ind-store.fix_bones.size())+10, 1));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +2, (int)12*(bone1ind-store.fix_bones.size())+11, 1));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +3, (int)12*(bone1ind-store.fix_bones.size())+0, p2[0]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +4, (int)12*(bone1ind-store.fix_bones.size())+1, p2[0]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +5, (int)12*(bone1ind-store.fix_bones.size())+2, p2[0]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +3, (int)12*(bone1ind-store.fix_bones.size())+3, p2[1]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +4, (int)12*(bone1ind-store.fix_bones.size())+4, p2[1]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +5, (int)12*(bone1ind-store.fix_bones.size())+5, p2[1]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +3, (int)12*(bone1ind-store.fix_bones.size())+6, p2[2]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +4, (int)12*(bone1ind-store.fix_bones.size())+7, p2[2]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +5, (int)12*(bone1ind-store.fix_bones.size())+8, p2[2]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +3, (int)12*(bone1ind-store.fix_bones.size())+9, 1));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +4, (int)12*(bone1ind-store.fix_bones.size())+10, 1));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +5, (int)12*(bone1ind-store.fix_bones.size())+11, 1));
                }
                if(bone2ind>=store.fix_bones.size()){
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +0, (int)12*(bone2ind - store.fix_bones.size())+0, -p1[0]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +1, (int)12*(bone2ind - store.fix_bones.size())+1, -p1[0]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +2, (int)12*(bone2ind - store.fix_bones.size())+2, -p1[0]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +0, (int)12*(bone2ind - store.fix_bones.size())+3, -p1[1]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +1, (int)12*(bone2ind - store.fix_bones.size())+4, -p1[1]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +2, (int)12*(bone2ind - store.fix_bones.size())+5, -p1[1]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +0, (int)12*(bone2ind - store.fix_bones.size())+6, -p1[2]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +1, (int)12*(bone2ind - store.fix_bones.size())+7, -p1[2]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +2, (int)12*(bone2ind - store.fix_bones.size())+8, -p1[2]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +0, (int)12*(bone2ind - store.fix_bones.size())+9, -1));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +1, (int)12*(bone2ind - store.fix_bones.size())+10,-1));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +2, (int)12*(bone2ind - store.fix_bones.size())+11,-1));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +3, (int)12*(bone2ind - store.fix_bones.size())+0, -p2[0]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +4, (int)12*(bone2ind - store.fix_bones.size())+1, -p2[0]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +5, (int)12*(bone2ind - store.fix_bones.size())+2, -p2[0]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +3, (int)12*(bone2ind - store.fix_bones.size())+3, -p2[1]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +4, (int)12*(bone2ind - store.fix_bones.size())+4, -p2[1]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +5, (int)12*(bone2ind - store.fix_bones.size())+5, -p2[1]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +3, (int)12*(bone2ind - store.fix_bones.size())+6, -p2[2]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +4, (int)12*(bone2ind - store.fix_bones.size())+7, -p2[2]));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +5, (int)12*(bone2ind - store.fix_bones.size())+8, -p2[2]));

                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +3, (int)12*(bone2ind - store.fix_bones.size())+9, -1));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +4, (int)12*(bone2ind - store.fix_bones.size())+10,-1));
                    joint_trips.push_back(Trip(6*hingejoints + 3*socketjoints +5, (int)12*(bone2ind - store.fix_bones.size())+11,-1));
                }
                hingejoints +=1;

            }else{
                RowVector3d p1 = joint.row(0);
                if(bone1ind>=store.fix_bones.size()){
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+0, (int)12*(bone1ind - store.fix_bones.size())+0, p1[0]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+1, (int)12*(bone1ind - store.fix_bones.size())+1, p1[0]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+2, (int)12*(bone1ind - store.fix_bones.size())+2, p1[0]));

                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+0, (int)12*(bone1ind - store.fix_bones.size())+3, p1[1]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+1, (int)12*(bone1ind - store.fix_bones.size())+4, p1[1]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+2, (int)12*(bone1ind - store.fix_bones.size())+5, p1[1]));

                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+0, (int)12*(bone1ind - store.fix_bones.size())+6, p1[2]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+1, (int)12*(bone1ind - store.fix_bones.size())+7, p1[2]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+2, (int)12*(bone1ind - store.fix_bones.size())+8, p1[2]));

                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+0, (int)12*(bone1ind - store.fix_bones.size())+9, 1));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+1, (int)12*(bone1ind - store.fix_bones.size())+10,1));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+2, (int)12*(bone1ind - store.fix_bones.size())+11,1));

                }
                if(bone2ind>=store.fix_bones.size()){
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+0, (int)12*(bone2ind - store.fix_bones.size())+0, -p1[0]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+1, (int)12*(bone2ind - store.fix_bones.size())+1, -p1[0]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+2, (int)12*(bone2ind - store.fix_bones.size())+2, -p1[0]));

                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+0, (int)12*(bone2ind - store.fix_bones.size())+3, -p1[1]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+1, (int)12*(bone2ind - store.fix_bones.size())+4, -p1[1]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+2, (int)12*(bone2ind - store.fix_bones.size())+5, -p1[1]));

                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+0, (int)12*(bone2ind - store.fix_bones.size())+6, -p1[2]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+1, (int)12*(bone2ind - store.fix_bones.size())+7, -p1[2]));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+2, (int)12*(bone2ind - store.fix_bones.size())+8, -p1[2]));

                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+0, (int)12*(bone2ind - store.fix_bones.size())+9, -1));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+1, (int)12*(bone2ind - store.fix_bones.size())+10,-1));
                    joint_trips.push_back(Trip(6*hingejoints+3*socketjoints+2, (int)12*(bone2ind - store.fix_bones.size())+11,-1));                    
                }
                socketjoints +=1;
            }
        }

        // for(int i =0; i<joint_trips.size(); i++){
        // 	cout<<joint_trips[i].row()<<", "<<joint_trips[i].col()<<", "<<joint_trips[i].value()<<endl;
        // }
      
        jointsY.resize(6*hingejoints + 3*socketjoints, store.Y.cols());
        jointsY.setFromTriplets(joint_trips.begin(), joint_trips.end());
}
}

#endif