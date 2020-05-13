#include "bone_vertices_projection.h"

using namespace Eigen;
using Store=exact::Store;

void exact::bone_vertices_projection(Store& store, SparseMatrix<double, Eigen::RowMajor>& mY){  
	//bone_tets should have fixed bones at the front, rest at the back
	VectorXd bone_or_muscle = VectorXd::Zero(3*store.V.rows()); //0 for muscle, 1,2 for each bone
	bone_or_muscle.array() += -1;
	for(int i=store.bone_tets.size() -1; i>=0; i--){ //go through in reverse so joints are part of the fixed bone
	    for(int j=0; j<store.bone_tets[i].size(); j++){
	        bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[0]) =  (i+1)*Vector3d::Ones();
	        bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[1]) =  (i+1)*Vector3d::Ones();
	        bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[2]) =  (i+1)*Vector3d::Ones();
	        bone_or_muscle.segment<3>(3*store.T.row(store.bone_tets[i][j])[3]) =  (i+1)*Vector3d::Ones(); 
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



