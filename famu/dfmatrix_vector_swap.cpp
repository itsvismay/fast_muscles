#include "dfmatrix_vector_swap.h"
#include <vector>
using namespace Eigen;
void famu::dFMatrix_Vector_Swap(SparseMatrix<double>& mat, VectorXd& vec){
    std::vector<Trip> mat_trips;

    for (int i=0; i<vec.size()/12; i++){

    	for(int j=0; j<4; j++){
	    	Vector3d seg = vec.segment<3>(12*i + 3*j);
	        mat_trips.push_back(Trip(12*i+3*j, 9*i+0, seg[0]));
	        mat_trips.push_back(Trip(12*i+3*j, 9*i+1, seg[1]));
	        mat_trips.push_back(Trip(12*i+3*j, 9*i+2, seg[2]));

	        mat_trips.push_back(Trip(12*i+3*j+1, 9*i+3, seg[0]));
	        mat_trips.push_back(Trip(12*i+3*j+1, 9*i+4, seg[1]));
	        mat_trips.push_back(Trip(12*i+3*j+1, 9*i+5, seg[2]));

	        mat_trips.push_back(Trip(12*i+3*j+2, 9*i+6, seg[0]));
	        mat_trips.push_back(Trip(12*i+3*j+2, 9*i+7, seg[1]));
	        mat_trips.push_back(Trip(12*i+3*j+2, 9*i+8, seg[2]));
	        
    	}
    }

    mat.resize(vec.size(), 9*(vec.size()/12));
    mat.setFromTriplets(mat_trips.begin(), mat_trips.end());

}