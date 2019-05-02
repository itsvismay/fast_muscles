
#include "construct_kkt_system.h"

using namespace famu;
using namespace Eigen;

typedef Eigen::Triplet<double> Trip;

std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> & M){
	std::vector<Eigen::Triplet<double>> v;
	for(int i = 0; i < M.outerSize(); i++){
		for(typename Eigen::SparseMatrix<double>::InnerIterator it(M,i); it; ++it){	
			v.emplace_back(it.row(),it.col(),it.value());
		}
	}
	return v;
}

void famu::construct_kkt_system_left(Eigen::SparseMatrix<double>& H, Eigen::SparseMatrix<double>& C, Eigen::SparseMatrix<double>& KKT_Left){
	KKT_Left.resize(H.rows()+C.rows(), H.rows()+C.rows());
	KKT_Left.setZero();

	std::vector<Trip> HTrips = to_triplets(H);
	std::vector<Trip> C_trips = to_triplets(C);

	for(int i=0; i<C_trips.size(); i++){
		int row = C_trips[i].row();
		int col = C_trips[i].col();
		double val = C_trips[i].value();
		HTrips.push_back(Trip(row+H.rows(), col, val));
		HTrips.push_back(Trip(col, row+H.cols(), val));
	}
	// HTrips.insert(HTrips.end(),C_trips.begin(), C_trips.end());
	KKT_Left.setFromTriplets(HTrips.begin(), HTrips.end());

}

void famu::construct_kkt_system_right(Eigen::VectorXd& top, Eigen::VectorXd& bottom, Eigen::VectorXd& KKT_right){
	if(KKT_right.size() != top.size()+bottom.size()){
		KKT_right.resize(top.size() + bottom.size());
	}
	KKT_right.head(top.size()) = top;
	KKT_right.tail(bottom.size()) = bottom; 
}