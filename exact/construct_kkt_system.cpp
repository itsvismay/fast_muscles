#include "construct_kkt_system.h"

using namespace exact;
using namespace Eigen;

typedef Eigen::Triplet<double> Trip;

std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double, Eigen::RowMajor> & M){
	std::vector<Eigen::Triplet<double>> v;
	for(int i = 0; i < M.outerSize(); i++){
		for(typename Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(M,i); it; ++it){	
			v.emplace_back(it.row(),it.col(),it.value());
		}
	}
	return v;
}

void exact::construct_kkt_system_left(Eigen::SparseMatrix<double , Eigen::RowMajor>& out, Eigen::SparseMatrix<double, Eigen::RowMajor>& TL, Eigen::SparseMatrix<double, Eigen::RowMajor>& TR, Eigen::SparseMatrix<double, Eigen::RowMajor>& BL, double constraint_stiffness){
	out.resize(TL.rows()+BL.rows(), TL.cols()+TR.cols());
	out.setZero();

	std::vector<Trip> TL_trips = to_triplets(TL);
	std::vector<Trip> TR_trips = to_triplets(TR);
	std::vector<Trip> BL_trips = to_triplets(TR);

	for(int i=0; i<TR_trips.size(); i++){
		int row = TR_trips[i].row();
		int col = TR_trips[i].col();
		double val = TR_trips[i].value();
		TL_trips.push_back(Trip(row, col+TL.cols(), val));
	}

	for(int i=0; i<BL_trips.size(); i++){
		int row = BL_trips[i].row();
		int col = BL_trips[i].col();
		double val = BL_trips[i].value();
		TL_trips.push_back(Trip(row+TL.rows(), col, val));
	}

	if(constraint_stiffness != 0){
		for(int i=0; i<BL.rows(); i++){
			int r = TL.rows() + i;
			TL_trips.push_back(Trip( r,r, constraint_stiffness));
		}
	}

	out.setFromTriplets(TL_trips.begin(), TL_trips.end());

}

void exact::construct_kkt_system_right(Eigen::VectorXd& top, Eigen::VectorXd& bottom, Eigen::VectorXd& KKT_right){
	if(KKT_right.size() != top.size()+bottom.size()){
		KKT_right.resize(top.size() + bottom.size());
	}
	KKT_right.head(top.size()) = top;
	KKT_right.tail(bottom.size()) = bottom; 
}