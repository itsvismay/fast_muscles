#ifndef MUSCLE_ENERGY_GRAD
#define MUSCLE_ENERGY_GRAD

#include "store.h"

using Store = exact::Store;
using namespace Eigen;

namespace exact
{
	namespace muscle{
		//output: energy
		//inputs: 
		//Fvec: deformation gradient
		//T: Tets
		//etc...
		double energy(VectorXd& Fvec,
					const MatrixXi& T,
					const VectorXd& rest_tet_vols,
					const MatrixXd& Uvec);

		void gradient(VectorXd& grad, 
					const VectorXd& Fvec,
					const MatrixXi& T,
					const VectorXd& rest_tet_vols,
					const MatrixXd& Uvec );

		void hessian(SparseMatrix<double, Eigen::RowMajor>& H, 
					const VectorXd& Fvec,
					const MatrixXi& T,
					const VectorXd& rest_tet_vols,
					const MatrixXd& Uvec);

		void set_muscle_mag(const int step);
	}
}

#endif