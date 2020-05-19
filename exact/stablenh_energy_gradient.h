#ifndef STABLENH_ENERGY_GRAD
#define STABLENH_ENERGY_GRAD

#include "store.h"

using namespace exact;
using namespace Eigen;
using Store = exact::Store;

namespace exact
{
	namespace stablenh
	{	

		//output: energy
		//inputs: deformation gradient
		double energy(Eigen::VectorXd& Fvec,
						const Eigen::MatrixXi& T,
						const Eigen::VectorXd& eY,
						const Eigen::VectorXd& eP,
						const Eigen::VectorXd& rest_tet_vols);
		void gradient(VectorXd& grad,
						const Eigen::VectorXd& Fvec,
						const Eigen::MatrixXi& T,
						const Eigen::VectorXd& eY,
						const Eigen::VectorXd& eP,
						const Eigen::VectorXd& rest_tet_vols);
		void hessian(Eigen::SparseMatrix<double, Eigen::RowMajor>& hess, 
						const Eigen::VectorXd& Fvec,
						const Eigen::MatrixXi& T,
						const Eigen::VectorXd& eY,
						const Eigen::VectorXd& eP,
						const Eigen::VectorXd& rest_tet_vols);

		//Treats bones as a single def gradient
		//output: energy
		//inputs: deformation gradient, tet is bone or muscle
		double energy(Eigen::VectorXd& Fvec,
						const Eigen::VectorXi& bone_or_muscle,
						const Eigen::MatrixXi& T,
						const Eigen::VectorXd& eY,
						const Eigen::VectorXd& eP,
						const Eigen::VectorXd& rest_tet_vols);
		void gradient(Eigen::VectorXd& grad, 
					const Eigen::VectorXd& Fvec, 
					const Eigen::VectorXi& bone_or_muscle,
					const Eigen::MatrixXi& T,
					const Eigen::VectorXd& eY,
					const Eigen::VectorXd& eP,
					const Eigen::VectorXd& rest_tet_vols,
					const std::vector<double>& bone_vols,
					const std::vector<Eigen::VectorXi>& muscle_tets,
					const std::vector<Eigen::VectorXi>& bone_tets);
		void hessian(Eigen::SparseMatrix<double, Eigen::RowMajor>& hess, 
					const Eigen::VectorXd& Fvec, 
					const Eigen::VectorXi& bone_or_muscle,
					const Eigen::MatrixXi& T,
					const Eigen::VectorXd& eY,
					const Eigen::VectorXd& eP,
					const Eigen::VectorXd& rest_tet_vols,
					const std::vector<double>& bone_vols,
					const std::vector<Eigen::VectorXi>& muscle_tets,
					const std::vector<Eigen::VectorXi>& bone_tets);

	}
}
#endif