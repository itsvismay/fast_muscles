#ifndef ACAP_SOLVE_ENERGY_GRADIENT
#define ACAP_SOLVE_ENERGY_GRADIENT

#include "store.h"
#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include<Eigen/LU>


using namespace famu;
using namespace std;
using Store = famu::Store;

namespace famu
{
	namespace acap
	{

		void mass_matrix_mesh(Eigen::SparseMatrix<double, Eigen::RowMajor>& M, Eigen::Ref<const Eigen::MatrixXi> T, double density, Eigen::Ref<const Eigen::VectorXd> v0);
		void mass_matrix_linear_tetrahedron(Eigen::Matrix<double, 12,12> &M, double density, double volume);
		void mesh_collisions(Store& store, Eigen::MatrixXd& DR);
		double energy(Store& store);

		double fastEnergy(Store& store, Eigen::VectorXd& dFvec);

		void fastGradient(Store& store, Eigen::VectorXd& grad);

		void fastHessian(Store& store, Eigen::SparseMatrix<double, Eigen::RowMajor>& hess, Eigen::MatrixXd& denseHess, bool include_dense=false);

		Eigen::VectorXd fd_gradient(Store& store);

		Eigen::MatrixXd fd_hessian(Store& store);

		void external_forces(Store& store, Eigen::VectorXd& f_ext, bool first=false);

		void solve(Store& store, Eigen::VectorXd& dFvec, bool solve1 = true);

		void setJacobian(Store& store, bool include_dense = false);

	}
}

#endif