#ifndef VERTEX_BOUNDARY_CONDITIONS 
#define VERTEX_BOUNDARY_CONDITIONS
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>

namespace famu
{
		void vertex_bc(std::vector<int>& mmov, 
			std::vector<int>& mfix, 
			Eigen::SparseMatrix<double, Eigen::RowMajor>& mFree, 
			Eigen::SparseMatrix<double, Eigen::RowMajor>& mConstrained,
			Eigen::MatrixXd& mV);
		void penalty_spring_bc(std::vector<std::pair<int,int>>& springs, Eigen::SparseMatrix<double, Eigen::RowMajor>& mP, Eigen::MatrixXd& mV);
		std::vector<int> getMaxVerts_Axis_Tolerance(Eigen::MatrixXi& mT, Eigen::MatrixXd& mV, int dim, double tolerance, Eigen::VectorXi& muscle);
		std::vector<int> getMinVerts_Axis_Tolerance(Eigen::MatrixXi& mT, Eigen::MatrixXd& mV, int dim, double tolerance, Eigen::VectorXi& muscle);
		std::vector<int> getMidVerts_Axis_Tolerance(Eigen::MatrixXd& mV, int dim, double tolerance, bool left);
		void make_closest_point_springs(Eigen::MatrixXi& mT, Eigen::MatrixXd& mV, Eigen::VectorXi& muscle, std::vector<int>& points, std::vector<std::pair<int,int>>& springs);

}

#endif