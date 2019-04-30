#ifndef STORE 
#define STORE 

#include <json.hpp>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <igl/Timer.h>
#include <Eigen/LU>
#include <Eigen/UmfPackSupport>

namespace famu{

	struct Store{
		nlohmann::json jinput;
		igl::Timer timer;

		Eigen::VectorXd muscle_mag;

		Eigen::MatrixXd V, discV;
		Eigen::MatrixXi T, discT, F;
		Eigen::MatrixXd Uvec;
		std::vector<std::string> fix_bones = {};
		std::vector<Eigen::VectorXi> bone_tets = {};
		std::vector<Eigen::VectorXi> muscle_tets = {};
		std::map<std::string, int> bone_name_index_map;
		std::map<std::string, int> muscle_name_index_map;
		std::vector< std::pair<std::vector<std::string>, Eigen::MatrixXd>> joint_bones_verts;
		Eigen::VectorXd relativeStiffness;
		Eigen::VectorXd eY, eP;

		std::vector<int> fixverts, movverts;
		std::vector<int> mfix, mmov;

		Eigen::SparseMatrix<double> dF;
		Eigen::SparseMatrix<double> D, _D;
		Eigen::SparseMatrix<double> C;
		Eigen::SparseMatrix<double> S;
		Eigen::SparseMatrix<double> ConstrainProjection, UnconstrainProjection;
		Eigen::SparseMatrix<double> StDtDS;

		Eigen::VectorXd dFvec;
		Eigen::VectorXd x, dx, x0;

		// Eigen::SparseLU<Eigen::SparseMatrix<double>> SPLU;
		Eigen::UmfPackLU<Eigen::SparseMatrix<double>> SPLU;
		
		//Fast Terms
		Eigen::VectorXd DSx0;
		Eigen::SparseMatrix<double> DSx0_mat;
		Eigen::SparseMatrix<double> StDt_dF_DSx0;
		Eigen::VectorXd x0tStDt_dF_DSx0;
		Eigen::SparseMatrix<double> x0tStDt_dF_dF_DSx0;
		Eigen::SparseMatrix<double> fastMuscles;

	};
}

#endif