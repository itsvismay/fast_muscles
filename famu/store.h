#ifndef STORE 
#define STORE 

#include <json.hpp>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <igl/Timer.h>
#include <Eigen/LU>
#include <Eigen/UmfPackSupport>
typedef Eigen::Triplet<double> Trip;

namespace famu{

	struct Store{
		nlohmann::json jinput;
		igl::Timer timer;

		Eigen::VectorXd muscle_mag;
		double alpha_arap = 1e3;
		double alpha_neo = 1;

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
		Eigen::SparseMatrix<double> JointConstraints;
		Eigen::SparseMatrix<double> Y;

		Eigen::VectorXd dFvec;
		Eigen::VectorXd x, dx, x0;

		// Eigen::SparseLU<Eigen::SparseMatrix<double>> SPLU;
		Eigen::UmfPackLU<Eigen::SparseMatrix<double>> SPLU;
		
		//Fast Terms
		double x0tStDtDSx0;
		Eigen::VectorXd x0tStDtDSY;
		Eigen::SparseMatrix<double> YtStDtDSY;
		Eigen::VectorXd x0tStDt_dF_DSx0;
		Eigen::SparseMatrix<double> YtStDt_dF_DSx0;
		Eigen::SparseMatrix<double> x0tStDt_dF_dF_DSx0;

		Eigen::VectorXd DSx0;
		Eigen::SparseMatrix<double> DSY;
		Eigen::SparseMatrix<double> StDtDS;
		Eigen::SparseMatrix<double> DSx0_mat;
		Eigen::SparseMatrix<double> fastMuscles;

	};
}

#endif