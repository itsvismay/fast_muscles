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
		Eigen::VectorXi bone_or_muscle;
		Eigen::VectorXd rest_tet_volume;
		
		double alpha_arap = 1e4;
		double alpha_neo = 1;

		Eigen::MatrixXd V, discV;
		Eigen::MatrixXi T, discT, F, discF;
		Eigen::MatrixXd Uvec;
		std::vector<std::string> fix_bones = {};
		std::vector<Eigen::VectorXi> bone_tets = {};
		std::vector<Eigen::VectorXi> muscle_tets = {};
		std::map<std::string, int> bone_name_index_map;
		std::map<std::string, int> muscle_name_index_map;
		std::vector< std::pair<std::vector<std::string>, Eigen::MatrixXd>> joint_bones_verts;
		Eigen::VectorXd relativeStiffness;
		Eigen::VectorXd eY, eP;
                // Young's Modulus per vertex (averaged from incident tets using eY)
		Eigen::VectorXd elogVY;

		std::vector<int> fixverts, movverts;
		std::vector<int> mfix, mmov;

		Eigen::SparseMatrix<double> dF;
		Eigen::SparseMatrix<double> D, _D;
		Eigen::SparseMatrix<double> C;
		Eigen::SparseMatrix<double> S;
		Eigen::SparseMatrix<double> ProjectF, PickBoneF;
		Eigen::SparseMatrix<double> ConstrainProjection, UnconstrainProjection;
		Eigen::SparseMatrix<double> JointConstraints, NullJ;
		Eigen::SparseMatrix<double> Y, Bx, Bf;
		Eigen::MatrixXd G;

		Eigen::VectorXd dFvec, BfI0;
		Eigen::VectorXd x, dx, x0, lambda2, acap_solve_result, acap_solve_rhs;

		
		// Eigen::SparseLU<Eigen::SparseMatrix<double>> SPLU;
		Eigen::UmfPackLU<Eigen::SparseMatrix<double>> ACAP_KKT_SPLU;
		Eigen::UmfPackLU<Eigen::SparseMatrix<double>> NM_SPLU;//TODO: optimize this away
		
		
		//Fast Terms
		double x0tStDtDSx0;
		Eigen::VectorXd x0tStDtDSY;
		Eigen::SparseMatrix<double> YtStDtDSY; //ddE/dxdx
		Eigen::VectorXd x0tStDt_dF_DSx0;
		Eigen::SparseMatrix<double> YtStDt_dF_DSx0; //ddE/dxdF
		Eigen::SparseMatrix<double> x0tStDt_dF_dF_DSx0; //ddE/dFdF

		Eigen::VectorXd DSx0;
		Eigen::SparseMatrix<double> DSY;
		Eigen::SparseMatrix<double> StDtDS;
		Eigen::SparseMatrix<double> DSx0_mat;
		std::vector<Eigen::SparseMatrix<double>> fastMuscles;
		Eigen::SparseMatrix<double> JacdxdF;

		//Hessians
		Eigen::SparseMatrix<double> acapHess, muscleHess, neoHess;
		Eigen::MatrixXd denseAcapHess, denseMuscleHess, denseNeoHess;

		//woodbury matrices
		//(A + BCD)-1
		//A = Hneo + Hmuscle + P'Z'ZP
		//B = -P'Z'DSYG
		//C = Inv(GtYtStDtDSYG)
		//D = G'Y'S'D'ZP
		Eigen::MatrixXd WoodB, WoodD, InvC, WoodC;
		Eigen::VectorXd eigenvalues;
		std::vector<int> contract_muscles;




	};
}

#endif
