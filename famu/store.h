#ifndef STORE 
#define STORE 
#include <json.hpp>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <igl/Timer.h>
#include <Eigen/LU>
#include <igl/writeOBJ.h>

#ifdef __linux__
#include <Eigen/PardisoSupport>
#include <omp.h>
#endif


#define NUM_MODES 48
typedef Eigen::Triplet<double> Trip;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;
typedef Eigen::Matrix<double, 9, NUM_MODES> Matrix9xModes;
typedef Eigen::Matrix<double, NUM_MODES, NUM_MODES> MatrixModesxModes;

namespace famu{

	struct Store{
		nlohmann::json jinput;
		igl::Timer timer;

		Eigen::VectorXd muscle_mag;
		Eigen::VectorXi bone_or_muscle;
		Eigen::VectorXd rest_tet_volume;
		
		double alpha_arap = 1e4;
		double alpha_neo = 1;
		double gradNormConvergence = 1e-4;

		Eigen::MatrixXd V, discV;
		Eigen::MatrixXi T, discT, F, discF;
		Eigen::MatrixXd Uvec;
		std::vector<std::string> fix_bones = {};
		std::vector<std::string> script_bones = {};
		std::vector<Eigen::VectorXi> bone_tets = {};
		std::vector<Eigen::VectorXi> muscle_tets = {};
		std::map<std::string, int> bone_name_index_map;
		std::map<std::string, int> muscle_name_index_map;
		std::vector< std::pair<std::vector<std::string>, Eigen::MatrixXd>> joint_bones_verts;
		Eigen::VectorXd relativeStiffness;
		Eigen::VectorXd eY, eP;
        
        // Young's Modulus per vertex (averaged from incident tets using eY)
		Eigen::VectorXd elogVY;
		std::vector<double> bone_vols;
		std::vector<int> fixverts, movverts;
		std::vector<int> mfix, mmov;

		std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> contact_components;

		Eigen::SparseMatrix<double, Eigen::RowMajor> dF;
		Eigen::SparseMatrix<double, Eigen::RowMajor> D, _D;
		Eigen::SparseMatrix<double, Eigen::RowMajor> C;
		Eigen::SparseMatrix<double, Eigen::RowMajor> S;
		Eigen::SparseMatrix<double, Eigen::RowMajor> ProjectF, RemFixedBones;
		Eigen::SparseMatrix<double, Eigen::RowMajor> ConstrainProjection, UnconstrainProjection;
		Eigen::SparseMatrix<double, Eigen::RowMajor> JointConstraints, NullJ;
		Eigen::SparseMatrix<double, Eigen::RowMajor> Y, Bx, Bf, Bsx, ScriptBonesY, PickBoneF;
		Eigen::MatrixXd G;

		Eigen::VectorXd dFvec, BfI0, I0, Bsx_vec;
		Eigen::VectorXd x, dx, x0, lambda2, acap_solve_result,acap_solve_result2, acap_solve_rhs, acap_solve_rhs2;

		
		#ifdef __linux__
		Eigen::PardisoLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> ACAP_KKT_SPLU, ACAP_KKT_SPLU2;
		Eigen::PardisoLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> NM_SPLU;//TODO: optimize this away
		#else
		Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> ACAP_KKT_SPLU, ACAP_KKT_SPLU2;
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> NM_SPLU;//TODO: optimize this away
		#endif


		Eigen::VectorXd acaptmp_sizex;
		Eigen::VectorXd acaptmp_sizedFvec1, acaptmp_sizedFvec2;
		
		
		//Fast Terms
		double x0tStDtDSx0;
		Eigen::VectorXd x0tStDtDSY, x0tStDtDSY2;
		Eigen::SparseMatrix<double, Eigen::RowMajor> YtStDtDSY; //ddE/dxdx
		Eigen::VectorXd x0tStDt_dF_DSx0;
		Eigen::SparseMatrix<double, Eigen::RowMajor> YtStDt_dF_DSx0, YtStDt_dF_DSx02; //ddE/dxdF
		Eigen::SparseMatrix<double, Eigen::RowMajor> x0tStDt_dF_dF_DSx0; //ddE/dFdF

		Eigen::VectorXd DSx0;
		Eigen::SparseMatrix<double, Eigen::RowMajor> DSY;
		Eigen::SparseMatrix<double, Eigen::RowMajor> StDtDS;
		Eigen::SparseMatrix<double, Eigen::RowMajor> DSx0_mat;
		std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> fastMuscles;
		Eigen::SparseMatrix<double, Eigen::RowMajor> JacdxdF;

		//Hessians
		Eigen::SparseMatrix<double, Eigen::RowMajor> acapHess, muscleHess, neoHess;
		Eigen::MatrixXd denseAcapHess, denseMuscleHess, denseNeoHess;

		//woodbury matrices
		//(A + BCD)-1
		//A = Hneo + Hmuscle + P'Z'ZP
		//B = -P'Z'DSYG
		//C = Inv(GtYtStDtDSYG)
		//D = G'Y'S'D'ZP
		std::vector<Eigen::LDLT<Matrix9d>> vecInvA;
		Eigen::MatrixXd WoodB, WoodD, InvC, WoodC;
		Eigen::VectorXd eigenvalues;
		std::vector<int> contract_muscles;
		std::vector<nlohmann::json> muscle_steps;

		//External Force matrices
		Eigen::VectorXd YtMg, ContactForce;
		Eigen::SparseMatrix<double, Eigen::RowMajor> ContactP, ContactP1, ContactP2, ContactHess;
		nlohmann::json joutput = {{"info",nlohmann::json::object()}, {"run", nlohmann::json::array()}, {"summary",nlohmann::json::object()}};

		int printState(int step, std::string name){
			std::string outputfile = jinput["output"];
			Eigen::VectorXd y = Y*x;
			Eigen::Map<Eigen::MatrixXd> newV(y.data(), V.cols(), V.rows());

			igl::writeOBJ(outputfile+"/"+name+std::to_string(step)+".obj", (newV.transpose()+V), F); //output mesh
		
		}

		int saveResults(){
			/*
			{	
				#Mesh info
				{T: ,
				 V: ,
				 F: ,
				 Threads: ,
				 convergence: ,
				 alpha: ,

				 },

				#Each static solve
				{total_ls_iters: ,
				 total_nm_iters: ,
				 total_nm_time: ,
				 total_woodbury_time: ,
				 total_ls_time: 
				 muscle_activations: ,
				 acap_grad: ,
				 muscle_grad: , 
				 neo_grad: ,
				 acap_E: ,
				 muscle_E: ,
				 neo_E
				 },

				 #summary
				 {
					total_sim_time:
					num_nm_its:
					num_ls_its:
					nm_time: 
					ls_time:
					wood_time:
					avg_nm_time:
					avg_wood_time: 
					avg_ls_time:
				 }
				
			}
			*/
			double total_sim_time = 0;
			int num_nm_its = 0;
			int num_ls_its = 0;
			double nm_time = 0;
			double ls_time = 0;
			double wood_time = 0;
			double avg_nm_time = 0;
			double avg_wood_time = 0;
			double avg_ls_time = 0;

			int ii=0;
			for(ii=0; ii<joutput["run"].size(); ii++){
				nlohmann::json step = joutput["run"][ii];
				total_sim_time += (double) step["total_nm_time"];
				num_nm_its += (int) step["total_nm_iters"];
				num_ls_its += (int) step["total_ls_iters"];
				nm_time += (double) step["total_nm_time"];
				ls_time += (double) step["total_ls_time"];
				wood_time += (double) step["total_woodbury_time"];
			}
			avg_nm_time = nm_time/ii;
			avg_wood_time = wood_time/ii;
			avg_ls_time = ls_time/ii;
			joutput["summary"]["total_sim_time"] = total_sim_time;
			joutput["summary"]["num_nm_its"] = num_nm_its;
			joutput["summary"]["num_ls_its"] = num_ls_its;
			joutput["summary"]["nm_time"] = nm_time;
			joutput["summary"]["ls_time"] = ls_time;
			joutput["summary"]["wood_time"] = wood_time;
			joutput["summary"]["avg_nm_time"] = avg_nm_time;
			joutput["summary"]["avg_wood_time"] = avg_wood_time;
			joutput["summary"]["avg_ls_time"] = avg_ls_time;


			std::string out = jinput["output"];
			std::ofstream o(out+"/results.json");
			o << std::setw(4) << joutput << std::endl;

		}



	};


}

#endif
