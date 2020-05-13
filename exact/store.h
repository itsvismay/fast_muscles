#ifndef STORE 
#define STORE 
#include <json.hpp>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <igl/Timer.h>
#include <Eigen/LU>
#include <igl/writeOBJ.h>
#include <Eigen/IterativeLinearSolvers>

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

namespace exact{

	struct Store{
		nlohmann::json jinput;
		igl::Timer timer;

		Eigen::MatrixXd V, discV, J;
		Eigen::MatrixXi T, discT, F, discF;
		Eigen::MatrixXd Uvec;
		std::vector<double> bone_vols;
		std::vector<std::string> fix_bones = {}, script_bones = {};
		std::vector<Eigen::VectorXi> bone_tets = {}, muscle_tets = {};
		std::map<std::string, int> bone_name_index_map, muscle_name_index_map;
		std::vector< std::pair<std::vector<std::string>, Eigen::MatrixXd>> joint_bones_verts;

		std::vector<std::pair<Eigen::MatrixXi, Eigen::MatrixXi>> contact_muscle_T_F;
		std::vector<std::pair<Eigen::MatrixXi, Eigen::MatrixXi>> contact_bone_T_F;

		Eigen::SparseMatrix<double, Eigen::RowMajor> Y, B, P, M, H_n, H_m, H_a;
		Eigen::VectorXd x0, b, rest_tet_vols, muscle_mag, eY, eP, elogVY, relativeStiffness,
			grad_n, grad_m;

		#ifdef __linux__
		#else
		#endif
		
		
		

		
		std::vector<int> contract_muscles, draw_points;
		std::vector<nlohmann::json> muscle_steps;


		nlohmann::json joutput = {{"info",nlohmann::json::object()}, {"run", nlohmann::json::array()}, {"summary",nlohmann::json::object()}};
		
		int printState(int step, std::string name, Eigen::VectorXd& y){
			std::string outputfile = jinput["output"];
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
