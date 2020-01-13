
#ifndef READ_CONFIG_FILES 
#define READ_CONFIG_FILES

#include <Eigen/Dense>
#include <iostream>
#include <json.hpp>
#include "store.h"



namespace famu
{

	void read_config_files(famu::Store& store,
						Eigen::MatrixXd& V, 
					    Eigen::MatrixXi& T, 
					    Eigen::MatrixXi& F, 
					    Eigen::MatrixXd& Uvec, 
					    std::map<std::string, int>& bone_name_index_map,
					    std::map<std::string, int>& muscle_name_index_map,
					    std::vector< std::pair<std::vector<std::string>, 
					    Eigen::MatrixXd>>& joint_bones_verts,
					    std::vector<Eigen::VectorXi>& bone_tets,
					    std::vector<Eigen::VectorXi>& muscle_tets,
					    std::vector<std::string>& fix_bones,
					    std::vector<std::string>& script_bones,
					    Eigen::VectorXd& relativeStiffness,
					    std::vector<int>& contract_muscles,
					    std::vector<nlohmann::json>& muscle_script_steps,
					    nlohmann::json& jinput);
}

#endif
