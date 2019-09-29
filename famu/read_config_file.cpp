
#include <igl/readOBJ.h>
#include <igl/readDMAT.h>
#include "read_config_files.h"
#include <fstream>
#include <ostream>



using namespace Eigen;
using json = nlohmann::json;
// json j_input;

void famu::read_config_files(Eigen::MatrixXd& V, 
                        Eigen::MatrixXi& T, 
                        Eigen::MatrixXi& F, 
                        Eigen::MatrixXd& Uvec, 
                        std::map<std::string, int>& bone_name_index_map,
                        std::map<std::string, int>& muscle_name_index_map,
                        std::vector< std::pair<std::vector<std::string>, 
                        Eigen::MatrixXd>>& joint_bones_verts,
                        std::vector<VectorXi>& bone_tets,
                        std::vector<VectorXi>& muscle_tets,
                        std::vector<std::string>& fix_bones,
                        Eigen::VectorXd& relativeStiffness,
                        std::vector<int>& contract_muscles,
                        json& j_input)
{
    std::cout<<"EIGEN"<<std::endl;
    std::cout<<EIGEN_MAJOR_VERSION<<std::endl;
    std::cout<<EIGEN_MINOR_VERSION<<std::endl;


    std::string datafile = j_input["data"];
    //Read Mesh
    igl::readDMAT(datafile+"/generated_files/tet_mesh_V.dmat", V);
    igl::readDMAT(datafile+"/generated_files/tet_mesh_T.dmat", T);
    igl::readDMAT(datafile+"/generated_files/combined_fiber_directions.dmat", Uvec);
    igl::readDMAT(datafile+"/generated_files/tet_is_tendon.dmat", relativeStiffness);

    //Read Geometry
    json j_geometries;
    std::ifstream muscle_config_file(datafile+"/config.json");
    muscle_config_file >> j_geometries;

    json j_muscles = j_geometries["muscles"];
    json j_bones = j_geometries["bones"];
    json j_joints = j_geometries["joints"];

    std::vector<std::string> fixed = j_input["fix_bones"];
    fix_bones.insert(fix_bones.end(), fixed.begin(), fixed.end());
    for(int t = 0; t<T.rows(); t++){
        //TODO: update to be parametrized by input mU
        Vector3d b = Uvec.row(t);
        if(b!=b || b.norm()==0){
            b = 0*Vector3d::UnitY();
            Uvec.row(t) = b;
        }
    }
   
    //these bones are fixed, store them at the front of the
    //vector and save (names, index)
    int count_index =0;
    for(int i=0; i<fix_bones.size(); i++){
        VectorXi bone_i;
        igl::readDMAT(datafile+"/generated_files/"+fix_bones[i]+"_bone_indices.dmat", bone_i);
        bone_tets.push_back(bone_i);
        bone_name_index_map[fix_bones[i]]  = count_index;
        j_bones.erase(fix_bones[i]);
        count_index +=1;
    }

    for(json::iterator it = j_bones.begin(); it != j_bones.end(); ++it){
        VectorXi bone_i;
        igl::readDMAT(datafile+"/generated_files/"+it.key()+"_bone_indices.dmat", bone_i);
        bone_tets.push_back(bone_i);
        bone_name_index_map[it.key()] = count_index;
        count_index +=1;
    }

    count_index = 0;
    for(json::iterator it = j_muscles.begin(); it != j_muscles.end(); ++it){
        VectorXi muscle_i;
        igl::readDMAT(datafile+"/generated_files/"+it.key()+"_muscle_indices.dmat", muscle_i);
        muscle_tets.push_back(muscle_i);
        muscle_name_index_map[it.key()] = count_index;
        count_index +=1;
    }

    count_index =0;
    for(json::iterator it = j_joints.begin(); it!= j_joints.end(); ++it){
        MatrixXd joint_i;
        MatrixXi joint_f;
        std::string joint_name = it.value()["location_obj"];
        igl::readOBJ(datafile+"/objs/"+joint_name, joint_i, joint_f);
        std::vector<std::string> bones = it.value()["bones"];
        joint_bones_verts.push_back(std::make_pair( bones, joint_i));
    }

    std::vector<std::string> contract_muscle_names = j_input["contract_muscles"];
    contract_muscles.clear();
    for(int i=0; i<contract_muscle_names.size(); i++){
        contract_muscles.push_back(muscle_name_index_map[contract_muscle_names[i]]);
    
    }

    if(relativeStiffness.size()==0){
        relativeStiffness = VectorXd::Ones(T.rows());
    }else{
        relativeStiffness *=10;
    }
}
