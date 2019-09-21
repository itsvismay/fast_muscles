
#include <igl/readOBJ.h>
#include <igl/readDMAT.h>
#include "read_config_files.h"
#include <fstream>
#include <ostream>
#include "store.h"
using Store = famu::Store;



using namespace Eigen;
using json = nlohmann::json;
// json j_input;

void famu::read_config_files(Store& store)
{
    std::cout<<"EIGEN"<<std::endl;
    std::cout<<EIGEN_MAJOR_VERSION<<std::endl;
    std::cout<<EIGEN_MINOR_VERSION<<std::endl;

    store.alpha_arap = store.jinput["alpha_arap"];
    store.alpha_neo = store.jinput["alpha_neo"];

    std::string datafile = store.jinput["data"];
    //Read Mesh
    igl::readDMAT(datafile+"/generated_files/tet_mesh_V.dmat", store.V);
    igl::readDMAT(datafile+"/generated_files/tet_mesh_T.dmat", store.T);
    igl::readDMAT(datafile+"/generated_files/combined_fiber_directions.dmat", store.Uvec);
    igl::readDMAT(datafile+"/generated_files/tet_is_tendon.dmat", store.relativeStiffness);
    // store.V /= 30;
    
    //Read Geometry
    json j_geometries;
    std::ifstream muscle_config_file(datafile+"/config.json");
    muscle_config_file >> j_geometries;

    json j_muscles = j_geometries["muscles"];
    json j_bones = j_geometries["bones"];
    json j_joints = j_geometries["joints"];

    std::vector<std::string> fixed = store.jinput["fix_bones"];
    store.fix_bones.insert(store.fix_bones.end(), fixed.begin(), fixed.end());
    for(int t = 0; t<store.T.rows(); t++){
        //TODO: update to be parametrized by input mU
        Vector3d b = store.Uvec.row(t);
        if(b!=b || b.norm()==0){
            b = 0*Vector3d::UnitY();
            store.Uvec.row(t) = b;
        }
    }
   
    //these bones are fixed, store them at the front of the
    //vector and save (names, index)
    int count_index =0;
    for(int i=0; i<store.fix_bones.size(); i++){
        VectorXi bone_i;
        igl::readDMAT(datafile+"/generated_files/"+store.fix_bones[i]+"_bone_indices.dmat", bone_i);
        store.bone_tets.push_back(bone_i);
        store.bone_name_index_map[store.fix_bones[i]]  = count_index;
        j_bones.erase(store.fix_bones[i]);
        count_index +=1;
    }

    for(json::iterator it = j_bones.begin(); it != j_bones.end(); ++it){
        VectorXi bone_i;
        igl::readDMAT(datafile+"/generated_files/"+it.key()+"_bone_indices.dmat", bone_i);
        store.bone_tets.push_back(bone_i);
        store.bone_name_index_map[it.key()] = count_index;
        count_index +=1;
    }

    count_index = 0;
    for(json::iterator it = j_muscles.begin(); it != j_muscles.end(); ++it){
        VectorXi muscle_i;
        igl::readDMAT(datafile+"/generated_files/"+it.key()+"_muscle_indices.dmat", muscle_i);
        store.muscle_tets.push_back(muscle_i);
        store.muscle_name_index_map[it.key()] = count_index;
        count_index +=1;
    }

    count_index =0;
    for(json::iterator it = j_joints.begin(); it!= j_joints.end(); ++it){
        MatrixXd joint_i;
        MatrixXi joint_f;
        std::string joint_name = it.value()["location_obj"];
        igl::readOBJ(datafile+"/objs/"+joint_name, joint_i, joint_f);
        std::vector<std::string> bones = it.value()["bones"];
        store.joint_bones_verts.push_back(std::make_pair( bones, joint_i));
    }

    std::vector<std::string> contract_muscle_names = store.jinput["contract_muscles"];
    store.contract_muscles.clear();
    for(int i=0; i<contract_muscle_names.size(); i++){
        store.contract_muscles.push_back(store.muscle_name_index_map[contract_muscle_names[i]]);
    
    }

    if(store.relativeStiffness.size()==0){
        store.relativeStiffness = VectorXd::Ones(store.T.rows());
    }else{
        store.relativeStiffness *=10;
    }
}
