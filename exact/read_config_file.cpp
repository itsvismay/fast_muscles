
#include <igl/readOBJ.h>
#include <igl/readDMAT.h>
#include <igl/boundary_facets.h>

#include "store.h"
#include "read_config_files.h"
#include <fstream>
#include <iostream>



using namespace Eigen;
using json = nlohmann::json;
// json j_input;

void exact::read_config_files(exact::Store& store)
{

    std::string datafile = store.jinput["data"];
    //Read Mesh
    igl::readDMAT(datafile+"/generated_files/tet_mesh_V.dmat", store.V);
    igl::readDMAT(datafile+"/generated_files/tet_mesh_T.dmat", store.T);
    igl::readDMAT(datafile+"/generated_files/combined_fiber_directions.dmat", store.Uvec);
    igl::readDMAT(datafile+"/generated_files/tet_is_tendon.dmat", store.relativeStiffness);
    igl::boundary_facets(store.T, store.F);
    //Read Geometry
    json j_geometries;
    std::ifstream muscle_config_file(datafile+"/config.json");
    muscle_config_file >> j_geometries;

    json j_muscles = j_geometries["muscles"];
    json j_bones = j_geometries["bones"];
    json j_joints = j_geometries["joints"];

    std::vector<std::string> fixed = store.jinput["fix_bones"];
    store.fix_bones.insert(store.fix_bones.end(), fixed.begin(), fixed.end());
    std::vector<std::string> scripted = store.jinput["script_bones"];
    store.script_bones.insert(store.script_bones.end(), scripted.begin(), scripted.end());

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
    for(int i=0; i<store.script_bones.size(); i++){
        VectorXi bone_i;
        igl::readDMAT(datafile+"/generated_files/"+store.script_bones[i]+"_bone_indices.dmat", bone_i);
        store.bone_tets.push_back(bone_i);
        store.bone_name_index_map[store.script_bones[i]]  = count_index;
        j_bones.erase(store.script_bones[i]);
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

    std::vector<nlohmann::json> temp = store.jinput["muscle_starting_strength"];
    store.muscle_steps = temp;

    if(store.relativeStiffness.size()==0){
        store.relativeStiffness = VectorXd::Ones(store.T.rows());
    }else{
        store.relativeStiffness *=10;
    }

    if(store.jinput["springk"]!=0){
        //contact is ON
        for(int i=0; i<store.muscle_tets.size(); i++){
            MatrixXi Ti = MatrixXi::Zero(store.muscle_tets[i].size(), 4);
            MatrixXi Fi;
            for(int k=0; k<store.muscle_tets[i].size(); k++){
                Ti.row(k) = store.T.row(store.muscle_tets[i][k]);
            }

            igl::boundary_facets(Ti, Fi);
            store.contact_muscle_T_F.push_back(std::make_pair(Ti, Fi));

        }

        for(int i=0; i<store.bone_tets.size(); i++){
            MatrixXi Ti = MatrixXi::Zero(store.bone_tets[i].size(), 4);
            MatrixXi Fi;
            for(int k=0; k<store.bone_tets[i].size(); k++){
                Ti.row(k) = store.T.row(store.bone_tets[i][k]);
            }

            igl::boundary_facets(Ti, Fi);
            store.contact_bone_T_F.push_back(std::make_pair(Ti, Fi));

        }
        
    }
}
