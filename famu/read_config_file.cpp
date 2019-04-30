
#include <igl/readOBJ.h>
#include <igl/readDMAT.h>
#include "read_config_files.h"



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
                        json& j_input)
{
    std::string datafile = j_input["data"];

    //Read Mesh
    igl::readDMAT(datafile+"/generated_files/tet_mesh_V.dmat", V);
    igl::readDMAT(datafile+"/generated_files/tet_mesh_T.dmat", T);
    igl::readDMAT(datafile+"/generated_files/combined_fiber_directions.dmat", Uvec);
    igl::readDMAT(datafile+"/generated_files/combined_relative_stiffness.dmat", relativeStiffness);
    
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
            b = Vector3d::UnitY();
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

    if(relativeStiffness.size()==0){
        relativeStiffness = VectorXd::Ones(T.rows());
    }else{
        // cout<<relativeStiffness.transpose()<<endl;
        // for(int i=0; i<relativeStiffness.size(); i++){
        //     if( relativeStiffness[i]>7){//V.row(T.row(i)[0])[1]>0
        //         relativeStiffness[i] = 1000;//*relativeStiffness[i]*relativeStiffness[i];
        //     }else{
        //         relativeStiffness[i] = 1;
        //     }
        // }

        // relativeStiffness = relativeStiffness/1e12;
        // for(int i=0; i<relativeStiffness.size(); i++){
        //     if( relativeStiffness[i]>3){//V.row(T.row(i)[0])[1]>0
        //         relativeStiffness[i] = 1000;//*relativeStiffness[i]*relativeStiffness[i];
        //     }else{
        //         relativeStiffness[i] = 1;
        //     }
        // }


        //Euclidean Distance tendons for biceps
        int axis = 1;
        double maxs = V.col(axis).maxCoeff();
        double mins = V.col(axis).minCoeff();
        for(int m=0; m<muscle_tets.size(); m++){
            for(int i=0; i<muscle_tets[m].size(); i++){
                int t= muscle_tets[m][i];
                if(fabs(V.row(T.row(t)[0])[axis] - maxs) < 3.5){
                    relativeStiffness[t] = 1;
                }else if(fabs(V.row(T.row(t)[0])[axis] - mins) < 3){
                    relativeStiffness[t] = 1;
                }else{
                    relativeStiffness[t] = 1;
                }
            }
        }
        // cout<<relativeStiffness.transpose()<<endl;

    }
}