#include "mesh.h"
#include "arap.h"
#include "redArap.h"
#include "elastic.h"
#include "solver.h"
#include "redSolver.h"
#include <Eigen/Core>
#include <iostream>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
#include <igl/readOBJ.h>
#include <igl/jet.h>
#include <igl/slice.h>
#include <igl/boundary_facets.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/Timer.h>

using namespace LBFGSpp;
using json = nlohmann::json;
using namespace Eigen;
using namespace std;
json j_input;


void readConfigFile(MatrixXd& V, 
    MatrixXi& T, MatrixXi& F, MatrixXd& Uvec, 
    std::map<std::string, int>& bone_name_index_map,
    std::map<std::string, int>& muscle_name_index_map,
    std::vector< std::pair<std::vector<std::string>, MatrixXd>>& joint_bones_verts,
    std::vector<VectorXi>& bone_tets,
    std::vector<VectorXi>& muscle_tets,
    std::vector<std::string>& fix_bones,
    VectorXd& relativeStiffness){
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

int main(int argc, char *argv[])
{
    std::cout<<"-----Configs-------"<<std::endl;
    std::ifstream input_file("../input/input.json");
    input_file >> j_input;
    
    MatrixXd V;
    MatrixXi T;
    MatrixXi F;    
    MatrixXd Uvec;
    std::vector<int> mov = {};
    
    std::vector<std::string> fix_bones = {};
    std::vector<VectorXi> bone_tets = {};
    std::vector<VectorXi> muscle_tets = {};
    std::map<std::string, int> bone_name_index_map;
    std::map<std::string, int> muscle_name_index_map;
    std::vector< std::pair<std::vector<std::string>, MatrixXd>> joint_bones_verts;
    VectorXd relativeStiffness;
    readConfigFile(V, T, F, Uvec, bone_name_index_map, muscle_name_index_map, joint_bones_verts, bone_tets, muscle_tets, fix_bones, relativeStiffness);  
    
    cout<<"---Record Mesh Setup Info"<<endl;
    cout<<"V size: "<<V.rows()<<endl;
    cout<<"T size: "<<T.rows()<<endl;
    cout<<"F size: "<<F.rows()<<endl;
    if(argc>1){
        j_input["number_modes"] =  stoi(argv[1]);
        j_input["number_rot_clusters"] =  stoi(argv[2]);
        j_input["number_skinning_handles"] =  stoi(argv[3]);
    }
    cout<<"NSH: "<<j_input["number_skinning_handles"]<<endl;
    cout<<"NRC: "<<j_input["number_rot_clusters"]<<endl;
    cout<<"MODES: "<<j_input["number_modes"]<<endl;
    std::string outputfile = j_input["output"];
    std::string namestring = to_string((int)j_input["number_modes"])+"modes"+to_string((int)j_input["number_rot_clusters"])+"clusters"+to_string((int)j_input["number_skinning_handles"])+"handles";

    igl::boundary_facets(T, F);


    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(
        V, 
        T, 
        Uvec, 
        fix_bones, 
        bone_tets, 
        muscle_tets, 
        bone_name_index_map, 
        muscle_name_index_map, 
        joint_bones_verts,
        relativeStiffness,  
        j_input);


    std::cout<<"-----Neo-------"<<std::endl;
    Elastic* neo = new Elastic(*mesh);

    std::cout<<"-----ARAP-----"<<std::endl;
    Arap* arap = new Arap(*mesh);
  
    std::cout<<"-----Solver-------"<<std::endl;
    int DIM = mesh->red_s().size();
    Rosenbrock f(DIM, mesh, arap, neo, j_input, false);

  

  
    LBFGSParam<double> param;
    param.epsilon = 1e-1;
    // param.max_iterations = 1000;
    // param.past = 2;
    // param.m = 5;
    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
    LBFGSSolver<double> solver(param);
    igl::Timer timer;

    // for(int i=0; i<5; i++){
    //     MatrixXd newV = mesh->continuousV();
    //     string datafile = j_input["data"];
    //     igl::writeOBJ(datafile+"test"+to_string(i)+".obj",newV,F);
        
    //     double fx =0;
    //     VectorXd ns = mesh->N().transpose()*mesh->red_s();
    //     int niter = solver.minimize(f, ns, fx);
    //     cout<<"End BFGS"<<", "<<niter<<endl;
    //     VectorXd reds = mesh->N()*ns + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
    //     for(int i=0; i<reds.size(); i++){
    //         mesh->red_s()[i] = reds[i];
    //     }
        
    //     neo->changeFiberMag(1.5);
    // }
    // exit(0);

	std::cout<<"-----Display-------"<<std::endl;
    igl::opengl::glfw::Viewer viewer;
    MatrixXd Colors = MatrixXd::Random(1000,3); // 3x3 Matrix filled with random numbers between (-1,1)
    Colors = (Colors + MatrixXd::Constant(1000,3,1.))*(1-1e-6)/2.; // add 1 to the matrix to have values between 0 and 2; multiply with range/2
    Colors = (Colors + MatrixXd::Constant(1000,3,1e-6)); //set LO as the lower bound (offset)
    MatrixXd SETCOLORSMAT = MatrixXd::Zero(V.rows(), 3);
  
    int kkkk = 0;
    double tttt = 0;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & viewer){   
     //    if(viewer.core.is_animating){
     //        if(kkkk<mesh->G().cols()){
     //            VectorXd x = 10*sin(tttt)*mesh->G().col(kkkk) + mesh->x0();
     //            Eigen::Map<Eigen::MatrixXd> newV(x.data(), V.cols(), V.rows());
     //            viewer.data().set_mesh(newV.transpose(), F);
     //            tttt+= 0.1;
     //        }
        // }
        return false;
    };

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers){   
 
        std::cout<<"Key down, "<<key<<std::endl;
        viewer.data().clear();
        
        if(key=='Q'){
            kkkk +=1;
            cout<<kkkk<<endl;
        }

        if(key=='A'){
            cout<<"here"<<endl;
            neo->changeFiberMag(j_input["multiplier_strength_each_step"]);
        }


        if(key==' '){
            
            // VectorXd ns = mesh->N().transpose()*mesh->red_s();
            // for(int i=0; i<ns.size()/6; i++){
            //     ns[6*i+0] -= 0.2;
            //     ns[6*i+1] -= 0.2;
            //     ns[6*i+2] -= 0.2;
            // }
            // VectorXd reds = mesh->N()*ns + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
            // for(int i=0; i<reds.size(); i++){
            //     mesh->red_s()[i] = reds[i];
            // }
            // arap->minimize(*mesh);


            double fx =0;
            VectorXd ns = mesh->N().transpose()*mesh->red_s();
            timer.start();

            int niter = solver.minimize(f, ns, fx);
            timer.stop();
            VectorXd reds = mesh->N()*ns + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
            
            for(int i=0; i<reds.size(); i++){
                mesh->red_s()[i] = reds[i];
            }
            cout<<"****QSsteptime: "<<timer.getElapsedTimeInMicroSec()<<", "<<niter<<endl;
        
        }

        
        if(key=='D'){
            
            // Draw disc mesh
            std::cout<<std::endl;
            MatrixXd& discV = mesh->discontinuousV();
            MatrixXi& discT = mesh->discontinuousT();
            for(int m=0; m< T.rows()/10; m++){
                // for(int i=0; i<muscle_tets[m].size()/10; i++){
                    // int t = muscle_tets[m][i];
                    int t= 10*m;
                    Vector4i e = discT.row(t);
                    // std::cout<<discT.row(i)<<std::endl<<std::endl;
                    // std::cout<<discV(Eigen::placeholders::all, discT.row(i))<<std::endl;
                    Matrix<double, 1,3> p0 = discV.row(e[0]);
                    Matrix<double, 1,3> p1 = discV.row(e[1]);
                    Matrix<double, 1,3> p2 = discV.row(e[2]);
                    Matrix<double, 1,3> p3 = discV.row(e[3]);

                    viewer.data().add_edges(p0,p1,Eigen::RowVector3d(1,0,1));
                    viewer.data().add_edges(p0,p2,Eigen::RowVector3d(1,0,1));
                    viewer.data().add_edges(p0,p3,Eigen::RowVector3d(1,0,1));
                    viewer.data().add_edges(p1,p2,Eigen::RowVector3d(1,0,1));
                    viewer.data().add_edges(p1,p3,Eigen::RowVector3d(1,0,1));
                    viewer.data().add_edges(p2,p3,Eigen::RowVector3d(1,0,1));
                }
            // }
        }
        
        // //Draw continuous mesh
        MatrixXd newV = mesh->continuousV();
        viewer.data().set_mesh(newV, F);

        viewer.data().compute_normals();
        igl::writeOBJ("fullmesh.obj", newV, F);
        

        if(key=='R'){
            for(int c=0; c<mesh->red_w().size()/3; c++){
                std::vector<int> cluster_elem = mesh->r_cluster_elem_map()[c];
                for(int e=0; e<cluster_elem.size(); e++){
                    SETCOLORSMAT.row(mesh->T().row(cluster_elem[e])[0]) = Colors.row(c);
                    SETCOLORSMAT.row(mesh->T().row(cluster_elem[e])[1]) = Colors.row(c);
                    SETCOLORSMAT.row(mesh->T().row(cluster_elem[e])[2]) = Colors.row(c);
                    SETCOLORSMAT.row(mesh->T().row(cluster_elem[e])[3]) = Colors.row(c);
                }
            }
            viewer.data().set_colors(SETCOLORSMAT);
        }

        if(key=='S'){
            SETCOLORSMAT.setZero();
            if(kkkk==mesh->sW().cols()/6){
                kkkk=0;
            }
            for(int i=0; i<mesh->T().rows(); i++){
                double weight = mesh->sW().col(6*kkkk)[6*i];
                if(weight > 0){
                    SETCOLORSMAT.row(mesh->T().row(i)[0]) = weight*Colors.row(5);
                    SETCOLORSMAT.row(mesh->T().row(i)[1]) = weight*Colors.row(5);
                    SETCOLORSMAT.row(mesh->T().row(i)[2]) = weight*Colors.row(5);
                    SETCOLORSMAT.row(mesh->T().row(i)[3]) = weight*Colors.row(5);

                }
            }
            viewer.data().set_colors(SETCOLORSMAT);
        }
        if(key=='V'){
            //Display tendon areas
            MatrixXd COLRS;
            VectorXd zz = 100*VectorXd::Ones(mesh->V().rows());
            for(int i=0; i<mesh->T().rows(); i++){
                zz[mesh->T().row(i)[0]] = relativeStiffness[i];
                zz[mesh->T().row(i)[1]] = relativeStiffness[i];
                zz[mesh->T().row(i)[2]] = relativeStiffness[i];
                zz[mesh->T().row(i)[3]] = relativeStiffness[i];
            }
            igl::jet(zz, true, COLRS);
            viewer.data().set_colors(COLRS);

            // VectorXd xi = mesh->G()*mesh->red_x();
            // for(int m=0; m<mesh->T().rows(); m++){
            //     Matrix3d Dm;
            //     for(int i=0; i<3; i++){
            //         Dm.col(i) = mesh->V().row(mesh->T().row(m)[i]) - mesh->V().row(mesh->T().row(m)[3]);
            //     }
            //     Matrix3d m_InvRefShapeMatrix = Dm.inverse();
                
            //     Matrix3d Ds;
            //     for(int i=0; i<3; i++)
            //     {
            //         Ds.col(i) = xi.segment<3>(3*mesh->T().row(m)[i]) - xi.segment<3>(3*mesh->T().row(m)[3]);
            //     }

            //     Matrix3d F = Matrix3d::Identity() + Ds*m_InvRefShapeMatrix;
            //     Matrix3d StS = F.transpose()*F;

            //     double s1 = mesh->sW().row(6*m+0)*mesh->red_s();
            //     double s2 = mesh->sW().row(6*m+1)*mesh->red_s();
            //     double s3 = mesh->sW().row(6*m+2)*mesh->red_s();
            //     double s4 = mesh->sW().row(6*m+3)*mesh->red_s();
            //     double s5 = mesh->sW().row(6*m+4)*mesh->red_s();
            //     double s6 = mesh->sW().row(6*m+5)*mesh->red_s();
            //     Matrix3d s_mat;
            //     s_mat<<s1,s4,s5,s4,s2,s6,s5,s6,s3;
            //     Matrix3d StS_mat = s_mat.transpose()*s_mat;
            //     double snorm = (StS_mat - Matrix3d::Identity()).norm();
               
            //     zz[mesh->T().row(m)[0]] += snorm;
            //     zz[mesh->T().row(m)[1]] += snorm;
            //     zz[mesh->T().row(m)[2]] += snorm; 
            //     zz[mesh->T().row(m)[3]] += snorm;
            // }
            // igl::jet(zz, true, COLRS);
            // viewer.data().set_colors(COLRS);
            // neo->Energy(*mesh);
        }
        //---------------- 

        if(key=='M'){
            MatrixXd COLRS;
            // cout<<relativeStiffness.transpose()<<endl;
            VectorXd zz = 100*VectorXd::Ones(mesh->V().rows());
            for(int i=0; i<muscle_tets[0].size() ; i++){
                zz[mesh->T().row(muscle_tets[0][i])[0]] = 0;
                zz[mesh->T().row(muscle_tets[0][i])[1]] = 0;
                zz[mesh->T().row(muscle_tets[0][i])[2]] = 0;
                zz[mesh->T().row(muscle_tets[0][i])[3]] = 0;
            }
            igl::jet(zz, true, COLRS);
            viewer.data().set_colors(COLRS);

        }

        //Draw fixed and moving points
        for(int i=0; i<mesh->fixed_verts().size(); i++){
            viewer.data().add_points(mesh->V().row(mesh->fixed_verts()[i]),Eigen::RowVector3d(1,0,0));
        }
        
        if(key=='J' || key== 'D'){
            //Draw joint points
            for(int i=0; i<joint_bones_verts.size(); i++){
                int bone1ind = bone_name_index_map[joint_bones_verts[i].first[0]];
                int bone2ind = bone_name_index_map[joint_bones_verts[i].first[1]];
                MatrixXd bone1Def = MatrixXd::Zero(3,4);
                MatrixXd bone2Def = MatrixXd::Zero(3,4);
                cout<<mesh->red_x().segment<40>(0).transpose()<<endl;
                if(bone1ind >= fix_bones.size()){
                    bone1Def.col(0) = mesh->red_x().segment<3>(12*(bone1ind-fix_bones.size())+0);
                    bone1Def.col(1) = mesh->red_x().segment<3>(12*(bone1ind-fix_bones.size())+3);
                    bone1Def.col(2) = mesh->red_x().segment<3>(12*(bone1ind-fix_bones.size())+6);
                    bone1Def.col(3) = mesh->red_x().segment<3>(12*(bone1ind-fix_bones.size())+9);
                }
                if(bone2ind >= fix_bones.size()){
                    bone2Def.col(0) = mesh->red_x().segment<3>(12*(bone2ind-fix_bones.size())+0);
                    bone2Def.col(1) = mesh->red_x().segment<3>(12*(bone2ind-fix_bones.size())+3);
                    bone2Def.col(2) = mesh->red_x().segment<3>(12*(bone2ind-fix_bones.size())+6);
                    bone2Def.col(3) = mesh->red_x().segment<3>(12*(bone2ind-fix_bones.size())+9);
                }

                cout<<"Joint"<<endl;
                cout<<bone1Def<<endl;
                cout<<bone2Def<<endl;
                //draw for B1
                Vector4d p4_11 = Vector4d::Ones();
                p4_11.segment<3>(0) = joint_bones_verts[i].second.row(0).transpose();
                RowVector3d p11 = bone1Def*p4_11 + joint_bones_verts[i].second.row(0).transpose();
                viewer.data().add_points(p11, Eigen::RowVector3d(0,0,1));
                if(joint_bones_verts[i].second.rows()>1){
                    Vector4d p4_2 = Vector4d::Ones();
                    p4_2.segment<3>(0) = joint_bones_verts[i].second.row(1).transpose();
                    RowVector3d p2 = bone1Def*p4_2 + joint_bones_verts[i].second.row(1).transpose();
                    viewer.data().add_points(p2, Eigen::RowVector3d(0,0,1));
                    viewer.data().add_edges(p11, p2, Eigen::RowVector3d(0,0,1));
                    
                }
               

                //draw for B2
                Vector4d p4_1 = Vector4d::Ones();
                p4_1.segment<3>(0) = joint_bones_verts[i].second.row(0).transpose();
                RowVector3d p1 = bone2Def*p4_1 + joint_bones_verts[i].second.row(0).transpose();
                viewer.data().add_points(p1, Eigen::RowVector3d(0,0,1));
                if(joint_bones_verts[i].second.rows()>1){
                    Vector4d p4_2 = Vector4d::Ones();
                    p4_2.segment<3>(0) = joint_bones_verts[i].second.row(1).transpose();
                    RowVector3d p2 = bone2Def*p4_2 + joint_bones_verts[i].second.row(1).transpose();
                    viewer.data().add_points(p2, Eigen::RowVector3d(0,0,1));
                    viewer.data().add_edges(p1, p2, Eigen::RowVector3d(0,0,1));
                    
                }

            }

            for(int i=0; i<joint_bones_verts.size(); i++){
                RowVector3d p1 = joint_bones_verts[i].second.row(0);//js.segment<3>(0);
                viewer.data().add_points(p1, Eigen::RowVector3d(0,0,0));
                if(joint_bones_verts[i].second.rows()>1){
                    RowVector3d p2 = joint_bones_verts[i].second.row(1);//js.segment<3>(3);
                    viewer.data().add_points(p2, Eigen::RowVector3d(0,0,0));
                    viewer.data().add_edges(p1, p2, Eigen::RowVector3d(0,0,0));
                    
                }
            }
        }
        
        return false;
    };

    viewer.data().set_mesh(V,F);
    viewer.data().show_lines = false;
    viewer.data().invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data().face_based = true;
    viewer.core.background_color = Eigen::Vector4f(1,1,1,0);
    // viewer.data().set_colors(SETCOLORSMAT);
    viewer.launch();

    return EXIT_SUCCESS;

    
    return 0;
}