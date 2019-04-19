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
#include <json.hpp>
#include <igl/Timer.h>
#include <sstream>
#include <iomanip>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>


#include "mesh.h"
#include "redArap.h"
#include "elastic.h"
#include "redSolver.h"



using json = nlohmann::json;

using namespace Eigen;
using namespace std;
json j_input;

RowVector3d red(1,0,0);
RowVector3d purple(1,0,1);
RowVector3d green(0,1,0);
RowVector3d black(0,0,0);
MatrixXd Colors;


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
    if(relativeStiffness.size()==0){
        relativeStiffness = VectorXd::Ones(T.rows());
    }else{
        relativeStiffness = relativeStiffness/1e12;
        for(int i=0; i<relativeStiffness.size(); i++){
            if(relativeStiffness[i]>5){
                relativeStiffness[i] = 1000;//*relativeStiffness[i]*relativeStiffness[i];
            }else{
                relativeStiffness[i] = 1;
            }
        }
    }
    
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
    cout<<"Number of bone edges: "<< count_index<<endl;
    count_index = 0;
    cout<<"Reading Configue::for muscle parts"<<endl;
    for(json::iterator it = j_muscles.begin(); it != j_muscles.end(); ++it){
        VectorXi muscle_i;
        igl::readDMAT(datafile+"/generated_files/"+it.key()+"_muscle_indices.dmat", muscle_i);
        muscle_tets.push_back(muscle_i);
        muscle_name_index_map[it.key()] = count_index;
        cout<<"Muscle part: "<<it.key()<<" inserted!"<<endl;
        count_index +=1;
    }
    cout<<"Reading Configue ends"<<endl;
    count_index =0;
    for(json::iterator it = j_joints.begin(); it!= j_joints.end(); ++it){
        MatrixXd joint_i;
        MatrixXi joint_f;
        std::string joint_name = it.value()["location_obj"];
        igl::readOBJ(datafile+"/objs/"+joint_name, joint_i, joint_f);
        std::vector<std::string> bones = it.value()["bones"];
        joint_bones_verts.push_back(std::make_pair( bones, joint_i));
    }
    cout<<"Number of joint edges: "<< count_index<<endl;

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

    //para
    //(MatrixXd& V, MatrixXi& T, MatrixXi& F, MatrixXd& Uvec, std::map<std::string, int>& bone_name_index_map, std::map<std::string, int>& muscle_name_index_map, std::vector< std::pair<std::vector<std::string>, MatrixXd>>& joint_bones_verts, std::vector<VectorXi>& bone_tets, std::vector<VectorXi>& muscle_tets, std::vector<std::string>& fix_bones, VectorXd& relativeStiffness)
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


    //Creating Meshes
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
    
    //MatrixXd is point
    //MatrixXi is edges(store two poinits index)
    /* Example of Bounding Box
    	for (unsigned i=0;i<E_box.rows(); ++i)
	    viewer.data().add_edges
	    (
	      V_box.row(E_box(i,0)),
	      V_box.row(E_box(i,1)),
	      Eigen::RowVector3d(1,0,0)
	    );
    */
	
    std::cout<<"-----ARAP-----"<<std::endl;
    Reduced_Arap* arap = new Reduced_Arap(*mesh);

    std::cout<<"-----Neo-------"<<std::endl;
    double start_strength = j_input["muscle_starting_strength"];
    std::vector<int> contract_muscles_at_ind = j_input["contract_muscles_at_index"];
    Elastic* neo = new Elastic(*mesh, start_strength, contract_muscles_at_ind);

    std::cout<<"-----Solver-------"<<std::endl;
    int DIM = mesh->red_s().size();
    RedSolver f(DIM, mesh, arap, neo, j_input);
    LBFGSParam<double> param;
    param.epsilon = 1e-1;
    
    if(j_input["bfgs_convergence_crit_fast"]){
        param.delta = 1e-5;
        param.past = 1;
    }

    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
    LBFGSSolver<double> solver(param);


    igl::Timer timer;

    //write something here
    int run =0;
    for(int run=0; run<j_input["QS_steps"]; run++){
        MatrixXd newV = mesh->continuousV();
        string datafile = j_input["data"];
        ostringstream out;
        out << std::internal << std::setfill('0') << std::setw(3) << run;
        igl::writeOBJ(outputfile+"/"+namestring+"/"+namestring+"animation"+out.str()+".obj",newV, F);
        igl::writeDMAT(outputfile+"/"+namestring+"/"+namestring+"animation"+out.str()+".dmat",newV);
        cout<<"     ---Quasi-Newton Step Info"<<endl;
        double fx =0;
        VectorXd ns = mesh->N().transpose()*mesh->red_s();
        timer.start();
        int niter = solver.minimize(f, ns, fx);
        timer.stop();
        cout<<"     BFGSIters: "<<niter<<endl;
        cout<<"     QSsteptime: "<<timer.getElapsedTimeInMicroSec()<<endl;
        VectorXd reds = mesh->N()*ns + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
        for(int i=0; i<reds.size(); i++){
            mesh->red_s()[i] = reds[i];
        }
        
        neo->changeFiberMag(j_input["multiplier_strength_each_step"]);
    }
    // exit(0);

    std::cout<<"-----Display-------"<<std::endl;
    igl::opengl::glfw::Viewer viewer;
    MatrixXd Colors = MatrixXd::Random(1000,3); // 3x3 Matrix filled with random numbers between (-1,1)
    Colors = (Colors + MatrixXd::Constant(1000,3,1.))*(1-1e-6)/2.; // add 1 to the matrix to have values between 0 and 2; multiply with range/2
    Colors = (Colors + MatrixXd::Constant(1000,3,1e-6)); //set LO as the lower bound (offset)
    MatrixXd SETCOLORSMAT = MatrixXd::Zero(V.rows(), 3);
  
    int kkkk = 0;
    double tttt = 0;

    //////////////////****Pre-Draw****///////////////////
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & viewer){   
        if(viewer.core.is_animating){
            if(kkkk<mesh->G().cols()){
                VectorXd x = 10*sin(tttt)*mesh->G().col(kkkk) + mesh->x0();
                Eigen::Map<Eigen::MatrixXd> newV(x.data(), V.cols(), V.rows());
                viewer.data().set_mesh(newV.transpose(), F);
                tttt+= 0.1;
            }
    	}
        //cout<<"Im here"<<endl;
        return false;
    };

    //////////////////****GUI Mouse Part****/////////////////////
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char m_key, int modifiers)
    {

        if (m_key == 0)
        {
 	    cout<<"Left"<<endl;

            // find the position of bone/muscle
	    // highlight if area is valid
	    // TODO
	    // mouse axies & bounding box?
            // bone_name_index_map / muscle_name_index_map
            // highlight colors
	    
            // Questions:
	    // loading bone and joint info from dictionaries?
		
	    //figure out position
	    //cout<<viewer.down_mouse_x<<" "<<viewer.down_mouse_y<<" "<<viewer.current_mouse_x<<" "<<viewer.current_mouse_y<<endl; 
	    // muscle_name_index_map: dictionary key names value index  
            // muscle_tets: vector index and vector of edges  
	    
        }
        else if (m_key == 1)
        {
            cout<<"Middle"<<endl;
	    // Deselect everything
            // TODO
            // restore colors
        }
        else 
        {
    	    cout<<"Right"<<endl;
	    //TBD

        }
        return false;
    };
    

    //////////////////****Key Part****/////////////////////
    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers){   
 
        std::cout<<"Key down, "<<key<<std::endl;
        viewer.data().clear();
        if(key=='Q'){
            kkkk +=1;
            cout<<kkkk<<endl;
        }

        // if(key=='A'){
        //     cout<<"here"<<endl;
        //     neo->changeFiberMag(j_input["multiplier_strength_each_step"]);

        // }


        // if(key==' '){
            
        //     // VectorXd ns = mesh->N().transpose()*mesh->red_s();
        //     // for(int i=0; i<ns.size()/6; i++){
        //     //     ns[6*i+1] -= 0.2;
        //     //     ns[6*i+2] += 0.2;
        //     //     ns[6*i+0] += 0.2;
        //     // }
        //     // arap->minimize(*mesh);

        //     double fx =0;
        //     VectorXd ns = mesh->N().transpose()*mesh->red_s();
        //     timer.start();
        //     int niter = solver.minimize(f, ns, fx);
        //     timer.stop();
        //     VectorXd reds = mesh->N()*ns + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
            
        //     for(int i=0; i<reds.size(); i++){
        //         mesh->red_s()[i] = reds[i];
        //     }
        //     cout<<"****QSsteptime: "<<timer.getElapsedTimeInMicroSec()<<", "<<niter<<endl;
        //     // arap->minimize(*mesh);
        // }

        
        if(key=='D'){
            
            // Draw disc mesh
            std::cout<<std::endl;
            MatrixXd& discV = mesh->discontinuousV();
            MatrixXi& discT = mesh->discontinuousT();
            for(int i=0; i<muscle_tets[0].size(); i++){
                Vector4i e = discT.row(muscle_tets[0][i]);
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
            viewer.data().add_points(Eigen::RowVector3d(-4.43164,-5.41006,0.745269), Eigen::RowVector3d(0,0,1));
            viewer.data().add_points(Eigen::RowVector3d(-1.08315,-3.25799,0.768346), Eigen::RowVector3d(0,0,1));
            viewer.data().add_points(Eigen::RowVector3d( 2.78361,-1.22904,0.615455), Eigen::RowVector3d(0,0,1));
            viewer.data().add_points(Eigen::RowVector3d( 4.99299,0.755414,0.878948), Eigen::RowVector3d(0,0,1));
            viewer.data().add_points(Eigen::RowVector3d(-6.34126,-6.59147,0.581683), Eigen::RowVector3d(0,0,1));
            viewer.data().add_points(Eigen::RowVector3d(0.862945,-2.24792,0.687048), Eigen::RowVector3d(0,0,1));
            viewer.data().add_points(Eigen::RowVector3d(-2.92221,-4.40623,0.764268), Eigen::RowVector3d(0,0,1));
  
        }
        // //Draw continuous mesh
        MatrixXd newV = mesh->continuousV();
        viewer.data().set_mesh(newV, F);

        viewer.data().compute_normals();
    
        

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
            // cout<<relativeStiffness.transpose()<<endl;
            VectorXd zz = VectorXd::Zero(mesh->V().rows());
            for(int i=0; i<mesh->T().rows(); i++){
                zz[mesh->T().row(i)[0]] = relativeStiffness[i];
                zz[mesh->T().row(i)[1]] = relativeStiffness[i];
                zz[mesh->T().row(i)[2]] = relativeStiffness[i];
                zz[mesh->T().row(i)[3]] = relativeStiffness[i];
            }
            igl::jet(zz, true, COLRS);
            viewer.data().set_colors(COLRS);
        }
        //---------------- 

        //Draw fixed and moving points
        for(int i=0; i<mesh->fixed_verts().size(); i++){
            viewer.data().add_points(mesh->V().row(mesh->fixed_verts()[i]),Eigen::RowVector3d(1,0,0));
        }
      
        //Draw joint points
        // for(int i=0; i<joint_bones_verts.size(); i++){
        //     RowVector3d p1 = joint_bones_verts[i].second.row(0);//js.segment<3>(0);
        //     viewer.data().add_points(p1, Eigen::RowVector3d(0,0,0));
        //     if(joint_bones_verts[i].second.rows()>1){
        //         RowVector3d p2 = joint_bones_verts[i].second.row(1);//js.segment<3>(3);
        //         viewer.data().add_points(p2, Eigen::RowVector3d(0,0,0));
        //         viewer.data().add_edges(p1, p2, Eigen::RowVector3d(0,0,0));
                
        //     }
        // }
        
        return false;
    };


    //////////////////****GUI Menu Part****/////////////////////
    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    // Customize the menu
    // TODO
    // link to somewhere
    // probably fix to neo->changeFiberMag?
    double input_box_max_muscle_strength = 0.1f; // Shared between two menus
    float slider_muscle_activiation = 0.1f; // 
    
    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {

        // Add new group
        if (ImGui::CollapsingHeader("Tool Box", ImGuiTreeNodeFlags_DefaultOpen))
        {
		// Expose variable directly ...
		ImGui::InputDouble("Muscle Strength", &input_box_max_muscle_strength, 0, 0, "%.4f");
		
		//ImGui::InputFloat("Activation level", &slider_muscle_activiation, 0, 0, "%.4f");

		//Sliding Bar
		ImGui::SliderFloat("Activation level", &slider_muscle_activiation, 0.f, 1.f, "%.4f", 1.f);

		// Add buttons for keys Q
		if (ImGui::Button("Key Q", ImVec2(-1,0)))
		{
		    viewer.data().clear();

		    kkkk +=1;
            	    cout<<"BUTTON Q"<<endl;

		    //Draw continuous mesh
		    MatrixXd newV = mesh->continuousV();
		    viewer.data().set_mesh(newV, F);

		    viewer.data().compute_normals();
    
		    for(int i=0; i<mesh->fixed_verts().size(); i++){
		        viewer.data().add_points(mesh->V().row(mesh->fixed_verts()[i]),Eigen::RowVector3d(1,0,0));
		    }
		}

		// Add buttons for keys D
		if (ImGui::Button("Key D", ImVec2(-1,0)))
		{
		    viewer.data().clear();

	            cout<<"BUTTON D"<<endl;
		    std::cout<<std::endl;
            	    MatrixXd& discV = mesh->discontinuousV();
                    MatrixXi& discT = mesh->discontinuousT();
                    for(int i=0; i<muscle_tets[0].size(); i++){
                	Vector4i e = discT.row(muscle_tets[0][i]);
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
            	    viewer.data().add_points(Eigen::RowVector3d(-4.43164,-5.41006,0.745269), Eigen::RowVector3d(0,0,1));
            	    viewer.data().add_points(Eigen::RowVector3d(-1.08315,-3.25799,0.768346), Eigen::RowVector3d(0,0,1));
                    viewer.data().add_points(Eigen::RowVector3d( 2.78361,-1.22904,0.615455), Eigen::RowVector3d(0,0,1));
                    viewer.data().add_points(Eigen::RowVector3d( 4.99299,0.755414,0.878948), Eigen::RowVector3d(0,0,1));
                    viewer.data().add_points(Eigen::RowVector3d(-6.34126,-6.59147,0.581683), Eigen::RowVector3d(0,0,1));
                    viewer.data().add_points(Eigen::RowVector3d(0.862945,-2.24792,0.687048), Eigen::RowVector3d(0,0,1));
                    viewer.data().add_points(Eigen::RowVector3d(-2.92221,-4.40623,0.764268), Eigen::RowVector3d(0,0,1));
		
		    //Draw continuous mesh
		    MatrixXd newV = mesh->continuousV();
		    viewer.data().set_mesh(newV, F);

		    viewer.data().compute_normals();
		     
		    for(int i=0; i<mesh->fixed_verts().size(); i++){
		        viewer.data().add_points(mesh->V().row(mesh->fixed_verts()[i]),Eigen::RowVector3d(1,0,0));
		    }
		}

		// Add buttons for keys R
		if (ImGui::Button("Key R", ImVec2(-1,0)))
		{
		    viewer.data().clear();	
		    MatrixXd newV = mesh->continuousV();
            	    viewer.data().set_mesh(newV, F);
		    viewer.data().compute_normals();
		    cout<<"BUTTON R"<<endl;

		    for(int c=0; c<mesh->red_w().size()/3; c++){
		        std::vector<int> cluster_elem = mesh->r_cluster_elem_map()[c];
		        for(int e=0; e<cluster_elem.size(); e++){
		            SETCOLORSMAT.row(mesh->T().row(cluster_elem[e])[0]) = Colors.row(c);
		            SETCOLORSMAT.row(mesh->T().row(cluster_elem[e])[1]) = Colors.row(c);
		            SETCOLORSMAT.row(mesh->T().row(cluster_elem[e])[2]) = Colors.row(c);
		            SETCOLORSMAT.row(mesh->T().row(cluster_elem[e])[3]) = Colors.row(c);
		        }
		    }

		    for(int i=0; i<mesh->fixed_verts().size(); i++){
		        viewer.data().add_points(mesh->V().row(mesh->fixed_verts()[i]),Eigen::RowVector3d(1,0,0));
		    }

		    viewer.data().set_colors(SETCOLORSMAT);
		}

		// Add buttons for keys S
		if (ImGui::Button("Key S", ImVec2(-1,0)))
		{
		    viewer.data().clear();
		    MatrixXd newV = mesh->continuousV();
            	    viewer.data().set_mesh(newV, F);
		    viewer.data().compute_normals();

		    cout<<"BUTTON S"<<endl;
		    SETCOLORSMAT.setZero();
		    for(int i=0; i<mesh->T().rows(); i++){
		        if(mesh->sW().col(6*kkkk)[6*i] > 0){
		            SETCOLORSMAT.row(mesh->T().row(i)[0]) = Colors.row(5);
		            SETCOLORSMAT.row(mesh->T().row(i)[1]) = Colors.row(5);
		            SETCOLORSMAT.row(mesh->T().row(i)[2]) = Colors.row(5);
		            SETCOLORSMAT.row(mesh->T().row(i)[3]) = Colors.row(5);

		        }
		    }
		    viewer.data().set_colors(SETCOLORSMAT);
		    for(int i=0; i<mesh->fixed_verts().size(); i++){
		        viewer.data().add_points(mesh->V().row(mesh->fixed_verts()[i]),Eigen::RowVector3d(1,0,0));
		    }

		}

		// Add buttons for keys V
		if (ImGui::Button("Key V", ImVec2(-1,0)))
		{
		    viewer.data().clear();
		    MatrixXd newV = mesh->continuousV();
            	    viewer.data().set_mesh(newV, F);
		    viewer.data().compute_normals();

		    cout<<"BUTTON S"<<endl;
		    //Display tendon areas
		    MatrixXd COLRS;
		    // cout<<relativeStiffness.transpose()<<endl;
		    VectorXd zz = VectorXd::Zero(mesh->V().rows());
		    for(int i=0; i<mesh->T().rows(); i++){
		        zz[mesh->T().row(i)[0]] = relativeStiffness[i];
		        zz[mesh->T().row(i)[1]] = relativeStiffness[i];
		        zz[mesh->T().row(i)[2]] = relativeStiffness[i];
		        zz[mesh->T().row(i)[3]] = relativeStiffness[i];
		    }
		    igl::jet(zz, true, COLRS);
		    viewer.data().set_colors(COLRS);
		    for(int i=0; i<mesh->fixed_verts().size(); i++){
		        viewer.data().add_points(mesh->V().row(mesh->fixed_verts()[i]),Eigen::RowVector3d(1,0,0));
		    }

		}

		// Add buttons for keys Space
		if (ImGui::Button("Key Space", ImVec2(-1,0)))
		{
		    viewer.data().clear();
            	    viewer.data().set_mesh(V, F);
		    //viewer.data().compute_normals();
		    cout<<"BUTTON Space"<<endl;
		}
		//status windows?
		
	  }
    };
    

    //////////////////****Viewer Init Part****/////////////////////
    viewer.data().set_mesh(V,F);
    viewer.data().show_lines = false;
    viewer.data().invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data().face_based = true;
    viewer.core.background_color = Eigen::Vector4f(1,1,1,0);
    // viewer.data().set_colors(SETCOLORSMAT);

    viewer.launch();
    return EXIT_SUCCESS;

}
