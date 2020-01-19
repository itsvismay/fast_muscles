#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/writeDMAT.h>
#include <igl/jet.h>
#include <igl/png/readPNG.h>
#include <imgui/imgui.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <json.hpp>

#include <sstream>
#include <iomanip>
#ifdef __linux__
#include <omp.h>
#endif

#include "../famu/setup_store.h"
#include "../famu/newton_solver.h"
#include "../famu/muscle_energy_gradient.h"
#include "../famu/draw_disc_mesh_functions.h"
#include "../famu/acap_solve_energy_gradient.h"


using namespace Eigen;
using namespace std;
using json = nlohmann::json;

using Store = famu::Store;
json j_input;


int main(int argc, char *argv[])
{
	int fancy_data_index,debug_data_index,discontinuous_data_index;
	std::cout<<"-----Configs-------"<<std::endl;
		std::string inputfile;
		int num_threads = 1;
		if(argc==3){
			num_threads = std::stoi(argv[2]);
			#ifdef __linux__
			omp_set_num_threads(num_threads);
			#endif
			std::ifstream input_file(argv[1]);
			input_file >> j_input;
			std::cout<<"Threads: "<<Eigen::nbThreads( )<<std::endl;
		}else{
			cout<<"Run as: ./pose-mesh-cheat input.json <threads>"<<endl;
			exit(0);
		}
		Eigen::initParallel();
	
		
    igl::Timer timer;

	famu::Store store;
	store.jinput = j_input;

    {
      std::string craps = "../";
      std::string data = store.jinput["data"];
      std::string out = store.jinput["output"];
      std::string material = store.jinput["material"];
      store.jinput["data"] = craps+data;
      store.jinput["output"] = craps+out;
      store.jinput["material"] = craps+material;
    }
	famu::setupStore(store);

	cout<<"--- External Forces Hard Coded Contact Matrices"<<endl;
    // famu::acap::adjointMethodExternalForces(store);
	
  std::cout<<"----POSE BONES MANUALLY ----"<<std::endl;
  //Scapula -> humerus -> forearm
  float test_h_p=-0.5, test_h_y=0, test_h_r=0;
  float test_f_p=-0.45, test_f_y=0, test_f_r=-0.44;
  int count_rotation = 0;
  int max_rotation_frame = 100;
  store.muscle_steps[0]["biceps"] = 100*max_rotation_frame;
  store.muscle_steps[0]["triceps"] = 5*max_rotation_frame;
  store.muscle_steps[0]["brachialis"] = 100*max_rotation_frame;
  store.muscle_steps[0]["front_deltoid"] = 5*max_rotation_frame;
  store.muscle_steps[0]["rear_deltoid"] = 5*max_rotation_frame;
  store.muscle_steps[0]["top_deltoid"] = 5*max_rotation_frame;

  max_rotation_frame = 100;
  for(int ii=0; ii<max_rotation_frame; ii++){

            famu::muscle::set_muscle_mag(store, 0);

            Eigen::AngleAxisd test_h_rot_xAngle((count_rotation%max_rotation_frame)*(test_h_p/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd test_h_rot_yAngle((count_rotation%max_rotation_frame)*(test_h_y/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd test_h_rot_zAngle((count_rotation%max_rotation_frame)*(test_h_r/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitZ());
            Eigen::Quaternion<double> q = test_h_rot_zAngle * test_h_rot_yAngle * test_h_rot_xAngle;
            Eigen::Matrix3d R = q.matrix();
            store.dFvec[9*store.bone_name_index_map["humerus"]+0] = R(0,0);
            store.dFvec[9*store.bone_name_index_map["humerus"]+1] = R(0,1);
            store.dFvec[9*store.bone_name_index_map["humerus"]+2] = R(0,2);
            store.dFvec[9*store.bone_name_index_map["humerus"]+3] = R(1,0);
            store.dFvec[9*store.bone_name_index_map["humerus"]+4] = R(1,1);
            store.dFvec[9*store.bone_name_index_map["humerus"]+5] = R(1,2);
            store.dFvec[9*store.bone_name_index_map["humerus"]+6] = R(2,0);
            store.dFvec[9*store.bone_name_index_map["humerus"]+7] = R(2,1);
            store.dFvec[9*store.bone_name_index_map["humerus"]+8] = R(2,2);
            cout<<"HERE2"<<endl;
           
           
            Eigen::AngleAxisd test_f_rot_xAngle((count_rotation%max_rotation_frame)*(test_f_p/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd test_f_rot_yAngle((count_rotation%max_rotation_frame)*(test_f_y/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd test_f_rot_zAngle((count_rotation%max_rotation_frame)*(test_f_r/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitZ());
            q = test_f_rot_zAngle * test_f_rot_yAngle * test_f_rot_xAngle;
            R = q.matrix();
            store.dFvec[9*store.bone_name_index_map["forearm"]+0] = R(0,0);
            store.dFvec[9*store.bone_name_index_map["forearm"]+1] = R(0,1);
            store.dFvec[9*store.bone_name_index_map["forearm"]+2] = R(0,2);
            store.dFvec[9*store.bone_name_index_map["forearm"]+3] = R(1,0);
            store.dFvec[9*store.bone_name_index_map["forearm"]+4] = R(1,1);
            store.dFvec[9*store.bone_name_index_map["forearm"]+5] = R(1,2);
            store.dFvec[9*store.bone_name_index_map["forearm"]+6] = R(2,0);
            store.dFvec[9*store.bone_name_index_map["forearm"]+7] = R(2,1);
            store.dFvec[9*store.bone_name_index_map["forearm"]+8] = R(2,2);
            cout<<"HERE3"<<endl;
            // famu::acap::solve(store, store.dFvec, false);
            cout<<"HERE4"<<endl;
            std::string name = "mesh";
            store.printState(ii, name);
            int niters =famu::newton_static_solve(store);



            count_rotation +=1;
  }
  store.saveResults();
  exit(0);



	std::cout<<"-----Display-------"<<std::endl;
    	igl::opengl::glfw::Viewer viewer;
    	int currentStep = 0;

      // Attach a menu plugin
      igl::opengl::glfw::imgui::ImGuiMenu menu;
      viewer.plugins.push_back(&menu);

      menu.callback_draw_custom_window = [&]()
      {
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 160), ImGuiSetCond_FirstUseEver);
        ImGui::Begin(
            "Bone GUI", nullptr,
            ImGuiWindowFlags_NoSavedSettings
        );

        static double bicep_activation = 0.0;
        ImGui::InputDouble("bicep activation", &bicep_activation, 0.01f, 1.0f, "%.2f");;
        static double tricep_activation = 0.0;
        ImGui::InputDouble("tricep activation", &tricep_activation, 0.01f, 1.0f, "%.2f");;
        static double brachialis_activation = 0.0;
        ImGui::InputDouble("brachialis_activation", &brachialis_activation, 0.01f, 1.0f, "%.2f");;
        static double front_delt_activation = 0.0;
        ImGui::InputDouble("front_delt activation", &front_delt_activation, 0.01f, 1.0f, "%.2f");;
        static double top_delt_activation = 0.0;
        ImGui::InputDouble("top_delt activation", &top_delt_activation, 0.01f, 1.0f, "%.2f");;
        static double rear_delt_activation = 0.0;
        ImGui::InputDouble("rear_delt activation", &rear_delt_activation, 0.01f, 1.0f, "%.2f");;
        static double contact_weight = 1;
        ImGui::InputDouble("contact weight", &contact_weight, 0.01f, 1.0f, "%.2f");;

        store.jinput["contact_force_weight"] = contact_weight;
        store.muscle_steps[0]["biceps"] = bicep_activation;
        store.muscle_steps[0]["triceps"] = tricep_activation;
        store.muscle_steps[0]["brachialis"] = brachialis_activation;
        store.muscle_steps[0]["front_deltoid"] = front_delt_activation;
        store.muscle_steps[0]["rear_deltoid"] = rear_delt_activation;
        store.muscle_steps[0]["top_deltoid"] = top_delt_activation;

        famu::muscle::set_muscle_mag(store, 0);

        json j_scripts = store.jinput["script_bones"];
        static float h_p=0, h_y=0, h_r=0;
        static float f_p=0, f_y=0, f_r=0;
        static float s_p=0, s_y=0, s_r=0;
          
        ImGui::SliderFloat("humerus _rot_x", &h_p, -2, 2);
        ImGui::SliderFloat("humerus _rot_y",   &h_y, -2, 2);
        ImGui::SliderFloat("humerus _rot_z",  &h_r, -2, 2);
        Eigen::AngleAxisd h_rot_xAngle((h_p)*M_PI, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd h_rot_yAngle((h_y)*M_PI, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd h_rot_zAngle((h_r)*M_PI, Eigen::Vector3d::UnitZ());
        Eigen::Quaternion<double> q = h_rot_zAngle * h_rot_yAngle * h_rot_xAngle;
        Eigen::Matrix3d R = q.matrix();
        store.dFvec[9*store.bone_name_index_map["humerus"]+0] = R(0,0);
        store.dFvec[9*store.bone_name_index_map["humerus"]+1] = R(0,1);
        store.dFvec[9*store.bone_name_index_map["humerus"]+2] = R(0,2);
        store.dFvec[9*store.bone_name_index_map["humerus"]+3] = R(1,0);
        store.dFvec[9*store.bone_name_index_map["humerus"]+4] = R(1,1);
        store.dFvec[9*store.bone_name_index_map["humerus"]+5] = R(1,2);
        store.dFvec[9*store.bone_name_index_map["humerus"]+6] = R(2,0);
        store.dFvec[9*store.bone_name_index_map["humerus"]+7] = R(2,1);
        store.dFvec[9*store.bone_name_index_map["humerus"]+8] = R(2,2);
        
        ImGui::SliderFloat("forearm _rot_x", &f_p, -2, 2);
        ImGui::SliderFloat("forearm _rot_y",   &f_y, -2, 2);
        ImGui::SliderFloat("forearm _rot_z",  &f_r, -2, 2);
        Eigen::AngleAxisd f_rot_xAngle((f_p)*M_PI, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd f_rot_yAngle((f_y)*M_PI, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd f_rot_zAngle((f_r)*M_PI, Eigen::Vector3d::UnitZ());
        q = f_rot_zAngle * f_rot_yAngle * f_rot_xAngle;
        R = q.matrix();
        store.dFvec[9*store.bone_name_index_map["forearm"]+0] = R(0,0);
        store.dFvec[9*store.bone_name_index_map["forearm"]+1] = R(0,1);
        store.dFvec[9*store.bone_name_index_map["forearm"]+2] = R(0,2);
        store.dFvec[9*store.bone_name_index_map["forearm"]+3] = R(1,0);
        store.dFvec[9*store.bone_name_index_map["forearm"]+4] = R(1,1);
        store.dFvec[9*store.bone_name_index_map["forearm"]+5] = R(1,2);
        store.dFvec[9*store.bone_name_index_map["forearm"]+6] = R(2,0);
        store.dFvec[9*store.bone_name_index_map["forearm"]+7] = R(2,1);
        store.dFvec[9*store.bone_name_index_map["forearm"]+8] = R(2,2);
        
        // ImGui::SliderFloat("sternum _rot_x", &s_p, -2, 2);
        // ImGui::SliderFloat("sternum _rot_y",   &s_y, -2, 2);
        // ImGui::SliderFloat("sternum _rot_z",  &s_r, -2, 2);
        // Eigen::AngleAxisd s_rot_xAngle((s_p)*M_PI, Eigen::Vector3d::UnitX());
        // Eigen::AngleAxisd s_rot_yAngle((s_y)*M_PI, Eigen::Vector3d::UnitY());
        // Eigen::AngleAxisd s_rot_zAngle((s_r)*M_PI, Eigen::Vector3d::UnitZ());
        // q = s_rot_zAngle * s_rot_yAngle * s_rot_xAngle;
        // R = q.matrix();
        // store.dFvec[9*store.bone_name_index_map["sternum"]+0] = R(0,0);
        // store.dFvec[9*store.bone_name_index_map["sternum"]+1] = R(0,1);
        // store.dFvec[9*store.bone_name_index_map["sternum"]+2] = R(0,2);
        // store.dFvec[9*store.bone_name_index_map["sternum"]+3] = R(1,0);
        // store.dFvec[9*store.bone_name_index_map["sternum"]+4] = R(1,1);
        // store.dFvec[9*store.bone_name_index_map["sternum"]+5] = R(1,2);
        // store.dFvec[9*store.bone_name_index_map["sternum"]+6] = R(2,0);
        // store.dFvec[9*store.bone_name_index_map["sternum"]+7] = R(2,1);
        // store.dFvec[9*store.bone_name_index_map["sternum"]+8] = R(2,2);
        
             
        ImGui::End();
      };

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers){   
        std::cout<<"Key down, "<<key<<std::endl;
        // given data show it as colors on debug mesh
        const auto set_colors_from_data = [&](const Eigen::VectorXd & zz)
        {
          MatrixXd COLRS;
          igl::jet(zz, true, COLRS);
          viewer.data_list[debug_data_index].set_colors(COLRS);
        };
        // If debug mesh is currently visible, turn it off and turn on fancy
        // mesh and return true; otherwise return false.
        const auto hide_debug = [&]()->bool
        {
          if(viewer.data_list[debug_data_index].show_faces)
          {
            viewer.data_list[debug_data_index].show_faces = false;
            viewer.data_list[fancy_data_index].show_faces = true;
            std::cout<<"hiding debug..."<<std::endl;
            return true;
          }
          viewer.data_list[debug_data_index].show_faces = true;
          viewer.data_list[fancy_data_index].show_faces = false;
          return false;
        };
        switch(key)
        {
          case ' ':
          {

            double fx = 0;
            int niters = 0;
            niters = famu::newton_static_solve(store);

            VectorXd y = store.Y*store.x;
        	  Eigen::Map<Eigen::MatrixXd> newV(y.data(), store.V.cols(), store.V.rows());
            viewer.data_list[fancy_data_index].set_vertices((newV.transpose()+store.V));
            viewer.data_list[debug_data_index].set_vertices((newV.transpose()+store.V));
            return true;
          }
          case ']':
          {
            
            cout<<"HERE"<<endl;
            Eigen::AngleAxisd test_h_rot_xAngle((count_rotation%max_rotation_frame)*(test_h_p/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd test_h_rot_yAngle((count_rotation%max_rotation_frame)*(test_h_y/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd test_h_rot_zAngle((count_rotation%max_rotation_frame)*(test_h_r/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitZ());
            Eigen::Quaternion<double> q = test_h_rot_zAngle * test_h_rot_yAngle * test_h_rot_xAngle;
            Eigen::Matrix3d R = q.matrix();
            store.dFvec[9*store.bone_name_index_map["humerus"]+0] = R(0,0);
            store.dFvec[9*store.bone_name_index_map["humerus"]+1] = R(0,1);
            store.dFvec[9*store.bone_name_index_map["humerus"]+2] = R(0,2);
            store.dFvec[9*store.bone_name_index_map["humerus"]+3] = R(1,0);
            store.dFvec[9*store.bone_name_index_map["humerus"]+4] = R(1,1);
            store.dFvec[9*store.bone_name_index_map["humerus"]+5] = R(1,2);
            store.dFvec[9*store.bone_name_index_map["humerus"]+6] = R(2,0);
            store.dFvec[9*store.bone_name_index_map["humerus"]+7] = R(2,1);
            store.dFvec[9*store.bone_name_index_map["humerus"]+8] = R(2,2);
            cout<<"HERE2"<<endl;
           

            Eigen::AngleAxisd test_f_rot_xAngle((count_rotation%max_rotation_frame)*(test_f_p/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd test_f_rot_yAngle((count_rotation%max_rotation_frame)*(test_f_y/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd test_f_rot_zAngle((count_rotation%max_rotation_frame)*(test_f_r/max_rotation_frame)*M_PI, Eigen::Vector3d::UnitZ());
            q = test_f_rot_zAngle * test_f_rot_yAngle * test_f_rot_xAngle;
            R = q.matrix();
            store.dFvec[9*store.bone_name_index_map["forearm"]+0] = R(0,0);
            store.dFvec[9*store.bone_name_index_map["forearm"]+1] = R(0,1);
            store.dFvec[9*store.bone_name_index_map["forearm"]+2] = R(0,2);
            store.dFvec[9*store.bone_name_index_map["forearm"]+3] = R(1,0);
            store.dFvec[9*store.bone_name_index_map["forearm"]+4] = R(1,1);
            store.dFvec[9*store.bone_name_index_map["forearm"]+5] = R(1,2);
            store.dFvec[9*store.bone_name_index_map["forearm"]+6] = R(2,0);
            store.dFvec[9*store.bone_name_index_map["forearm"]+7] = R(2,1);
            store.dFvec[9*store.bone_name_index_map["forearm"]+8] = R(2,2);
            cout<<"HERE3"<<endl;
            famu::acap::solve(store, store.dFvec, false);
            cout<<"HERE4"<<endl;

            VectorXd y = store.Y*store.x;
            Eigen::Map<Eigen::MatrixXd> newV(y.data(), store.V.cols(), store.V.rows());
            viewer.data_list[fancy_data_index].set_vertices((newV.transpose()+store.V));
            viewer.data_list[debug_data_index].set_vertices((newV.transpose()+store.V));
            
            count_rotation +=1;
            return true;
          }
          case 'A':
          case 'a':
          {

            famu::acap::solve(store, store.dFvec, false);
            // int niters = 0;
            // niters = famu::newton_static_solve(store);

            VectorXd y = store.Y*store.x;
            Eigen::Map<Eigen::MatrixXd> newV(y.data(), store.V.cols(), store.V.rows());
            viewer.data_list[fancy_data_index].set_vertices((newV.transpose()+store.V));
            viewer.data_list[debug_data_index].set_vertices((newV.transpose()+store.V));
            return true;
          }
          case 'C':
          case 'c':
          {
            std::cout<<"C..."<<std::endl;
            if(!hide_debug())
            {
              std::cout<<" C..."<<std::endl;
              VectorXd zz = VectorXd::Ones(store.V.rows());
              // probably want to have this visualization update with each press
              // of space ' '... I'd consider having a little lambda that will
              // update the geometry _and_ any active visualizations. Might want
              // to have an enum or something to tell which debug visualization
              // is active.
              VectorXd y = store.Y*store.x;
              for(int m=0; m<store.T.rows(); m++){
                Matrix3d Dm;
                for(int i=0; i<3; i++){
                  Dm.col(i) = store.V.row(store.T.row(m)[i]) - store.V.row(store.T.row(m)[3]);
                }
                Matrix3d m_InvRefShapeMatrix = Dm.inverse();

                Matrix3d Ds;
                for(int i=0; i<3; i++)
                {
                  Ds.col(i) = y.segment<3>(3*store.T.row(m)[i]) - y.segment<3>(3*store.T.row(m)[3]);
                }

                Matrix3d F = Matrix3d::Identity() + Ds*m_InvRefShapeMatrix;

                double snorm = (F.transpose()*F - Matrix3d::Identity()).norm();

                zz[store.T.row(m)[0]] += snorm;
                zz[store.T.row(m)[1]] += snorm;
                zz[store.T.row(m)[2]] += snorm; 
                zz[store.T.row(m)[3]] += snorm;
              }
              set_colors_from_data(zz);
            }
            return true;
          }
          case 'D':
          case 'd':
          {
            viewer.data_list[discontinuous_data_index].show_lines =
              !viewer.data_list[discontinuous_data_index].show_lines;
            if(viewer.data_list[discontinuous_data_index].show_lines)
            {
              famu::discontinuousV(store);
              viewer.data_list[discontinuous_data_index].set_vertices(store.discV);
            }
            return true;
          }
          case 'E':
          case 'e':
          {
            if(!hide_debug())
            {
              VectorXd zz = VectorXd::Ones(store.V.rows());
              //map ACAP energy over the meseh
              VectorXd ls = store.DSY*store.x + store.DSx0;
              VectorXd rs = store.DSx0_mat*store.ProjectF*store.dFvec;
              for(int i=0; i<store.T.rows(); i++){
                double enorm = (ls.segment<12>(12*i) - rs.segment<12>(12*i)).norm();

                zz[store.T.row(i)[0]] += enorm;
                zz[store.T.row(i)[1]] += enorm;
                zz[store.T.row(i)[2]] += enorm; 
                zz[store.T.row(i)[3]] += enorm;

              }
              set_colors_from_data(zz);
            }
            return true;
          }
          case 'S':
          case 's':
          {
            if(!hide_debug())
            {
              VectorXd zz = VectorXd::Ones(store.V.rows());
              //map strains
              VectorXd fulldFvec = store.ProjectF*store.dFvec;
              for(int m=0; m<store.T.rows(); m++){
                Matrix3d F = Map<Matrix3d>(fulldFvec.segment<9>(9*m).data()).transpose();
                double snorm = (F.transpose()*F - Matrix3d::Identity()).norm();

                zz[store.T.row(m)[0]] += snorm;
                zz[store.T.row(m)[1]] += snorm;
                zz[store.T.row(m)[2]] += snorm; 
                zz[store.T.row(m)[3]] += snorm;
	      }
              set_colors_from_data(zz);
            }
            return true;
          }
          case 'V':
          case 'v':
          {
            if(!hide_debug())
            {
              VectorXd zz = VectorXd::Ones(store.V.rows());
              //Display tendon areas
              for(int i=0; i<store.T.rows(); i++){
                zz[store.T.row(i)[0]] = store.relativeStiffness[i];
                zz[store.T.row(i)[1]] = store.relativeStiffness[i];
                zz[store.T.row(i)[2]] = store.relativeStiffness[i];
                zz[store.T.row(i)[3]] = store.relativeStiffness[i];
              }
              set_colors_from_data(zz);
            }
            return true;
          }
        }

        viewer.data().points = Eigen::MatrixXd(0,6);
        viewer.data().lines = Eigen::MatrixXd(0,9);
 
        return false;
    };

  fancy_data_index = viewer.selected_data_index;
  viewer.data_list[fancy_data_index].set_mesh(store.V, store.F);
  viewer.data_list[fancy_data_index].show_lines = false;
  viewer.data_list[fancy_data_index].invert_normals = true;
  viewer.data_list[fancy_data_index].set_face_based(false);
  viewer.append_mesh();
  debug_data_index = viewer.selected_data_index;
  viewer.data_list[debug_data_index].set_mesh(store.V, store.F);
  viewer.data_list[debug_data_index].show_faces = false;
  viewer.data_list[debug_data_index].invert_normals = true;
  viewer.data_list[debug_data_index].show_lines = false;
  viewer.append_mesh();
  discontinuous_data_index = viewer.selected_data_index;
  viewer.data_list[discontinuous_data_index].set_mesh(store.discV, store.discF);
  viewer.data_list[discontinuous_data_index].show_lines = true;
  viewer.data_list[discontinuous_data_index].show_faces = false;
  // set fancy rendered mesh to be selected.
  viewer.selected_data_index = fancy_data_index;


  // must be called before messing with shaders
  viewer.launch_init(true,false);
  std::cout<<R"(
		fd_famu:
		  C,c  Show continuous mesh's strain
		  D,d  Toggle discontinous mesh wireframe
		  E,e  Show ACAP energy (interpolated on the continuous mesh)
		  S,s  Show discontinuous mesh's strain (interpolated on the continuous mesh)
		  V,v  Tendon vs. muscle vis
		)";

  // Send Young's modulus data in via color channel
  {
    Eigen::MatrixXd C(store.V.rows(),3);
    for(int i = 0;i<store.V.rows();i++)
    {
      if(store.elogVY(i) < 0.5*(60000 + 1.2e9))
      {
        C.row(i) = Eigen::RowVector3d(1,0,0);
      }else if(store.elogVY(i) < 0.5*(1.2e9 + 1.0e10))
      {
        C.row(i) = Eigen::RowVector3d(0.99,0.99,1);
      }else
      {
        C.row(i) = Eigen::RowVector3d(0.85,0.85,0.8);
      }
    }
    viewer.data_list[fancy_data_index].set_colors(store.elogVY.replicate(1,3));
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;
    // @Vismay, perhaps include this path in the json?
    igl::png::readPNG(store.jinput["material"],R,G,B,A);
    viewer.data_list[fancy_data_index].set_texture(R,G,B,A);
    viewer.data_list[fancy_data_index].show_texture = true;
    // must be called before messing with shaders
    viewer.data_list[fancy_data_index].meshgl.init();
    igl::opengl::destroy_shader_program(
      viewer.data_list[fancy_data_index].meshgl.shader_mesh);
    {
      std::string mesh_vertex_shader_string =
		R"(#version 150
		uniform mat4 view;
		uniform mat4 proj;
		uniform mat4 normal_matrix;
		in vec3 position;
		in vec3 normal;
		// Color
		in vec3 Kd;
		// Young's modulus
		out float elogY;
		out vec3 normal_eye;

		void main()
		{
		  normal_eye = normalize(vec3 (normal_matrix * vec4 (normal, 0.0)));
		  gl_Position = proj * view * vec4(position, 1.0);
		  elogY = Kd.r;
		})";

      std::string mesh_fragment_shader_string =
		R"(#version 150
		in vec3 normal_eye;
		// Young's modulus
		in float elogY;
		out vec4 outColor;
		uniform sampler2D tex;
		void main()
		{
		  vec2 uv = normalize(normal_eye).xy * vec2(0.5/3.0,0.5);
		  float t_tendon = clamp( (elogY-4.7782)/(9.0792-4.7782) , 0.0 , 1.0);
		  float t_bone =   clamp( (elogY-9.0092)/(10.000-9.0792) , 0.0 , 1.0);
		  outColor = mix(
		      texture(tex, uv + vec2(0.5/3.0,0.5)),
		      texture(tex, uv + vec2(1.5/3.0,0.5)),
		      t_tendon);
		  outColor = mix( outColor,   texture(tex, uv + vec2(2.5/3.0,0.5)),t_bone);
		  //outColor.a = 1.0;
		})";

      igl::opengl::create_shader_program(
        mesh_vertex_shader_string,
        mesh_fragment_shader_string,
        {},
        viewer.data_list[fancy_data_index].meshgl.shader_mesh);
    }
  }



  viewer.core().is_animating = false;
  viewer.core().background_color = Eigen::Vector4f(1,1,1,0);

  viewer.launch_rendering(true);
  viewer.launch_shut();

}
