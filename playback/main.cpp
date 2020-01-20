#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
#include <igl/readOBJ.h>
#include <igl/jet.h>
#include <igl/png/readPNG.h>
#include <igl/png/writePNG.h>
#include <igl/volume.h>
#include <igl/slice.h>
#include <igl/boundary_facets.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/destroy_shader_program.h>
#include <igl/opengl/create_shader_program.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/remove_unreferenced.h>
#include <igl/list_to_matrix.h>
#include <imgui/imgui.h>
#include <igl/null.h>
#include <json.hpp>

#include <sstream>
#include <iomanip>

#include "../famu/store.h"
#include "../famu/read_config_files.h"

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

using Store = famu::Store;
json j_input;

int main(int argc, char *argv[])
{
	std::cout<<"-----Configs-------"<<std::endl;	
		std::string inputfile;
		if(argc<1){
			cout<<"Run as: ./playback <path-to-input>/input.json <path-to-objs>/ num"<<endl;
			exit(0);
		}
		std::ifstream input_file(argv[1]);
		input_file >> j_input;
    #ifdef __linux__
    omp_set_num_threads(1);
    #endif
		
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


		famu::read_config_files(store,store.V, 
								store.T, 
								store.F, 
								store.Uvec, 
								store.bone_name_index_map, 
								store.muscle_name_index_map, 
								store.joint_bones_verts, 
								store.bone_tets, 
								store.muscle_tets, 
								store.fix_bones, 
                store.script_bones,
								store.relativeStiffness,
								store.contract_muscles,
								store.muscle_steps,
                store.jinput);  
	
		


	cout<<"---Record Mesh Setup Info"<<endl;
		cout<<"V size: "<<store.V.rows()<<endl;
		cout<<"T size: "<<store.T.rows()<<endl;
		cout<<"F size: "<<store.F.rows()<<endl;
		store.jinput["number_modes"] = NUM_MODES;
		std::string outputfile = j_input["output"];
		igl::boundary_facets(store.T, store.F);

  cout<<"--READ OBJ FILES"<<endl;
    std::string files_to_read(argv[2]);
    int start_int = 0;
    int fin_int = stoi(argv[3]);
    cout<<files_to_read<<endl;
    cout<<fin_int<<endl;
    MatrixXd newV = store.V;
    MatrixXi newF = store.F;


	cout<<"---Set Mesh Params"<<store.x.size()<<endl;
		//YM, poissons
		store.eY = 1e10*VectorXd::Ones(store.T.rows());
		store.eP = 0.49*VectorXd::Ones(store.T.rows());
		store.muscle_mag = VectorXd::Zero(store.T.rows());
		for(int m=0; m<store.muscle_tets.size(); m++){
			for(int t=0; t<store.muscle_tets[m].size(); t++){
				if(store.relativeStiffness[store.muscle_tets[m][t]]>1){
					store.eY[store.muscle_tets[m][t]] = 1.2e9;
				}else{
					store.eY[store.muscle_tets[m][t]] = 60000;
				}
			}
		}

        {
          store.elogVY = Eigen::VectorXd::Zero(store.V.rows());
          // volume associated with each vertex
          Eigen::VectorXd Vvol = Eigen::VectorXd::Zero(store.V.rows());
          Eigen::VectorXd Tvol;
          igl::volume(store.V,store.T,Tvol);
          // loop over tets
          for(int i = 0;i<store.T.rows();i++)
          {
            const double vol4 = Tvol(i)/4.0;
            for(int j = 0;j<4;j++)
            {
              Vvol(store.T(i,j)) += vol4;
              store.elogVY(store.T(i,j)) += vol4*log10(store.eY(i));
            }
          }
          // loop over vertices to divide to take average
          for(int i = 0;i<store.V.rows();i++)
          {
            store.elogVY(i) /= Vvol(i);
          }
        }



	std::cout<<"-----Display-------"<<std::endl;
		  int fancy_data_index,debug_data_index,discontinuous_data_index;
    	igl::opengl::glfw::Viewer viewer;

      // Attach a menu plugin
      igl::opengl::glfw::imgui::ImGuiMenu menu;
      viewer.plugins.push_back(&menu);

      // Customize the menu
      double doubleVariable = 0.1f; // Shared between two menus
      // Draw additional windows
      menu.callback_draw_custom_window = [&]()
      {
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiSetCond_FirstUseEver);
        ImGui::Begin(
            "New Window", nullptr,
            ImGuiWindowFlags_NoSavedSettings
        );

    

        static int i1=start_int;
        ImGui::SliderInt("slider int", &i1, start_int, fin_int);
        cout<<i1<<endl;
        // load the mesh
        igl::readOBJ(files_to_read+to_string(i1)+".obj",newV, newF);
        viewer.data_list[fancy_data_index].set_vertices(newV);        
        ImGui::End();
      };


    	int currentStep = 0;
        //render out current view
        // Allocate temporary buffers for 1280x800 image
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(1920,1280);
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(1920,1280);
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(1920,1280);
        Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(1920,1280);
        
    	viewer.callback_post_draw= [&](igl::opengl::glfw::Viewer & viewer) {
        if(viewer.core().is_animating){
          igl::readOBJ(files_to_read+to_string(currentStep)+".obj",newV, newF);
          viewer.data_list[fancy_data_index].set_vertices(newV);

          std::stringstream out_file;


          // Draw the scene in the buffers
          viewer.core().draw_buffer(viewer.data(),false,R,G,B,A);

          // Save it to a PNG
          out_file<<"vid_"<<std::setfill('0') << std::setw(5) <<currentStep<<".png";

          std::string out = store.jinput["output"];
          out = out+"/"+ out_file.str();
          std::cout<<out<<std::endl;
          igl::png::writePNG(R,G,B,A,out);
          if(currentStep==fin_int)
            exit(0);
          currentStep += 1;
        }

  	    return false;
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
  // viewer.data_list[discontinuous_data_index].set_mesh(store.discV, store.discF);
  // viewer.data_list[discontinuous_data_index].show_lines = true;
  // viewer.data_list[discontinuous_data_index].show_faces = false;
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
