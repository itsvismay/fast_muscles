#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/writeDMAT.h>
#include <igl/jet.h>
#include <igl/png/readPNG.h>
#include <imgui/imgui.h>
#include <json.hpp>

#include <sstream>
#include <iomanip>
// #include <omp.h>

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
			cout<<"Run as: ./famu input.json <threads>"<<endl;
			exit(0);
		}
		Eigen::initParallel();
	
		
    igl::Timer timer;

		famu::Store store;
		store.jinput = j_input;
		famu::setupStore(store);


	cout<<"--- External Forces Hard Coded Contact Matrices"<<endl;
    // famu::acap::adjointMethodExternalForces(store);
	
  std::cout<<"----POSE BONES MANUALLY ----"<<std::endl;
  //Scapula -> humerus -> forearm
    Matrix3d R;
    R<<0.7071, 0.7071, 0,
      -0.7071, 0.7071, 0,
      0, 0, 1;
    R = R*R*R;



	std::cout<<"-----Display-------"<<std::endl;
    	igl::opengl::glfw::Viewer viewer;
    	int currentStep = 0;
    	viewer.callback_post_draw= [&](igl::opengl::glfw::Viewer & viewer) {
	    
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
          case 'A':
          case 'a':
          {
              cout<<"Before: "<<famu::acap::fastEnergy(store, store.dFvec)<<endl;
              store.dFvec[18+0] = R(0,0);
              store.dFvec[18+1] = R(0,1);
              store.dFvec[18+2] = R(0,2);
              store.dFvec[18+3] = R(1,0);
              store.dFvec[18+4] = R(1,1);
              store.dFvec[18+5] = R(1,2);
              store.dFvec[18+6] = R(2,0);
              store.dFvec[18+7] = R(2,1);
              store.dFvec[18+8] = R(2,2);

              store.BfI0 = store.Bf*store.dFvec;
              famu::acap::solve(store, store.dFvec);
              cout<<"After: "<<famu::acap::fastEnergy(store, store.dFvec)<<endl;
            // if(currentStep>=store.muscle_steps.size()){
            //   currentStep = 0;
            // }else{
            //   famu::muscle::set_muscle_mag(store, currentStep);
            //   currentStep += 1;
            // }
            // return true;
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



  viewer.core.is_animating = false;
  viewer.core.background_color = Eigen::Vector4f(1,1,1,0);

  viewer.launch_rendering(true);
  viewer.launch_shut();

}