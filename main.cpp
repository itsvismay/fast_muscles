#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>

#include <json.hpp>

#include "mesh.h"

using json = nlohmann::json;

using namespace Eigen;
using namespace std;
json j_input;

int main(int argc, char *argv[])
{
    std::cout<<"-----Configs-------"<<std::endl;
    json j_config_parameters;
    std::ifstream i("../input/input.json");
    i >> j_input;

    double youngs_mod = j_input["youngs"];
    double poisson = j_input["poissons"];
    double gravity = j_input["gravity"];
    
    MatrixXd V;
    MatrixXi T;
    MatrixXi F;
    igl::readMESH(j_input["mesh_file"], V, T, F);

    VectorXi fix;
    VectorXi mov;

    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(T, V, fix, mov);

    std::cout<<"-----ARAP-------"<<std::endl;
   


    
   



    std::cout<<"-----Display-------"<<std::endl;
    igl::opengl::glfw::Viewer viewer;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & viewer)
    {   
        if(viewer.core.is_animating)
        {
            // MatrixXd newV = mesh->getCurrentVerts();
            // viewer.data().set_vertices(newV);
    	}
        return false;
    };

	viewer.data().set_mesh(V,F);
    viewer.data().show_lines = true;
    viewer.data().invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data().face_based = true;
    // viewer.data.set_colors(C);

    viewer.launch();
    return EXIT_SUCCESS;

}
