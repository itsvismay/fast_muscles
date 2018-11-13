#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>

#include <json.hpp>

#include "mesh.h"
#include "arap.h"

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

    std::vector<int> fix = {0,3};
    std::sort (fix.begin(), fix.end());
    std::vector<int> mov = {};
    std::sort (mov.begin(), mov.end());

    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(T, V, fix, mov);

    std::cout<<"-----ARAP-------"<<std::endl;
    Arap* arap = new Arap(*mesh);
    arap->minimize(*mesh);


    igl::opengl::glfw::Viewer viewer;
    std::cout<<"-----Display-------"<<std::endl;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & viewer)
    {   
        if(viewer.core.is_animating)
        {
            
    	}
        return false;
    };

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers)
    {   //Doing things
        VectorXd& s = mesh->s();
        for(int i=0; i<mesh->T().rows(); i++){
            s[6*i+1] += 0.1;
        }
        mesh->setGlobalF(false, true, false);
        // std::cout<<mesh->s()<<std::endl;
        //----------------

        //Draw continuous mesh
        // MatrixXd newV = mesh->continuousV();
        // viewer.data().set_vertices(newV);
        //Draw disc mesh
        std::cout<<std::endl;
        MatrixXd discV = mesh->discontinuousV();
        MatrixXi discT = mesh->discontinuousT();
        for(int i=0; i<discT.rows(); i++){
            Vector4i e = discT.row(i);
            // std::cout<<discT.row(i)<<std::endl<<std::endl;
            // std::cout<<discV(Eigen::placeholders::all, discT.row(i))<<std::endl;
            MatrixXd p0 = discV.row(e[0]);
            MatrixXd p1 = discV.row(e[1]);
            MatrixXd p2 = discV.row(e[2]);
            MatrixXd p3 = discV.row(e[3]);
            viewer.data().add_edges(p0,p1,Eigen::RowVector3d(1,0,1));
            viewer.data().add_edges(p0,p2,Eigen::RowVector3d(1,0,1));
            viewer.data().add_edges(p0,p3,Eigen::RowVector3d(1,0,1));
            viewer.data().add_edges(p1,p2,Eigen::RowVector3d(1,0,1));
            viewer.data().add_edges(p1,p3,Eigen::RowVector3d(1,0,1));
            viewer.data().add_edges(p2,p3,Eigen::RowVector3d(1,0,1));
        }
        //Draw fixed and moving points
        for(int i=0; i<fix.size(); i++){
            viewer.data().add_points(mesh->V().row(fix[i]),Eigen::RowVector3d(1,0,0));
        }
        for(int i=0; i<mov.size(); i++){
            viewer.data().add_points(mesh->V().row(mov[i]),Eigen::RowVector3d(0,1,0));
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
