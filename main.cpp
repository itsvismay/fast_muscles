#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readDMAT.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>
#include <igl/boundary_facets.h>


#include "mesh.h"
#include "arap.h"
#include "elastic.h"


using json = nlohmann::json;

using namespace Eigen;
using namespace std;
json j_input;

RowVector3d red(1,0,0);
RowVector3d purple(1,0,1);
RowVector3d green(0,1,0);
RowVector3d black(0,0,0);
MatrixXd Colors;

std::vector<int> getMaxVerts_Axis_Tolerance(MatrixXd& mV, int dim, double tolerance=1e-5){
    auto maxX = mV.col(dim).maxCoeff();
    std::vector<int> maxV;
    for(unsigned int ii=0; ii<mV.rows(); ++ii) {

        if(fabs(mV(ii,dim) - maxX) < tolerance) {
            maxV.push_back(ii);
            std::cout<<ii;
        }
    }
    return maxV;
}

std::vector<int> getMinVerts_Axis_Tolerance(MatrixXd& mV, int dim, double tolerance=1e-5){
    auto maxX = mV.col(dim).minCoeff();
    std::vector<int> maxV;
    for(unsigned int ii=0; ii<mV.rows(); ++ii) {

        if(fabs(mV(ii,dim) - maxX) < tolerance) {
            maxV.push_back(ii);
            std::cout<<ii;
        }
    }
    return maxV;
}

int main(int argc, char *argv[])
{
    std::cout<<"-----Configs-------"<<std::endl;
    json j_config_parameters;
    std::ifstream i("../input/input.json");
    i >> j_input;
    
    MatrixXd V;
    MatrixXi T;
    MatrixXi F;
    igl::readDMAT(j_input["v_file"], V);
    igl::readDMAT(j_input["t_file"], T);
    igl::boundary_facets(T, F);
    cout<<"V size: "<<V.rows()<<endl;
    cout<<"T size: "<<T.rows()<<endl;
    cout<<"F size: "<<F.rows()<<endl;
    
    std::vector<int> fix = getMaxVerts_Axis_Tolerance(V, 1);
    std::sort (fix.begin(), fix.end());
    std::vector<int> mov = getMinVerts_Axis_Tolerance(V, 1);
    std::sort (mov.begin(), mov.end());
    std::vector<int> bones = {};
    // exit(0);


    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(T, V, fix, mov,bones, j_input);
    
    std::cout<<"-----ARAP-----"<<std::endl;
    std::cout<<"-----Neo-------"<<std::endl;
    std::cout<<"-----Solver-------"<<std::endl;


    std::cout<<"-----Display-------"<<std::endl;
    igl::opengl::glfw::Viewer viewer;
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & viewer){   
        if(viewer.core.is_animating){
            
    	}
        return false;
    };

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers){   
        std::cout<<"Key down, "<<key<<std::endl;
        viewer.data().clear();
        

        if(key==' '){
          
        }
        
        //----------------
        //Draw continuous mesh
        MatrixXd newV = mesh->continuousV();
        viewer.data().set_mesh(newV, F);
        
        //Draw disc mesh
        // std::cout<<std::endl;
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
            viewer.data().add_points(newV.row(mov[i]),Eigen::RowVector3d(0,1,0));
        }
        // viewer.data().set_colors(Colors);
        
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
