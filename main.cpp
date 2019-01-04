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
#include "solver.h"



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
    MatrixXd Uvec;
    VectorXi muscle1;
    VectorXi bone1;
    VectorXi bone2;
    VectorXi joint1;




    std::string datafile = j_input["data"];
    igl::readDMAT(j_input["v_file"], V);
    igl::readDMAT(j_input["t_file"], T);
    igl::readDMAT(datafile+"simple_joint/generated_files/combined_fiber_directions.dmat", Uvec);
    igl::readDMAT(datafile+"simple_joint/generated_files/muscle_muscle_indices.dmat", muscle1);
    igl::readDMAT(datafile+"simple_joint/generated_files/top_bone_bone_indices.dmat", bone1);
    igl::readDMAT(datafile+"simple_joint/generated_files/bottom_bone_bone_indices.dmat", bone2);
    igl::readDMAT(datafile+"simple_joint/generated_files/joint_indices.dmat", joint1);

    igl::boundary_facets(T, F);
    cout<<"V size: "<<V.rows()<<endl;
    cout<<"T size: "<<T.rows()<<endl;
    cout<<"F size: "<<F.rows()<<endl;
    
    std::vector<int> fix = getMaxVerts_Axis_Tolerance(V, 1);
    std::vector<int> fix2 = getMaxVerts_Axis_Tolerance(V, 0, 0.5);
    fix.insert(fix.end(), fix2.begin(), fix2.end());
    std::sort (fix.begin(), fix.end());
    

    std::vector<int> mov = {};//getMinVerts_Axis_Tolerance(V, 1);
    // std::sort (mov.begin(), mov.end());
    
    std::vector<int> bones = {};
    for(int i=0; i<bone1.size(); i++){
        bones.push_back(bone1[i]);
    }
    for(int i=0; i<bone2.size(); i++){
        bones.push_back(bone2[i]);
    }
    // for(int i=0; i<joint1.size(); i++){
    //     bones.push_back(joint1[i]);
    // }


    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(T, V, fix, mov,bones, muscle1,Uvec,  j_input);
    
    std::cout<<"-----ARAP-----"<<std::endl;
    Arap* arap = new Arap(*mesh);

    std::cout<<"-----Neo-------"<<std::endl;
    Elastic* neo = new Elastic(*mesh);

    std::cout<<"-----Solver-------"<<std::endl;
    int DIM = mesh->red_s().size();
    Rosenbrock f(DIM, mesh, arap, neo, j_input);
    LBFGSParam<double> param;
    param.epsilon = 1e-1;
    // param.max_iterations = 1000;
    // param.past = 2;
    // param.m = 5;
    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
    LBFGSSolver<double> solver(param);

    for(int i=0; i<5; i++){
        MatrixXd newV = mesh->continuousV();
        string datafile = j_input["data"];
        igl::writeOBJ(datafile+"simple_joint"+to_string(i)+".obj",newV,F);
        
        double fx =0;
        VectorXd ns = mesh->N().transpose()*mesh->red_s();
        int niter = solver.minimize(f, ns, fx);
        cout<<"End BFGS"<<", "<<niter<<endl;
        VectorXd reds = mesh->N()*ns + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
        for(int i=0; i<reds.size(); i++){
            mesh->red_s()[i] = reds[i];
        }
        
        neo->changeFiberMag(5);
    }
    exit(0);

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
          double fx =0;
            VectorXd ns = mesh->N().transpose()*mesh->red_s();
            cout<<"NS"<<endl;
            int niter = solver.minimize(f, ns, fx);
            VectorXd reds = mesh->N()*ns + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
            
            for(int i=0; i<reds.size(); i++){
                mesh->red_s()[i] = reds[i];
            }
    
        }
        
        //----------------
        //Draw continuous mesh
        MatrixXd newV = mesh->continuousV();
        viewer.data().set_mesh(newV, F);
        
        //Draw disc mesh
        // std::cout<<std::endl;
        // MatrixXd& discV = mesh->discontinuousV();
        // MatrixXi& discT = mesh->discontinuousT();
        // for(int i=0; i<discT.rows(); i++){
        //     Vector4i e = discT.row(i);
        //     // std::cout<<discT.row(i)<<std::endl<<std::endl;
        //     // std::cout<<discV(Eigen::placeholders::all, discT.row(i))<<std::endl;
        //     Matrix<double, 1,3> p0 = discV.row(e[0]);
        //     Matrix<double, 1,3> p1 = discV.row(e[1]);
        //     Matrix<double, 1,3> p2 = discV.row(e[2]);
        //     Matrix<double, 1,3> p3 = discV.row(e[3]);

        //     viewer.data().add_edges(p0,p1,Eigen::RowVector3d(1,0,1));
        //     viewer.data().add_edges(p0,p2,Eigen::RowVector3d(1,0,1));
        //     viewer.data().add_edges(p0,p3,Eigen::RowVector3d(1,0,1));
        //     viewer.data().add_edges(p1,p2,Eigen::RowVector3d(1,0,1));
        //     viewer.data().add_edges(p1,p3,Eigen::RowVector3d(1,0,1));
        //     viewer.data().add_edges(p2,p3,Eigen::RowVector3d(1,0,1));
        // }

        //Draw fixed and moving points
        for(int i=0; i<fix.size(); i++){
            viewer.data().add_points(mesh->V().row(fix[i]),Eigen::RowVector3d(1,0,0));
        }
        for(int i=0; i<mov.size(); i++){
            viewer.data().add_points(newV.row(mov[i]),Eigen::RowVector3d(0,1,0));
        }
        // // viewer.data().set_colors(Colors);
        
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
