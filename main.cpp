#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>
#include <igl/boundary_facets.h>


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
    igl::readDMAT(datafile+"simple_joint/generated_files/tet_mesh_V.dmat", V);
    igl::readDMAT(datafile+"simple_joint/generated_files/tet_mesh_T.dmat", T);
    igl::readDMAT(datafile+"simple_joint/generated_files/combined_fiber_directions.dmat", Uvec);
    igl::readDMAT(datafile+"simple_joint/generated_files/muscle_muscle_indices.dmat", muscle1);
    igl::readDMAT(datafile+"simple_joint/generated_files/top_bone_bone_indices.dmat", bone1);
    igl::readDMAT(datafile+"simple_joint/generated_files/bottom_bone_bone_indices.dmat", bone2);
    igl::readDMAT(datafile+"simple_joint/generated_files/joint_indices.dmat", joint1);
    // igl::readDMAT(datafile+"output/combined_V.dmat", V);
    // igl::readDMAT(datafile+"output/combined_T.dmat", T);
    // igl::readDMAT(datafile+"output/fiber_directions.dmat", Uvec);
    // igl::readDMAT(datafile+"output/muscle_I.dmat", muscle1);
    // igl::readDMAT(datafile+"output/bone_1_I.dmat", bone1);
    // igl::readDMAT(datafile+"output/bone_2_I.dmat", bone2);

    igl::boundary_facets(T, F);
    
    std::vector<int> fix = getMaxVerts_Axis_Tolerance(V, 1);
    std::vector<int> fix2 = getMaxVerts_Axis_Tolerance(V, 0, 0.5);
    fix.insert(fix.end(), fix2.begin(), fix2.end());
    std::sort (fix.begin(), fix.end());
    

    std::vector<int> mov = {};//getMinVerts_Axis_Tolerance(V, 1);
    // std::sort (mov.begin(), mov.end());
    
    std::vector<VectorXi> bones = {bone1, bone2};

    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(T, V, fix, mov,bones, muscle1,Uvec,  j_input);
    
    std::cout<<"-----ARAP-----"<<std::endl;
    Reduced_Arap* arap = new Reduced_Arap(*mesh);

    std::cout<<"-----Neo-------"<<std::endl;
    Elastic* neo = new Elastic(*mesh);

    std::cout<<"-----Solver-------"<<std::endl;
    int DIM = mesh->red_s().size();
    RedSolver f(DIM, mesh, arap, neo, j_input);
    LBFGSParam<double> param;
    param.epsilon = 1e-1;
    // param.max_iterations = 1000;
    // param.past = 2;
    // param.m = 5;
    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
    LBFGSSolver<double> solver(param);

    cout<<"---Record Mesh Setup Info"<<endl;
    cout<<"V size: "<<V.rows()<<endl;
    cout<<"T size: "<<T.rows()<<endl;
    cout<<"F size: "<<F.rows()<<endl;
    cout<<"NSH: "<<j_input["number_skinning_handles"]<<endl;
    cout<<"NRC: "<<j_input["number_rotation_clusters"]<<endl;


    int run =0;
    for(int run=0; run<6; run++){
        MatrixXd newV = mesh->continuousV();
        string datafile = j_input["data"];
        igl::writeOBJ(datafile+"modes-skinned-clustered"+to_string(run)+".obj",newV, F);
        igl::writeDMAT(datafile+"modes-skinned-clustered_V"+to_string(run)+".dmat",newV);
        cout<<"---Quasi-Newton Step Info"<<endl;
        double fx =0;
        VectorXd ns = mesh->N().transpose()*mesh->red_s();
        int niter = solver.minimize(f, ns, fx);
        cout<<"BFGSIters: "<<niter<<endl;
        VectorXd reds = mesh->N()*ns + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
        for(int i=0; i<reds.size(); i++){
            mesh->red_s()[i] = reds[i];
        }
        
        neo->changeFiberMag(2);
    }

    exit(0);

    std::cout<<"-----Display-------"<<std::endl;
    igl::opengl::glfw::Viewer viewer;
    MatrixXd Colors = MatrixXd::Random(100,3); // 3x3 Matrix filled with random numbers between (-1,1)
    Colors = (Colors + MatrixXd::Constant(100,3,1.))*(1-1e-6)/2.; // add 1 to the matrix to have values between 0 and 2; multiply with range/2
    Colors = (Colors + MatrixXd::Constant(100,3,1e-6)); //set LO as the lower bound (offset)
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & viewer){   
        if(viewer.core.is_animating){
            
    	}
        return false;
    };

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers){   
        std::cout<<"Key down, "<<key<<std::endl;
        viewer.data().clear();
        if(key=='A'){
            cout<<"here"<<endl;
            neo->changeFiberMag(2);
        }


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
        
        for(int c=0; c<mesh->red_w().size()/3; c++){
            std::vector<int> cluster_elem = mesh->r_cluster_elem_map()[c];
            for(int e=0; e<cluster_elem.size(); e++){
                viewer.data().add_points(newV.row(mesh->T().row(cluster_elem[e])[0]), Colors.row(c));
                viewer.data().add_points(newV.row(mesh->T().row(cluster_elem[e])[1]), Colors.row(c));
                viewer.data().add_points(newV.row(mesh->T().row(cluster_elem[e])[2]), Colors.row(c));
                viewer.data().add_points(newV.row(mesh->T().row(cluster_elem[e])[3]), Colors.row(c));
            }
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
