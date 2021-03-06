#include "mesh.h"
#include "arap.h"
#include "redArap.h"
#include "elastic.h"
#include "solver.h"
#include "redSolver.h"
#include <Eigen/Core>
#include <iostream>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/readDMAT.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/setdiff.h>

using namespace LBFGSpp;
using json = nlohmann::json;
using namespace Eigen;
using namespace std;
json j_input;

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
        }
    }
    return maxV;
}
void getMaxTets_Axis_Tolerance(std::vector<int>& ibones, MatrixXd& mV, MatrixXi& mT, double dim, double tolerance = 1-5){
	auto maxX = mV.col(dim).maxCoeff();
	for(int i=0; i< mT.rows(); i++){
		Vector3d centre = (mV.row(mT.row(i)[0])+ mV.row(mT.row(i)[1]) + mV.row(mT.row(i)[2])+ mV.row(mT.row(i)[3]))/4.0;
		if (fabs(centre[dim] - maxX)< tolerance){
			ibones.push_back(i);
		}
	}
}

void getMinTets_Axis_Tolerance(std::vector<int>& ibones, MatrixXd& mV, MatrixXi& mT, double dim, double tolerance = 1-5){
	auto maxX = mV.col(dim).minCoeff();
	for(int i=0; i< mT.rows(); i++){
		Vector3d centre = (mV.row(mT.row(i)[0])+ mV.row(mT.row(i)[1]) + mV.row(mT.row(i)[2])+ mV.row(mT.row(i)[3]))/4.0;
		if (fabs(centre[dim] - maxX)< tolerance){
			ibones.push_back(i);
		}
	}
}


int main()
{
	std::cout<<"-----Configs-------"<<std::endl;
    json j_config_parameters;
    std::ifstream i("../input/input.json");
    i >> j_input;
    if(!j_input["reduced"]){
        std::cout<<"USE unreduced code"<<endl;
        exit(0);
    }

    
    MatrixXd V;
    MatrixXi T;
    MatrixXi F;
    igl::readMESH(j_input["mesh_file"], V, T, F);
    cout<<"V size: "<<V.rows()<<endl;
    cout<<"T size: "<<T.rows()<<endl;
    cout<<"F size: "<<F.rows()<<endl;
    
    std::vector<int> fix = getMaxVerts_Axis_Tolerance(V, 1);
    std::sort (fix.begin(), fix.end());
    std::vector<int> mov = {};//getMinVerts_Axis_Tolerance(V, 1);
    std::sort (mov.begin(), mov.end());

    std::vector<MatrixXd> joints;
    std::vector<int> bone1={};
    getMaxTets_Axis_Tolerance(bone1, V, T, 1, 3);
    std::vector<int> bone2={};
    getMinTets_Axis_Tolerance(bone2, V, T, 1, 3);
    VectorXi bone1vec = VectorXi::Map(bone1.data(), bone1.size());
    VectorXi bone2vec = VectorXi::Map(bone2.data(), bone2.size());
    VectorXi bonesvec(bone1vec.size() + bone2vec.size());
    bonesvec<< bone1vec,bone2vec;
    VectorXi all(T.rows());
    MatrixXd Uvec(all.size(), 3);
    for(int i=0; i<T.rows(); i++){
        all[i] = i;
        Uvec.row(i) = Vector3d::UnitY();
    }
    VectorXi muscle1;
    VectorXi shit;
    igl::setdiff(all, bonesvec, muscle1, shit);

    std::vector<VectorXi> muscles = {muscle1};
    std::vector<VectorXi> bones = {bone1vec, bone2vec};
    std::vector<int> fix_bones = {0};

    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(T, V, fix_bones, mov, bones, muscles, joints, Uvec, j_input);

    std::cout<<"-----Neo-------"<<std::endl;
    Elastic* neo = new Elastic(*mesh);

    std::cout<<"-----ARAP-----"<<std::endl;
    Reduced_Arap* arap = new Reduced_Arap(*mesh);
    
    std::cout<<"-----Solver-------"<<std::endl;
    int DIM = mesh->red_s().size();
    
    RedSolver f(DIM, mesh, arap, neo, j_input, true);
  

  
    LBFGSParam<double> param;
    param.epsilon = 1e-1;
    // param.max_iterations = 1000;
    // param.past = 2;
    // param.m = 5;
    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
    LBFGSSolver<double> solver(param);

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

    igl::opengl::glfw::Viewer viewer;
    std::cout<<"-----Display-------"<<std::endl;
    MatrixXd Colors = MatrixXd::Random(100,3); // 3x3 Matrix filled with random numbers between (-1,1)
    Colors = (Colors + MatrixXd::Constant(100,3,1.))*(1-1e-6)/2.; // add 1 to the matrix to have values between 0 and 2; multiply with range/2
    Colors = (Colors + MatrixXd::Constant(100,3,1e-6)); //set LO as the lower bound (offset)
    double tttt = 0;
    int kkkk = 0;
    VectorXd testY = VectorXd::Zero(mesh->Y().cols());
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & viewer)
    {   
        if(viewer.core.is_animating)
        {   
            // viewer.data().clear();
            // if(kkkk<mesh->G().cols()){
            //     VectorXd x = 10*sin(tttt)*mesh->G().col(kkkk) + mesh->x0();
            //     Eigen::Map<Eigen::MatrixXd> newV(x.data(), V.cols(), V.rows());
            //     viewer.data().set_mesh(newV.transpose(), F);
            //     tttt+= 0.1;
            // }
    	}
            
        
        return false;
    };

    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int modifiers)
    {   
        kkkk +=1;

        std::cout<<"Key down, "<<key<<std::endl;
        viewer.data().clear();
        
        //----------------
        if(key=='A'){
            cout<<"here"<<endl;
            neo->changeFiberMag(2);
        }

        if(key==' '){
      		VectorXd& dx = mesh->dx();
            
            // VectorXd ns = mesh->N().transpose()*mesh->red_s();
            // for(int i=0; i<ns.size()/6; i++){
            //     ns[6*i+1] += 0.1;
            //     ns[6*i+0] += 0.1;
            // }
            
            // arap->minimize(*mesh);
            
            double fx =0;
            VectorXd ns = mesh->N().transpose()*mesh->red_s();
            int niter = solver.minimize(f, ns, fx);
            std::cout<<"niter "<<niter<<std::endl;

            VectorXd reds = mesh->N()*ns + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
            for(int i=0; i<reds.size(); i++){
                mesh->red_s()[i] = reds[i];
            }
        }
        // ----------------
        // Draw continuous mesh
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
        // //Draw fixed and moving points
        // for(int i=0; i<fix.size(); i++){
        //     viewer.data().add_points(mesh->V().row(fix[i]),Eigen::RowVector3d(1,0,0));
        // }
        // for(int i=0; i<mov.size(); i++){
        //     viewer.data().add_points(newV.row(mov[i]),Eigen::RowVector3d(0,1,0));
        // }
        // for(int c=0; c<mesh->red_w().size()/3; c++){
        //     std::vector<int> cluster_elem = mesh->r_cluster_elem_map()[c];
        //     for(int e=0; e<cluster_elem.size(); e++){
        //         viewer.data().add_points(newV.row(mesh->T().row(cluster_elem[e])[0]), Colors.row(c));
        //         viewer.data().add_points(newV.row(mesh->T().row(cluster_elem[e])[1]), Colors.row(c));
        //         viewer.data().add_points(newV.row(mesh->T().row(cluster_elem[e])[2]), Colors.row(c));
        //         viewer.data().add_points(newV.row(mesh->T().row(cluster_elem[e])[3]), Colors.row(c));
        //     }
        // }

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

    
    return 0;
}