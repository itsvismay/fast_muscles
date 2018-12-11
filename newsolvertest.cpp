#include "mesh.h"
#include "arap.h"
#include "elastic.h"
#include <Eigen/Core>
#include <iostream>
#include <LBFGS.h>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>

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

class Rosenbrock
{
private:
    int n;
    Mesh* mesh;
    Arap* arap;
    Elastic* elas;

public:
    Rosenbrock(int n_, Mesh* m, Arap* a, Elastic* e) : n(n_) {
    	mesh = m;
        arap = a;
        elas = e;
    }

    VectorXd Full_FD_Grad(Mesh& mesh, Arap& arap, double E0, double eps){
        VectorXd z = mesh.red_x();
        VectorXd fake = VectorXd::Zero(mesh.red_s().size());
        for(int i=0; i<fake.size(); i++){
            mesh.red_s()[i] += 0.5*eps;
            mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);
            double Eleft = arap.Energy(mesh);
            mesh.red_s()[i] -= 0.5*eps;
            
            mesh.red_s()[i] -= 0.5*eps;
            mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);
            double Eright = arap.Energy(mesh);
            mesh.red_s()[i] += 0.5*eps;
            fake[i] = (Eleft - Eright)/eps;
        }
        mesh.setGlobalF(false, true, false);
        // std::cout<<"FUll fake: "<<fake.transpose()<<std::endl;
        return fake;
    }

    double operator()(const VectorXd& x, VectorXd& grad)
    {
        for(int i=0; i<x.size(); i++){
            mesh->red_s()[i] = x[i];
        }
        std::cout<<"updated s"<<std::endl;
        std::cout<<x.transpose()<<std::endl;
        mesh->setGlobalF(false, true, false);
        arap->minimize(*mesh);

        double Earap = arap->Energy(*mesh);
        double Eneo = 0; //elas->Energy(*mesh);
        std::cout<<"neo: "<<Eneo<<", "<<"arap: "<<Earap<<std::endl;

        // VectorXd pegrad = elas->PEGradient(*mesh);
        VectorXd arapgrad = arap->Jacobians(*mesh);
        // VectorXd fake = Full_FD_Grad(*mesh, *arap, fx, 1e-5);
        // if ((arapgrad-fake).norm()>0.001){
        // 	std::cout<<arapgrad<<std::endl<<std::endl;
        // 	std::cout<<fake<<std::endl<<std::endl;
        // 	exit(0);
        // }

        for(int i=0; i< x.size(); i++){
        	// grad[i] = pegrad[i];
            grad[i] = arapgrad[i];
            // grad[i] = fake[i];
        }

        double fx = Earap + Eneo;
        assert( ! std::isnan(fx) );
        return fx;
    }
};

int main()
{
	std::cout<<"-----Configs-------"<<std::endl;
    json j_config_parameters;
    std::ifstream i("../input/input.json");
    i >> j_input;

    
    MatrixXd V;
    MatrixXi T;
    MatrixXi F;
    igl::readMESH(j_input["mesh_file"], V, T, F);
    
    std::vector<int> fix = getMaxVerts_Axis_Tolerance(V, 1);
    std::sort (fix.begin(), fix.end());
    std::vector<int> mov = {1,7};
    std::sort (mov.begin(), mov.end());

    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(T, V, fix, mov, j_input);

    std::cout<<"-----ARAP-----"<<std::endl;
    Arap* arap = new Arap(*mesh);

    std::cout<<"-----Neo-------"<<std::endl;
    Elastic* neo = new Elastic(*mesh);

    std::cout<<"-----Solver-------"<<std::endl;
    int DIM = mesh->red_s().size();
    Rosenbrock f(DIM, mesh, arap, neo);
    LBFGSParam<double> param;
    LBFGSSolver<double> solver(param);


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
    {   std::cout<<"Key down, "<<key<<std::endl;
        viewer.data().clear();
        
        //Doing things
        // VectorXd& s = mesh->red_s();
        // for(int i=0; i<s.size()/6; i++){
        //     s[6*i+1] += 0.1;
        // }
        // mesh->setGlobalF(false, true, false);
        // arap->minimize(*mesh);
        // std::cout<<arap->Jacobians(*mesh)<<std::endl;
        // //----------------

        if(key==' '){
      		VectorXd& dx = mesh->dx();
		    for(int i=0; i<mov.size(); i++){
		        dx[3*mov[i]+1] += 3;
		    }

		    double fx =0;
		    VectorXd ns = mesh->red_s();
		    int niter = solver.minimize(f, ns, fx);
		    for(int i=0; i<ns.size(); i++){
		        mesh->red_s()[i] = ns[i];
		    }
		    mesh->setGlobalF(false, true, false);

			std::cout<<"new s"<<std::endl;
			std::cout<<ns.transpose()<<std::endl;
			std::cout<<"niter "<<niter<<std::endl;
        }
        
        //----------------
        //Draw continuous mesh
        MatrixXd newV = mesh->continuousV();
        viewer.data().set_mesh(newV, F);
        
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

    
    return 0;
}