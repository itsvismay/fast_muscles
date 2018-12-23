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

class Rosenbrock
{
private:
    int n;
    Mesh* mesh;
    Arap* arap;
    Elastic* elas;
    double alpha_arap = 1;
    double alpha_neo = 1;

public:
    Rosenbrock(int n_, Mesh* m, Arap* a, Elastic* e, json& j_input) : n(n_) {
    	mesh = m;
        arap = a;
        elas = e;
        alpha_arap = j_input["alpha_arap"];
        alpha_neo = j_input["alpha_neo"];

    }

    VectorXd Full_ARAP_Grad(Mesh& mesh, Arap& arap, Elastic& elas, double E0, double eps){
        VectorXd z = mesh.red_x();
        VectorXd fake = VectorXd::Zero(mesh.red_s().size());
        for(int i=0; i<fake.size(); i++){
            mesh.red_s()[i] += 0.5*eps;
            mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);
            double Eleft = alpha_arap*arap.Energy(mesh);
            mesh.red_s()[i] -= 0.5*eps;
            
            mesh.red_s()[i] -= 0.5*eps;
            mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);
            double Eright = alpha_arap*arap.Energy(mesh);
            mesh.red_s()[i] += 0.5*eps;
            fake[i] = (Eleft - Eright)/eps;
        }
        mesh.setGlobalF(false, true, false);
        // std::cout<<"FUll fake: "<<fake.transpose()<<std::endl;
        return fake;
    }

    VectorXd Full_NEO_Grad(Mesh& mesh, Arap& arap, Elastic& elas, double E0, double eps){
        VectorXd fake = VectorXd::Zero(mesh.red_s().size());
        for(int i=0; i<fake.size(); i++){
            mesh.red_s()[i] += 0.5*eps;
            mesh.setGlobalF(false, true, false);
            double Eleft = alpha_neo*elas.Energy(mesh);
            mesh.red_s()[i] -= 0.5*eps;
            
            mesh.red_s()[i] -= 0.5*eps;
            mesh.setGlobalF(false, true, false);
            double Eright = alpha_neo*elas.Energy(mesh);
            mesh.red_s()[i] += 0.5*eps;
            fake[i] = (Eleft - Eright)/eps;
        }
        mesh.setGlobalF(false, true, false);
        // std::cout<<"FUll fake: "<<fake.transpose()<<std::endl;
        return fake;
    }
    VectorXd WikipediaEnergy_grad(Mesh& mesh, Elastic& elas, double eps){
	    VectorXd fake = VectorXd::Zero(mesh.red_s().size());
	    for(int i=0; i<fake.size(); i++){
	        mesh.red_s()[i] += 0.5*eps;
	        double Eleft = elas.WikipediaEnergy(mesh);
	        mesh.red_s()[i] -= 0.5*eps;
	        
	        mesh.red_s()[i] -= 0.5*eps;
	        double Eright = elas.WikipediaEnergy(mesh);
	        mesh.red_s()[i] += 0.5*eps;
	        fake[i] = (Eleft - Eright)/eps;
	    }
	    // mesh.setGlobalF(false, true, false);
	    // std::cout<<"FUll fake: "<<fake.transpose()<<std::endl;
	    return fake;
	}

    double operator()(const VectorXd& x, VectorXd& grad, bool computeGrad = true)
    {
  		VectorXd reds = mesh->N()*x + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
	    for(int i=0; i<reds.size(); i++){
	        mesh->red_s()[i] = reds[i];
	    }

        mesh->setGlobalF(false, true, false);
        arap->minimize(*mesh);
        double Eneo = alpha_neo*elas->Energy(*mesh);
        double Earap = alpha_arap*arap->Energy(*mesh);
        double fx = Eneo + Earap;

        if(computeGrad){        
	        VectorXd pegrad = alpha_neo*mesh->N().transpose()*elas->PEGradient(*mesh);
	        VectorXd arapgrad = alpha_arap*mesh->N().transpose()*arap->Jacobians(*mesh);
	        
	        // VectorXd pegrad = alpha_neo*elas->PEGradient(*mesh);
	        // VectorXd arapgrad = alpha_arap*arap->Jacobians(*mesh);

	        // VectorXd fake_arap = Full_ARAP_Grad(*mesh, *arap,*elas, fx, 1e-5);
	        // if ((arapgrad-fake_arap).norm()>0.001){
	        // 	std::cout<<"fake arap issues"<<std::endl;
	        // 	std::cout<<arapgrad.transpose()<<std::endl<<std::endl;
	        // 	std::cout<<fake_arap.transpose()<<std::endl<<std::endl;
	        // 	exit(0);
	        // }

	        // VectorXd fake = alpha_neo*WikipediaEnergy_grad(*mesh, *elas, 1e-5);
	        // if ((pegrad-fake_neo).norm()>0.001){
	        // 	std::cout<<"fake physics issues"<<std::endl;
	        // 	std::cout<<x.transpose()<<std::endl;
	        // 	std::cout<<arapgrad.transpose()<<std::endl<<std::endl;
	        // 	std::cout<<fake_neo.transpose()<<std::endl<<std::endl;
	        // 	exit(0);
	        // }

	        for(int i=0; i< x.size(); i++){
	            grad[i] = arapgrad[i];
	        	grad[i] += pegrad[i];
	            // grad[i] = fake[i];
	        }
	        	std::cout<<Eneo<<", "<<Earap<<", "<<Eneo+Earap<<", "<<grad.norm()<<std::endl;
	        // std::cout<<pegrad.head(12).transpose()<<std::endl<<std::endl;
	        // std::cout<<arapgrad.head(12).transpose()<<std::endl<<std::endl;
        }


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
    cout<<"V size: "<<V.rows()<<endl;
    cout<<"T size: "<<T.rows()<<endl;
    cout<<"F size: "<<F.rows()<<endl;
    
    std::vector<int> fix = getMaxVerts_Axis_Tolerance(V, 1);
    std::sort (fix.begin(), fix.end());
    std::vector<int> mov = getMinVerts_Axis_Tolerance(V, 1);
    std::sort (mov.begin(), mov.end());
    std::vector<int> bones = {};
    // getMaxTets_Axis_Tolerance(bones, V, T, 1, 3);
    // getMinTets_Axis_Tolerance(bones, V, T, 1, 3);

    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(T, V, fix, mov,bones, j_input);

    std::cout<<"-----ARAP-----"<<std::endl;
    Arap* arap = new Arap(*mesh);


    // std::cout<<"-----Neo-------"<<std::endl;
    // Elastic* neo = new Elastic(*mesh);

    // std::cout<<"-----Solver-------"<<std::endl;
    // int DIM = mesh->red_s().size();
    // Rosenbrock f(DIM, mesh, arap, neo, j_input);
    // LBFGSParam<double> param;
    // // param.epsilon = 1e-1;
    // // param.max_iterations = 1000;
    // // param.past = 2;
    // // param.m = 5;
    // param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
    // LBFGSSolver<double> solver(param);


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
		        dx[3*mov[i]+1] -= 3;
		    }
            // for(int i=0; i<mesh->red_s().size()/6; i++){
            //     mesh->red_s()[6*i+1] += 0.1;
            // }
            arap->minimize(*mesh);
            cout<<"dx"<<endl;
            cout<<dx.transpose()<<endl;
            cout<<"rot"<<endl;
            cout<<mesh->red_r().transpose()<<endl;
            cout<<"x"<<endl;
            cout<<mesh->red_x().transpose()<<endl;
		    // double fx =0;
		    // VectorXd ns = mesh->N().transpose()*mesh->red_s();
		    // cout<<"NS"<<endl;
		    // int niter = solver.minimize(f, ns, fx);
		    // VectorXd reds = mesh->N()*ns + mesh->AN()*mesh->AN().transpose()*mesh->red_s();
		    // // VectorXd reds = mesh->red_s();
		    // // cout<<reds.size()<<endl;
		    // // cout<<mesh->T().rows()*6<<endl;
		    // // cout<<mesh->bones().transpose()<<endl;
		    // // int niter = solver.minimize(f, reds, fx);
		    
		    // for(int i=0; i<reds.size(); i++){
	     //        mesh->red_s()[i] = reds[i];
	     //    }
		    // mesh->setGlobalF(false, true, false);

			// std::cout<<"new s"<<std::endl;
			// std::cout<<reds.transpose()<<std::endl;
			// std::cout<<"niter "<<niter<<std::endl;
        }
        
        //----------------
        //Draw continuous mesh
        MatrixXd newV = mesh->continuousV();
        viewer.data().set_mesh(newV, F);
        
        //Draw disc mesh
        std::cout<<std::endl;
        MatrixXd& discV = mesh->discontinuousV();
        MatrixXi& discT = mesh->discontinuousT();
        for(int i=0; i<discT.rows(); i++){
            Vector4i e = discT.row(i);
            // std::cout<<discT.row(i)<<std::endl<<std::endl;
            // std::cout<<discV(Eigen::placeholders::all, discT.row(i))<<std::endl;
            Matrix<double, 1,3> p0 = discV.row(e[0]);
            Matrix<double, 1,3> p1 = discV.row(e[1]);
            Matrix<double, 1,3> p2 = discV.row(e[2]);
            Matrix<double, 1,3> p3 = discV.row(e[3]);

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