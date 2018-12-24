#include "mesh.h"
#include "arap.h"
#include "elastic.h"
#include <LBFGS.h>

using namespace LBFGSpp;
using json = nlohmann::json;
using namespace Eigen;
using namespace std;

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

	        // VectorXd fake_arap = mesh->N().transpose()*Full_ARAP_Grad(*mesh, *arap,*elas, fx, 1e-5);
	        // if ((arapgrad-fake_arap).norm()>0.01){
	        // 	std::cout<<"fake arap issues"<<std::endl;
	        // 	std::cout<<arapgrad.transpose()<<std::endl<<std::endl;
	        // 	std::cout<<fake_arap.transpose()<<std::endl<<std::endl;
         //        cout<<"s"<<endl;
         //        std::cout<<x.transpose()<<endl<<endl;
         //        cout<<"r"<<endl;
         //        cout<<mesh->red_r().transpose()<<endl<<endl;
         //        cout<<"x"<<endl;
         //        cout<<mesh->red_x().transpose()<<endl<<endl;
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
	        	grad[i] = pegrad[i];
                grad[i] += arapgrad[i];
	            // grad[i] = fake[i];
	        }
	        std::cout<<Eneo<<", "<<Earap<<", "<<Eneo+Earap<<", "<<grad.norm()<<std::endl;
	        // std::cout<<pegrad.head(12).transpose()<<std::endl<<std::endl;
	        // std::cout<<arapgrad.head(12).transpose()<<std::endl<<std::endl;
        }


        return fx;
    }
};
