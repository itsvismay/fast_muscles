#include "mesh.h"
#include "redArap.h"
#include "elastic.h"
#include <LBFGS.h>
#include <igl/Timer.h>

using namespace LBFGSpp;
using json = nlohmann::json;
using namespace Eigen;
using namespace std;

class RedSolver
{
private:
    int n;
    Mesh* mesh;
    Reduced_Arap* arap;
    Elastic* elas;
    double alpha_arap = 1;
    double alpha_neo = 1;
    double eps = 1e-6;
    bool stest = false;
    igl::Timer timer;
    bool terminate;

public:
    RedSolver(int n_, Mesh* m, Reduced_Arap* a, Elastic* e, json& j_input, bool test=false) : n(n_) {
    	mesh = m;
        arap = a;
        elas = e;
        alpha_arap = j_input["alpha_arap"];
        alpha_neo = j_input["alpha_neo"];
        stest = test;

    }
    VectorXd get_w(VectorXd& r0, VectorXd& r){
        VectorXd w = VectorXd::Zero(r0.size()/3);
        for(int i=0; i<r0.size()/9; i++){
            Matrix3d R0, R;
            R0<<r0[9*i+0],r0[9*i+1],r0[9*i+2],
                r0[9*i+3],r0[9*i+4],r0[9*i+5],
                r0[9*i+6],r0[9*i+7],r0[9*i+8];
            R<<r[9*i+0],r[9*i+1],r[9*i+2],
                r[9*i+3],r[9*i+4],r[9*i+5],
                r[9*i+6],r[9*i+7],r[9*i+8];

            Matrix3d exp_brac_w = R0.transpose()*R;
            Matrix3d brac_w = exp_brac_w.log();
            
            w[3*i+0] = brac_w(2,1);
            w[3*i+1] = brac_w(0,2);
            w[3*i+2] = brac_w(1,0);
        }

        return w;
    }
    //CHECK E,x-------------
    VectorXd Ex(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
        VectorXd z = mesh.red_x();
        VectorXd fake = VectorXd::Zero(z.size());
        #pragma omp parallel for
        for(int i=0; i<fake.size(); i++){
            z[i] += 0.5*eps;
            double Eleft = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
            z[i] -= 0.5*eps;
            z[i] -= 0.5*eps;
            double Eright = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
            z[i] += 0.5*eps;
            fake[i] = (Eleft - Eright)/eps;
        }
        return fake;
    }
    //-----------------------

    //CHECK E,r-------------
    VectorXd Er(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
        VectorXd z = mesh.red_x();
        VectorXd fake = VectorXd::Zero(mesh.red_w().size());
        for(int i=0; i<fake.size(); i++){
            mesh.red_w()[i] += 0.5*eps;
            // mesh.setGlobalF(true, false, false);
            double Eleft = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
            mesh.red_w()[i] -= 0.5*eps;
            mesh.red_w()[i] -= 0.5*eps;
            // mesh.setGlobalF(true, false, false);
            double Eright = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
            mesh.red_w()[i] += 0.5*eps;
            fake[i] = (Eleft - Eright)/(eps);
        }
        // mesh.setGlobalF(true, false, false);
        return fake;
    }
    //-----------------------

    //CHECK E,s-------------
    VectorXd Es(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
        VectorXd z = mesh.red_x();
        VectorXd fake = VectorXd::Zero(mesh.red_s().size());
        for(int i=0; i<fake.size(); i++){
            mesh.red_s()[i] += 0.5*eps;
            double Eleft = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
            mesh.red_s()[i] -= 0.5*eps;
            
            mesh.red_s()[i] -= 0.5*eps;
            double Eright = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
            mesh.red_s()[i] += 0.5*eps;
            fake[i] = (Eleft - Eright)/eps;
        }
        return fake;
    }
    //-----------------------

    //CHECK Exx--------------
    MatrixXd Exx(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_x().size());
        VectorXd z = mesh.red_x();
        for(int i=0; i<fake.rows(); i++){
            for(int j=0; j<fake.cols(); j++){
                z[i] += eps;
                z[j] += eps;
                double Eij = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
                z[i] -= eps;
                z[j] -= eps;

                z[i] += eps;
                double Ei = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
                z[i] -= eps;

                z[j] += eps;
                double Ej = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
                z[j] -= eps;
                fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
            }
        }
        
        return fake;
    }
    //-----------------------


    //CHECK Exr/Erx-------------
    MatrixXd Exr(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_w().size());
        VectorXd z = mesh.red_x();

        for(int i=0; i<fake.rows(); i++){
            for(int j=0; j<fake.cols(); j++){
                mesh.red_w()[j] += eps;
                z[i] += eps;
                // mesh.setGlobalF(true, false, false);
                double Eij = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
                mesh.red_w()[j] -= eps;
                z[i] -= eps;

                mesh.red_w()[j] += eps;
                // mesh.setGlobalF(true, false, false);
                double Ei = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
                mesh.red_w()[j] -= eps;

                z[i] += eps;
                // mesh.setGlobalF(true, false, false);
                double Ej = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
                z[i] -= eps;

                fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
            }
        }
        // mesh.setGlobalF(true, false, false);
        return fake;
    }
    //-----------------------

    //CHECK Exs-------------
    MatrixXd Exs(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_s().size());
        VectorXd z = mesh.red_x();

        for(int i=0; i<fake.rows(); i++){
            for(int j=0; j<fake.cols(); j++){
                mesh.red_s()[j] += eps;
                z[i] += eps;
                // mesh.setGlobalF(false, true, false);
                double Eij = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
                mesh.red_s()[j] -= eps;
                z[i] -= eps;

                mesh.red_s()[j] += eps;
                // mesh.setGlobalF(false, true, false);
                double Ei = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
                mesh.red_s()[j] -= eps;

                z[i] += eps;
                // mesh.setGlobalF(false, true, false);
                double Ej = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
                z[i] -= eps;

                fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
            }
        }
        // mesh.setGlobalF(false, true, false);
        return fake;
    }
    //-----------------------

    //CHECK Err--------------
    MatrixXd Err(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_w().size());

        for(int i=0; i<fake.rows(); i++){
            for(int j=0; j<fake.cols(); j++){
                mesh.red_w()[j] += eps;
                mesh.red_w()[i] += eps;
                // mesh.setGlobalF(true, false, false);
                double Eij = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
                mesh.red_w()[j] -= eps;
                mesh.red_w()[i] -= eps;

                mesh.red_w()[j] += eps;
                // mesh.setGlobalF(true, false, false);
                double Ei = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
                mesh.red_w()[j] -= eps;

                mesh.red_w()[i] += eps;
                // mesh.setGlobalF(true, false, false);
                double Ej = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
                mesh.red_w()[i] -= eps;

                fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
            }
        }
        // mesh.setGlobalF(true, false, false);
        
        return fake;
    }
    //-----------------------

    //CHECK Ers--------------
    MatrixXd Ers(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_s().size());

        for(int i=0; i<fake.rows(); i++){
            for(int j=0; j<fake.cols(); j++){
                mesh.red_w()[i] += eps;
                mesh.red_s()[j] += eps;
                // mesh.setGlobalF(true, true, false);
                double Eij = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
                mesh.red_s()[j] -= eps;
                mesh.red_w()[i] -= eps;

                mesh.red_w()[i] += eps;
                // mesh.setGlobalF(true, true, false);
                double Ei = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
                mesh.red_w()[i] -= eps;

                mesh.red_s()[j] += eps;
                // mesh.setGlobalF(true, true, false);
                double Ej = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
                mesh.red_s()[j] -= eps;

                fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
            }
        }
        // mesh.setGlobalF(true, true, false);

        return fake;
    }
    //-----------------------

    VectorXd Full_ARAP_Grad(Mesh& mesh, Reduced_Arap& arap, Elastic& elas, double E0, double eps){
        VectorXd fake = VectorXd::Zero(mesh.red_s().size());
        for(int i=0; i<fake.size(); i++){
            mesh.red_s()[i] += 0.5*eps;
            // mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);
            double Eleft = alpha_arap*arap.Energy(mesh);
            mesh.red_s()[i] -= 0.5*eps;
            
            mesh.red_s()[i] -= 0.5*eps;
            // mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);
            double Eright = alpha_arap*arap.Energy(mesh);
            mesh.red_s()[i] += 0.5*eps;
            fake[i] = (Eleft - Eright)/eps;
        }
        arap.minimize(mesh);
        // mesh.setGlobalF(false, true, false);
        // std::cout<<"FUll fake: "<<fake.transpose()<<std::endl;
        return fake;
    }

    VectorXd Full_NEO_Grad(Mesh& mesh, Reduced_Arap& arap, Elastic& elas, double E0, double eps){
        VectorXd fake = VectorXd::Zero(mesh.red_s().size());
        for(int i=0; i<fake.size(); i++){
            mesh.red_s()[i] += 0.5*eps;
            // mesh.setGlobalF(false, true, false);
            double Eleft = alpha_neo*elas.Energy(mesh);
            mesh.red_s()[i] -= 0.5*eps;
            
            mesh.red_s()[i] -= 0.5*eps;
            // mesh.setGlobalF(false, true, false);
            double Eright = alpha_neo*elas.Energy(mesh);
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
        double Eneo = alpha_neo*elas->Energy(*mesh);
        // timer.start();
        bool converged = arap->minimize(*mesh);

        // timer.stop();
        // double arap_time = timer.getElapsedTimeInMilliSec();
        // cout<<"     ---LineSearch Info"<<endl;
        // cout<<"     ARAPConverged: "<<converged<<endl;
        // cout<<"     ARAPTime: "<<arap_time<<endl;

        double Earap = alpha_arap*arap->Energy(*mesh);
        double fx = Eneo + Earap;


        if(computeGrad){
            VectorXd pegrad = alpha_neo*mesh->N().transpose()*elas->PEGradient(*mesh);
            timer.start();
            VectorXd arapgrad = alpha_arap*mesh->N().transpose()*arap->Jacobians(*mesh);
            timer.stop();
            double arap_grad_time = timer.getElapsedTimeInMilliSec();

           

            
            if(stest){
                VectorXd fake_arap = mesh->N().transpose()*Full_ARAP_Grad(*mesh, *arap,*elas, fx, eps);
                if ((arapgrad-fake_arap).norm()>10){
                    double E0 = arap->Energy(*mesh);
                   
                    std::cout<<"fake arap issues"<<std::endl;
                    std::cout<<arapgrad.transpose()<<std::endl<<std::endl;
                    std::cout<<fake_arap.transpose()<<std::endl<<std::endl;
                    cout<<"s"<<endl;
                    std::cout<<x.transpose()<<endl<<endl;
                    cout<<"r"<<endl;
                    cout<<mesh->red_r().transpose()<<endl<<endl;
                    cout<<"x"<<endl;
                    cout<<mesh->red_x().transpose()<<endl<<endl;
                    cout<<"-------------------------------------"<<endl;
                    cout<<"Ex"<<endl;
                    VectorXd fakeEx = Ex(*mesh, *arap, E0, eps);
                    cout<<(arap->Ex().transpose()-fakeEx.transpose()).norm()<<endl<<endl;

                    cout<<"Er"<<endl;
                    VectorXd fakeEr = Er(*mesh, *arap, E0, eps);
                    cout<<(arap->Er().transpose()-fakeEr.transpose()).norm()<<endl<<endl;

                    cout<<"Es"<<endl;
                    VectorXd fakeEs = Es(*mesh, *arap,E0, eps);
                    cout<<arap->Es().transpose()<<endl<<endl;
                    cout<<fakeEs.transpose()<<endl;
                    cout<<(arap->Es().transpose() - fakeEs.transpose()).norm()<<endl<<endl;
                    

                    cout<<"Exx"<<endl;
                    MatrixXd fakeExx = Exx(*mesh, *arap, E0, eps);
                    cout<<(fakeExx-arap->Exx()).norm()<<endl<<endl;
                    cout<<endl<<endl;

                    MatrixXd fakeExr = Exr(*mesh, *arap, E0, eps);
                    cout<<"Exr"<<endl;
                    cout<<(fakeExr-MatrixXd(arap->Exr())).norm()<<endl<<endl;
                    cout<<endl<<endl;

                    cout<<"Exs"<<endl;
                    MatrixXd fakeExs = Exs(*mesh, *arap, E0, eps);
                    cout<<(fakeExs-MatrixXd(arap->Exs())).norm()<<endl<<endl;
                    cout<<endl;

                    cout<<"Err"<<endl;
                    MatrixXd fakeErr = Err(*mesh, *arap, E0, eps);
                    cout<<(fakeErr-MatrixXd(arap->Err())).norm()<<endl<<endl;
                    cout<<endl;

                    cout<<"Ers"<<endl;
                    MatrixXd fakeErs = Ers(*mesh, *arap, E0, eps);
                    cout<<(fakeErs-MatrixXd(arap->Ers())).norm()<<endl<<endl;
                    cout<<endl;
                    exit(0);
                }  
            }
            
            for(int i=0; i< x.size(); i++){
                grad[i] = pegrad[i];
                grad[i] += arapgrad[i];
            }
            cout<<"---BFGS Info"<<endl;
            cout<<"NeoEnergy: "<<Eneo<<endl;
            cout<<"NeoGradNorm: "<<pegrad.norm()<<endl;
            cout<<"ArapEnergy: "<<Earap<<endl;
            cout<<"ARAPGradNorm: "<<arapgrad.norm()<<endl;
            cout<<"TotalGradNorm: "<<grad.norm()<<endl;


	       // cout<<arapgrad.transpose()<<endl;
        }

        return fx;
    }
};
