#ifndef SOLVER
#define SOLVER

#include <iostream>
#include "mesh.h"
#include "arap.h"
#include "elastic.h"
#include "cppoptlib/meta.h"
#include "cppoptlib/boundedproblem.h"
#include "cppoptlib/solver/lbfgsbsolver.h"
#include "cppoptlib/solver/neldermeadsolver.h"

using namespace cppoptlib;
using Eigen::VectorXd;
template<typename T>
class StaticSolve : public BoundedProblem<T> {
private:
    Mesh* mesh;
    Arap* arap;
    Elastic* elas;
    double alpha_neo =1;
    double alpha_arap = 1; 

public:
    using typename cppoptlib::BoundedProblem<T>::Scalar;
    using typename cppoptlib::BoundedProblem<T>::TVector;

    StaticSolve(int dim, Mesh* m, Arap* a, Elastic* e): BoundedProblem<T>(dim) {
        mesh = m;
        arap = a;
        elas = e;

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
    VectorXd Ex(Mesh& mesh, Arap& arap, double E0, double eps){
        VectorXd z = mesh.red_x();
        VectorXd fake = VectorXd::Zero(z.size());
        #pragma omp parallel for
        for(int i=0; i<fake.size(); i++){
            z[i] += 0.5*eps;
            double Eleft = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
            z[i] -= 0.5*eps;
            z[i] -= 0.5*eps;
            double Eright = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
            z[i] += 0.5*eps;
            fake[i] = (Eleft - Eright)/eps;
        }
        return fake;
    }
    //-----------------------

    //CHECK E,r-------------
    VectorXd Er(Mesh& mesh, Arap& arap, double E0, double eps){
        VectorXd z = mesh.red_x();
        VectorXd fake = VectorXd::Zero(mesh.red_w().size());
        for(int i=0; i<fake.size(); i++){
            mesh.red_w()[i] += 0.5*eps;
            mesh.setGlobalF(true, false, false);
            double Eleft = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
            mesh.red_w()[i] -= 0.5*eps;
            mesh.red_w()[i] -= 0.5*eps;
            mesh.setGlobalF(true, false, false);
            double Eright = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
            mesh.red_w()[i] += 0.5*eps;
            fake[i] = (Eright - Eleft)/(eps);
        }
        mesh.setGlobalF(true, false, false);

        return fake;
    }
    //-----------------------

    //CHECK E,s-------------
    VectorXd Es(Mesh& mesh, Arap& arap, double E0, double eps){
        VectorXd z = mesh.red_x();
        VectorXd fake = VectorXd::Zero(mesh.red_s().size());
        for(int i=0; i<fake.size(); i++){
            mesh.red_s()[i] += 0.5*eps;
            mesh.setGlobalF(false, true, false);
            double Eleft = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
            mesh.red_s()[i] -= 0.5*eps;
            
            mesh.red_s()[i] -= 0.5*eps;
            mesh.setGlobalF(false, true, false);
            double Eright = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
            mesh.red_s()[i] += 0.5*eps;
            fake[i] = (Eleft - Eright)/eps;
        }
        mesh.setGlobalF(false, true, false);
        return fake;
    }
    //-----------------------

    //CHECK Exx--------------
    MatrixXd Exx(Mesh& mesh, Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_x().size());
        VectorXd z = mesh.red_x();
        // std::cout<<z.transpose()<<std::endl;
        for(int i=0; i<fake.rows(); i++){
            for(int j=0; j<fake.cols(); j++){
                z[i] += eps;
                z[j] += eps;
                double Eij = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
                z[i] -= eps;
                z[j] -= eps;

                z[i] += eps;
                double Ei = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
                z[i] -= eps;

                z[j] += eps;
                double Ej = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
                z[j] -= eps;
                fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
            }
        }
        return fake;
    }

    //-----------------------

    //CHECK Exr/Erx-------------
    MatrixXd Exr(Mesh& mesh, Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_w().size());
        VectorXd z = mesh.red_x();

        for(int i=0; i<fake.rows(); i++){
            for(int j=0; j<fake.cols(); j++){
                mesh.red_w()[j] += eps;
                z[i] += eps;
                mesh.setGlobalF(true, false, false);
                double Eij = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
                mesh.red_w()[j] -= eps;
                z[i] -= eps;

                mesh.red_w()[j] += eps;
                mesh.setGlobalF(true, false, false);
                double Ei = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
                mesh.red_w()[j] -= eps;

                z[i] += eps;
                mesh.setGlobalF(true, false, false);
                double Ej = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
                z[i] -= eps;

                fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
            }
        }
        mesh.setGlobalF(true, false, false);
        return fake;
    }
    //-----------------------

    //CHECK Exs-------------
    MatrixXd Exs(Mesh& mesh, Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_s().size());
        VectorXd z = mesh.red_x();

        for(int i=0; i<fake.rows(); i++){
            for(int j=0; j<fake.cols(); j++){
                mesh.red_s()[j] += eps;
                z[i] += eps;
                mesh.setGlobalF(false, true, false);
                double Eij = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
                mesh.red_s()[j] -= eps;
                z[i] -= eps;

                mesh.red_s()[j] += eps;
                mesh.setGlobalF(false, true, false);
                double Ei = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
                mesh.red_s()[j] -= eps;

                z[i] += eps;
                mesh.setGlobalF(false, true, false);
                double Ej = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
                z[i] -= eps;

                fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
            }
        }
        mesh.setGlobalF(false, true, false);
        return fake;
    }
    //-----------------------

    //CHECK Err--------------
    MatrixXd Err(Mesh& mesh, Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_w().size());

        for(int i=0; i<fake.rows(); i++){
            for(int j=0; j<fake.cols(); j++){
                mesh.red_w()[j] += eps;
                mesh.red_w()[i] += eps;
                mesh.setGlobalF(true, false, false);
                double Eij = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
                mesh.red_w()[j] -= eps;
                mesh.red_w()[i] -= eps;

                mesh.red_w()[j] += eps;
                mesh.setGlobalF(true, false, false);
                double Ei = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
                mesh.red_w()[j] -= eps;

                mesh.red_w()[i] += eps;
                mesh.setGlobalF(true, false, false);
                double Ej = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
                mesh.red_w()[i] -= eps;

                fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
            }
        }
        mesh.setGlobalF(true, false, false);
        return fake;
    }
    //-----------------------

    //CHECK Ers--------------
    MatrixXd Ers(Mesh& mesh, Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_s().size());
        fake.setZero();
        for(int i=0; i<fake.rows(); i++){
            for(int j=0; j<fake.cols(); j++){
                mesh.red_w()[i] += eps;
                mesh.red_s()[j] += eps;
                mesh.setGlobalF(true, true, false);
                double Eij = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
                mesh.red_s()[j] -= eps;
                mesh.red_w()[i] -= eps;

                mesh.red_w()[i] += eps;
                mesh.setGlobalF(true, true, false);
                double Ei = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
                mesh.red_w()[i] -= eps;

                mesh.red_s()[j] += eps;
                mesh.setGlobalF(true, true, false);
                double Ej = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
                mesh.red_s()[j] -= eps;

                fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
            }
        }
        mesh.setGlobalF(true, true, false);
        return fake;
    }
    //-----------------------

    //CHECK Ers with dEds for energy--------------
    MatrixXd Ers_part1(Mesh& mesh, Arap& arap, double E0, double eps){
        MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_s().size());

        for(int i=0; i<fake.rows(); i++){
                mesh.red_w()[i] += 0.5*eps;
                mesh.setGlobalF(true, false, false);
                VectorXd Eleft = arap.dEds(mesh);
                mesh.red_w()[i] -= 0.5*eps;

                mesh.red_w()[i] -= 0.5*eps;
                mesh.setGlobalF(true, false, false);
                VectorXd Eright = arap.dEds(mesh);
                mesh.red_w()[i] += 0.5*eps;

                fake.row(i) = (Eleft-Eright)/eps;
        }
        mesh.setGlobalF(true, false, false);
        return fake;
    }
    //-----------------------

    //CHECK dgds-----------------------
    MatrixXd Jac_dgds(Mesh& mesh, Arap& arap, double E0, double eps){
        // arap.Jacobians(mesh);
        
        MatrixXd dgds = MatrixXd::Zero(mesh.red_x().size(), mesh.red_s().size());
        VectorXd z0 = mesh.red_x();
        VectorXd& z = mesh.red_x();

        for(int i=0; i<mesh.red_s().size(); i++){
            mesh.red_x(z0);

            mesh.red_s()[i] += 0.5*eps;
            mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);

            VectorXd dgds_left = mesh.red_x() + VectorXd::Zero(z.size());

            mesh.red_s()[i] -= 0.5*eps;
            mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);


            mesh.red_s()[i] -= 0.5*eps;
            mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);
            
            VectorXd dgds_right = mesh.red_x() + VectorXd::Zero(z.size());

            mesh.red_s()[i] += 0.5*eps;
            mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);

            dgds.col(i) = (dgds_left - dgds_right)/eps;
        }
        std::cout<<"dgds"<<std::endl;
        // std::cout<<dgds<<std::endl;
        return dgds;
    }
    //---------------------------------

    //CHECK drds-----------------------
    MatrixXd Jac_drds(Mesh& mesh, Arap& arap, double E0, double eps){
        // VectorXd dEds = arap.Jacobians(mesh);

        MatrixXd drds = MatrixXd::Zero(mesh.red_w().size(), mesh.red_s().size());
        VectorXd vecR0, vecR;
        for(int i=0; i<mesh.red_s().size(); i++){

            mesh.red_s()[i] += 0.5*eps;
            mesh.setGlobalF(false, true, false);
            vecR0 = mesh.red_r();
            arap.minimize(mesh);
            vecR = mesh.red_r();

            VectorXd drds_left = get_w(vecR0, vecR);

            mesh.red_s()[i] -= 0.5*eps;
            mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);


            mesh.red_s()[i] -= 0.5*eps;
            mesh.setGlobalF(false, true, false);
            vecR0 = mesh.red_r();
            arap.minimize(mesh);
            vecR = mesh.red_r();
            
            VectorXd drds_right = get_w(vecR0, vecR);

            mesh.red_s()[i] += 0.5*eps;
            mesh.setGlobalF(false, true, false);
            arap.minimize(mesh);

            drds.col(i) = (drds_left - drds_right)/eps;
        }
        std::cout<<"drds"<<std::endl;
        // std::cout<<drds<<std::endl;
        return drds;
    }
    //---------------------------------

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
    
    double value(const TVector &x) {
        for(int i=0; i<x.size(); i++){
            mesh->red_s()[i] = x[i];
        }
        mesh->setGlobalF(false, true, false);

        arap->minimize(*mesh);
        double Eneo = 0;// alpha_neo*elas->Energy(*mesh);
        double Earap = alpha_arap*arap->Energy(*mesh);
                std::cout<<"Energy"<<std::endl;
                std::cout<<"neo: "<<alpha_neo*Eneo<<", arap: "<<alpha_arap*Earap<<std::endl;
                // std::cout<<mesh->red_s().transpose()<<std::endl;
        return Eneo + Earap;
    }
    void gradient(const TVector &x, TVector &grad) {
        for(int i=0; i<x.size(); i++){
            mesh->red_s()[i] = x[i];
        }
        mesh->setGlobalF(false, true, false);
        arap->minimize(*mesh);

        // // VectorXd pegrad = elas->PEGradient(*mesh);
        VectorXd arapgrad = arap->Jacobians(*mesh);
        double eps = 1e-4;
        double E0 = arap->Energy(*mesh);

        VectorXd full_fake = Full_FD_Grad(*mesh, *arap, E0, eps);
        std::cout<<"FD GRAD"<<std::endl;
        std::cout<<(full_fake - arapgrad).norm()<<std::endl;
        
        VectorXd dEdr = Er(*mesh,*arap,E0, eps);
        VectorXd dEdx = Ex(*mesh,*arap,E0, eps);
        VectorXd dEds = Es(*mesh,*arap,E0, eps);
        MatrixXd dEdxx = Exx(*mesh, *arap, E0, eps);
        MatrixXd dEdxr = Exr(*mesh, *arap, E0, eps);
        MatrixXd dEdxs = Exs(*mesh, *arap, E0, eps);
        MatrixXd dEdrr = Err(*mesh, *arap, E0, eps);
        MatrixXd dEdrs = Ers_part1(*mesh, *arap, E0, eps);
        MatrixXd lhs_left(dEdxx.rows()+dEdxr.cols(), dEdxx.cols());
        lhs_left<<dEdxx, dEdxr.transpose();
        MatrixXd lhs_right(dEdxr.rows() + dEdrr.rows() , dEdxr.cols());
        lhs_right<<dEdxr, dEdrr; 
        MatrixXd rhs(dEdxs.rows()+dEdrs.rows(), dEdxs.cols());
        rhs<<-1*dEdxs, -1*dEdrs;
        MatrixXd CG = MatrixXd(mesh->AB().transpose())*mesh->G();
        MatrixXd col1(lhs_left.rows()+CG.rows(), lhs_left.cols());
        col1<<lhs_left, CG;
        MatrixXd col2(lhs_right.rows()+CG.rows(), lhs_right.cols());
        col2<<lhs_right,MatrixXd::Zero(CG.rows(), lhs_right.cols());
        MatrixXd col3(CG.cols()+CG.rows()+dEdrr.rows(), CG.rows());
        col3<<CG.transpose(),MatrixXd::Zero(CG.rows()+dEdrr.rows(), CG.rows());
        MatrixXd KKT_constrains(rhs.rows() + CG.rows(), rhs.cols());
        KKT_constrains<<rhs,MatrixXd::Zero(CG.rows(), rhs.cols());
        MatrixXd JacKKT(col1.rows(), col1.rows());
        JacKKT<<col1, col2, col3;
        MatrixXd results = JacKKT.fullPivLu().solve(KKT_constrains).topRows(rhs.rows());
        MatrixXd dgds = results.topRows(dEdxx.rows());
        MatrixXd drds = results.bottomRows(dEdrr.rows());
        VectorXd fake =  dgds.transpose()*dEdx + drds.transpose()*dEdr + dEds;
        // std::cout<<"DEDs"<<std::endl;
        // std::cout<<(fake - arapgrad).norm()<<std::endl;

        // std::cout<<"Energy: "<<E0<<" - "<<arap->Energy(*mesh)<<std::endl;
        // std::cout<<"Hessians"<<std::endl;
        // std::cout<<"Exx "<<(dEdxx - arap->Exx()).norm()<<std::endl;
        // std::cout<<"Exr "<<(dEdxr - arap->Exr()).norm()<<std::endl;
        // std::cout<<"Exs "<<(dEdxs - arap->Exs()).norm()<<std::endl;
        // std::cout<<"Ers "<<(dEdrs - arap->Ers()).norm()<<std::endl;
        // std::cout<<"Err "<<(dEdrr - arap->Err()).norm()<<std::endl;
        // std::cout<<"Gradients"<<std::endl;
        // std::cout<<"Er "<<(dEdr - arap->Er()).norm()<<std::endl;
        // std::cout<<"Es "<<(dEds - arap->Es()).norm()<<std::endl;
        // std::cout<<"check deds"<<std::endl;
        // std::cout<<dEds.transpose()<<std::endl;
        // std::cout<<arap->Es().transpose()<<std::endl;
        // std::cout<<"check deds"<<std::endl;
        // std::cout<<(dEdx - arap->Ex()).norm()<<std::endl<<std::endl;


        if((full_fake - arapgrad).norm()>0.1){
            std::cout<<full_fake.transpose()<<std::endl<<std::endl;
            std::cout<<arapgrad.transpose()<<std::endl<<std::endl;
            std::cout<<fake.transpose()<<std::endl<<std::endl;

            
        //     // std::cout<<fake.transpose()<<std::endl<<std::endl;
        //     // std::cout<<arapgrad.transpose()<<std::endl<<std::endl;
            std::cout<<"reds"<<std::endl;
            std::cout<<mesh->red_s().transpose()<<std::endl;
            std::cout<<"redx"<<std::endl;
            std::cout<<mesh->red_x().transpose()<<std::endl;
            std::cout<<"redr"<<std::endl;
            std::cout<<mesh->red_r().transpose()<<std::endl;
            std::cout<<"redw"<<std::endl;
            std::cout<<mesh->red_w().transpose()<<std::endl;
            exit(0);
        }
        for(int i=0; i< x.size(); i++){
            // grad[i] = alpha_neo*pegrad[i];
            grad[i] = alpha_arap*full_fake[i];
        }

        // exit(0);

    }

    bool callback(const Criteria<T> &state, const TVector &x) {
        // std::cout << "(" << std::setw(2) << state.iterations << ")"
        //           << " ||dx|| = " << std::fixed << std::setw(8) << std::setprecision(4) << state.gradNorm
        //           << " ||x|| = "  << std::setw(6) << x.norm()
        //           << " f(x) = "   << std::setw(8) << value(x)
        //           << " x = [" << std::setprecision(8) << x.transpose() << "]" << std::endl;
        return true;
    }
};

#endif