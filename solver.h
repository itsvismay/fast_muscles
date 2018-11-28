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
    double alpha_arap = 1e2; 

public:
    using typename cppoptlib::BoundedProblem<T>::Scalar;
    using typename cppoptlib::BoundedProblem<T>::TVector;

    StaticSolve(int dim, Mesh* m, Arap* a, Elastic* e): BoundedProblem<T>(dim) {
        mesh = m;
        arap = a;
        elas = e;

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
                std::cout<<mesh->s().transpose()<<std::endl;
        return Eneo + Earap;
    }
    // void gradient(const TVector &x, TVector &grad) {
    //     std::cout<<"Grad"<<std::endl;
    //     for(int i=0; i<x.size(); i++){
    //         mesh->s()[i] = x[i];
    //     }
    //     mesh->setGlobalF(false, true, false);

    //     // VectorXd pegrad = elas->PEGradient(*mesh);
    //     VectorXd arapgrad = arap->FDGrad(*mesh);

    //     for(int i=0; i< x.size(); i++){
    //         // grad[i] = alpha_neo*pegrad[i];
    //         grad[i] = alpha_arap*arapgrad[i];
    //     }

    // }

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