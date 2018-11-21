#ifndef SOLVER
#define SOLVER

#include <iostream>
#include "mesh.h"
#include "arap.h"
#include "elastic.h"
#include "cppoptlib/meta.h"
#include "cppoptlib/boundedproblem.h"
#include "cppoptlib/solver/lbfgsbsolver.h"

using namespace cppoptlib;
using Eigen::VectorXd;
template<typename T>
class StaticSolve : public BoundedProblem<T> {
private:
    Mesh* mesh;
    Arap* arap;
    Elastic* elas;
    double alpha_neo =1e4;
    double alpha_arap = 1e1; 

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
            mesh->s()[i] = x[i];
        }

        arap->minimize(*mesh);
        double Eneo = elas->Energy(*mesh);
        double Earap = arap->Energy(*mesh);
        std::cout<<"Energy"<<std::endl;
        std::cout<<Eneo<<", "<<Earap<<std::endl;
        // std::cout<<x.transpose()<<std::endl;
        return alpha_neo*Eneo + alpha_arap*Earap;
    }
    void gradient(const TVector &x, TVector &grad) {
        std::cout<<"Grad"<<std::endl;
        for(int i=0; i<x.size(); i++){
            mesh->s()[i] = x[i];
        }

        VectorXd pegrad = elas->PEGradient(*mesh);
        VectorXd arapgrad = arap->FDGrad(*mesh);

        for(int i=0; i< pegrad.size(); i++){
            grad[i] = alpha_neo*pegrad[i] + alpha_arap*arapgrad[i];
        }
    }
};

#endif