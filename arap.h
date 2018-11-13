#ifndef ARAP
#define ARAP

#include "mesh.h"


using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;

class Arap
{

protected:

public:
	Arap(Mesh& m){ 

	}

	inline double Energy(Mesh& m){
		VectorXd PAx = m.P()*m.A()*m.x();
		VectorXd FPAx0 = m.xbar();
		return 0.5*PAx.transpose()*FPAx0;
	}

	VectorXd dEdx(Mesh& m){
		VectorXd PAx = m.P()*m.A()*m.x();
		VectorXd FPAx0 = m.xbar();
		VectorXd res = (m.P()*m.A()).transpose()*(PAx - FPAx0);
		return res;
	}

	void itT(Mesh& m){

	}

	void itR(Mesh& m){

	}

	void minimize(Mesh& m){
		print(" + ARAP minimize");
		
		VectorXd Ex0 = dEdx(m);
		for(int i=0; i< 100; i++){
			itT(m);
			itR(m);
			m.setGlobalF(true, false, false);
			
			VectorXd Ex = dEdx(m);
			if ((Ex - Ex0).norm()){
				print(" - ARAP minimize");
				return;
			}
			Ex0 = Ex;
		}
	}



	template<class T>
    inline void print(T a){ std::cout<<a<<std::endl; }

};

#endif