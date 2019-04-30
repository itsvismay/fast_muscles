#include "stablenh_energy_gradient.h"
#include "store.h"
using namespace Eigen;
using Store = famu::Store;

double famu::stablenh::energy(Store& store, Eigen::VectorXd& dFvec){
	double stableNHEnergy = 0;


	for(int t=0; t<dFvec.size()/9; t++){
		double youngsModulus = store.eY[t];
		double poissonsRatio = store.eP[t];
		double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
		double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
		Matrix3d F = Map<Matrix3d>(dFvec.segment<9>(9*t).data()).transpose();
		double I1 = (F.transpose()*F).trace();
		double J = F.determinant();
		double alpha = (1 + (C1/D1) - (C1/(D1*4)));
		double W = 0.5*C1*(I1 -3) + 0.5*D1*(J-alpha)*(J-alpha) - 0.5*C1*log(I1 + 1);
		stableNHEnergy += W;
	}

	return stableNHEnergy;
}

void famu::stablenh::gradient(Store& store, Eigen::VectorXd& grad){

	for(int t=0; t<store.T.rows(); t++){
		double youngsModulus = store.eY[t];
		double poissonsRatio = store.eP[t];
		double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
		double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
		
		double s1 = store.dFvec[9*t+0];
		double s2 = store.dFvec[9*t+1];
		double s3 = store.dFvec[9*t+2];
		double s4 = store.dFvec[9*t+3];
		double s5 = store.dFvec[9*t+4];
		double s6 = store.dFvec[9*t+5];
		double s7 = store.dFvec[9*t+6];
		double s8 = store.dFvec[9*t+7];
		double s9 = store.dFvec[9*t+8];

		VectorXd tet_grad(9);
		tet_grad[0] = 1.*C1*s1 - (1.*C1*s1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2)) + 
    		1.*(s6*s8 - 1.*s5*s9)*(0.75*C1 + D1*(1 + s3*s5*s7 - 1.*s2*s6*s7 - 1.*s3*s4*s8 + s1*s6*s8 + s2*s4*s9 - 1.*s1*s5*s9));
		tet_grad[1] = 1.*C1*s2 + 1.*D1*(s6*s7 - s4*s9)*(-1 - (0.75*C1)/D1 - s3*s5*s7 + s2*s6*s7 + s3*s4*s8 - s1*s6*s8 - s2*s4*s9 + s1*s5*s9) - 
    		(1.*C1*s2)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));
		tet_grad[2] = 1.*C1*s3 - (1.*C1*s3)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2)) + 
    		1.*(s5*s7 - 1.*s4*s8)*(0.75*C1 + D1*(1 + s3*s5*s7 - 1.*s2*s6*s7 - 1.*s3*s4*s8 + s1*s6*s8 + s2*s4*s9 - 1.*s1*s5*s9));
		tet_grad[3] = 1.*C1*s4 + 1.*D1*(s3*s8 - s2*s9)*(-1 - (0.75*C1)/D1 - s3*s5*s7 + s2*s6*s7 + s3*s4*s8 - s1*s6*s8 - s2*s4*s9 + s1*s5*s9) - 
    		(1.*C1*s4)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));
		tet_grad[4] = 1.*C1*s5 - (1.*C1*s5)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2)) + 
    		1.*(s3*s7 - 1.*s1*s9)*(0.75*C1 + D1*(1 + s3*s5*s7 - 1.*s2*s6*s7 - 1.*s3*s4*s8 + s1*s6*s8 + s2*s4*s9 - 1.*s1*s5*s9));
		tet_grad[5] = 1.*C1*s6 + 1.*D1*(s2*s7 - s1*s8)*(-1 - (0.75*C1)/D1 - s3*s5*s7 + s2*s6*s7 + s3*s4*s8 - s1*s6*s8 - s2*s4*s9 + s1*s5*s9) - 
    		(1.*C1*s6)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));
		tet_grad[6] = 1.*C1*s7 - (1.*C1*s7)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2)) + 
    		1.*(s3*s5 - 1.*s2*s6)*(0.75*C1 + D1*(1 + s3*s5*s7 - 1.*s2*s6*s7 - 1.*s3*s4*s8 + s1*s6*s8 + s2*s4*s9 - 1.*s1*s5*s9));
		tet_grad[7] = 1.*C1*s8 + 1.*D1*(s3*s4 - s1*s6)*(-1 - (0.75*C1)/D1 - s3*s5*s7 + s2*s6*s7 + s3*s4*s8 - s1*s6*s8 - s2*s4*s9 + s1*s5*s9) - 
    		(1.*C1*s8)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));
		tet_grad[8] = 1.*C1*s9 - (1.*C1*s9)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2)) + 
    		1.*(s2*s4 - 1.*s1*s5)*(0.75*C1 + D1*(1 + s3*s5*s7 - 1.*s2*s6*s7 - 1.*s3*s4*s8 + s1*s6*s8 + s2*s4*s9 - 1.*s1*s5*s9));		
   
		grad.segment<9>(9*t) += tet_grad;
	}
}

VectorXd famu::stablenh::fd_gradient(Store& store){
	Eigen::VectorXd fake = Eigen::VectorXd::Zero(store.dFvec.size());
	double eps = 0.00001;
	for(int i=0; i<store.dFvec.size(); i++){
		store.dFvec[i] += 0.5*eps;
		// setDF(store.dFvec, store.dF);
		double Eleft = energy(store, store.dFvec);
		store.dFvec[i] -= 0.5*eps;

		store.dFvec[i] -= 0.5*eps;
		// setDF(store.dFvec, store.dF);
		double Eright = energy(store, store.dFvec);
		store.dFvec[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	return fake;
}