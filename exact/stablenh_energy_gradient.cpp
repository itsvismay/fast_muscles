#include "stablenh_energy_gradient.h"
#include "omp.h"
#include <iostream>
using namespace Eigen;
typedef Eigen::Triplet<double> Trip;


double exact::stablenh::energy(Eigen::VectorXd& Fvec,
								const Eigen::MatrixXi& T,
								const Eigen::VectorXd& eY,
								const Eigen::VectorXd& eP,
								const Eigen::VectorXd& rest_tet_vols){
	double stableNHEnergy = 0;

	#pragma omp parallel
	{	
		double sNHpriv = 0;

		#pragma omp for 
		for(int i=0; i<T.rows(); i++){
			int t = i;
			double youngsModulus = eY[t];
			double poissonsRatio = eP[t];
			double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
			double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
			
			int f_index = t;

			Matrix3d F = Map<Matrix3d>(Fvec.segment<9>(9*f_index).data()).transpose();
			double I1 = (F.transpose()*F).trace();
			double J = F.determinant();
			double alpha = (1 + (C1/D1) - (C1/(D1*4)));
			double W = 0.5*C1*(I1 -3) + 0.5*D1*(J-alpha)*(J-alpha) - 0.5*C1*log(I1 + 1);
			sNHpriv += W*rest_tet_vols[t];
		}

		#pragma omp critical
		{
			stableNHEnergy += sNHpriv;
		}
	} 
	

	return stableNHEnergy;
}

void exact::stablenh::gradient( Eigen::VectorXd& grad, 
								const Eigen::VectorXd& Fvec,
								const Eigen::MatrixXi& T,
								const Eigen::VectorXd& eY,
								const Eigen::VectorXd& eP,
								const Eigen::VectorXd& rest_tet_vols){
	grad.setZero();

	#pragma omp parallel for
	for(int i=0; i<T.rows(); i++){
		int t = i;
		double youngsModulus = eY[t];
		double poissonsRatio = eP[t];
		double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
		double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
		
		int f_index = t;

		double s1 = Fvec[9*f_index + 0];
		double s2 = Fvec[9*f_index + 1];
		double s3 = Fvec[9*f_index + 2];
		double s4 = Fvec[9*f_index + 3];
		double s5 = Fvec[9*f_index + 4];
		double s6 = Fvec[9*f_index + 5];
		double s7 = Fvec[9*f_index + 6];
		double s8 = Fvec[9*f_index + 7];
		double s9 = Fvec[9*f_index + 8];

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
   
		grad.segment<9>(9*f_index)  += rest_tet_vols[t]*tet_grad;
	}
}
void exact::stablenh::hessian( Eigen::SparseMatrix<double, Eigen::RowMajor>& hess, 
								const Eigen::VectorXd& Fvec,
								const Eigen::MatrixXi& T,
								const Eigen::VectorXd& eY,
								const Eigen::VectorXd& eP,
								const Eigen::VectorXd& rest_tet_vols){
		hess.setZero();
		std::vector<Trip> hess_trips(9*9*T.rows());
		std::mutex door;
		int idx = 0;
		#pragma omp parallel for shared(idx)
		for(int t=0; t<T.rows(); t++){
			double youngsModulus = eY[t];
			double poissonsRatio = eP[t];
			double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
			double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
			
			int f_index = t;

			double s1 = Fvec[9*f_index + 0];
			double s2 = Fvec[9*f_index + 1];
			double s3 = Fvec[9*f_index + 2];
			double s4 = Fvec[9*f_index + 3];
			double s5 = Fvec[9*f_index + 4];
			double s6 = Fvec[9*f_index + 5];
			double s7 = Fvec[9*f_index + 6];
			double s8 = Fvec[9*f_index + 7];
			double s9 = Fvec[9*f_index + 8];

			Eigen::Matrix<double, 9,9> ddw(9,9);
				ddw(0,0) = 1.*C1 + 1.*D1*std::pow(s6*s8 - 1.*s5*s9,2) + (2.*C1*std::pow(s1,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(1,0) = -1.*D1*(s6*s7 - 1.*s4*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s2)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,0) = 1.*D1*(s5*s7 - 1.*s4*s8)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,0) = -1.*D1*(s3*s8 - 1.*s2*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,0) = -0.75*C1*s9 + D1*s9*(-1. + 1.*s2*s6*s7 - 2.*s1*s6*s8 - 1.*s2*s4*s9 + 2.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 - 2.*s5*s7*s9 + 1.*s4*s8*s9) + (2.*C1*s1*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,0) = D1*s8*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s2*s5*s7 + 1.*s2*s4*s8 - 2.*s1*s5*s8)*s9 + C1*(0.75*s8 + (2.*s1*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,0) = 1.*D1*(s3*s5 - 1.*s2*s6)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,0) = D1*s6*(1. + 1.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s3*s4*s5 + 1.*s2*s4*s6 - 2.*s1*s5*s6)*s9 + C1*(0.75*s6 + (2.*s1*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,0) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(-1.*s3*s7 + 2.*s1*s9) + s5*(-1. + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 - 2.*s2*s4*s9)) + C1*(-0.75*s5 + (2.*s1*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,1) = 1.*D1*(-1.*s6*s7 + s4*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s2)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,1) = 1.*C1 + 1.*D1*std::pow(s6*s7 - 1.*s4*s9,2) + (2.*C1*std::pow(s2,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(2,1) = 1.*D1*(s5*s7 - 1.*s4*s8)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,1) = 0.75*C1*s9 + D1*s9*(1. - 2.*s2*s6*s7 + 1.*s1*s6*s8 + 2.*s2*s4*s9 - 1.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 + 1.*s5*s7*s9 - 2.*s4*s8*s9) + (2.*C1*s2*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,1) = 1.*D1*(s3*s7 - 1.*s1*s9)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,1) = D1*s7*(-1. - 1.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s4*s7 + 1.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(-0.75*s7 + (2.*s2*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,1) = D1*s6*(-1. - 2.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s3*s4*s5 - 2.*s2*s4*s6 + 1.*s1*s5*s6)*s9 + C1*(-0.75*s6 + (2.*s2*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,1) = -1.*D1*(s3*s4 - 1.*s1*s6)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,1) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(-1.*s3*s8 + 2.*s2*s9) + s4*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s4 + (2.*s2*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,2) = 1.*D1*(s5*s7 - 1.*s4*s8)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,2) = -1.*D1*(s5*s7 - 1.*s4*s8)*(s6*s7 - 1.*s4*s9) + (2.*C1*s2*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,2) = 1.*C1 + 1.*D1*std::pow(s5*s7 - 1.*s4*s8,2) + (2.*C1*std::pow(s3,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(3,2) = D1*s8*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s2*s5*s7 - 2.*s2*s4*s8 + 1.*s1*s5*s8)*s9 + C1*(-0.75*s8 + (2.*s3*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(4,2) = D1*s7*(1. + 2.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s4*s7 - 2.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(0.75*s7 + (2.*s3*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(5,2) = -1.*D1*(s2*s7 - 1.*s1*s8)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,2) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(2.*s3*s7 - 1.*s1*s9) + s5*(1. - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 + 1.*s2*s4*s9)) + C1*(0.75*s5 + (2.*s3*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,2) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(2.*s3*s8 - 1.*s2*s9) + s4*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s4 + (2.*s3*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,2) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,3) = 1.*D1*(-1.*s3*s8 + s2*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,3) = 0.75*C1*s9 + D1*s9*(1. - 2.*s2*s6*s7 + 1.*s1*s6*s8 + 2.*s2*s4*s9 - 1.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 + 1.*s5*s7*s9 - 2.*s4*s8*s9) + (2.*C1*s2*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,3) = D1*s8*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s2*s5*s7 - 2.*s2*s4*s8 + 1.*s1*s5*s8)*s9 + C1*(-0.75*s8 + (2.*s3*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,3) = 1.*C1 + 1.*D1*std::pow(s3*s8 - 1.*s2*s9,2) + (2.*C1*std::pow(s4,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(4,3) = 1.*D1*(s3*s7 - 1.*s1*s9)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,3) = -1.*D1*(s2*s7 - 1.*s1*s8)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,3) = 1.*D1*(s3*s5 - 1.*s2*s6)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,3) = D1*s3*(-1. - 1.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s3*s4 + 1.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(-0.75*s3 + (2.*s4*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,3) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(-1.*s6*s7 + 2.*s4*s9) + s2*(1. + 1.*s3*s5*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s2 + (2.*s4*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,4) = -0.75*C1*s9 + D1*s9*(-1. + 1.*s2*s6*s7 - 2.*s1*s6*s8 - 1.*s2*s4*s9 + 2.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 - 2.*s5*s7*s9 + 1.*s4*s8*s9) + (2.*C1*s1*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,4) = -1.*D1*(s3*s7 - 1.*s1*s9)*(s6*s7 - 1.*s4*s9) + (2.*C1*s2*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,4) = D1*s7*(1. + 2.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s4*s7 - 2.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(0.75*s7 + (2.*s3*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,4) = -1.*D1*(s3*s7 - 1.*s1*s9)*(s3*s8 - 1.*s2*s9) + (2.*C1*s4*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,4) = 1.*C1 + 1.*D1*std::pow(s3*s7 - 1.*s1*s9,2) + (2.*C1*std::pow(s5,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(5,4) = -1.*D1*(s2*s7 - 1.*s1*s8)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,4) = D1*s3*(1. + 2.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s3*s4 - 2.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(0.75*s3 + (2.*s5*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,4) = -1.*D1*(s3*s4 - 1.*s1*s6)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,4) = 1.*D1*s2*s3*s4*s7 + D1*s1*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8 - 2.*s2*s4*s9 + 2.*s1*s5*s9) + C1*(-0.75*s1 + (2.*s5*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,5) = D1*s8*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s2*s5*s7 + 1.*s2*s4*s8 - 2.*s1*s5*s8)*s9 + C1*(0.75*s8 + (2.*s1*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(1,5) = D1*s7*(-1. - 1.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s4*s7 + 1.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(-0.75*s7 + (2.*s2*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(2,5) = 1.*D1*(-1.*s2*s7 + s1*s8)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,5) = -1.*D1*(s2*s7 - 1.*s1*s8)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,5) = 1.*D1*(-1.*s2*s7 + s1*s8)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,5) = 1.*C1 + 1.*D1*std::pow(s2*s7 - 1.*s1*s8,2) + (2.*C1*std::pow(s6,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(6,5) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(2.*s6*s7 - 1.*s4*s9) + s2*(-1. - 2.*s3*s5*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s2 + (2.*s6*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,5) = 1.*D1*s2*s3*s4*s7 + D1*s1*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8 + 1.*s2*s4*s9 - 1.*s1*s5*s9) + C1*(0.75*s1 + (2.*s6*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,5) = 1.*D1*(s2*s4 - 1.*s1*s5)*(-1.*s2*s7 + s1*s8) + (2.*C1*s6*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,6) = 1.*D1*(s3*s5 - 1.*s2*s6)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,6) = D1*s6*(-1. - 2.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s3*s4*s5 - 2.*s2*s4*s6 + 1.*s1*s5*s6)*s9 + C1*(-0.75*s6 + (2.*s2*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(2,6) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(2.*s3*s7 - 1.*s1*s9) + s5*(1. - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 + 1.*s2*s4*s9)) + C1*(0.75*s5 + (2.*s3*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,6) = -1.*D1*(s3*s5 - 1.*s2*s6)*(s3*s8 - 1.*s2*s9) + (2.*C1*s4*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,6) = D1*s3*(1. + 2.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s3*s4 - 2.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(0.75*s3 + (2.*s5*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(5,6) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(2.*s6*s7 - 1.*s4*s9) + s2*(-1. - 2.*s3*s5*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s2 + (2.*s6*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,6) = 1.*C1 + 1.*D1*std::pow(s3*s5 - 1.*s2*s6,2) + (2.*C1*std::pow(s7,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(7,6) = -1.*D1*(s3*s4 - 1.*s1*s6)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,6) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,7) = D1*s6*(1. + 1.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s3*s4*s5 + 1.*s2*s4*s6 - 2.*s1*s5*s6)*s9 + C1*(0.75*s6 + (2.*s1*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(1,7) = -1.*D1*(s3*s4 - 1.*s1*s6)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,7) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(2.*s3*s8 - 1.*s2*s9) + s4*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s4 + (2.*s3*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,7) = D1*s3*(-1. - 1.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s3*s4 + 1.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(-0.75*s3 + (2.*s4*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(4,7) = 1.*D1*(-1.*s3*s4 + s1*s6)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,7) = 1.*D1*s2*s3*s4*s7 + D1*s1*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8 + 1.*s2*s4*s9 - 1.*s1*s5*s9) + C1*(0.75*s1 + (2.*s6*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,7) = 1.*D1*(-1.*s3*s4 + s1*s6)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,7) = 1.*C1 + 1.*D1*std::pow(s3*s4 - 1.*s1*s6,2) + (2.*C1*std::pow(s8,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(8,7) = 1.*D1*(s2*s4 - 1.*s1*s5)*(-1.*s3*s4 + s1*s6) + (2.*C1*s8*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,8) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(-1.*s3*s7 + 2.*s1*s9) + s5*(-1. + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 - 2.*s2*s4*s9)) + C1*(-0.75*s5 + (2.*s1*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(1,8) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(-1.*s3*s8 + 2.*s2*s9) + s4*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s4 + (2.*s2*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(2,8) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,8) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(-1.*s6*s7 + 2.*s4*s9) + s2*(1. + 1.*s3*s5*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s2 + (2.*s4*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(4,8) = 1.*D1*s2*s3*s4*s7 + D1*s1*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8 - 2.*s2*s4*s9 + 2.*s1*s5*s9) + C1*(-0.75*s1 + (2.*s5*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(5,8) = -1.*D1*(s2*s4 - 1.*s1*s5)*(s2*s7 - 1.*s1*s8) + (2.*C1*s6*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,8) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,8) = -1.*D1*(s2*s4 - 1.*s1*s5)*(s3*s4 - 1.*s1*s6) + (2.*C1*s8*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,8) = 1.*C1 + 1.*D1*std::pow(s2*s4 - 1.*s1*s5,2) + (2.*C1*std::pow(s9,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

			//FIX SVD
			Eigen::SelfAdjointEigenSolver<Matrix<double, 9, 9>> es(ddw);
			Eigen::Matrix<double, 9, 9> DiagEval = es.eigenvalues().real().asDiagonal();
	        Eigen::Matrix<double, 9, 9> Evec = es.eigenvectors().real();
	        
	        for (int i = 0; i < 9; ++i) {
	            if (es.eigenvalues()[i]<1e-6) {
	                DiagEval(i,i) = 1e-3;
	            }
	        }

	        ddw = Evec * DiagEval * Evec.transpose();

	        {
		        std::lock_guard<std::mutex> lg(door);
				for(int i=0; i<ddw.rows(); i++){
					for(int j=0; j<ddw.cols(); j++){
						hess_trips[idx] = Trip(9*f_index + i, 9*f_index + j, rest_tet_vols[t]*ddw(i,j));
						idx += 1;
					}
				}
	        }
		}
		hess.setFromTriplets(hess_trips.begin(), hess_trips.end());	
}

double exact::stablenh::energy(Eigen::VectorXd& Fvec, 
							const Eigen::VectorXi& bone_or_muscle, 
							const Eigen::MatrixXi& T,
							const Eigen::VectorXd& eY,
							const Eigen::VectorXd& eP,
							const Eigen::VectorXd& rest_tet_vols){
	double stableNHEnergy = 0;

	#pragma omp parallel
	{	
		double sNHpriv = 0;

		#pragma omp for 
		for(int i=0; i<T.rows(); i++){
			int t = i;
			double youngsModulus = eY[t];
			double poissonsRatio = eP[t];
			double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
			double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
			
			int f_index = bone_or_muscle[t];

			Matrix3d F = Map<Matrix3d>(Fvec.segment<9>(9*f_index).data()).transpose();
			double I1 = (F.transpose()*F).trace();
			double J = F.determinant();
			double alpha = (1 + (C1/D1) - (C1/(D1*4)));
			double W = 0.5*C1*(I1 -3) + 0.5*D1*(J-alpha)*(J-alpha) - 0.5*C1*log(I1 + 1);
			sNHpriv += W*rest_tet_vols[t];
		}

		#pragma omp critical
		{
			stableNHEnergy += sNHpriv;
		}
	} 
	

	return stableNHEnergy;
}

void exact::stablenh::gradient( Eigen::VectorXd& grad, 
								const Eigen::VectorXd& Fvec, 
								const Eigen::VectorXi& bone_or_muscle,
								const Eigen::MatrixXi& T,
								const Eigen::VectorXd& eY,
								const Eigen::VectorXd& eP,
								const Eigen::VectorXd& rest_tet_vols,
								const std::vector<double>& bone_vols,
								const std::vector<Eigen::VectorXi>& muscle_tets,
								const std::vector<Eigen::VectorXi>& bone_tets){
	grad.setZero();

	for(int m=0; m<muscle_tets.size(); m++){
		#pragma omp parallel for
		for(int i=0; i<muscle_tets[m].size(); i++){
			int t = muscle_tets[m][i];
			double youngsModulus = eY[t];
			double poissonsRatio = eP[t];
			double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
			double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
			
			int f_index = bone_or_muscle[t];

			double s1 = Fvec[9*f_index + 0];
			double s2 = Fvec[9*f_index + 1];
			double s3 = Fvec[9*f_index + 2];
			double s4 = Fvec[9*f_index + 3];
			double s5 = Fvec[9*f_index + 4];
			double s6 = Fvec[9*f_index + 5];
			double s7 = Fvec[9*f_index + 6];
			double s8 = Fvec[9*f_index + 7];
			double s9 = Fvec[9*f_index + 8];

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
	   
			grad.segment<9>(9*f_index)  += rest_tet_vols[t]*tet_grad;
		}
	}

	for(int m=0; m<bone_tets.size(); m++){
		double s1 = Fvec[9*m + 0];
		double s2 = Fvec[9*m + 1];
		double s3 = Fvec[9*m + 2];
		double s4 = Fvec[9*m + 3];
		double s5 = Fvec[9*m + 4];
		double s6 = Fvec[9*m + 5];
		double s7 = Fvec[9*m + 6];
		double s8 = Fvec[9*m + 7];
		double s9 = Fvec[9*m + 8];
		int t = bone_tets[m][0];
		double youngsModulus = eY[t];
		double poissonsRatio = eP[t];

		double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
		double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
		
		// int f_index = bone_or_muscle[t];


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

		
		grad.segment<9>(9*m) +=  bone_vols[m]*tet_grad;

	}
}

void exact::stablenh::hessian(Eigen::SparseMatrix<double, Eigen::RowMajor>& hess, 
								const Eigen::VectorXd& Fvec, 
								const Eigen::VectorXi& bone_or_muscle,
								const Eigen::MatrixXi& T,
								const Eigen::VectorXd& eY,
								const Eigen::VectorXd& eP,
								const Eigen::VectorXd& rest_tet_vols,
								const std::vector<double>& bone_vols,
								const std::vector<Eigen::VectorXi>& muscle_tets,
								const std::vector<Eigen::VectorXi>& bone_tets){
	hess.setZero();
	std::vector<Trip> hess_trips;
	hess_trips.reserve(9*9*T.rows());
	for(int m=0; m<muscle_tets.size(); m++){
		#pragma omp parallel for
		for(int i=0; i<muscle_tets[m].size(); i++){
			int t = muscle_tets[m][i];
			double youngsModulus = eY[t];
			double poissonsRatio = eP[t];
			double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
			double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
			
			int f_index = bone_or_muscle[t];

			double s1 = Fvec[9*f_index + 0];
			double s2 = Fvec[9*f_index + 1];
			double s3 = Fvec[9*f_index + 2];
			double s4 = Fvec[9*f_index + 3];
			double s5 = Fvec[9*f_index + 4];
			double s6 = Fvec[9*f_index + 5];
			double s7 = Fvec[9*f_index + 6];
			double s8 = Fvec[9*f_index + 7];
			double s9 = Fvec[9*f_index + 8];

			Eigen::Matrix<double, 9,9> ddw(9,9);
				ddw(0,0) = 1.*C1 + 1.*D1*std::pow(s6*s8 - 1.*s5*s9,2) + (2.*C1*std::pow(s1,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(1,0) = -1.*D1*(s6*s7 - 1.*s4*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s2)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,0) = 1.*D1*(s5*s7 - 1.*s4*s8)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,0) = -1.*D1*(s3*s8 - 1.*s2*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,0) = -0.75*C1*s9 + D1*s9*(-1. + 1.*s2*s6*s7 - 2.*s1*s6*s8 - 1.*s2*s4*s9 + 2.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 - 2.*s5*s7*s9 + 1.*s4*s8*s9) + (2.*C1*s1*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,0) = D1*s8*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s2*s5*s7 + 1.*s2*s4*s8 - 2.*s1*s5*s8)*s9 + C1*(0.75*s8 + (2.*s1*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,0) = 1.*D1*(s3*s5 - 1.*s2*s6)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,0) = D1*s6*(1. + 1.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s3*s4*s5 + 1.*s2*s4*s6 - 2.*s1*s5*s6)*s9 + C1*(0.75*s6 + (2.*s1*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,0) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(-1.*s3*s7 + 2.*s1*s9) + s5*(-1. + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 - 2.*s2*s4*s9)) + C1*(-0.75*s5 + (2.*s1*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,1) = 1.*D1*(-1.*s6*s7 + s4*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s2)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,1) = 1.*C1 + 1.*D1*std::pow(s6*s7 - 1.*s4*s9,2) + (2.*C1*std::pow(s2,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(2,1) = 1.*D1*(s5*s7 - 1.*s4*s8)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,1) = 0.75*C1*s9 + D1*s9*(1. - 2.*s2*s6*s7 + 1.*s1*s6*s8 + 2.*s2*s4*s9 - 1.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 + 1.*s5*s7*s9 - 2.*s4*s8*s9) + (2.*C1*s2*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,1) = 1.*D1*(s3*s7 - 1.*s1*s9)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,1) = D1*s7*(-1. - 1.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s4*s7 + 1.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(-0.75*s7 + (2.*s2*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,1) = D1*s6*(-1. - 2.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s3*s4*s5 - 2.*s2*s4*s6 + 1.*s1*s5*s6)*s9 + C1*(-0.75*s6 + (2.*s2*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,1) = -1.*D1*(s3*s4 - 1.*s1*s6)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,1) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(-1.*s3*s8 + 2.*s2*s9) + s4*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s4 + (2.*s2*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,2) = 1.*D1*(s5*s7 - 1.*s4*s8)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,2) = -1.*D1*(s5*s7 - 1.*s4*s8)*(s6*s7 - 1.*s4*s9) + (2.*C1*s2*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,2) = 1.*C1 + 1.*D1*std::pow(s5*s7 - 1.*s4*s8,2) + (2.*C1*std::pow(s3,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(3,2) = D1*s8*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s2*s5*s7 - 2.*s2*s4*s8 + 1.*s1*s5*s8)*s9 + C1*(-0.75*s8 + (2.*s3*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(4,2) = D1*s7*(1. + 2.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s4*s7 - 2.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(0.75*s7 + (2.*s3*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(5,2) = -1.*D1*(s2*s7 - 1.*s1*s8)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,2) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(2.*s3*s7 - 1.*s1*s9) + s5*(1. - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 + 1.*s2*s4*s9)) + C1*(0.75*s5 + (2.*s3*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,2) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(2.*s3*s8 - 1.*s2*s9) + s4*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s4 + (2.*s3*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,2) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,3) = 1.*D1*(-1.*s3*s8 + s2*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,3) = 0.75*C1*s9 + D1*s9*(1. - 2.*s2*s6*s7 + 1.*s1*s6*s8 + 2.*s2*s4*s9 - 1.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 + 1.*s5*s7*s9 - 2.*s4*s8*s9) + (2.*C1*s2*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,3) = D1*s8*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s2*s5*s7 - 2.*s2*s4*s8 + 1.*s1*s5*s8)*s9 + C1*(-0.75*s8 + (2.*s3*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,3) = 1.*C1 + 1.*D1*std::pow(s3*s8 - 1.*s2*s9,2) + (2.*C1*std::pow(s4,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(4,3) = 1.*D1*(s3*s7 - 1.*s1*s9)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,3) = -1.*D1*(s2*s7 - 1.*s1*s8)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,3) = 1.*D1*(s3*s5 - 1.*s2*s6)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,3) = D1*s3*(-1. - 1.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s3*s4 + 1.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(-0.75*s3 + (2.*s4*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,3) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(-1.*s6*s7 + 2.*s4*s9) + s2*(1. + 1.*s3*s5*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s2 + (2.*s4*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,4) = -0.75*C1*s9 + D1*s9*(-1. + 1.*s2*s6*s7 - 2.*s1*s6*s8 - 1.*s2*s4*s9 + 2.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 - 2.*s5*s7*s9 + 1.*s4*s8*s9) + (2.*C1*s1*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,4) = -1.*D1*(s3*s7 - 1.*s1*s9)*(s6*s7 - 1.*s4*s9) + (2.*C1*s2*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,4) = D1*s7*(1. + 2.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s4*s7 - 2.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(0.75*s7 + (2.*s3*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,4) = -1.*D1*(s3*s7 - 1.*s1*s9)*(s3*s8 - 1.*s2*s9) + (2.*C1*s4*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,4) = 1.*C1 + 1.*D1*std::pow(s3*s7 - 1.*s1*s9,2) + (2.*C1*std::pow(s5,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(5,4) = -1.*D1*(s2*s7 - 1.*s1*s8)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,4) = D1*s3*(1. + 2.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s3*s4 - 2.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(0.75*s3 + (2.*s5*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,4) = -1.*D1*(s3*s4 - 1.*s1*s6)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,4) = 1.*D1*s2*s3*s4*s7 + D1*s1*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8 - 2.*s2*s4*s9 + 2.*s1*s5*s9) + C1*(-0.75*s1 + (2.*s5*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,5) = D1*s8*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s2*s5*s7 + 1.*s2*s4*s8 - 2.*s1*s5*s8)*s9 + C1*(0.75*s8 + (2.*s1*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(1,5) = D1*s7*(-1. - 1.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s4*s7 + 1.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(-0.75*s7 + (2.*s2*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(2,5) = 1.*D1*(-1.*s2*s7 + s1*s8)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,5) = -1.*D1*(s2*s7 - 1.*s1*s8)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,5) = 1.*D1*(-1.*s2*s7 + s1*s8)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,5) = 1.*C1 + 1.*D1*std::pow(s2*s7 - 1.*s1*s8,2) + (2.*C1*std::pow(s6,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(6,5) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(2.*s6*s7 - 1.*s4*s9) + s2*(-1. - 2.*s3*s5*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s2 + (2.*s6*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,5) = 1.*D1*s2*s3*s4*s7 + D1*s1*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8 + 1.*s2*s4*s9 - 1.*s1*s5*s9) + C1*(0.75*s1 + (2.*s6*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,5) = 1.*D1*(s2*s4 - 1.*s1*s5)*(-1.*s2*s7 + s1*s8) + (2.*C1*s6*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,6) = 1.*D1*(s3*s5 - 1.*s2*s6)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,6) = D1*s6*(-1. - 2.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s3*s4*s5 - 2.*s2*s4*s6 + 1.*s1*s5*s6)*s9 + C1*(-0.75*s6 + (2.*s2*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(2,6) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(2.*s3*s7 - 1.*s1*s9) + s5*(1. - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 + 1.*s2*s4*s9)) + C1*(0.75*s5 + (2.*s3*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,6) = -1.*D1*(s3*s5 - 1.*s2*s6)*(s3*s8 - 1.*s2*s9) + (2.*C1*s4*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,6) = D1*s3*(1. + 2.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s3*s4 - 2.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(0.75*s3 + (2.*s5*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(5,6) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(2.*s6*s7 - 1.*s4*s9) + s2*(-1. - 2.*s3*s5*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s2 + (2.*s6*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,6) = 1.*C1 + 1.*D1*std::pow(s3*s5 - 1.*s2*s6,2) + (2.*C1*std::pow(s7,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(7,6) = -1.*D1*(s3*s4 - 1.*s1*s6)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,6) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,7) = D1*s6*(1. + 1.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s3*s4*s5 + 1.*s2*s4*s6 - 2.*s1*s5*s6)*s9 + C1*(0.75*s6 + (2.*s1*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(1,7) = -1.*D1*(s3*s4 - 1.*s1*s6)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,7) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(2.*s3*s8 - 1.*s2*s9) + s4*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s4 + (2.*s3*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,7) = D1*s3*(-1. - 1.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s3*s4 + 1.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(-0.75*s3 + (2.*s4*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(4,7) = 1.*D1*(-1.*s3*s4 + s1*s6)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,7) = 1.*D1*s2*s3*s4*s7 + D1*s1*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8 + 1.*s2*s4*s9 - 1.*s1*s5*s9) + C1*(0.75*s1 + (2.*s6*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,7) = 1.*D1*(-1.*s3*s4 + s1*s6)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,7) = 1.*C1 + 1.*D1*std::pow(s3*s4 - 1.*s1*s6,2) + (2.*C1*std::pow(s8,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(8,7) = 1.*D1*(s2*s4 - 1.*s1*s5)*(-1.*s3*s4 + s1*s6) + (2.*C1*s8*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,8) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(-1.*s3*s7 + 2.*s1*s9) + s5*(-1. + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 - 2.*s2*s4*s9)) + C1*(-0.75*s5 + (2.*s1*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(1,8) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(-1.*s3*s8 + 2.*s2*s9) + s4*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s4 + (2.*s2*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(2,8) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,8) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(-1.*s6*s7 + 2.*s4*s9) + s2*(1. + 1.*s3*s5*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s2 + (2.*s4*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(4,8) = 1.*D1*s2*s3*s4*s7 + D1*s1*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8 - 2.*s2*s4*s9 + 2.*s1*s5*s9) + C1*(-0.75*s1 + (2.*s5*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(5,8) = -1.*D1*(s2*s4 - 1.*s1*s5)*(s2*s7 - 1.*s1*s8) + (2.*C1*s6*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,8) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,8) = -1.*D1*(s2*s4 - 1.*s1*s5)*(s3*s4 - 1.*s1*s6) + (2.*C1*s8*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,8) = 1.*C1 + 1.*D1*std::pow(s2*s4 - 1.*s1*s5,2) + (2.*C1*std::pow(s9,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

			//FIX SVD
			Eigen::SelfAdjointEigenSolver<Matrix<double, 9, 9>> es(ddw);
			Eigen::Matrix<double, 9, 9> DiagEval = es.eigenvalues().real().asDiagonal();
	        Eigen::Matrix<double, 9, 9> Evec = es.eigenvectors().real();
	        
	        for (int i = 0; i < 9; ++i) {
	            if (es.eigenvalues()[i]<1e-6) {
	                DiagEval(i,i) = 1e-3;
	            }
	        }

	        ddw = Evec * DiagEval * Evec.transpose();

			for(int i=0; i<ddw.rows(); i++){
				for(int j=0; j<ddw.cols(); j++){
					hess_trips.push_back(Trip(9*f_index + i, 9*f_index + j, rest_tet_vols[t]*ddw(i,j)));
				}
			}

		}
	}
	for(int m=0; m<bone_tets.size(); m++){
			double s1 = Fvec[9*m + 0];
			double s2 = Fvec[9*m + 1];
			double s3 = Fvec[9*m + 2];
			double s4 = Fvec[9*m + 3];
			double s5 = Fvec[9*m + 4];
			double s6 = Fvec[9*m + 5];
			double s7 = Fvec[9*m + 6];
			double s8 = Fvec[9*m + 7];
			double s9 = Fvec[9*m + 8];
			int t = bone_tets[m][0];
			double youngsModulus = eY[t];
			double poissonsRatio = eP[t];

			double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
			double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));

			Eigen::Matrix<double, 9,9> ddw(9,9);
				ddw(0,0) = 1.*C1 + 1.*D1*std::pow(s6*s8 - 1.*s5*s9,2) + (2.*C1*std::pow(s1,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(1,0) = -1.*D1*(s6*s7 - 1.*s4*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s2)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,0) = 1.*D1*(s5*s7 - 1.*s4*s8)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,0) = -1.*D1*(s3*s8 - 1.*s2*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,0) = -0.75*C1*s9 + D1*s9*(-1. + 1.*s2*s6*s7 - 2.*s1*s6*s8 - 1.*s2*s4*s9 + 2.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 - 2.*s5*s7*s9 + 1.*s4*s8*s9) + (2.*C1*s1*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,0) = D1*s8*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s2*s5*s7 + 1.*s2*s4*s8 - 2.*s1*s5*s8)*s9 + C1*(0.75*s8 + (2.*s1*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,0) = 1.*D1*(s3*s5 - 1.*s2*s6)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,0) = D1*s6*(1. + 1.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s3*s4*s5 + 1.*s2*s4*s6 - 2.*s1*s5*s6)*s9 + C1*(0.75*s6 + (2.*s1*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,0) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(-1.*s3*s7 + 2.*s1*s9) + s5*(-1. + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 - 2.*s2*s4*s9)) + C1*(-0.75*s5 + (2.*s1*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,1) = 1.*D1*(-1.*s6*s7 + s4*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s2)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,1) = 1.*C1 + 1.*D1*std::pow(s6*s7 - 1.*s4*s9,2) + (2.*C1*std::pow(s2,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(2,1) = 1.*D1*(s5*s7 - 1.*s4*s8)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,1) = 0.75*C1*s9 + D1*s9*(1. - 2.*s2*s6*s7 + 1.*s1*s6*s8 + 2.*s2*s4*s9 - 1.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 + 1.*s5*s7*s9 - 2.*s4*s8*s9) + (2.*C1*s2*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,1) = 1.*D1*(s3*s7 - 1.*s1*s9)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,1) = D1*s7*(-1. - 1.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s4*s7 + 1.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(-0.75*s7 + (2.*s2*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,1) = D1*s6*(-1. - 2.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s3*s4*s5 - 2.*s2*s4*s6 + 1.*s1*s5*s6)*s9 + C1*(-0.75*s6 + (2.*s2*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,1) = -1.*D1*(s3*s4 - 1.*s1*s6)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,1) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(-1.*s3*s8 + 2.*s2*s9) + s4*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s4 + (2.*s2*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,2) = 1.*D1*(s5*s7 - 1.*s4*s8)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,2) = -1.*D1*(s5*s7 - 1.*s4*s8)*(s6*s7 - 1.*s4*s9) + (2.*C1*s2*s3)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,2) = 1.*C1 + 1.*D1*std::pow(s5*s7 - 1.*s4*s8,2) + (2.*C1*std::pow(s3,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(3,2) = D1*s8*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s2*s5*s7 - 2.*s2*s4*s8 + 1.*s1*s5*s8)*s9 + C1*(-0.75*s8 + (2.*s3*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(4,2) = D1*s7*(1. + 2.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s4*s7 - 2.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(0.75*s7 + (2.*s3*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(5,2) = -1.*D1*(s2*s7 - 1.*s1*s8)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,2) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(2.*s3*s7 - 1.*s1*s9) + s5*(1. - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 + 1.*s2*s4*s9)) + C1*(0.75*s5 + (2.*s3*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,2) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(2.*s3*s8 - 1.*s2*s9) + s4*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s4 + (2.*s3*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,2) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,3) = 1.*D1*(-1.*s3*s8 + s2*s9)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,3) = 0.75*C1*s9 + D1*s9*(1. - 2.*s2*s6*s7 + 1.*s1*s6*s8 + 2.*s2*s4*s9 - 1.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 + 1.*s5*s7*s9 - 2.*s4*s8*s9) + (2.*C1*s2*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,3) = D1*s8*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s2*s5*s7 - 2.*s2*s4*s8 + 1.*s1*s5*s8)*s9 + C1*(-0.75*s8 + (2.*s3*s4)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,3) = 1.*C1 + 1.*D1*std::pow(s3*s8 - 1.*s2*s9,2) + (2.*C1*std::pow(s4,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(4,3) = 1.*D1*(s3*s7 - 1.*s1*s9)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,3) = -1.*D1*(s2*s7 - 1.*s1*s8)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,3) = 1.*D1*(s3*s5 - 1.*s2*s6)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,3) = D1*s3*(-1. - 1.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s3*s4 + 1.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(-0.75*s3 + (2.*s4*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,3) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(-1.*s6*s7 + 2.*s4*s9) + s2*(1. + 1.*s3*s5*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s2 + (2.*s4*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,4) = -0.75*C1*s9 + D1*s9*(-1. + 1.*s2*s6*s7 - 2.*s1*s6*s8 - 1.*s2*s4*s9 + 2.*s1*s5*s9) + D1*s3*(1.*s6*s7*s8 - 2.*s5*s7*s9 + 1.*s4*s8*s9) + (2.*C1*s1*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,4) = -1.*D1*(s3*s7 - 1.*s1*s9)*(s6*s7 - 1.*s4*s9) + (2.*C1*s2*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,4) = D1*s7*(1. + 2.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s4*s7 - 2.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(0.75*s7 + (2.*s3*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,4) = -1.*D1*(s3*s7 - 1.*s1*s9)*(s3*s8 - 1.*s2*s9) + (2.*C1*s4*s5)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,4) = 1.*C1 + 1.*D1*std::pow(s3*s7 - 1.*s1*s9,2) + (2.*C1*std::pow(s5,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(5,4) = -1.*D1*(s2*s7 - 1.*s1*s8)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,4) = D1*s3*(1. + 2.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s3*s4 - 2.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(0.75*s3 + (2.*s5*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,4) = -1.*D1*(s3*s4 - 1.*s1*s6)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,4) = 1.*D1*s2*s3*s4*s7 + D1*s1*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8 - 2.*s2*s4*s9 + 2.*s1*s5*s9) + C1*(-0.75*s1 + (2.*s5*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(0,5) = D1*s8*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s2*s5*s7 + 1.*s2*s4*s8 - 2.*s1*s5*s8)*s9 + C1*(0.75*s8 + (2.*s1*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(1,5) = D1*s7*(-1. - 1.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s4*s7 + 1.*s1*s5*s7 + 1.*s1*s4*s8)*s9 + C1*(-0.75*s7 + (2.*s2*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(2,5) = 1.*D1*(-1.*s2*s7 + s1*s8)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,5) = -1.*D1*(s2*s7 - 1.*s1*s8)*(-1.*s3*s8 + s2*s9) + (2.*C1*s4*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,5) = 1.*D1*(-1.*s2*s7 + s1*s8)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s6)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,5) = 1.*C1 + 1.*D1*std::pow(s2*s7 - 1.*s1*s8,2) + (2.*C1*std::pow(s6,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(6,5) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(2.*s6*s7 - 1.*s4*s9) + s2*(-1. - 2.*s3*s5*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s2 + (2.*s6*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(7,5) = 1.*D1*s2*s3*s4*s7 + D1*s1*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8 + 1.*s2*s4*s9 - 1.*s1*s5*s9) + C1*(0.75*s1 + (2.*s6*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(8,5) = 1.*D1*(s2*s4 - 1.*s1*s5)*(-1.*s2*s7 + s1*s8) + (2.*C1*s6*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,6) = 1.*D1*(s3*s5 - 1.*s2*s6)*(s6*s8 - 1.*s5*s9) + (2.*C1*s1*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(1,6) = D1*s6*(-1. - 2.*s3*s5*s7 + 2.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8) + D1*(1.*s3*s4*s5 - 2.*s2*s4*s6 + 1.*s1*s5*s6)*s9 + C1*(-0.75*s6 + (2.*s2*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(2,6) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(2.*s3*s7 - 1.*s1*s9) + s5*(1. - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 + 1.*s2*s4*s9)) + C1*(0.75*s5 + (2.*s3*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,6) = -1.*D1*(s3*s5 - 1.*s2*s6)*(s3*s8 - 1.*s2*s9) + (2.*C1*s4*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(4,6) = D1*s3*(1. + 2.*s3*s5*s7 - 2.*s2*s6*s7 - 1.*s3*s4*s8 + 1.*s1*s6*s8) + D1*(1.*s2*s3*s4 - 2.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(0.75*s3 + (2.*s5*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(5,6) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(2.*s6*s7 - 1.*s4*s9) + s2*(-1. - 2.*s3*s5*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s2 + (2.*s6*s7)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,6) = 1.*C1 + 1.*D1*std::pow(s3*s5 - 1.*s2*s6,2) + (2.*C1*std::pow(s7,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(7,6) = -1.*D1*(s3*s4 - 1.*s1*s6)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,6) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,7) = D1*s6*(1. + 1.*s3*s5*s7 - 1.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8) + D1*(1.*s3*s4*s5 + 1.*s2*s4*s6 - 2.*s1*s5*s6)*s9 + C1*(0.75*s6 + (2.*s1*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(1,7) = -1.*D1*(s3*s4 - 1.*s1*s6)*(-1.*s6*s7 + s4*s9) + (2.*C1*s2*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(2,7) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(2.*s3*s8 - 1.*s2*s9) + s4*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 - 2.*s1*s6*s8 + 1.*s1*s5*s9)) + C1*(-0.75*s4 + (2.*s3*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(3,7) = D1*s3*(-1. - 1.*s3*s5*s7 + 1.*s2*s6*s7 + 2.*s3*s4*s8 - 2.*s1*s6*s8) + D1*(-2.*s2*s3*s4 + 1.*s1*s3*s5 + 1.*s1*s2*s6)*s9 + C1*(-0.75*s3 + (2.*s4*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(4,7) = 1.*D1*(-1.*s3*s4 + s1*s6)*(s3*s7 - 1.*s1*s9) + (2.*C1*s5*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(5,7) = 1.*D1*s2*s3*s4*s7 + D1*s1*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 - 2.*s3*s4*s8 + 2.*s1*s6*s8 + 1.*s2*s4*s9 - 1.*s1*s5*s9) + C1*(0.75*s1 + (2.*s6*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(6,7) = 1.*D1*(-1.*s3*s4 + s1*s6)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s8)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,7) = 1.*C1 + 1.*D1*std::pow(s3*s4 - 1.*s1*s6,2) + (2.*C1*std::pow(s8,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

				ddw(8,7) = 1.*D1*(s2*s4 - 1.*s1*s5)*(-1.*s3*s4 + s1*s6) + (2.*C1*s8*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(0,8) = D1*(1.*s2*s4*s6*s8 + std::pow(s5,2)*(-1.*s3*s7 + 2.*s1*s9) + s5*(-1. + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 2.*s1*s6*s8 - 2.*s2*s4*s9)) + C1*(-0.75*s5 + (2.*s1*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(1,8) = D1*(1.*s1*s5*s6*s7 + std::pow(s4,2)*(-1.*s3*s8 + 2.*s2*s9) + s4*(1. + 1.*s3*s5*s7 - 2.*s2*s6*s7 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s4 + (2.*s2*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(2,8) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s5*s7 - 1.*s4*s8) + (2.*C1*s3*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(3,8) = D1*(1.*s1*s3*s5*s8 + std::pow(s2,2)*(-1.*s6*s7 + 2.*s4*s9) + s2*(1. + 1.*s3*s5*s7 - 2.*s3*s4*s8 + 1.*s1*s6*s8 - 2.*s1*s5*s9)) + C1*(0.75*s2 + (2.*s4*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(4,8) = 1.*D1*s2*s3*s4*s7 + D1*s1*(-1. - 2.*s3*s5*s7 + 1.*s2*s6*s7 + 1.*s3*s4*s8 - 1.*s1*s6*s8 - 2.*s2*s4*s9 + 2.*s1*s5*s9) + C1*(-0.75*s1 + (2.*s5*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2));

				ddw(5,8) = -1.*D1*(s2*s4 - 1.*s1*s5)*(s2*s7 - 1.*s1*s8) + (2.*C1*s6*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(6,8) = 1.*D1*(s2*s4 - 1.*s1*s5)*(s3*s5 - 1.*s2*s6) + (2.*C1*s7*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(7,8) = -1.*D1*(s2*s4 - 1.*s1*s5)*(s3*s4 - 1.*s1*s6) + (2.*C1*s8*s9)/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2);

				ddw(8,8) = 1.*C1 + 1.*D1*std::pow(s2*s4 - 1.*s1*s5,2) + (2.*C1*std::pow(s9,2))/std::pow(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2),2) - (1.*C1)/(1 + std::pow(s1,2) + std::pow(s2,2) + std::pow(s3,2) + std::pow(s4,2) + std::pow(s5,2) + std::pow(s6,2) + std::pow(s7,2) + std::pow(s8,2) + std::pow(s9,2));

			//FIX SVD
			Eigen::SelfAdjointEigenSolver<Matrix<double, 9, 9>> es(ddw);
			Eigen::Matrix<double, 9, 9> DiagEval = es.eigenvalues().real().asDiagonal();
	        Eigen::Matrix<double, 9, 9> Evec = es.eigenvectors().real();
	        
	        for (int i = 0; i < 9; ++i) {
	            if (es.eigenvalues()[i]<1e-6) {
	                DiagEval(i,i) = 1e-3;
	            }
	        }

	        ddw = Evec * DiagEval * Evec.transpose();

			for(int i=0; i<ddw.rows(); i++){
				for(int j=0; j<ddw.cols(); j++){
					hess_trips.push_back(Trip(9*m + i, 9*m + j, bone_vols[m]*ddw(i,j)));
				}
			}

	}
	
	
	hess.setFromTriplets(hess_trips.begin(), hess_trips.end());
}
