#include "stablenh_energy_gradient.h"
#include "store.h"
#include <iostream>
#include <omp.h>
using namespace Eigen;
using Store = famu::Store;
typedef Eigen::Triplet<double> Trip;


double famu::stablenh::continuous_energy(const Store& store, VectorXd& x){
	double stableNHEnergy = 0;
	VectorXd y = store.Y*x+store.x0;
	#pragma omp parallel
	{	
		double sNHpriv = 0;
		for(int t = 0; t<store.T.rows(); t++){
			Vector4i verts_index = store.T.row(t);

			Matrix3d Dm;
	        for(int i=0; i<3; i++)
	        {
	            Dm.col(i) = store.x0.segment<3>(3*verts_index(i)) - store.x0.segment<3>(3*verts_index(3));
	        }
	        Matrix3d m_InvRefShapeMatrix = Dm.inverse();
	        

			Matrix3d Ds;
	        for(int i=0; i<3; i++)
	        {
	            Ds.col(i) = y.segment<3>(3*verts_index(i)) - y.segment<3>(3*verts_index(3));
	        }

	        double youngsModulus = store.eY[t];
			double poissonsRatio = store.eP[t];
			double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
			double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));

	        Matrix3d F = Ds*m_InvRefShapeMatrix;
	        double I1 = (F.transpose()*F).trace();
			double J = F.determinant();
			double alpha = (1 + (C1/D1) - (C1/(D1*4)));
			double W = 0.5*C1*(I1 -3) + 0.5*D1*(J-alpha)*(J-alpha) - 0.5*C1*log(I1 + 1);
			sNHpriv += W*store.rest_tet_volume[t];

		}
		#pragma omp critical
		{
			stableNHEnergy += sNHpriv;
		}
	}
	return stableNHEnergy;
}

double famu::stablenh::energy(const Store& store, Eigen::VectorXd& dFvec){
	double stableNHEnergy = 0;

	for(int m=0; m<store.muscle_tets.size(); m++){
		#pragma omp parallel
		{	
			double sNHpriv = 0;

			#pragma omp for 
			for(int i=0; i<store.muscle_tets[m].size(); i++){
				int t = store.muscle_tets[m][i];
				double youngsModulus = store.eY[t];
				double poissonsRatio = store.eP[t];
				double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
				double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
				
				int f_index = store.bone_or_muscle[t];

				Matrix3d F = Map<Matrix3d>(dFvec.segment<9>(9*f_index).data()).transpose();
				double I1 = (F.transpose()*F).trace();
				double J = F.determinant();
				double alpha = (1 + (C1/D1) - (C1/(D1*4)));
				double W = 0.5*C1*(I1 -3) + 0.5*D1*(J-alpha)*(J-alpha) - 0.5*C1*log(I1 + 1);
				sNHpriv += W*store.rest_tet_volume[t];
			}

			#pragma omp critical
			{
				stableNHEnergy += sNHpriv;
			}
		} 
	}

	for(int m=0; m<store.bone_tets.size(); m++){
		int t = store.bone_tets[m][0];
		double youngsModulus = store.eY[t];
		double poissonsRatio = store.eP[t];
		double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
		double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));

		Matrix3d F = Map<Matrix3d>(dFvec.segment<9>(9*m).data()).transpose();
		double I1 = (F.transpose()*F).trace();
		double J = F.determinant();
		double alpha = (1 + (C1/D1) - (C1/(D1*4)));
		double W = 0.5*C1*(I1 -3) + 0.5*D1*(J-alpha)*(J-alpha) - 0.5*C1*log(I1 + 1);

		stableNHEnergy += store.bone_vols[m]*W;

	}

	return store.alpha_neo*stableNHEnergy;
}

void famu::stablenh::gradient(const Store& store, Eigen::VectorXd& grad){
	grad.setZero();

	for(int m=0; m<store.muscle_tets.size(); m++){
		#pragma omp parallel for
		for(int i=0; i<store.muscle_tets[m].size(); i++){
			int t = store.muscle_tets[m][i];
			double youngsModulus = store.eY[t];
			double poissonsRatio = store.eP[t];
			double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
			double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
			
			int f_index = store.bone_or_muscle[t];

			double s1 = store.dFvec[9*f_index + 0];
			double s2 = store.dFvec[9*f_index + 1];
			double s3 = store.dFvec[9*f_index + 2];
			double s4 = store.dFvec[9*f_index + 3];
			double s5 = store.dFvec[9*f_index + 4];
			double s6 = store.dFvec[9*f_index + 5];
			double s7 = store.dFvec[9*f_index + 6];
			double s8 = store.dFvec[9*f_index + 7];
			double s9 = store.dFvec[9*f_index + 8];

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
	   
			grad.segment<9>(9*f_index)  += store.alpha_neo*store.rest_tet_volume[t]*tet_grad;
		}
	}

	for(int m=0; m<store.bone_tets.size(); m++){
		double s1 = store.dFvec[9*m + 0];
		double s2 = store.dFvec[9*m + 1];
		double s3 = store.dFvec[9*m + 2];
		double s4 = store.dFvec[9*m + 3];
		double s5 = store.dFvec[9*m + 4];
		double s6 = store.dFvec[9*m + 5];
		double s7 = store.dFvec[9*m + 6];
		double s8 = store.dFvec[9*m + 7];
		double s9 = store.dFvec[9*m + 8];
		int t = store.bone_tets[m][0];
		double youngsModulus = store.eY[t];
		double poissonsRatio = store.eP[t];

		double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
		double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
		
		// int f_index = store.bone_or_muscle[t];


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

    	
		grad.segment<9>(9*m) +=  store.alpha_neo*store.bone_vols[m]*tet_grad;

	}
}

void famu::stablenh::hessian(const Store& store, Eigen::SparseMatrix<double, Eigen::RowMajor>& hess, Eigen::MatrixXd& denseHess, bool dense){


	if(dense){
		denseHess.setZero();
		for(int m=0; m<store.muscle_tets.size(); m++){
			#pragma omp parallel for
			for(int i=0; i<store.muscle_tets[m].size(); i++){
				int t = store.muscle_tets[m][i];
				double youngsModulus = store.eY[t];
				double poissonsRatio = store.eP[t];
				double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
				double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
				
				int f_index = store.bone_or_muscle[t];

				double s1 = store.dFvec[9*f_index + 0];
				double s2 = store.dFvec[9*f_index + 1];
				double s3 = store.dFvec[9*f_index + 2];
				double s4 = store.dFvec[9*f_index + 3];
				double s5 = store.dFvec[9*f_index + 4];
				double s6 = store.dFvec[9*f_index + 5];
				double s7 = store.dFvec[9*f_index + 6];
				double s8 = store.dFvec[9*f_index + 7];
				double s9 = store.dFvec[9*f_index + 8];

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

		        denseHess.block<9,9>(9*f_index, 0) += store.rest_tet_volume[t]*ddw;

			}
		}

		for(int m=0; m<store.bone_tets.size(); m++){
			double s1 = store.dFvec[9*m + 0];
			double s2 = store.dFvec[9*m + 1];
			double s3 = store.dFvec[9*m + 2];
			double s4 = store.dFvec[9*m + 3];
			double s5 = store.dFvec[9*m + 4];
			double s6 = store.dFvec[9*m + 5];
			double s7 = store.dFvec[9*m + 6];
			double s8 = store.dFvec[9*m + 7];
			double s9 = store.dFvec[9*m + 8];
			int t = store.bone_tets[m][0];
			double youngsModulus = store.eY[t];
			double poissonsRatio = store.eP[t];

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

	        denseHess.block<9,9>(9*m, 0) += store.bone_vols[m]*ddw;
		}

		denseHess *= store.alpha_neo;



	}else{

		hess.setZero();
		std::vector<Trip> hess_trips;
		hess_trips.reserve(9*9*store.T.rows());
		for(int t=0; t<store.T.rows(); t++){
			double youngsModulus = store.eY[t];
			double poissonsRatio = store.eP[t];
			double C1 = youngsModulus/(2.0*(1.0+poissonsRatio));
			double D1 = (youngsModulus*poissonsRatio)/((1.0+poissonsRatio)*(1.0-2.0*poissonsRatio));
			
			int f_index = store.bone_or_muscle[t];

			double s1 = store.dFvec[9*f_index + 0];
			double s2 = store.dFvec[9*f_index + 1];
			double s3 = store.dFvec[9*f_index + 2];
			double s4 = store.dFvec[9*f_index + 3];
			double s5 = store.dFvec[9*f_index + 4];
			double s6 = store.dFvec[9*f_index + 5];
			double s7 = store.dFvec[9*f_index + 6];
			double s8 = store.dFvec[9*f_index + 7];
			double s9 = store.dFvec[9*f_index + 8];

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
					hess_trips.push_back(Trip(9*f_index + i, 9*f_index + j, store.rest_tet_volume[t]*ddw(i,j)));
				}
			}
		
		}
		hess.setFromTriplets(hess_trips.begin(), hess_trips.end());
		hess = store.alpha_neo*hess;
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

MatrixXd famu::stablenh::fd_hessian(Store& store){
	MatrixXd fake = MatrixXd::Zero(store.dFvec.size(), store.dFvec.size());
	VectorXd dFvec = store.dFvec;
	double eps = 1e-3;
	double E0 = famu::stablenh::energy(store, dFvec);
	for(int i=0; i<17; i++){
		for(int j=0; j<17; j++){
			dFvec[i] += eps;
			dFvec[j] += eps;
			double Eij = famu::stablenh::energy(store, dFvec);
			dFvec[i] -= eps;
			dFvec[j] -= eps;

			dFvec[i] += eps;
			double Ei = famu::stablenh::energy(store, dFvec);
			dFvec[i] -=eps;

			dFvec[j] += eps;
			double Ej = famu::stablenh::energy(store, dFvec);
			dFvec[j] -=eps;
			
			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	return fake;
}