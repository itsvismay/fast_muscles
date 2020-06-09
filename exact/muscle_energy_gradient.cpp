#include "muscle_energy_gradient.h"
#include <iostream>
#include <mutex>
#include <omp.h>
using namespace Eigen;

double exact::muscle::energy(VectorXd& Fvec,
							const MatrixXi& T,
							const VectorXd& rest_tet_vols,
							const MatrixXd& Uvec){
	double MuscleEnergy = 0;

	#pragma omp parallel
	{	
		double me_priv = 0;

		#pragma omp for	
		for(int t=0; t<T.rows(); t++){
			int f_index = t;
			Matrix3d F = Map<Matrix3d>(Fvec.segment<9>(9*f_index).data()).transpose();
			//Muscle fiber directions for this tet Uvec.row(t)
			Vector3d y = Uvec.row(t).transpose();
			Vector3d z = F*y;
			double W = 0.5*rest_tet_vols[t]*(z.dot(z));
			me_priv += W;
		}

		#pragma omp critical
		{
			MuscleEnergy += me_priv;
		}
	}
	
	return MuscleEnergy;
}

void exact::muscle::gradient(VectorXd& grad, 
							const VectorXd& Fvec,
							const MatrixXi& T,
							const VectorXd& rest_tet_vols,
							const MatrixXd& Uvec ){
	grad.setZero();

	#pragma omp parallel for
	for(int t=0; t<T.rows(); t++){
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

		Vector3d y = Uvec.row(t).transpose();
		double u1 = y[0];
		double u2 = y[1];
		double u3 = y[2];
		double a = 1; //muscle_mag[t];

		VectorXd tet_grad(9);

		tet_grad[0] = 0.5*a*(s2*u1*u2 + s3*u1*u3 + u1*(2*s1*u1 + s2*u2 + s3*u3));
		tet_grad[1] = 0.5*a*(s1*u1*u2 + s3*u2*u3 + u2*(s1*u1 + 2*s2*u2 + s3*u3));
		tet_grad[2] = 0.5*a*(s1*u1*u3 + s2*u2*u3 + u3*(s1*u1 + s2*u2 + 2*s3*u3));

		tet_grad[3] = 0.5*a*(s5*u1*u2 + s6*u1*u3 + u1*(2*s4*u1 + s5*u2 + s6*u3));
		tet_grad[4] = 0.5*a*(s4*u1*u2 + s6*u2*u3 + u2*(s4*u1 + 2*s5*u2 + s6*u3));
		tet_grad[5] = 0.5*a*(s4*u1*u3 + s5*u2*u3 + u3*(s4*u1 + s5*u2 + 2*s6*u3));

		tet_grad[6] = 0.5*a*(s8*u1*u2 + s9*u1*u3 + u1*(2*s7*u1 + s8*u2 + s9*u3));
		tet_grad[7] = 0.5*a*(s7*u1*u2 + s9*u2*u3 + u2*(s7*u1 + 2*s8*u2 + s9*u3));
		tet_grad[8] = 0.5*a*(s7*u1*u3 + s8*u2*u3 + u3*(s7*u1 + s8*u2 + 2*s9*u3));

		grad.segment<9>(9*f_index) += rest_tet_vols[t]*tet_grad;
	}
}

void exact::muscle::hessian(SparseMatrix<double, Eigen::RowMajor>& H, 
							const VectorXd& Fvec,
							const MatrixXi& T,
							const VectorXd& rest_tet_vols,
							const MatrixXd& Uvec){
	H.setZero();

	std::vector<Trip> mat_trips;
	SparseMatrix<double, Eigen::RowMajor> mat;

	//#pragma omp parallel for shared(idx)
	for(int t=0; t<T.rows(); t++){
			int f_index = t;
			Vector3d u = std::sqrt(rest_tet_vols[t])*Uvec.row(t);
				

		{
		    //std::lock_guard<std::mutex> lg(door);
			mat_trips.push_back(Trip(3*t+0, 9*f_index + 0, u[0]));
			mat_trips.push_back(Trip(3*t+0, 9*f_index + 1, u[1]));
			mat_trips.push_back(Trip(3*t+0, 9*f_index + 2, u[2]));

			mat_trips.push_back(Trip(3*t+1, 9*f_index + 3, u[0]));
			mat_trips.push_back(Trip(3*t+1, 9*f_index + 4, u[1]));
			mat_trips.push_back(Trip(3*t+1, 9*f_index + 5, u[2]));

			mat_trips.push_back(Trip(3*t+2, 9*f_index + 6, u[0]));
			mat_trips.push_back(Trip(3*t+2, 9*f_index + 7, u[1]));
			mat_trips.push_back(Trip(3*t+2, 9*f_index + 8, u[2]));
		}

	}
	mat.resize(3*T.rows(), Fvec.size());
	mat.setFromTriplets(mat_trips.begin(), mat_trips.end());
	H = mat.transpose()*mat;
}



void exact::muscle::set_muscle_mag(const int step){
	// store.muscle_mag.setZero();
	// store.contract_muscles.clear();
	// nlohmann::json activations_at_step = store.muscle_steps[step];
	// for(nlohmann::json::iterator it = activations_at_step.begin(); it != activations_at_step.end(); ++it){
	// 	store.contract_muscles.push_back(store.muscle_name_index_map[it.key()]);
	// 	for(int j=0; j<store.muscle_tets[store.muscle_name_index_map[it.key()]].size(); j++){
	// 		int t = store.muscle_tets[store.muscle_name_index_map[it.key()]][j];
	// 		store.muscle_mag[t] = it.value();
	// 	}
	// }
	//exact::muscle::setupFastMuscles(store);
	//famu::muscle::fastHessian(store, store.muscleHess, store.denseMuscleHess);
}





