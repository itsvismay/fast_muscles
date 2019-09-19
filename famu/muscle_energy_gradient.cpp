#include "muscle_energy_gradient.h"
#include "store.h"
using Store = famu::Store;
using namespace Eigen;

void famu::muscle::setupFastMuscles(Store& store){
	store.fastMuscles.clear();
	
	std::vector<Trip> mat_trips;
	for(int i=0; i<store.muscle_tets.size(); i++){
		SparseMatrix<double, Eigen::RowMajor> mat;
		for(int j=0; j<store.muscle_tets[i].size(); j++){
			int t = store.muscle_tets[i][j];
			Vector3d u = std::sqrt(store.muscle_mag[t]*store.rest_tet_volume[t])*store.Uvec.row(t);
			
			int f_index = store.bone_or_muscle[t];

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

		mat.resize(3*store.T.rows(), store.dFvec.size());
		mat.setFromTriplets(mat_trips.begin(), mat_trips.end());
		
		store.fastMuscles.push_back(mat.transpose()*mat);
	}
}

// double famu::muscle::fastEnergy(Store& store, VectorXd& dFvec){
// 	double W =0;
// 	for(int i=0; i<store.contract_muscles.size(); i++){
// 		W += 0.5*dFvec.transpose()*store.fastMuscles[store.contract_muscles[i]]*dFvec;
// 	}
// 	return W;
// }

// void famu::muscle::fastGradient(Store& store, VectorXd& grad){
// 	for(int i=0; i<store.contract_muscles.size(); i++){
// 		grad += store.fastMuscles[store.contract_muscles[i]]*store.dFvec;
// 	}
// }

double famu::muscle::energy(Store& store, VectorXd& dFvec){
	double MuscleEnergy = 0;

	for(int i=0; i<store.contract_muscles.size(); i++){
		#pragma omp parallel
		{	
			double me_priv = 0;

			#pragma omp for	
			for(int j=0; j<store.muscle_tets[store.contract_muscles[i]].size(); j++){
				int t = store.muscle_tets[store.contract_muscles[i]][j];
				int f_index = store.bone_or_muscle[t];
				Matrix3d F = Map<Matrix3d>(dFvec.segment<9>(9*f_index).data()).transpose();
				Vector3d y = store.Uvec.row(t).transpose();
				Vector3d z = F*y;
				double W = 0.5*store.muscle_mag[t]*store.rest_tet_volume[t]*(z.dot(z));
				me_priv += W;
			}

			#pragma omp critical
			{
				MuscleEnergy += me_priv;
			}
		}
	}
	return MuscleEnergy;
}

void famu::muscle::gradient(Store& store, VectorXd& grad){
	grad.setZero();

	for(int i=0; i<store.contract_muscles.size(); i++){
		#pragma omp parallel for
		for(int j=0; j<store.muscle_tets[store.contract_muscles[i]].size(); j++){
			int t = store.muscle_tets[store.contract_muscles[i]][j];
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

			Vector3d y = store.Uvec.row(t).transpose();
			double u1 = y[0];
			double u2 = y[1];
			double u3 = y[2];
			double a = store.muscle_mag[t];

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

			grad.segment<9>(9*f_index) += store.rest_tet_volume[t]*tet_grad;
		}
	}
}

void famu::muscle::fastHessian(Store& store, SparseMatrix<double, Eigen::RowMajor>& hess, Eigen::MatrixXd& denseHess){
	hess.setZero();
	for(int i=0; i<store.contract_muscles.size(); i++){
		hess += store.fastMuscles[store.contract_muscles[i]];
	}

	if(store.jinput["woodbury"]){
		//set dense block diag hess if woodbury
	}
}

VectorXd famu::muscle::fd_gradient(Store& store){
	VectorXd fake = VectorXd::Zero(store.dFvec.size());
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

MatrixXd famu::muscle::fd_hessian(Store& store){
	MatrixXd fake = MatrixXd::Zero(store.dFvec.size(), store.dFvec.size());
	VectorXd dFvec = store.dFvec;
	double eps = 1e-3;
	double E0 = energy(store, dFvec);
	for(int i=0; i<11; i++){
		for(int j=0; j<11; j++){
			dFvec[i] += eps;
			dFvec[j] += eps;
			double Eij = energy(store, dFvec);
			dFvec[i] -= eps;
			dFvec[j] -= eps;

			dFvec[i] += eps;
			double Ei = energy(store, dFvec);
			dFvec[i] -=eps;

			dFvec[j] += eps;
			double Ej = energy(store, dFvec);
			dFvec[j] -=eps;
			
			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	return fake;
}