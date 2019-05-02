#ifndef MUSCLE_ENERGY_GRAD
#define MUSCLE_ENERGY_GRAD

#include "store.h"

using namespace famu;
using Store = famu::Store;

namespace famu
{
	namespace muscle{

		void setupFastMuscles(Store& store, SparseMatrix<double>& mat){
			std::vector<Trip> mat_trips;
			for(int t=0; t<store.T.rows(); t++){
				Vector3d u = store.Uvec.row(t);
				mat_trips.push_back(Trip(3*t+0, 9*t+0, u[0]));
				mat_trips.push_back(Trip(3*t+0, 9*t+1, u[1]));
				mat_trips.push_back(Trip(3*t+0, 9*t+2, u[2]));

				mat_trips.push_back(Trip(3*t+1, 9*t+3, u[0]));
				mat_trips.push_back(Trip(3*t+1, 9*t+4, u[1]));
				mat_trips.push_back(Trip(3*t+1, 9*t+5, u[2]));

				mat_trips.push_back(Trip(3*t+2, 9*t+6, u[0]));
				mat_trips.push_back(Trip(3*t+2, 9*t+7, u[1]));
				mat_trips.push_back(Trip(3*t+2, 9*t+8, u[2]));
			}

			mat.resize(3*store.T.rows(), 9*store.T.rows());
			mat.setFromTriplets(mat_trips.begin(), mat_trips.end());
		}

		double fastEnergy(Store& store, VectorXd& dFvec){

			return store.alpha_neo*0.5*dFvec.transpose()*store.fastMuscles*dFvec;
		}

		void fastGradient(Store& store, VectorXd& grad){
			grad += store.alpha_neo*store.fastMuscles*store.dFvec;
		}

		double energy(Store& store, VectorXd& dFvec){
			double MuscleEnergy = 0;

			#pragma omp parallel for
			for(int t=0; t<dFvec.size()/9; t++){
				Matrix3d F = Map<Matrix3d>(dFvec.segment<9>(9*t).data()).transpose();
				Vector3d y = store.Uvec.row(t).transpose();
				Vector3d z = F*y;
				double W = 0.5*store.muscle_mag[t]*(z.dot(z));
				MuscleEnergy += W;
			}
			return store.alpha_neo*MuscleEnergy;
		}

		void gradient(Store& store, VectorXd& grad){

			for(int t=0; t<store.T.rows(); t++){
				double s1 = store.dFvec[9*t+0];
				double s2 = store.dFvec[9*t+1];
				double s3 = store.dFvec[9*t+2];
				double s4 = store.dFvec[9*t+3];
				double s5 = store.dFvec[9*t+4];
				double s6 = store.dFvec[9*t+5];
				double s7 = store.dFvec[9*t+6];
				double s8 = store.dFvec[9*t+7];
				double s9 = store.dFvec[9*t+8];
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

				grad.segment<9>(9*t) += store.alpha_neo*tet_grad;
			}
		}

		VectorXd fd_gradient(Store& store){
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

	}
}

#endif