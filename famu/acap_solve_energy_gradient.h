#ifndef ACAP_SOLVE_ENERGY_GRADIENT
#define ACAP_SOLVE_ENERGY_GRADIENT

#include "store.h"
#include <igl/writeDMAT.h>
#include <igl/readDMAT.h>
#include<Eigen/LU>


using namespace famu;
using namespace std;
using Store = famu::Store;

namespace famu
{
	namespace acap
	{
		std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> & M){
			std::vector<Eigen::Triplet<double>> v;
			for(int i = 0; i < M.outerSize(); i++){
				for(typename Eigen::SparseMatrix<double>::InnerIterator it(M,i); it; ++it){	
					v.emplace_back(it.row(),it.col(),it.value());
				}
			}
			return v;
		}

		double energy(Store& store){
			SparseMatrix<double> DS = store.D*store.S;
			double E1 =  0.5*(store.D*store.S*(store.x+store.x0) - store.dF*store.DSx0).squaredNorm();

			double E2 = 0.5*store.x.transpose()*store.StDtDS*store.x;
			double E3 = store.x0.transpose()*store.StDtDS*store.x;
			double E4 = 0.5*store.x0.transpose()*store.StDtDS*store.x0;
			double E5 = -store.x.transpose()*DS.transpose()*store.dF*DS*store.x0;
			double E6 = -store.x0.transpose()*DS.transpose()*store.dF*DS*store.x0;
			double E7 = 0.5*(store.dF*store.DSx0).transpose()*(store.dF*store.DSx0);
			double E8 = E2+E3+E4+E5+E6+E7;
			assert(fabs(E1 - E8)< 1e-6);
			return E1;
		}

		double fastEnergy(Store& store, VectorXd& dFvec){
			double E1 = 0.5*store.x0tStDtDSx0;
			double E2 = store.x0tStDtDSY.dot(store.x);
			double E3 = 0.5*store.x.transpose()*store.YtStDtDSY*store.x;
			double E4 = -store.x0tStDt_dF_DSx0.dot(dFvec);
			double E5 = -store.x.transpose()*store.YtStDt_dF_DSx0*dFvec;
			double E6 = 0.5*dFvec.transpose()*store.x0tStDt_dF_dF_DSx0*dFvec;
		 
			double E9 = E1+E2+E3+E4+E5+E6;
			return E9;
		}

		void fastGradient(Store& store, VectorXd& grad){
			grad = -store.x0tStDt_dF_DSx0;
			grad += -store.x.transpose()*store.YtStDt_dF_DSx0;
			grad += store.dFvec.transpose()*store.x0tStDt_dF_dF_DSx0;
			grad -= store.Bf.transpose()*store.lambda2;
		}

		void fastHessian(Store& store, SparseMatrix<double>& hess){
			hess.setZero();
			hess = store.x0tStDt_dF_dF_DSx0; //PtZtZP

			if(!store.jinput["woodbury"]){
				hess -= store.YtStDt_dF_DSx0.transpose()*store.JacdxdF;
			}

		}

		VectorXd fd_gradient(Store& store){
			VectorXd fake = VectorXd::Zero(store.dFvec.size());
			VectorXd dFvec = store.dFvec;
			double eps = 0.00001;
			for(int i=0; i<dFvec.size(); i++){
				dFvec[i] += 0.5*eps;
				double Eleft = fastEnergy(store, dFvec);
				dFvec[i] -= 0.5*eps;

				dFvec[i] -= 0.5*eps;
				double Eright = fastEnergy(store, dFvec);
				dFvec[i] += 0.5*eps;
				fake[i] = (Eleft - Eright)/eps;
			}
			return fake;
		}

		MatrixXd fd_hessian(Store& store){
			MatrixXd fake = MatrixXd::Zero(store.dFvec.size(), store.dFvec.size());
			VectorXd dFvec = store.dFvec;
			double eps = 1e-3;
			double E0 = fastEnergy(store, dFvec);
			for(int i=0; i<11; i++){
				for(int j=0; j<11; j++){
					dFvec[i] += eps;
					dFvec[j] += eps;
					double Eij = fastEnergy(store, dFvec);
					dFvec[i] -= eps;
					dFvec[j] -= eps;

					dFvec[i] += eps;
					double Ei = fastEnergy(store, dFvec);
					dFvec[i] -=eps;

					dFvec[j] += eps;
					double Ej = fastEnergy(store, dFvec);
					dFvec[j] -=eps;

					fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
				}
			}
			return fake;
		}

		void solve(Store& store, VectorXd& dFvec){

			VectorXd top = store.YtStDt_dF_DSx0*dFvec - store.x0tStDtDSY;
			VectorXd zer = VectorXd::Zero(store.JointConstraints.rows());
			VectorXd bone_def = store.Bf*store.dFvec - store.BfI0;

			VectorXd KKT_right(top.size() + zer.size() + bone_def.size());
			KKT_right<<top, zer, bone_def;
			// igl::writeDMAT("rhs.dmat", KKT_right);

			VectorXd result = store.ACAP_KKT_SPLU.solve(KKT_right);
			store.x = result.head(top.size());
			store.lambda2 = result.tail(bone_def.size());
		
		}

		void setJacobian(Store& store){
			// std::string outputfile = store.jinput["output"];
			// igl::readDMAT(outputfile+"/dxdF.dmat", store.JacdxdF);
			// igl::readDMAT(outputfile+"/dlambdadF.dmat", store.JacdlambdadF);
			// if(store.JacdxdF.rows()!=0){
			// 	return;
			// }

			//Sparse jacobian
			if(store.jinput["sparseJac"]){

				SparseMatrix<double> rhs = store.YtStDt_dF_DSx0.leftCols(9*store.bone_tets.size());
				MatrixXd top = MatrixXd(rhs);
				MatrixXd zer = MatrixXd(store.JointConstraints.rows(), top.cols());
				MatrixXd bone_def = MatrixXd(store.Bf.leftCols(9*store.bone_tets.size()));
				MatrixXd KKT_right(top.rows() + zer.rows() + bone_def.rows(), top.cols());
				KKT_right<<top,zer, bone_def;

				MatrixXd result = store.ACAP_KKT_SPLU.solve(KKT_right).topRows(top.rows());
				if(result!=result){
					cout<<"ACAP Jacobian result has nans"<<endl;
					exit(0);
				}
				SparseMatrix<double> spRes = result.sparseView();

				std::vector<Trip> res_trips = to_triplets(spRes);
				SparseMatrix<double> spJac(spRes.rows(), store.dFvec.size());
				spJac.setFromTriplets(res_trips.begin(), res_trips.end());
				store.JacdxdF = spJac;
				cout<<"jac dims: "<<store.JacdxdF.rows()<<", "<<store.JacdxdF.cols()<<", "<<store.JacdxdF.nonZeros()<<endl;


			}else{

				if(!store.jinput["woodbury"]){

					SparseMatrix<double> rhs = store.YtStDt_dF_DSx0;
					MatrixXd top = MatrixXd(rhs);
					MatrixXd zer = MatrixXd(store.JointConstraints.rows(), top.cols());
					MatrixXd bone_def = MatrixXd(store.Bf);
					MatrixXd KKT_right(top.rows() + zer.rows()+ bone_def.rows(), top.cols());
					KKT_right<<top,zer, bone_def;

					MatrixXd result = store.ACAP_KKT_SPLU.solve(KKT_right).topRows(top.rows());
					if(result!=result){
						cout<<"ACAP Jacobian result has nans"<<endl;
						exit(0);
					}
					

					// MatrixXd rhs = MatrixXd(store.YtStDt_dF_DSx0);
					// UmfPackLU<SparseMatrix<double>> SPLU;
					// SPLU.compute(store.YtStDtDSY);
					// MatrixXd result = SPLU.solve(rhs);
					// cout<<result.rows()<<", "<<result.cols()<<endl;
					

					SparseMatrix<double> spRes = (result).sparseView();
					store.JacdxdF = spRes;
					cout<<"jac dims: "<<store.JacdxdF.rows()<<", "<<store.JacdxdF.cols()<<", "<<store.JacdxdF.nonZeros()<<endl;

				}
				


				// MatrixXd GtYtStDtDSYG = store.G.transpose()*store.YtStDtDSY*store.G;
				// MatrixXd rhs = store.G.transpose()*store.YtStDt_dF_DSx0;

				// // UmfPackLU<SparseMatrix<double>> SPLU;
				// FullPivLU<MatrixXd> SPLU;
				// SPLU.compute(GtYtStDtDSYG);
				// MatrixXd result = SPLU.solve(rhs);
				// cout<<store.G.rows()<<", "<<store.G.cols()<<endl;
				// cout<<result.rows()<<", "<<result.cols()<<endl;
				// SparseMatrix<double> spG = store.G.sparseView();
				// SparseMatrix<double> spRes = (result).sparseView();
				// store.JacdxdF = spG*spRes;
				// cout<<"jac dims: "<<store.JacdxdF.rows()<<", "<<store.JacdxdF.cols()<<", "<<store.JacdxdF.nonZeros()<<endl;

				// exit(0);
			}

		}

		
	}

}

#endif