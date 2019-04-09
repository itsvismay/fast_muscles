#ifndef ELASTIC
#define ELASTIC

#include "mesh.h"
#include "math.h"

using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;

class Elastic
{

protected:
	double muscle_fibre_mag = 1.5e6;
	double rho = 6.4; 
	VectorXd sW1, sW2, sW3, sW4, sW5, sW6, muscle_forces, elastic_forces;
	std::vector<int> contract_muscles = {};

	std::vector<MatrixXd> aFastMuscles;

public:
	Elastic(Mesh& m, double strength=10000, std::vector<int> tocontract={0}){
		if(m.T().rows()*6 == m.red_s().size()){
			sW1 = VectorXd::Zero(m.red_s().size());
			sW2 = VectorXd::Zero(m.red_s().size());
			sW3 = VectorXd::Zero(m.red_s().size());
			sW4 = VectorXd::Zero(m.red_s().size());
			sW5 = VectorXd::Zero(m.red_s().size());
			sW6 = VectorXd::Zero(m.red_s().size());
		}

		muscle_forces = VectorXd::Zero(m.red_s().size());
		elastic_forces = VectorXd::Zero(m.red_s().size());

		muscle_fibre_mag = strength;
		for(int i=0; i<tocontract.size(); i++){
			contract_muscles.push_back(tocontract[i]);
		}


		cout<<"Pre-process Muscles"<<endl;
		setupFastMuscles(m);
	}

	void setupFastMuscles(Mesh& mesh){
		for(int m=0; m<mesh.muscle_vecs().size(); m++){
			//for each muscle, preprocess the muscle energy equation
			std::vector<Trip> uS_trips;
			for(int i=0; i<mesh.muscle_vecs()[m].size(); i++){
				int t = mesh.muscle_vecs()[m][i];
				if(mesh.relativeStiffness()[t]>10){
					continue;
				}
				Vector3d u = mesh.Uvecs().row(t);
				uS_trips.push_back(Trip(3*t+0, 6*t+0 , u[0]));
				uS_trips.push_back(Trip(3*t+0, 6*t+3 , u[1]));
				uS_trips.push_back(Trip(3*t+0, 6*t+4 , u[2]));

				uS_trips.push_back(Trip(3*t+1, 6*t+3 , u[0]));
				uS_trips.push_back(Trip(3*t+1, 6*t+1 , u[1]));
				uS_trips.push_back(Trip(3*t+1, 6*t+5 , u[2]));

				uS_trips.push_back(Trip(3*t+2, 6*t+4 , u[0]));
				uS_trips.push_back(Trip(3*t+2, 6*t+5 , u[1]));
				uS_trips.push_back(Trip(3*t+2, 6*t+2 , u[2]));
				
			}
			SparseMatrix<double> uS(3*mesh.T().rows(), 6*mesh.T().rows());
			uS.setFromTriplets(uS_trips.begin(), uS_trips.end());
			aFastMuscles.push_back((uS*mesh.sW()).transpose()*(uS*mesh.sW()));

		}
	}

	double MuscleElementEnergy(const VectorXd& w1, const VectorXd& w2, const VectorXd& w3, const VectorXd& w4, const VectorXd& w5, const VectorXd& w6,  const VectorXd& rs, Vector3d& u){
		double s1 = w1.dot(rs);
		double s2 = w2.dot(rs);
		double s3 = w3.dot(rs);
		double s4 = w4.dot(rs);
		double s5 = w5.dot(rs);
		double s6 = w6.dot(rs);
		double u1 = u[0];
		double u2 = u[1];
		double u3 = u[2];


		double W1 = 0.5*muscle_fibre_mag*((s5*u1 + s6*u2 + s3*u3)*(s5*u1 + s6*u2 + s3*u3) + (s1*u1 + s4*u2 + s5*u3)*(s1*u1 + s4*u2 + s5*u3) + 
   					(s4*u1 + s2*u2 + s6*u3)*(s4*u1 + s2*u2 + s6*u3));
		return W1;
	}

	double MuscleEnergy(Mesh& mesh){
		double En = 0;
		VectorXd& rs = mesh.red_s();
		VectorXd& bones = mesh.bones();
		
		// for(int q=0; q<contract_muscles.size(); q++){
		// 	if(contract_muscles[q]>=mesh.muscle_vecs().size()){
		// 		continue;
		// 	}
		// 	for(int i=0; i<mesh.muscle_vecs()[contract_muscles[q]].size(); i++){
		// 		int t = mesh.muscle_vecs()[contract_muscles[q]][i];
		// 		// if(bones[t]>=0){
		// 		// 	continue;
		// 		// }

	 //            Vector3d u = mesh.Uvecs().row(t);
		        

		// 		if(rs.size()==6*mesh.T().rows()){
		// 			sW1[6*t+0] += 1;
		// 			sW2[6*t+1] += 1;
		// 			sW3[6*t+2] += 1;
		// 			sW4[6*t+3] += 1;
		// 			sW5[6*t+4] += 1;
		// 			sW6[6*t+5] += 1;
		//         	En += MuscleElementEnergy(sW1,sW2,sW3,sW4,sW5,sW6, rs, u);
		//         	sW1[6*t+0] -= 1;
		// 			sW2[6*t+1] -= 1;
		// 			sW3[6*t+2] -= 1;
		// 			sW4[6*t+3] -= 1;
		// 			sW5[6*t+4] -= 1;
		// 			sW6[6*t+5] -= 1;
		// 		}else{
	 //            	En += MuscleElementEnergy(mesh.sW().row(6*t+0),mesh.sW().row(6*t+1),mesh.sW().row(6*t+2),mesh.sW().row(6*t+3),mesh.sW().row(6*t+4),mesh.sW().row(6*t+5), rs, u);
					
		// 		}
		// 	}
		// }
		for(int q=0; q<contract_muscles.size(); q++){
			if(contract_muscles[q]>=mesh.muscle_vecs().size()){
				continue;
			}
			En += 0.5*muscle_fibre_mag*mesh.red_s().transpose()*aFastMuscles[contract_muscles[q]]*mesh.red_s();
		}
		return En;
	}

	void MuscleElementForce(VectorXd& force, const VectorXd& w1, const VectorXd& w2, const VectorXd& w3, const VectorXd& w4, const VectorXd& w5, const VectorXd& w6,  const VectorXd& rs, Vector3d& u){
		double s1 = w1.dot(rs);
		double s2 = w2.dot(rs);
		double s3 = w3.dot(rs);
		double s4 = w4.dot(rs);
		double s5 = w5.dot(rs);
		double s6 = w6.dot(rs);
		double u1 = u[0];
		double u2 = u[1];
		double u3 = u[2];

		double a = muscle_fibre_mag;
		double t_0 = w6.dot(rs);
		double t_1 = w5.dot(rs);
		double t_2 = w4.dot(rs);
		double t_3 = w2.dot(rs);
		double t_4 = w3.dot(rs);
		double t_5 = (((u1 * t_1) + (u2 * t_0)) + (u3 * t_4));
		double t_6 = w1.dot(rs);
		double t_7 = (((u1 * t_6) + (u2 * t_2)) + (u3 * t_1));
		double t_8 = (((u1 * t_2) + (u2 * t_3)) + (u3 * t_0));
		double t_9 = (0.5 * a);
		double t_10 = rs.dot(w6);
		double t_11 = (t_9 * (u1 * u2));
		double t_12 = (t_9 * std::pow(u2 , 2));
		double t_13 = rs.dot(w4);
		double t_14 = (t_9 * (u2 * u3));
		double t_15 = rs.dot(w5);
		double t_16 = rs.dot(w2);
		double t_17 = rs.dot(w3);
		double t_18 = (((u1 * t_15) + (u2 * t_10)) + (u3 * t_17));
		double t_19 = (t_9 * std::pow(u1 , 2));
		double t_20 = rs.dot(w1);
		double t_21 = (t_9 * (u1 * u3));
		double t_22 = (((u1 * t_20) + (u2 * t_13)) + (u3 * t_15));
		double t_23 = (t_11 * t_13);
		double t_24 = (((u1 * t_13) + (u2 * t_16)) + (u3 * t_10));
		double t_25 = (t_21 * t_15);
		double t_26 = (t_9 * std::pow(u3 , 2));
		double t_27 = (t_14 * t_10);
		// double functionValue = (t_9 * (((u2 * (((t_5 * t_0) + (t_7 * t_2)) + (t_8 * t_3))) + (u1 * (((t_5 * t_1) + (t_7 * t_6)) + (t_8 * t_2)))) + (u3 * (((t_5 * t_4) + (t_7 * t_1)) + (t_8 * t_0)))));
		force += (((((((((((((((((((((((((((((((((((((t_11 * t_10) * w5) + ((t_12 * t_10) * w6)) + (t_27 * w3)) + ((t_9 * (u2 * t_18)) * w6)) + (t_23 * w1)) + ((t_12 * t_13) * w4)) + ((t_14 * t_13) * w5)) + ((t_9 * (u2 * t_22)) * w4)) + ((t_11 * t_16) * w4)) + ((t_12 * t_16) * w2)) + ((t_14 * t_16) * w6)) + ((t_9 * (u2 * t_24)) * w2)) + ((t_19 * t_15) * w5)) + ((t_11 * t_15) * w6)) + (t_25 * w3)) + ((t_9 * (u1 * t_18)) * w5)) + ((t_19 * t_20) * w1)) + ((t_11 * t_20) * w4)) + ((t_21 * t_20) * w5)) + ((t_9 * (u1 * t_22)) * w1)) + ((t_19 * t_13) * w4)) + (t_23 * w2)) + ((t_21 * t_13) * w6)) + ((t_9 * (u1 * t_24)) * w4)) + ((t_21 * t_17) * w5)) + ((t_14 * t_17) * w6)) + ((t_26 * t_17) * w3)) + ((t_9 * (u3 * t_18)) * w3)) + (t_25 * w1)) + ((t_14 * t_15) * w4)) + ((t_26 * t_15) * w5)) + ((t_9 * (u3 * t_22)) * w5)) + ((t_21 * t_10) * w4)) + (t_27 * w2)) + ((t_26 * t_10) * w6)) + ((t_9 * (u3 * t_24)) * w6));

		return;
	}

	void MuscleForce(Mesh& mesh){
		muscle_forces.setZero();
		VectorXd& rs = mesh.red_s();
		VectorXd& bones = mesh.bones();

    	for(int q=0; q<contract_muscles.size(); q++){
			if(contract_muscles[q]>=mesh.muscle_vecs().size()){
				continue;
			}
			muscle_forces += muscle_fibre_mag*aFastMuscles[contract_muscles[q]]*mesh.red_s();
		}
	}

	double StableNeoEnergy(Mesh& mesh){
		double EnMuscle = 0;
		double EnTendon = 0;
		VectorXd& eY = mesh.eYoungs();
		VectorXd& eP = mesh.ePoissons();
		VectorXd& bones = mesh.bones();
		VectorXd& rs = mesh.red_s();

		#pragma omp parallel for
		for(int q=0; q<contract_muscles.size(); q++){
			if(contract_muscles[q]>=mesh.muscle_vecs().size()){
				continue;
			}
			cout<<"contracting "<< contract_muscles[q]<<endl;
			for(int i=0; i<mesh.muscle_vecs()[contract_muscles[q]].size(); i++){
				int t = mesh.muscle_vecs()[contract_muscles[q]][i];
			
	            double C1 = eY[t]/(2.0*(1.0+eP[t]));
	            double D1 = (eY[t]*eP[t])/((1.0+eP[t])*(1.0-2.0*eP[t]));
	            if(rs.size()==6*mesh.T().rows()){
					sW1[6*t+0] += 1;
					sW2[6*t+1] += 1;
					sW3[6*t+2] += 1;
					sW4[6*t+3] += 1;
					sW5[6*t+4] += 1;
					sW6[6*t+5] += 1;
		        	EnMuscle += StableNeoElementEnergy(sW1,sW2,sW3,sW4,sW5,sW6, rs, C1, D1);
		        	sW1[6*t+0] -= 1;
					sW2[6*t+1] -= 1;
					sW3[6*t+2] -= 1;
					sW4[6*t+3] -= 1;
					sW5[6*t+4] -= 1;
					sW6[6*t+5] -= 1;
				}else{
	            	double En = StableNeoElementEnergy(mesh.sW().row(6*t+0),mesh.sW().row(6*t+1),mesh.sW().row(6*t+2),mesh.sW().row(6*t+3),mesh.sW().row(6*t+4),mesh.sW().row(6*t+5), rs, C1, D1);
					if(mesh.relativeStiffness()[t]>100){
						EnTendon += En;
					}else{
						EnMuscle += En;
					}
				}
			}
		}
		std::cout<<"Muscle Energy: "<<EnMuscle<<std::endl;
		std::cout<<"Tendon Energy: "<<EnTendon<<std::endl;
		return EnMuscle + EnTendon;
	}

	double StableNeoElementEnergy(const VectorXd& w0, const VectorXd& w1, const VectorXd& w2, const VectorXd& w3, const VectorXd& w4, const VectorXd& w5,  const VectorXd& rs, double C1, double D1){
		double s0 = w0.dot(rs);
		double s1 = w1.dot(rs);
		double s2 = w2.dot(rs);
		double s3 = w3.dot(rs);
		double s4 = w4.dot(rs);
		double s5 = w5.dot(rs);
		
		double I1 = s0*s0 + s1*s1 + s2*s2 + 2*s3*s3 + 2*s4*s4 + 2*s5*s5;
		double J = s0*s1*s2 - s2*s3*s3 - s1*s4*s4 + 2*s3*s4*s5 - s0*s5*s5;


		if(s0<0 || s1<0 || s2<0 || J< 0){
			return 1e40;
		}

		double alpha = (1 + (C1/D1) - (C1/(D1*4)));
		double W = 0.5*C1*(I1 -3) + 0.5*D1*(J-alpha)*(J-alpha) - 0.5*C1*log(I1 + 1);
		
		// if(W<-1e-5){
		// 	std::cout<<"Stable Neo: Negative  energy"<<std::endl;
		// 	std::cout<<"W: "<<W<<std::endl;
		// 	std::cout<<"S: "<<s0<<", "<<s1<<", "<<s2<<", "<<s3<<", "<<s4<<", "<<s5<<std::endl;
		// 	std::cout<<"I1, J, log(I1 +1): "<<I1<<", "<<J<<", "<<log(I1+1)<<std::endl;
		// 	std::cout<<"Term1: "<<0.5*C1*(I1 -3)<<std::endl;
		// 	std::cout<<"Term2: "<<0.5*D1*(J-alpha)*(J-alpha)<<std::endl;
		// 	std::cout<<"Term3: "<<0.5*C1*log(I1 + 1)<<std::endl;
		// 	exit(0);
		// }else if(W<0){
		// 	return 0;
		// }

		if (W!=W){
			cout<<"NAN in Stable Neo energy"<<endl;
			cout<<I1<<endl;
			cout<<J<<endl;
			cout<<w0.transpose()<<endl;
			cout<<w1.transpose()<<endl;
			cout<<w2.transpose()<<endl;
			cout<<w3.transpose()<<endl;
			cout<<w4.transpose()<<endl;
			cout<<w5.transpose()<<endl;
			exit(0);
		}
		return W;
	}

	double StableNeoElementForce(VectorXd& force, const VectorXd& w0, const VectorXd& w1, const VectorXd& w2, const VectorXd& w3, const VectorXd& w4, const VectorXd& w5,  const VectorXd& rs, double C1, double D1){
		
		double alpha = (1 + (C1/D1) - (C1/(D1*4)));

		double t_0 = w0.dot(rs);
		double t_1 = w1.dot(rs);
		double t_2 = w2.dot(rs);
		double t_3 = w3.dot(rs);
		double t_4 = w4.dot(rs);
		double t_5 = w5.dot(rs);
		double t_6 = (2 * t_3);
		double t_7 = (t_0 * t_0);
		double t_8 = (t_1 * t_1);
		double t_9 = (t_2 * t_2);
		double t_10 = (t_6 * t_3);
		double t_11 = ((2 * t_4) * t_4);
		double t_12 = ((2 * t_5) * t_5);
		double t_13 = (2 * C1);
		double t_14 = rs.dot(w0);
		double t_15 = rs.dot(w1);
		double t_16 = rs.dot(w2);
		double t_17 = rs.dot(w3);
		double t_18 = rs.dot(w4);
		double t_19 = rs.dot(w5);
		double t_20 = (2 * t_17);
		double t_21 = (((((((t_14 * t_15) * t_16) - (t_16 * (t_17 * t_17))) - (t_15 * (t_18 * t_18))) + (t_20 * (t_18 * t_19))) - (t_14 * (t_19 * t_19))) - alpha);
		double t_22 = (D1 * t_21);
		double t_23 = (t_22 * t_14);
		double t_24 = ((2 * D1) * t_21);
		double t_25 = (t_24 * t_18);
		double t_26 = (t_24 * t_19);
		double t_27 = ((((((1 + (t_14 * t_14)) + (t_15 * t_15)) + (t_16 * t_16)) + (t_20 * t_17)) + ((2 * t_18) * t_18)) + ((2 * t_19) * t_19));
		double t_28 = (C1 / t_27);
		double t_29 = (t_13 / t_27);
		double functionValue = ((((0.5 * C1) * ((((((t_7 - 3) + t_8) + t_9) + t_10) + t_11) + t_12)) + ((0.5 * D1) * std::pow((((((((t_0 * t_1) * t_2) - (t_2 * (t_3 * t_3))) - (t_1 * (t_4 * t_4))) + (t_6 * (t_4 * t_5))) - (t_0 * (t_5 * t_5))) - alpha), 2))) - (0.5 * (C1 * log(((((((1 + t_7) + t_8) + t_9) + t_10) + t_11) + t_12)))));
	    force += (((((((((((((((((C1 * t_14) * w0) + ((C1 * t_15) * w1)) + ((C1 * t_16) * w2)) + ((t_13 * t_17) * w3)) + ((t_13 * t_18) * w4)) + ((t_13 * t_19) * w5)) + ((t_22 * t_15) * (t_16 * w0))) + (t_23 * (t_16 * w1))) + (t_23 * (t_15 * w2))) - (((t_22 * t_17) * (t_17 * w2)) + ((t_24 * t_17) * (t_16 * w3)))) - (((t_22 * t_18) * (t_18 * w1)) + (t_25 * (t_15 * w4)))) + (t_25 * (t_19 * w3))) + (t_26 * (t_17 * w4))) + (t_25 * (t_17 * w5))) - (((t_22 * t_19) * (t_19 * w0)) + (t_26 * (t_14 * w5)))) - (((((((t_28 * t_14) * w0) + ((t_28 * t_15) * w1)) + ((t_28 * t_16) * w2)) + ((t_29 * t_17) * w3)) + ((t_29 * t_18) * w4)) + ((t_29 * t_19) * w5)));



		double s0 = w0.dot(rs);
		double s1 = w1.dot(rs);
		double s2 = w2.dot(rs);
		double s3 = w3.dot(rs);
		double s4 = w4.dot(rs);
		double s5 = w5.dot(rs);
		double I1 = s0*s0 + s1*s1 + s2*s2 + 2*s3*s3 + 2*s4*s4 + 2*s5*s5;
		double J = s0*s1*s2 - s2*s3*s3 - s1*s4*s4 + 2*s3*s4*s5 - s0*s5*s5;
		if(s0<0 || s1<0 || s2<0 || J< 0){
			return 1e40;
		}
		double W = 0.5*C1*(I1 -3) + 0.5*D1*(J-alpha)*(J-alpha) - 0.5*C1*log(I1 + 1);
		// if(fabs(W- functionValue) > 1e-5){
		// 	cout<<"Energy values dont agree"<<endl;
		// 	cout<<"W: "<<W<<endl;
		// 	cout<<"func: "<<functionValue<<endl;
		// 	exit(0);
		// }
		
	}

	double StableNeoForce(Mesh& mesh){
		elastic_forces.setZero();
		VectorXd& eY = mesh.eYoungs();
		VectorXd& eP = mesh.ePoissons();
		VectorXd& bones = mesh.bones();
		VectorXd& rs = mesh.red_s();

		Matrix3d c;
		#pragma omp parallel for
		for(int q=0; q<contract_muscles.size(); q++){
			if(contract_muscles[q]>=mesh.muscle_vecs().size()){
				continue;
			}
			for(int i=0; i<mesh.muscle_vecs()[contract_muscles[q]].size(); i++){
				int t = mesh.muscle_vecs()[contract_muscles[q]][i];
				
	            double C1 = eY[t]/(2.0*(1.0+eP[t]));
	            double D1 = (eY[t]*eP[t])/((1.0+eP[t])*(1.0-2.0*eP[t]));
	            if(rs.size()==6*mesh.T().rows()){
					sW1[6*t+0] += 1;
					sW2[6*t+1] += 1;
					sW3[6*t+2] += 1;
					sW4[6*t+3] += 1;
					sW5[6*t+4] += 1;
					sW6[6*t+5] += 1;
		        	StableNeoElementForce(elastic_forces, sW1,sW2,sW3,sW4,sW5,sW6, rs, C1, D1);
		        	sW1[6*t+0] -= 1;
					sW2[6*t+1] -= 1;
					sW3[6*t+2] -= 1;
					sW4[6*t+3] -= 1;
					sW5[6*t+4] -= 1;
					sW6[6*t+5] -= 1;
				}else{
	            	StableNeoElementForce(elastic_forces, mesh.sW().row(6*t+0),mesh.sW().row(6*t+1),mesh.sW().row(6*t+2),mesh.sW().row(6*t+3),mesh.sW().row(6*t+4),mesh.sW().row(6*t+5), rs, C1, D1);
				}
			}
		}
	}

	double WikipediaElementEnergy(const VectorXd& w0, const VectorXd& w1, const VectorXd& w2, const VectorXd& w3, const VectorXd& w4, const VectorXd& w5,  const VectorXd& rs, double C1, double D1){
		double s0 = w0.dot(rs);
		double s1 = w1.dot(rs);
		double s2 = w2.dot(rs);
		double s3 = w3.dot(rs);
		double s4 = w4.dot(rs);
		double s5 = w5.dot(rs);
		double I1 = s0*s0 + s1*s1 + s2*s2 + 2*s3*s3 + 2*s4*s4 + 2*s5*s5;
		double J = s0*s1*s2 - s2*s3*s3 - s1*s4*s4 + 2*s3*s4*s5 - s0*s5*s5;
		//Energy Terms for MatrixCalc.org
		//I1 = w0'*rs*w0'*rs + w1'*rs*w1'*rs + w2'*rs*w2'*rs + 2*w3'*rs*w3'*rs + 2*w4'*rs*w4'*rs + 2*w5'*rs*w5'*rs
		//J =  w0'*rs*w1'*rs*w2'*rs - w2'*rs*w3'*rs*w3'*rs - w1'*rs*w4'*rs*w4'*rs + 2*w3'*rs*w4'*rs*w5'*rs - w0'*rs*w5'*rs*w5'*rs
		//Term1 (C1*((J^-2/3) * I1   - 3))
		//Term2 (D1*(J-1)*(J-1))

		if(s0<0 || s1<0 || s2<0 || J< 0){
			return 1e40;
		}

		double I1bar = std::pow(J, -2/3.0)*I1;
		double W = C1*(I1bar -3) + D1*(J-1)*(J-1);
		if(W<-1e-5){
			std::cout<<"Negative energy"<<std::endl;
			std::cout<<"W: "<<W<<std::endl;
			std::cout<<"I1, log(J), I1bar: "<<I1<<", "<<log(J)<<", "<<I1bar<<std::endl;
			std::cout<<"term1: "<<C1*(I1bar -3)<<std::endl;
			std::cout<<"term2: "<<D1*(J-1)*(J-1)<<std::endl;
			exit(0);
		}else if(W<0){
			return 0;
		}

		if (W!=W){
			cout<<"NAN in Wikipedia energy"<<endl;
			cout<<I1<<endl;
			cout<<J<<endl;
			cout<<I1bar<<endl;
			cout<<w0.transpose()<<endl;
			cout<<w1.transpose()<<endl;
			cout<<w2.transpose()<<endl;
			cout<<w3.transpose()<<endl;
			cout<<w4.transpose()<<endl;
			cout<<w5.transpose()<<endl;
			exit(0);
		}
		return W;
	}

	double WikipediaEnergy(Mesh& mesh){
		double En = 0;
		VectorXd& eY = mesh.eYoungs();
		VectorXd& eP = mesh.ePoissons();
		VectorXd& bones = mesh.bones();
		VectorXd& rs = mesh.red_s();

		#pragma omp parallel for
		for(int t =0; t<mesh.T().rows(); t++){
			if(mesh.bones()[t]>=0){
				continue;
			}
            double C1 = 0.5*eY[t]/(2.0*(1.0+eP[t]));
            double D1 = 0.5*(eY[t]*eP[t])/((1.0+eP[t])*(1.0-2.0*eP[t]));
            if(rs.size()==6*mesh.T().rows()){
				sW1[6*t+0] += 1;
				sW2[6*t+1] += 1;
				sW3[6*t+2] += 1;
				sW4[6*t+3] += 1;
				sW5[6*t+4] += 1;
				sW6[6*t+5] += 1;
	        	En += WikipediaElementEnergy(sW1,sW2,sW3,sW4,sW5,sW6, rs, C1, D1);
	        	sW1[6*t+0] -= 1;
				sW2[6*t+1] -= 1;
				sW3[6*t+2] -= 1;
				sW4[6*t+3] -= 1;
				sW5[6*t+4] -= 1;
				sW6[6*t+5] -= 1;
			}else{
            	En += WikipediaElementEnergy(mesh.sW().row(6*t+0),mesh.sW().row(6*t+1),mesh.sW().row(6*t+2),mesh.sW().row(6*t+3),mesh.sW().row(6*t+4),mesh.sW().row(6*t+5), rs, C1, D1);
			}
		}
		return En;
	}

	void WikipediaElementForce(VectorXd& force, const VectorXd& w0, const VectorXd& w1, const VectorXd& w2, const VectorXd& w3, const VectorXd& w4, const VectorXd& w5,  const VectorXd& rs, double C1, double D1){
		double t_0 = w2.dot(rs);
		double t_1 = w3.dot(rs);
		double t_2 = w1.dot(rs);
		double t_3 = w4.dot(rs);
		double t_4 = w0.dot(rs);
		double t_5 = w5.dot(rs);
		double t_6 = (2.0 * t_1);
		double t_7 = rs.dot(w2);
		double t_8 = rs.dot(w3);
		double t_9 = rs.dot(w1);
		double t_10 = rs.dot(w4);
		double t_11 = rs.dot(w0);
		double t_12 = rs.dot(w5);
		double t_13 = (2.0 / 3.0);
		double t_14 = -t_13;
		double t_15 = (2.0 * C1);
		double t_16 = (2.0 * t_8);
		double t_17 = ((((((t_11 * t_9) * t_7) - (t_7 * (t_8 * t_8))) - (t_9 * (t_10 * t_10))) + (t_16 * (t_10 * t_12))) - (t_11 * (t_12 * t_12)));
		double t_18 = (std::pow(t_17, -(1.0 + t_13)) * ((((((t_11 * t_11) + (t_9 * t_9)) + (t_7 * t_7)) + (t_16 * t_8)) + ((2 * t_10) * t_10)) + ((2 * t_12) * t_12)));
		double t_19 = ((t_15 * t_18) / 3.0);
		double t_20 = (t_19 * t_11);
		double t_21 = (4.0 * C1);
		double t_22 = ((t_21 * t_18) / 3.0);
		double t_23 = (t_22 * t_10);
		double t_24 = (t_22 * t_12);
		double t_25 = std::pow(t_17, t_14);
		double t_26 = (t_15 * t_25);
		double t_27 = (t_21 * t_25);
		// functionValue = (C1 * (((((((((t_4 * t_2) * t_0) - (t_0 * (t_1 * t_1))) - (t_2 * (t_3 * t_3))) + (t_6 * (t_3 * t_5))) - (t_4 * (t_5 * t_5))) ** t_14) * ((((((t_4 * t_4) + (t_2 * t_2)) + (t_0 * t_0)) + (t_6 * t_1)) + ((2 * t_3) * t_3)) + ((2 * t_5) * t_5))) - 3))
		force += ((((((((t_26 * t_11) * w0) - ((((((((((t_19 * t_9) * (t_7 * w0)) + (t_20 * (t_7 * w1))) + (t_20 * (t_9 * w2))) - (((t_19 * t_8) * (t_8 * w2)) + ((t_22 * t_8) * (t_7 * w3)))) - (((t_19 * t_10) * (t_10 * w1)) + (t_23 * (t_9 * w4)))) + (t_23 * (t_12 * w3))) + (t_24 * (t_8 * w4))) + (t_23 * (t_8 * w5))) - (((t_19 * t_12) * (t_12 * w0)) + (t_24 * (t_11 * w5))))) + ((t_26 * t_9) * w1)) + ((t_26 * t_7) * w2)) + ((t_27 * t_8) * w3)) + ((t_27 * t_10) * w4)) + ((t_27 * t_12) * w5));

		t_0 = w2.dot(rs);
		t_1 = w3.dot(rs);
		t_2 = w1.dot(rs);
		t_3 = w4.dot(rs);
		t_4 = w0.dot(rs);
		t_5 = w5.dot(rs);
		t_6 = rs.dot(w2);
		t_7 = rs.dot(w3);
		t_8 = rs.dot(w1);
		t_9 = rs.dot(w4);
		t_10 = rs.dot(w0);
		t_11 = rs.dot(w5);
		t_12 = (((((((t_10 * t_8) * t_6) - 1) - (t_6 * (t_7 * t_7))) - (t_8 * (t_9 * t_9))) + ((2 * t_7) * (t_9 * t_11))) - (t_10 * (t_11 * t_11)));
		t_13 = ((2 * D1) * t_12);
		t_14 = (t_13 * t_10);
		t_15 = ((4 * D1) * t_12);
		t_16 = (t_15 * t_9);
		t_17 = (t_15 * t_11);
		// functionValue = (D1 * ((((((((t_4 * t_2) * t_0) - 1) - (t_0 * (t_1 * t_1))) - (t_2 * (t_3 * t_3))) + ((2 * t_1) * (t_3 * t_5))) - (t_4 * (t_5 * t_5))) ** 2))
		force += ((((((((((t_13 * t_8) * (t_6 * w0)) + (t_14 * (t_6 * w1))) + (t_14 * (t_8 * w2))) - (((t_13 * t_7) * (t_7 * w2)) + ((t_15 * t_7) * (t_6 * w3)))) - (((t_13 * t_9) * (t_9 * w1)) + (t_16 * (t_8 * w4)))) + (t_16 * (t_11 * w3))) + (t_17 * (t_7 * w4))) + (t_16 * (t_7 * w5))) - (((t_13 * t_11) * (t_11 * w0)) + (t_17 * (t_10 * w5))));
		
	}

	void WikipediaForce(Mesh& mesh){
		elastic_forces.setZero();
		VectorXd& eY = mesh.eYoungs();
		VectorXd& eP = mesh.ePoissons();
		VectorXd& bones = mesh.bones();
		VectorXd& rs = mesh.red_s();

		Matrix3d c;
		#pragma omp parallel for
		for(int t =0; t<mesh.T().rows(); t++){
			if(mesh.bones()[t]>=0){
				continue;
			}
            double C1 = 0.5*eY[t]/(2.0*(1.0+eP[t]));
            double D1 = 0.5*(eY[t]*eP[t])/((1.0+eP[t])*(1.0-2.0*eP[t]));
            if(rs.size()==6*mesh.T().rows()){
				sW1[6*t+0] += 1;
				sW2[6*t+1] += 1;
				sW3[6*t+2] += 1;
				sW4[6*t+3] += 1;
				sW5[6*t+4] += 1;
				sW6[6*t+5] += 1;
	        	WikipediaElementForce(elastic_forces, sW1,sW2,sW3,sW4,sW5,sW6, rs, C1, D1);
	        	sW1[6*t+0] -= 1;
				sW2[6*t+1] -= 1;
				sW3[6*t+2] -= 1;
				sW4[6*t+3] -= 1;
				sW5[6*t+4] -= 1;
				sW6[6*t+5] -= 1;
			}else{
            	WikipediaElementForce(elastic_forces, mesh.sW().row(6*t+0),mesh.sW().row(6*t+1),mesh.sW().row(6*t+2),mesh.sW().row(6*t+3),mesh.sW().row(6*t+4),mesh.sW().row(6*t+5), rs, C1, D1);
			}
		}
	}


	double Energy(Mesh& m){
		double Elas =  StableNeoEnergy(m);
		double Muscle = MuscleEnergy(m);
		cout<<"Muscle Energy: "<< Muscle<<endl;
		return Elas + Muscle;
	}

	VectorXd PEGradient(Mesh& m){
		StableNeoForce(m);
		MuscleForce(m);
		return muscle_forces+elastic_forces;
	}

	void changeFiberMag(double multiplier){
		muscle_fibre_mag += multiplier;
		cout<<"muscle fiber mag"<<endl;
		cout<<muscle_fibre_mag<<endl;
	}

	void changeContractMuscle(int i){
		cout<<"contracting muscles"<<endl;
		if(i==0){
			contract_muscles.clear();
		}else{
			contract_muscles.push_back(i-1);
		}
		for(int c=0; c<contract_muscles.size(); c++){
			cout<<contract_muscles[c]<<", ";
		}
		cout<<endl;
	}

	template<typename DataType>
	inline DataType stablePow(DataType a, DataType b) {
		return static_cast<DataType> (std::pow(std::cbrt(static_cast<DataType>(a)),static_cast<DataType>(b)));
	}

};


#endif