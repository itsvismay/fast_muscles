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
	double muscle_fibre_mag = 100000;
	double rho = 6.4; 
	VectorXd forces;

public:
	Elastic(Mesh& m){

		forces.resize(m.red_s().size());
		forces.setZero();
	}

	double MuscleElementEnergy(Matrix3d& c, Vector3d& u){
		return 0.5*muscle_fibre_mag*((c*u).dot(c*u));
	}

	double MuscleEnergy(Mesh& mesh){
		double En = 0;
		VectorXd s = mesh.sW()*mesh.red_s();
		Matrix3d c;
		for(int t=0; t<mesh.T().rows(); t++){
			c.coeffRef(0, 0) = s[6*t + 0];
            c.coeffRef(1, 1) = s[6*t + 1];
            c.coeffRef(2, 2) = s[6*t + 2];
            c.coeffRef(0, 1) = s[6*t + 3];
            c.coeffRef(1, 0) = s[6*t + 3];
            c.coeffRef(0, 2) = s[6*t + 4];
            c.coeffRef(2, 0) = s[6*t + 4];
            c.coeffRef(1, 2) = s[6*t + 5];
            c.coeffRef(2, 1) = s[6*t + 5];
            Vector3d u;
            u<<0,1,0;
            En += MuscleElementEnergy(c, u);
		}
		return En;
	}

	VectorXd MuscleElementForce(Matrix3d& c, Vector3d& u){
		double s1 = c(0,0);
		double s2 = c(1,1);
		double s3 = c(2,2);
		double s4 = c(0,1);
		double s5 = c(0,2);
		double s6 = c(1,2);
		double u1 = u(0);
		double u2 = u(1);
		double u3 = u(2);
		double a = muscle_fibre_mag;

		VectorXd f_m(6);
		f_m << 0.5*a*(s4*u1*u2 + s5*u1*u3 + u1*(2*s1*u1 + s4*u2 + s5*u3)),
		   0.5*a*(s4*u1*u2 + s6*u2*u3 + u2*(s4*u1 + 2*s2*u2 + s6*u3)),
		   0.5*a*(s5*u1*u3 + s6*u2*u3 + u3*(s5*u1 + s6*u2 + 2*s3*u3)),
		   0.5*a*((s6*u1 + s5*u2)*u3 + u2*(s1*u1 + s2*u1 + 2*s4*u2 + s5*u3) + 
		      u1*(2*s4*u1 + s1*u2 + s2*u2 + s6*u3)),
		   0.5*a*(u1*(2*s5*u1 + s6*u2 + s1*u3 + s3*u3) + u2*(s6*u1 + s4*u3) + 
		      u3*(s1*u1 + s3*u1 + s4*u2 + 2*s5*u3)),
		   0.5*a*(u2*(s5*u1 + 2*s6*u2 + s2*u3 + s3*u3) + u1*(s5*u2 + s4*u3) + 
	      	u3*(s4*u1 + s2*u2 + s3*u2 + 2*s6*u3));
		return f_m;
	}

	void MuscleForce(Mesh& mesh){
		VectorXd s = mesh.sW()*mesh.red_s();
		Matrix3d c;
		for(int t=0; t<mesh.T().rows(); t++){
			c.coeffRef(0, 0) = s[6*t + 0];
            c.coeffRef(1, 1) = s[6*t + 1];
            c.coeffRef(2, 2) = s[6*t + 2];
            c.coeffRef(0, 1) = s[6*t + 3];
            c.coeffRef(1, 0) = s[6*t + 3];
            c.coeffRef(0, 2) = s[6*t + 4];
            c.coeffRef(2, 0) = s[6*t + 4];
            c.coeffRef(1, 2) = s[6*t + 5];
            c.coeffRef(2, 1) = s[6*t + 5];
            Vector3d u;
            u<<0,1,0;
            forces.segment<6>(6*t) += MuscleElementForce(c, u);
        }
	}

	double WikipediaElementEnergy(Matrix3d& c, double C1, double D1){
		double I1 = (c.transpose()*c).trace();
		double J = c.determinant();
		double I1bar = std::pow(J, -2/3.0)*I1;

		if(c(0,0)<0 || c(1,1)<0 || c(2,2)<0 || J< 0){
			return 1e40;
		}
		// double W = C1*(I1 -3 -2*log(J)) + D1*(log(J)*log(J));
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
		return W;
	}

	double WikipediaEnergy(Mesh& mesh){
		double En = 0;

		VectorXd& eY = mesh.eYoungs();
		VectorXd& eP = mesh.ePoissons();

		VectorXd s = mesh.sW()*mesh.red_s();
		// std::cout<<s.transpose()<<std::endl;
		Matrix3d c;
		for(int t =0; t<mesh.T().rows(); t++){
			c.coeffRef(0, 0) = s[6*t + 0];
            c.coeffRef(1, 1) = s[6*t + 1];
            c.coeffRef(2, 2) = s[6*t + 2];
            c.coeffRef(0, 1) = s[6*t + 3];
            c.coeffRef(1, 0) = s[6*t + 3];
            c.coeffRef(0, 2) = s[6*t + 4];
            c.coeffRef(2, 0) = s[6*t + 4];
            c.coeffRef(1, 2) = s[6*t + 5];
            c.coeffRef(2, 1) = s[6*t + 5];
            double C1 = 0.5*eY[t]/(2.0*(1.0+eP[t]));
            double D1 = 0.5*(eY[t]*eP[t])/((1.0+eP[t])*(1.0-2.0*eP[t]));
            En += WikipediaElementEnergy(c, C1, D1);
		}
		return En;
	}

	VectorXd WikipediaElementForce(Matrix3d& c, double C1, double D1){
		double s1 = c(0,0);
		double s2 = c(1,1);
		double s3 = c(2,2);
		double s4 = c(0,1);
		double s5 = c(0,2);
		double s6 = c(1,2);
		double J = c.determinant();

		VectorXd f_e(6);
		if(c(0,0)<0 || c(1,1)<0 || c(2,2)<0 || J< 0){
			f_e<<1e40, 1e40, 1e40, 1e40, 1e40, 1e40;
			return f_e;
		}

		// f_e<<C1*(1 - (2*(s2*s3 - pow(s6,2)))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2))) + (2*D1*(s2*s3 - pow(s6,2))*log(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)),
		// C1*(1 - (2*(s1*s3 - pow(s5,2)))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2))) + (2*D1*(s1*s3 - pow(s5,2))*log(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)),
		// C1*(1 - (2*(s1*s2 - pow(s4,2)))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2))) + (2*D1*(s1*s2 - pow(s4,2))*log(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)),
		// (-2*C1*(-2*s3*s4 + 2*s5*s6))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)) + (2*D1*(-2*s3*s4 + 2*s5*s6)*log(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)),
		// (-2*C1*(-2*s2*s5 + 2*s4*s6))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)) + (2*D1*(-2*s2*s5 + 2*s4*s6)*log(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)),
		// (-2*C1*(2*s4*s5 - 2*s1*s6))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)) + (2*D1*(2*s4*s5 - 2*s1*s6)*log(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)))/(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2));
		f_e<<2*D1*(s2*s3 - pow(s6,2))*(-1 + s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - 
       	s1*pow(s6,2)) + C1*((-0.6666666666666666*(pow(s1,2) + pow(s2,2) + pow(s3,2))*
          (s2*s3 - pow(s6,2)))/
        pow(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2),
         1.6666666666666665) + (2*s1)/
        pow(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2),
         0.6666666666666666)),2*D1*(s1*s3 - pow(s5,2))*
     	(-1 + s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)) + 
    	C1*((-0.6666666666666666*(pow(s1,2) + pow(s2,2) + pow(s3,2))*(s1*s3 - pow(s5,2)))/
        pow(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2),
         1.6666666666666665) + (2*s2)/
        pow(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2),
         0.6666666666666666)),2*D1*(s1*s2 - pow(s4,2))*
     	(-1 + s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)) + 
    	C1*((-0.6666666666666666*(pow(s1,2) + pow(s2,2) + pow(s3,2))*(s1*s2 - pow(s4,2)))/
        pow(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2),
         1.6666666666666665) + (2*s3)/
        pow(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2),
         0.6666666666666666)),2*D1*(-2*s3*s4 + 2*s5*s6)*
     	(-1 + s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)) - 
    	(0.6666666666666666*C1*(pow(s1,2) + pow(s2,2) + pow(s3,2))*(-2*s3*s4 + 2*s5*s6))/
     	pow(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2),
      	1.6666666666666665),2*D1*(-2*s2*s5 + 2*s4*s6)*
     	(-1 + s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)) - 
    	(0.6666666666666666*C1*(pow(s1,2) + pow(s2,2) + pow(s3,2))*(-2*s2*s5 + 2*s4*s6))/
     	pow(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2),
      	1.6666666666666665),2*D1*(2*s4*s5 - 2*s1*s6)*
     	(-1 + s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2)) - 
    	(0.6666666666666666*C1*(pow(s1,2) + pow(s2,2) + pow(s3,2))*(2*s4*s5 - 2*s1*s6))/
     	pow(s1*s2*s3 - s3*pow(s4,2) - s2*pow(s5,2) + 2*s4*s5*s6 - s1*pow(s6,2),
      	1.6666666666666665);


		return f_e;
	}

	void WikipediaForce(Mesh& mesh){		
		VectorXd& eY = mesh.eYoungs();
		VectorXd& eP = mesh.ePoissons();

		VectorXd s = mesh.sW()*mesh.red_s();

		Matrix3d c;
		for(int t =0; t<mesh.T().rows(); t++){
			c.coeffRef(0, 0) = s[6*t + 0];
            c.coeffRef(1, 1) = s[6*t + 1];
            c.coeffRef(2, 2) = s[6*t + 2];
            c.coeffRef(0, 1) = s[6*t + 3];
            c.coeffRef(1, 0) = s[6*t + 3];
            c.coeffRef(0, 2) = s[6*t + 4];
            c.coeffRef(2, 0) = s[6*t + 4];
            c.coeffRef(1, 2) = s[6*t + 5];
            c.coeffRef(2, 1) = s[6*t + 5];
            double C1 = 0.5*eY[t]/(2.0*(1.0+eP[t]));
            double D1 = 0.5*(eY[t]*eP[t])/((1.0+eP[t])*(1.0-2.0*eP[t]));
            forces.segment<6>(6*t) += WikipediaElementForce(c, C1, D1);
		}
	}

	double Energy(Mesh& m){
		double Elas =  WikipediaEnergy(m);
		double Muscle = 0;// MuscleEnergy(m);
		// std::cout<<"	elas: "<<Elas<<", muscle: "<<Muscle<<std::endl;
		return Elas + Muscle;
	}

	VectorXd PEGradient(Mesh& m){
		forces.setZero();
		WikipediaForce(m);
		// MuscleForce(m);
		return forces;
	}

};


#endif