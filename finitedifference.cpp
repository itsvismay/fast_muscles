#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>
#include <unsupported/Eigen/NumericalDiff>


#include <json.hpp>

#include "mesh.h"
#include "arap.h"
#include "redArap.h"
#include "elastic.h"
// #include "solver.h"


using json = nlohmann::json;

using namespace Eigen;
using namespace std;
json j_input;

std::vector<int> getMaxVerts_Axis_Tolerance(MatrixXd& mV, int dim, double tolerance=1e-5){
    auto maxX = mV.col(dim).maxCoeff();
    std::vector<int> maxV;
    for(unsigned int ii=0; ii<mV.rows(); ++ii) {

        if(fabs(mV(ii,dim) - maxX) < tolerance) {
            maxV.push_back(ii);
        }
    }
    return maxV;
}

std::vector<int> getMinVerts_Axis_Tolerance(MatrixXd& mV, int dim, double tolerance=1e-5){
    auto maxX = mV.col(dim).minCoeff();
    std::vector<int> maxV;
    for(unsigned int ii=0; ii<mV.rows(); ++ii) {

        if(fabs(mV(ii,dim) - maxX) < tolerance) {
            maxV.push_back(ii);
        }
    }
    return maxV;
}

VectorXd get_w(VectorXd& r0, VectorXd& r){
	VectorXd w = VectorXd::Zero(r0.size()/3);
	for(int i=0; i<r0.size()/9; i++){
		Matrix3d R0, R;
		R0<<r0[9*i+0],r0[9*i+1],r0[9*i+2],
			r0[9*i+3],r0[9*i+4],r0[9*i+5],
			r0[9*i+6],r0[9*i+7],r0[9*i+8];
		R<<r[9*i+0],r[9*i+1],r[9*i+2],
			r[9*i+3],r[9*i+4],r[9*i+5],
			r[9*i+6],r[9*i+7],r[9*i+8];

		Matrix3d exp_brac_w = R0.transpose()*R;
		Matrix3d brac_w = exp_brac_w.log();
		
		w[3*i+0] = brac_w(2,1);
		w[3*i+1] = brac_w(0,2);
		w[3*i+2] = brac_w(1,0);
	}

	return w;
}
//CHECK E,x-------------
VectorXd Ex(Mesh& mesh, Arap& arap, double E0, double eps){
	VectorXd z = mesh.red_x();
	VectorXd fake = VectorXd::Zero(z.size());

	for(int i=0; i<fake.size(); i++){
		z[i] += 0.5*eps;
		double Eleft = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
		z[i] -= 0.5*eps;
		z[i] -= 0.5*eps;
		double Eright = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
		z[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	return fake;
}
//-----------------------

//CHECK E,r-------------
VectorXd Er(Mesh& mesh, Arap& arap, double E0, double eps){
	VectorXd z = mesh.red_x();
	VectorXd fake = VectorXd::Zero(mesh.red_w().size());
	for(int i=0; i<fake.size(); i++){
		mesh.red_w()[i] += 0.5*eps;
		// mesh.setGlobalF(true, false, false);
		double Eleft = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
		mesh.red_w()[i] -= 0.5*eps;
		mesh.red_w()[i] -= 0.5*eps;
		// mesh.setGlobalF(true, false, false);
		double Eright = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
		mesh.red_w()[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/(eps);
	}
	// mesh.setGlobalF(true, false, false);
	return fake;
}
//-----------------------

//CHECK E,s-------------
VectorXd Es(Mesh& mesh, Arap& arap, double E0, double eps){
	VectorXd z = mesh.red_x();
	VectorXd fake = VectorXd::Zero(mesh.red_s().size());
	for(int i=0; i<fake.size(); i++){
		mesh.red_s()[i] += 0.5*eps;
		// mesh.setGlobalF(false, true, false);
		double Eleft = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
		mesh.red_s()[i] -= 0.5*eps;
		
		mesh.red_s()[i] -= 0.5*eps;
		// mesh.setGlobalF(false, true, false);
		double Eright = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
		mesh.red_s()[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	// mesh.setGlobalF(false, true, false);
	return fake;
}
//-----------------------

//CHECK Exx--------------
MatrixXd Exx(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_x().size());
	VectorXd z = mesh.red_x();
	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			z[i] += eps;
			z[j] += eps;
			double Eij = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			z[i] -= eps;
			z[j] -= eps;

			z[i] += eps;
			double Ei = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			z[i] -= eps;

			z[j] += eps;
			double Ej = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			z[j] -= eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	
    return fake;
}

//-----------------------

//CHECK Exr/Erx-------------
MatrixXd Exr(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_w().size());
	VectorXd z = mesh.red_x();

	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			mesh.red_w()[j] += eps;
			z[i] += eps;
			// mesh.setGlobalF(true, false, false);
			double Eij = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_w()[j] -= eps;
			z[i] -= eps;

			mesh.red_w()[j] += eps;
			// mesh.setGlobalF(true, false, false);
			double Ei = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_w()[j] -= eps;

			z[i] += eps;
			// mesh.setGlobalF(true, false, false);
			double Ej = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			z[i] -= eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	// mesh.setGlobalF(true, false, false);
	return fake;
}
//-----------------------

//CHECK Exs-------------
MatrixXd Exs(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_s().size());
	VectorXd z = mesh.red_x();

	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			mesh.red_s()[j] += eps;
			z[i] += eps;
			// mesh.setGlobalF(false, true, false);
			double Eij = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_s()[j] -= eps;
			z[i] -= eps;

			mesh.red_s()[j] += eps;
			// mesh.setGlobalF(false, true, false);
			double Ei = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_s()[j] -= eps;

			z[i] += eps;
			// mesh.setGlobalF(false, true, false);
			double Ej = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			z[i] -= eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	// mesh.setGlobalF(false, true, false);
	return fake;
}
//-----------------------

//CHECK Err--------------
MatrixXd Err(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_w().size());

	for(int i=0; i<fake.rows(); i++){
        for(int j=0; j<fake.cols(); j++){
			mesh.red_w()[j] += eps;
            mesh.red_w()[i] += eps;
            // mesh.setGlobalF(true, false, false);
            double Eij = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
            mesh.red_w()[j] -= eps;
            mesh.red_w()[i] -= eps;

            mesh.red_w()[j] += eps;
            // mesh.setGlobalF(true, false, false);
            double Ei = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
            mesh.red_w()[j] -= eps;

            mesh.red_w()[i] += eps;
            // mesh.setGlobalF(true, false, false);
            double Ej = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
            mesh.red_w()[i] -= eps;

            fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
        }
	}
	// mesh.setGlobalF(true, false, false);
	
    return fake;
}
//-----------------------

//CHECK Ers--------------
MatrixXd Ers(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_s().size());

	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			mesh.red_w()[i] += eps;
			mesh.red_s()[j] += eps;
			// mesh.setGlobalF(true, true, false);
			double Eij = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_s()[j] -= eps;
			mesh.red_w()[i] -= eps;

			mesh.red_w()[i] += eps;
			// mesh.setGlobalF(true, true, false);
			double Ei = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_w()[i] -= eps;

			mesh.red_s()[j] += eps;
			// mesh.setGlobalF(true, true, false);
			double Ej = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_s()[j] -= eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	// mesh.setGlobalF(true, true, false);

	return fake;
}
//-----------------------

int checkARAP(Mesh& mesh, Arap& arap){
	double eps = j_input["fd_eps"];
	double E0 = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
	cout<<"E0: "<<E0<<endl;

    arap.Gradients(mesh);
    cout<<"Ex"<<endl;
    VectorXd fakeEx = Ex(mesh, arap, E0, eps);
    cout<<fakeEx<<endl;
    cout<<(arap.Ex().transpose()-fakeEx.transpose()).norm()<<endl<<endl;
    
    cout<<"Er"<<endl;
    VectorXd fakeEr = Er(mesh, arap, E0, eps);
    cout<<fakeEr.transpose()<<endl;
    cout<<arap.Er().transpose()<<endl;
    cout<<(arap.Er().transpose()-fakeEr.transpose()).norm()<<endl<<endl;
    
    cout<<"Es"<<endl;
	VectorXd fakeEs = Es(mesh, arap,E0, eps);
	cout<<fakeEs.transpose()<<endl;
    cout<<arap.Es().transpose()<<endl;
    cout<<(arap.Es().transpose() - fakeEs.transpose()).norm()<<endl<<endl;

    arap.Hessians(mesh);
	MatrixXd fakeExx = Exx(mesh, arap, E0, eps);
    cout<<"Exx"<<endl;
    cout<<(fakeExx-MatrixXd(arap.Exx())).norm()<<endl<<endl;
    cout<<endl<<endl;

    MatrixXd fakeExr = Exr(mesh, arap, E0, eps);
    cout<<"Exr"<<endl;
    cout<<(fakeExr-MatrixXd(arap.Exr())).norm()<<endl<<endl;
    cout<<endl<<endl;

    cout<<"Exs"<<endl;
    MatrixXd fakeExs = Exs(mesh, arap, E0, eps);
    cout<<(fakeExs-MatrixXd(arap.Exs())).norm()<<endl<<endl;
    cout<<endl;

    cout<<"Err"<<endl;
    MatrixXd fakeErr = Err(mesh, arap, E0, eps);
    cout<<(fakeErr-MatrixXd(arap.Err())).norm()<<endl<<endl;
    cout<<endl;

    cout<<"Ers"<<endl;
    MatrixXd fakeErs = Ers(mesh, arap, E0, eps);
    cout<<fakeErs<<endl<<endl;
    cout<<arap.Ers()<<endl;
    cout<<(fakeErs-MatrixXd(arap.Ers())).norm()<<endl<<endl;
    cout<<endl;
}

// VectorXd WikipediaEnergy_grad(Mesh& mesh, Elastic& elas, double E0, double eps){
//     VectorXd fake = VectorXd::Zero(mesh.red_s().size());
//     cout<<"start E0:"<<E0<<endl;
//     for(int i=0; i<fake.size(); i++){
//         mesh.red_s()[i] += 0.5*eps;
//         double Eleft = elas.WikipediaEnergy(mesh);
//         mesh.red_s()[i] -= 0.5*eps;
        
//         mesh.red_s()[i] -= 0.5*eps;
//         double Eright = elas.WikipediaEnergy(mesh);
//         mesh.red_s()[i] += 0.5*eps;
//         fake[i] = (Eleft - Eright)/(eps);
//     }
    // mesh.setGlobalF(false, true, false);
//     // std::cout<<"FUll fake: "<<fake.transpose()<<std::endl;
//     return fake;
// }

// VectorXd MuscleEnergy_grad(Mesh& mesh, Elastic& elas, double E0, double eps){
//     VectorXd fake = VectorXd::Zero(mesh.red_s().size());
//     cout<<"start E0:"<<E0<<endl;
//     for(int i=0; i<fake.size(); i++){
//         mesh.red_s()[i] += 0.5*eps;
//         double Eleft = elas.MuscleEnergy(mesh);
//         mesh.red_s()[i] -= 0.5*eps;
        
//         mesh.red_s()[i] -= 0.5*eps;
//         double Eright = elas.MuscleEnergy(mesh);
//         mesh.red_s()[i] += 0.5*eps;
//         fake[i] = (Eleft - Eright)/(eps);
//     }
    // mesh.setGlobalF(false, true, false);
//     // std::cout<<"FUll fake: "<<fake.transpose()<<std::endl;
//     return fake;
// }

// int checkElastic(Mesh& mesh, Elastic& elas){
// 	double eps = j_input["fd_eps"];
// 	double E0 = elas.WikipediaEnergy(mesh);
// 	std::cout<<"Energy0: "<<E0<<std::endl;
	
// 	cout<<"Real wikipedia grad"<<endl;
// 	VectorXd real = elas.WikipediaForce(mesh);
// 	std::cout<<real.transpose()<<std::endl;
// 	std::cout<<"Energy0: "<<elas.WikipediaEnergy(mesh)<<std::endl;

// 	cout<<"Fake wikipedia grad"<<endl;
// 	VectorXd fake = WikipediaEnergy_grad(mesh, elas, E0, eps);
// 	cout<<(fake-real).norm()<<endl;
// 	cout<<fake.transpose()<<endl;
// 	cout<<"E1 "<<elas.WikipediaEnergy(mesh)<<endl<<endl;

// 	///////Muscles
// 	E0 = elas.MuscleEnergy(mesh);
// 	std::cout<<"Energy0: "<<E0<<std::endl;
	
// 	cout<<"Real wikipedia grad"<<endl;
// 	real = elas.MuscleForce(mesh);
// 	std::cout<<real.transpose()<<std::endl;
// 	std::cout<<"Energy0: "<<elas.MuscleEnergy(mesh)<<std::endl;

// 	cout<<"Fake wikipedia grad"<<endl;
// 	fake = MuscleEnergy_grad(mesh, elas, E0, eps);
// 	cout<<(fake-real).norm()<<endl;
// 	cout<<fake.transpose()<<endl;
// 	cout<<"E1 "<<elas.MuscleEnergy(mesh)<<endl<<endl;
// }

int main(int argc, char *argv[]){
    std::cout<<"-----Configs-------"<<std::endl;
    json j_config_parameters;
    std::ifstream i("../input/input.json");
    i >> j_input;

    
    MatrixXd V;
    MatrixXi T;
    MatrixXi F;
    igl::readMESH(j_input["mesh_file"], V, T, F);
    
    std::vector<int> fix = getMaxVerts_Axis_Tolerance(V, 1);
    std::sort (fix.begin(), fix.end());
    std::vector<int> mov = {1,7};
    std::sort (mov.begin(), mov.end());
    std::vector<int> bones = {};

    std::cout<<"-----Mesh-------"<<std::endl;
    VectorXi muscle1;
    MatrixXd Uvec;
    Mesh* mesh = new Mesh(T, V, fix, mov, bones, muscle1, Uvec, j_input);

    std::cout<<"-----ARAP-----"<<std::endl;
    Arap* arap = new Arap(*mesh);

    std::cout<<"-----Neo-------"<<std::endl;
    Elastic* neo = new Elastic(*mesh);

    std::vector<double> s = {1.02964,     0.877578,
          1.02964, -8.11786e-10,
            2.28826e-11,  1.01396e-09,      1.02964,
                 0.877578,      1.02964 , -1.66318e-09, 
                  1.02164e-11, -1.02721e-10,      1.02964,
                       0.877578,      1.02964, -1.35768e-09,
                         1.30983e-10,  2.67604e-09,
     1.02964 ,    0.877578    ,  1.02964 ,-2.09268e-09 , 
     6.54942e-11,  2.42438e-09,      1.02964,     0.877578,
           1.02964,   3.6373e-10,  1.80095e-10, -8.03432e-11,
      1.02964,     0.877578,      1.02964 , 8.49187e-10,
        2.51311e-10,  3.18955e-10};

    std::vector<double> r = {0.99957,   0.0292967 ,-0.00158559,
      -0.0293113,    0.999519 , -0.0100828,  0.00128944 ,  0.0101249,    0.999948 ,
         0.999677,   0.0253299, -0.00203604,  -0.0253488 ,
        0.99963, -0.00987051,  0.00178527,  0.00991893 ,   0.999949};

    std::vector<double> x = { -0.394831 ,     2.29085,   -0.0674824 ,
       -0.386983 ,     2.34076 ,   -0.209718,  -9.8391e-17,
        -5.24752e-16 ,  3.2797e-17  ,          0  ,          0  ,
                  0 ,-1.90823e-16 ,
    	-3.39287e-16, -2.81781e-17, -1.81941e-16, -5.13116e-16 , 5.35026e-17,    -0.534805    ,  2.51089 ,   -0.259069 ,   -0.542944 ,     2.45913 ,   -0.111564};
    
    for(int i=0; i<s.size(); i++){
    	mesh->red_s()[i] = s[i];
    }

    for(int i=0; i<r.size(); i++){
    	mesh->red_r()[i] = r[i];
    }

    for(int i=0; i<x.size(); i++){
    	mesh->red_x()[i] = x[i];
    }
    arap->minimize(*mesh);
    checkARAP(*mesh, *arap);
    // checkElastic(*mesh, *neo);

}