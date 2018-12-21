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
	VectorXd real = arap.dEdx(mesh);
	VectorXd z = mesh.red_x();
	VectorXd fake = VectorXd::Zero(real.size());
	#pragma omp parallel for
	for(int i=0; i<fake.size(); i++){
		z[i] += 0.5*eps;
		double Eleft = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
		z[i] -= 0.5*eps;
		z[i] -= 0.5*eps;
		double Eright = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
		z[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	std::cout<<"Ex error:"<<(real-fake).norm()<<std::endl;	
	return fake;
}
//-----------------------

//CHECK E,r-------------
VectorXd Er(Mesh& mesh, Arap& arap, double E0, double eps){
	VectorXd real = arap.dEdr(mesh);
	VectorXd z = mesh.red_x();
	VectorXd fake = VectorXd::Zero(real.size());
	for(int i=0; i<fake.size(); i++){
		mesh.red_w()[i] += 0.5*eps;
		mesh.setGlobalF(true, false, false);
		double Eleft = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
		mesh.red_w()[i] -= 0.5*eps;
		mesh.red_w()[i] -= 0.5*eps;
		mesh.setGlobalF(true, false, false);
		double Eright = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
		mesh.red_w()[i] += 0.5*eps;
		fake[i] = (Eright - Eleft)/(eps);
	}
	mesh.setGlobalF(true, false, false);
	std::cout<<"Er error:"<<(real-fake).norm()<<std::endl;	
	// std::cout<<fake.transpose()<<std::endl;
	// std::cout<<real.transpose()<<std::endl;
	return fake;
}
//-----------------------

//CHECK E,s-------------
VectorXd Es(Mesh& mesh, Arap& arap, double E0, double eps){
	VectorXd real = arap.dEds(mesh);
	VectorXd z = mesh.red_x();
	VectorXd fake = VectorXd::Zero(real.size());
	for(int i=0; i<fake.size(); i++){
		mesh.red_s()[i] += 0.5*eps;
		mesh.setGlobalF(false, true, false);
		double Eleft = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
		mesh.red_s()[i] -= 0.5*eps;
		
		mesh.red_s()[i] -= 0.5*eps;
		mesh.setGlobalF(false, true, false);
		double Eright = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
		mesh.red_s()[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	mesh.setGlobalF(false, true, false);
	std::cout<<"Es error:"<<(real-fake).norm()<<std::endl;	
	// std::cout<<fake.transpose()<<std::endl;
	// std::cout<<std::endl;
	// std::cout<<real.transpose()<<std::endl;
	return fake;
}
//-----------------------

//CHECK Exx--------------
MatrixXd Exx(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd& real = arap.Exx();
	MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_x().size());
	VectorXd z = mesh.red_x();
	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			z[i] += eps;
			z[j] += eps;
			double Eij = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
			z[i] -= eps;
			z[j] -= eps;

			z[i] += eps;
			double Ei = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
			z[i] -= eps;

			z[j] += eps;
			double Ej = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
			z[j] -= eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	std::cout<<"Exx"<<std::endl;
	std::cout<<(fake- real).norm()<<std::endl;
	return fake;
}

//-----------------------

//CHECK Exr/Erx-------------
MatrixXd Exr(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd& real = arap.Exr(mesh);
	MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_w().size());
	VectorXd z = mesh.red_x();

	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			mesh.red_w()[j] += eps;
			z[i] += eps;
			mesh.setGlobalF(true, false, false);
			double Eij = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
			mesh.red_w()[j] -= eps;
			z[i] -= eps;

			mesh.red_w()[j] += eps;
			mesh.setGlobalF(true, false, false);
			double Ei = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
			mesh.red_w()[j] -= eps;

			z[i] += eps;
			mesh.setGlobalF(true, false, false);
			double Ej = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
			z[i] -= eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	mesh.setGlobalF(true, false, false);
	std::cout<<"Exr"<<std::endl;
	// std::cout<<real<<std::endl<<std::endl;
	// std::cout<<fake<<std::endl;
	std::cout<<(fake- real).norm()<<std::endl;
	return fake;
}
//-----------------------

//CHECK Exs-------------
MatrixXd Exs(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd& real = arap.Exs(mesh);
	MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_s().size());
	VectorXd z = mesh.red_x();

	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			mesh.red_s()[j] += eps;
			z[i] += eps;
			mesh.setGlobalF(false, true, false);
			double Eij = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
			mesh.red_s()[j] -= eps;
			z[i] -= eps;

			mesh.red_s()[j] += eps;
			mesh.setGlobalF(false, true, false);
			double Ei = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
			mesh.red_s()[j] -= eps;

			z[i] += eps;
			mesh.setGlobalF(false, true, false);
			double Ej = arap.Energy(mesh, z, mesh.GR(), mesh.GS(), mesh.GU());
			z[i] -= eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	mesh.setGlobalF(false, true, false);
	std::cout<<"Exs"<<std::endl;
	std::cout<<real<<std::endl;
	std::cout<<fake<<std::endl;
	std::cout<<(fake- real).norm()<<std::endl;
	return fake;
}
//-----------------------

//CHECK Err--------------
MatrixXd Err(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd& real = arap.Err(mesh);
	MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_w().size());

	for(int i=0; i<fake.rows(); i++){
			mesh.red_w()[i] += 0.5*eps;
			mesh.setGlobalF(true, false, false);
			VectorXd Eleft = arap.dEdr(mesh);
			mesh.red_w()[i] -= 0.5*eps;

			mesh.red_w()[i] -= 0.5*eps;
			mesh.setGlobalF(true, false, false);
			VectorXd Eright = arap.dEdr(mesh);
			mesh.red_w()[i] += 0.5*eps;

			fake.row(i) = (Eleft-Eright)/eps;
	}
	mesh.setGlobalF(true, false, false);
	std::cout<<"Err"<<std::endl;
	std::cout<<fake<<std::endl;
	std::cout<<real<<std::endl;
	std::cout<<(fake- real).norm()<<std::endl;
	return fake;
}
//-----------------------

//CHECK Ers--------------
MatrixXd Ers(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd& real = arap.Ers(mesh);
	MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_s().size());

	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			mesh.red_w()[i] += eps;
			mesh.red_s()[j] += eps;
			mesh.setGlobalF(true, true, false);
			double Eij = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
			mesh.red_s()[j] -= eps;
			mesh.red_w()[i] -= eps;

			mesh.red_w()[i] += eps;
			mesh.setGlobalF(true, true, false);
			double Ei = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
			mesh.red_w()[i] -= eps;

			mesh.red_s()[j] += eps;
			mesh.setGlobalF(true, true, false);
			double Ej = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
			mesh.red_s()[j] -= eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}
	mesh.setGlobalF(true, true, false);
	std::cout<<"Ers"<<std::endl;
	// std::cout<<fake<<std::endl;
	// std::cout<<real<<std::endl;
	std::cout<<(fake- real).norm()<<std::endl;
	return fake;
}
//-----------------------

//CHECK Ers with dEds for energy--------------
MatrixXd Ers_part1(Mesh& mesh, Arap& arap, double E0, double eps){
	MatrixXd& real = arap.Ers(mesh);
	MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_s().size());

	for(int i=0; i<fake.rows(); i++){
			mesh.red_w()[i] += 0.5*eps;
			mesh.setGlobalF(true, false, false);
			VectorXd Eleft = arap.dEds(mesh);
			mesh.red_w()[i] -= 0.5*eps;

			mesh.red_w()[i] -= 0.5*eps;
			mesh.setGlobalF(true, false, false);
			VectorXd Eright = arap.dEds(mesh);
			mesh.red_w()[i] += 0.5*eps;

			fake.row(i) = (Eleft-Eright)/eps;
	}
	mesh.setGlobalF(true, false, false);
	std::cout<<"Ers_part1"<<std::endl;
	// std::cout<<fake<<std::endl;
	std::cout<<(fake- real).norm()<<std::endl;
	return fake;
}
//-----------------------

//CHECK dgds-----------------------
MatrixXd Jac_dgds(Mesh& mesh, Arap& arap, double E0, double eps){
	// arap.Jacobians(mesh);
	
	MatrixXd dgds = MatrixXd::Zero(mesh.red_x().size(), mesh.red_s().size());
	VectorXd z0 = mesh.red_x();
	VectorXd& z = mesh.red_x();

	for(int i=0; i<mesh.red_s().size(); i++){
		mesh.red_x(z0);

		mesh.red_s()[i] += 0.5*eps;
		mesh.setGlobalF(false, true, false);
		arap.minimize(mesh);

		VectorXd dgds_left = mesh.red_x() + VectorXd::Zero(z.size());

		mesh.red_s()[i] -= 0.5*eps;
		mesh.setGlobalF(false, true, false);
		arap.minimize(mesh);


		mesh.red_s()[i] -= 0.5*eps;
		mesh.setGlobalF(false, true, false);
		arap.minimize(mesh);
		
		VectorXd dgds_right = mesh.red_x() + VectorXd::Zero(z.size());

		mesh.red_s()[i] += 0.5*eps;
		mesh.setGlobalF(false, true, false);
		arap.minimize(mesh);

		dgds.col(i) = (dgds_left - dgds_right)/eps;
	}
	std::cout<<"dgds"<<std::endl;
	// std::cout<<dgds<<std::endl;
	return dgds;
}
//---------------------------------

//CHECK drds-----------------------
MatrixXd Jac_drds(Mesh& mesh, Arap& arap, double E0, double eps){
	// VectorXd dEds = arap.Jacobians(mesh);

	MatrixXd drds = MatrixXd::Zero(mesh.red_w().size(), mesh.red_s().size());
	VectorXd vecR0, vecR;
	for(int i=0; i<mesh.red_s().size(); i++){

		mesh.red_s()[i] += 0.5*eps;
		mesh.setGlobalF(false, true, false);
		vecR0 = mesh.red_r();
		arap.minimize(mesh);
		vecR = mesh.red_r();

		VectorXd drds_left = get_w(vecR0, vecR);

		mesh.red_s()[i] -= 0.5*eps;
		mesh.setGlobalF(false, true, false);
		arap.minimize(mesh);


		mesh.red_s()[i] -= 0.5*eps;
		mesh.setGlobalF(false, true, false);
		vecR0 = mesh.red_r();
		arap.minimize(mesh);
		vecR = mesh.red_r();
		
		VectorXd drds_right = get_w(vecR0, vecR);

		mesh.red_s()[i] += 0.5*eps;
		mesh.setGlobalF(false, true, false);
		arap.minimize(mesh);

		drds.col(i) = (drds_left - drds_right)/eps;
	}
	std::cout<<"drds"<<std::endl;
	// std::cout<<drds<<std::endl;
	return drds;
}
//---------------------------------

VectorXd Full_FD_Grad(Mesh& mesh, Arap& arap, double E0, double eps){
    VectorXd fake = VectorXd::Zero(mesh.red_s().size());
    for(int i=0; i<fake.size(); i++){
        mesh.red_s()[i] += 0.5*eps;
        mesh.setGlobalF(false, true, false);
        arap.minimize(mesh);
        double Eleft = arap.Energy(mesh);
        mesh.red_s()[i] -= 0.5*eps;
        
        mesh.red_s()[i] -= 0.5*eps;
        mesh.setGlobalF(false, true, false);
        arap.minimize(mesh);
        double Eright = arap.Energy(mesh);
        mesh.red_s()[i] += 0.5*eps;
        fake[i] = (Eleft - Eright)/eps;
    }
    mesh.setGlobalF(false, true, false);
    // std::cout<<"FUll fake: "<<fake.transpose()<<std::endl;
    return fake;
}

int checkARAP(Mesh& mesh, Arap& arap){
	double eps = j_input["fd_eps"];
	double E0 = arap.Energy(mesh);
	cout<<"E0: "<<E0<<endl;
	// Es(mesh, arap,E0, eps);
	// Er(mesh, arap, E0, eps);
	// Ex(mesh, arap, E0, eps);
	// Exr(mesh, arap, E0, eps);
	Exs(mesh, arap, E0, eps);
	// Err(mesh, arap, E0, eps);
	// Ers(mesh, arap, E0, eps);
}

VectorXd WikipediaEnergy_grad(Mesh& mesh, Elastic& elas, double E0, double eps){
    VectorXd fake = VectorXd::Zero(mesh.red_s().size());
    cout<<"start E0:"<<E0<<endl;
    for(int i=0; i<fake.size(); i++){
        mesh.red_s()[i] += 0.5*eps;
        double Eleft = elas.WikipediaEnergy(mesh);
        mesh.red_s()[i] -= 0.5*eps;
        
        mesh.red_s()[i] -= 0.5*eps;
        double Eright = elas.WikipediaEnergy(mesh);
        mesh.red_s()[i] += 0.5*eps;
        fake[i] = (Eleft - Eright)/(eps);
    }
    // mesh.setGlobalF(false, true, false);
    // std::cout<<"FUll fake: "<<fake.transpose()<<std::endl;
    return fake;
}

VectorXd MuscleEnergy_grad(Mesh& mesh, Elastic& elas, double E0, double eps){
    VectorXd fake = VectorXd::Zero(mesh.red_s().size());
    cout<<"start E0:"<<E0<<endl;
    for(int i=0; i<fake.size(); i++){
        mesh.red_s()[i] += 0.5*eps;
        double Eleft = elas.MuscleEnergy(mesh);
        mesh.red_s()[i] -= 0.5*eps;
        
        mesh.red_s()[i] -= 0.5*eps;
        double Eright = elas.MuscleEnergy(mesh);
        mesh.red_s()[i] += 0.5*eps;
        fake[i] = (Eleft - Eright)/(eps);
    }
    // mesh.setGlobalF(false, true, false);
    // std::cout<<"FUll fake: "<<fake.transpose()<<std::endl;
    return fake;
}

int checkElastic(Mesh& mesh, Elastic& elas){
	double eps = j_input["fd_eps"];
	double E0 = elas.WikipediaEnergy(mesh);
	std::cout<<"Energy0: "<<E0<<std::endl;
	
	cout<<"Real wikipedia grad"<<endl;
	VectorXd real = elas.WikipediaForce(mesh);
	std::cout<<real.transpose()<<std::endl;
	std::cout<<"Energy0: "<<elas.WikipediaEnergy(mesh)<<std::endl;

	cout<<"Fake wikipedia grad"<<endl;
	VectorXd fake = WikipediaEnergy_grad(mesh, elas, E0, eps);
	cout<<(fake-real).norm()<<endl;
	cout<<fake.transpose()<<endl;
	cout<<"E1 "<<elas.WikipediaEnergy(mesh)<<endl<<endl;

	///////Muscles
	E0 = elas.MuscleEnergy(mesh);
	std::cout<<"Energy0: "<<E0<<std::endl;
	
	cout<<"Real wikipedia grad"<<endl;
	real = elas.MuscleForce(mesh);
	std::cout<<real.transpose()<<std::endl;
	std::cout<<"Energy0: "<<elas.MuscleEnergy(mesh)<<std::endl;

	cout<<"Fake wikipedia grad"<<endl;
	fake = MuscleEnergy_grad(mesh, elas, E0, eps);
	cout<<(fake-real).norm()<<endl;
	cout<<fake.transpose()<<endl;
	cout<<"E1 "<<elas.MuscleEnergy(mesh)<<endl<<endl;
}

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
    Mesh* mesh = new Mesh(T, V, fix, mov, bones, j_input);

    std::cout<<"-----ARAP-----"<<std::endl;
    Arap* arap = new Arap(*mesh);

    std::cout<<"-----Neo-------"<<std::endl;
    Elastic* neo = new Elastic(*mesh);


    // for(int i=0; i<mesh->red_s().size()/6; i++)
   	// 	mesh->red_s()[6*i+1] -= 0.1;
   	// // cout<<mesh->red_s().transpose()<<endl;
    // mesh->setGlobalF(false, true, false);
    MatrixXd Exr1 = arap->Exr(*mesh);
    MatrixXd Exr2 = MatrixXd::Zero(mesh->red_x().size(), mesh->red_w().size());
    arap->Exr_max(*mesh, Exr2);
    // cout<<Exr2<<endl<<endl;
    // cout<<Exr1<<endl<<endl;
    // cout<<Exr1 - Exr2<<endl;
    // checkARAP(*mesh, *arap);

    // checkElastic(*mesh, *neo);

}