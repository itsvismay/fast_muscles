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
// #include "elastic.h"
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

int checkARAP(Mesh& mesh, Arap& arap){
	double eps = j_input["fd_eps"];
	double E0 = arap.Energy(mesh);
	std::cout<<"Energy0: "<<E0<<std::endl;
	//CHECK E,x-------------
	auto Ex = [&mesh, &arap, E0, eps](){
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
	};
	//-----------------------

	//CHECK E,r-------------
	auto Er = [&mesh, &arap, E0, eps](){
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
			fake[i] = (Eleft - Eright)/eps;
		}
		mesh.setGlobalF(true, false, false);
		std::cout<<"Er error:"<<(real-fake).norm()<<std::endl;	
		std::cout<<fake.transpose()<<std::endl;
		std::cout<<real.transpose()<<std::endl;
	};
	//-----------------------

	//CHECK E,s-------------
	auto Es = [&mesh, &arap, E0, eps](){
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
	};
	//-----------------------

	//CHECK Exx--------------
	auto Exx = [&mesh, &arap, E0, eps](){
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
	};

	//-----------------------

	//CHECK Exr/Erx-------------
	auto Exr = [&mesh, &arap, E0, eps](){
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
		std::cout<<real<<std::endl<<std::endl;
		std::cout<<fake<<std::endl;
		std::cout<<(fake- real).norm()<<std::endl;
	};
	//-----------------------

	//CHECK Ers with dEds for energy--------------
	auto Exr_part1 = [&mesh, &arap, E0, eps](){
		MatrixXd& real = arap.Exr(mesh);
		MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_w().size());
		
		for(int i=0; i<fake.rows(); i++){
				mesh.red_x()[i] += 0.5*eps;
				VectorXd Eleft = arap.dEdr(mesh);
				mesh.red_x()[i] -= 0.5*eps;

				mesh.red_x()[i] -= 0.5*eps;
				VectorXd Eright = arap.dEdr(mesh);
				mesh.red_x()[i] += 0.5*eps;

				fake.row(i) = (Eleft-Eright)/eps;
		}
		std::cout<<"Exr_part1"<<std::endl;
		std::cout<<real<<std::endl<<std::endl;
		std::cout<<fake<<std::endl;
		std::cout<<(fake- real).norm()<<std::endl;
	};
	//-----------------------

	//CHECK Ers with dEds for energy--------------
	auto Exr_part2 = [&mesh, &arap, E0, eps](){
		MatrixXd& real = arap.Exr(mesh);
		MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_w().size());
		
		for(int i=0; i<fake.cols(); i++){
				mesh.red_w()[i] += 0.5*eps;
				mesh.setGlobalF(true, false, false);
				VectorXd Eleft = arap.dEdx(mesh);
				mesh.red_w()[i] -= 0.5*eps;

				mesh.red_w()[i] -= 0.5*eps;
				mesh.setGlobalF(true, false, false);
				VectorXd Eright = arap.dEdx(mesh);
				mesh.red_w()[i] += 0.5*eps;

				fake.col(i) = (Eleft-Eright)/eps;
		}
		std::cout<<"Exr_part1"<<std::endl;
		std::cout<<real<<std::endl<<std::endl;
		std::cout<<fake<<std::endl;
		std::cout<<(fake- real).norm()<<std::endl;
	};
	//-----------------------

	//CHECK Exs-------------
	auto Exs = [&mesh, &arap, E0, eps](){
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
		// std::cout<<real<<std::endl;
		std::cout<<(fake- real).norm()<<std::endl;
	};
	//-----------------------

	//CHECK Err--------------
	auto Err = [&mesh, &arap, E0, eps](){
		MatrixXd& real = arap.Err(mesh);
		MatrixXd fake = MatrixXd::Zero(mesh.red_r().size(), mesh.red_r().size());

		for(int i=0; i<fake.rows(); i++){
			for(int j=0; j<fake.cols(); j++){
				mesh.red_r()[j] += eps;
				mesh.red_r()[i] += eps;
				mesh.setGlobalF(true, false, false);
				double Eij = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
				mesh.red_r()[j] -= eps;
				mesh.red_r()[i] -= eps;

				mesh.red_r()[j] += eps;
				mesh.setGlobalF(true, false, false);
				double Ei = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
				mesh.red_r()[j] -= eps;

				mesh.red_r()[i] += eps;
				mesh.setGlobalF(true, false, false);
				double Ej = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
				mesh.red_r()[i] -= eps;

				fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
			}
		}
		mesh.setGlobalF(true, false, false);
		std::cout<<"Err"<<std::endl;
		// std::cout<<fake<<std::endl;
		std::cout<<(fake- real).norm()<<std::endl;
	};
	//-----------------------

	//CHECK Ers--------------
	auto Ers = [&mesh, &arap, E0, eps](){
		MatrixXd& real = arap.Ers(mesh);
		MatrixXd fake = MatrixXd::Zero(mesh.red_r().size(), mesh.red_s().size());

		for(int i=0; i<fake.rows(); i++){
			for(int j=0; j<fake.cols(); j++){
				mesh.red_r()[i] += eps;
				mesh.red_s()[j] += eps;
				mesh.setGlobalF(true, true, false);
				double Eij = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
				mesh.red_s()[j] -= eps;
				mesh.red_r()[i] -= eps;

				mesh.red_r()[i] += eps;
				mesh.setGlobalF(true, true, false);
				double Ei = arap.Energy(mesh, mesh.red_x(), mesh.GR(), mesh.GS(), mesh.GU());
				mesh.red_r()[i] -= eps;

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
		std::cout<<(fake- real).norm()<<std::endl;
	};
	//-----------------------

	//CHECK Ers with dEds for energy--------------
	auto Ers_part1 = [&mesh, &arap, E0, eps](){
		MatrixXd& real = arap.Ers(mesh);
		MatrixXd fake = MatrixXd::Zero(mesh.red_r().size(), mesh.red_s().size());

		for(int i=0; i<fake.rows(); i++){
				mesh.red_r()[i] += 0.5*eps;
				mesh.setGlobalF(true, false, false);
				VectorXd Eleft = arap.dEds(mesh);
				mesh.red_r()[i] -= 0.5*eps;

				mesh.red_r()[i] -= 0.5*eps;
				mesh.setGlobalF(true, false, false);
				VectorXd Eright = arap.dEds(mesh);
				mesh.red_r()[i] += 0.5*eps;

				fake.row(i) = (Eleft-Eright)/eps;
		}
		mesh.setGlobalF(true, false, false);
		std::cout<<"Ers_part1"<<std::endl;
		// std::cout<<fake<<std::endl;
		std::cout<<(fake- real).norm()<<std::endl;
	};
	//-----------------------

	//CHECK Ers with dEdr for energy--------------
	auto Ers_part2 = [&mesh, &arap, E0, eps](){
		MatrixXd& real = arap.Ers(mesh);
		MatrixXd fake = MatrixXd::Zero(mesh.red_r().size(), mesh.red_s().size());

		for(int i=0; i<fake.cols(); i++){
				mesh.red_s()[i] += 0.5*eps;
				mesh.setGlobalF(false, true, false);
				VectorXd Eleft = arap.dEdr(mesh);
				mesh.red_s()[i] -= 0.5*eps;

				mesh.red_s()[i] -= 0.5*eps;
				mesh.setGlobalF(false, true, false);
				VectorXd Eright = arap.dEdr(mesh);
				mesh.red_s()[i] += 0.5*eps;

				fake.col(i) = (Eleft-Eright)/eps;
		}
		mesh.setGlobalF(false, true, false);
		std::cout<<"Ers_part2"<<std::endl;
		// std::cout<<fake<<std::endl;
		std::cout<<(fake- real).norm()<<std::endl;
	};
	//-----------------------

	//CHECK dgds-----------------------
	auto Jac_dgds = [&mesh, &arap, E0, eps](){
		arap.Jacobians(mesh);
		
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
		std::cout<<dgds<<std::endl;
	};
	//---------------------------------

	//CHECK drds-----------------------
	auto Jac_drds = [&mesh, &arap, E0, eps](){
		MatrixXd drds = MatrixXd::Zero(mesh.red_r().size(), mesh.red_s().size());

		for(int i=0; i<mesh.red_s().size(); i++){

			mesh.red_s()[i] += 0.5*eps;
			mesh.setGlobalF(false, true, false);
			arap.minimize(mesh);

			VectorXd drds_left = mesh.red_r();

			mesh.red_s()[i] -= 0.5*eps;
			mesh.setGlobalF(false, true, false);
			arap.minimize(mesh);


			mesh.red_s()[i] -= 0.5*eps;
			mesh.setGlobalF(false, true, false);
			arap.minimize(mesh);
			
			VectorXd drds_right = mesh.red_r();

			mesh.red_s()[i] += 0.5*eps;
			mesh.setGlobalF(false, true, false);
			arap.minimize(mesh);

			drds.col(i) = (drds_left - drds_right)/eps;
		}
		std::cout<<"drds"<<std::endl;
		std::cout<<drds<<std::endl;
	};
	//---------------------------------


	Ex();
	Er();
	Es();
	// Exx();
	// Exr();
	Exr_part1();
	Exr_part2();
	// Exs();
	// Err();
	// Ers();
	// Ers_part1();
	// Ers_part2();

	// Jac_dgds();
	// Jac_drds();
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
    std::vector<int> mov = {};
    std::sort (mov.begin(), mov.end());

    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(T, V, fix, mov, j_input);

    std::cout<<"-----ARAP-----"<<std::endl;
    Arap* arap = new Arap(*mesh);

    for(int i=0; i<mesh->red_s().size()/6; i++){
    	mesh->red_s()[6*i+1] += 0.1;
    }
    mesh->setGlobalF(false, true, false);
    arap->minimize(*mesh);

    checkARAP(*mesh, *arap);

    // std::cout<<"-----Neo-------"<<std::endl;
    // Elastic* neo = new Elastic(*mesh);


}