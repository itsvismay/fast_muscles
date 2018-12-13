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
	// std::cout<<real<<std::endl;
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
	// std::cout<<fake<<std::endl;
	// std::cout<<real<<std::endl;
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
	arap.minimize(mesh);
	double eps = j_input["fd_eps"];
	double E0 = arap.Energy(mesh);
	std::cout<<"Energy0: "<<E0<<std::endl;
	
	cout<<"Real ARAP Jac"<<endl;
	arap.minimize(mesh);
	VectorXd real = arap.Jacobians(mesh);
	std::cout<<"Energy0: "<<arap.Energy(mesh)<<std::endl;
	// cout<<mesh.red_s().transpose()<<endl;
	// cout<<mesh.red_r().transpose()<<endl;
	// cout<<mesh.red_x().transpose()<<endl;
	// cout<<mesh.red_w().transpose()<<endl;
	cout<<real.transpose()<<endl<<endl;

	// // cout<<"fake Jac 1"<<endl;
	// VectorXd dEdr = Er(mesh,arap,E0, eps);
	// VectorXd dEdx = Ex(mesh,arap,E0, eps);
	// VectorXd dEds = Es(mesh,arap,E0, eps);
	// MatrixXd dgds = Jac_dgds(mesh,arap,E0, eps);
	// MatrixXd drds = Jac_drds(mesh,arap,E0, eps) ;
	// VectorXd fake1 =  dgds.transpose()*dEdx + drds.transpose()*dEdr + dEds;
	// std::cout<<"Energy0: "<<arap.Energy(mesh)<<std::endl;
	// cout<<mesh.red_s().transpose()<<endl;
	// cout<<mesh.red_r().transpose()<<endl;
	// cout<<mesh.red_x().transpose()<<endl;
	// cout<<mesh.red_w().transpose()<<endl;
	// // cout<<fake1.transpose()<<endl<<endl;

	cout<<"fake Jac 2"<<endl;
	VectorXd fake2 = Full_FD_Grad(mesh, arap, E0, eps);
	std::cout<<"Energy0: "<<arap.Energy(mesh)<<std::endl;
	// cout<<mesh.red_s().transpose()<<endl;
	// cout<<mesh.red_r().transpose()<<endl;
	// cout<<mesh.red_x().transpose()<<endl;
	// cout<<mesh.red_w().transpose()<<endl;
	cout<<fake2.transpose()<<endl<<endl;

	// cout<<"fake Jac 3"<<endl;
 //    MatrixXd dEdxx = Exx(mesh, arap, E0, eps);
 //    MatrixXd dEdxr = Exr(mesh, arap, E0, eps);
 //    MatrixXd dEdxs = Exs(mesh, arap, E0, eps);
 //    MatrixXd dEdrr = Err(mesh, arap, E0, eps);
 //    MatrixXd dEdrs = Ers_part1(mesh, arap, E0, eps);
 //    MatrixXd lhs_left(dEdxx.rows()+dEdxr.cols(), dEdxx.cols());
 //    lhs_left<<dEdxx, dEdxr.transpose();
 //    MatrixXd lhs_right(dEdxr.rows() + dEdrr.rows() , dEdxr.cols());
 //    lhs_right<<dEdxr, dEdrr; 
 //    MatrixXd rhs(dEdxs.rows()+dEdrs.rows(), dEdxs.cols());
 //    rhs<<-1*dEdxs, -1*dEdrs;
 //    MatrixXd CG = MatrixXd(mesh.AB().transpose())*mesh.G();
 //    MatrixXd col1(lhs_left.rows()+CG.rows(), lhs_left.cols());
 //    col1<<lhs_left, CG;
 //    MatrixXd col2(lhs_right.rows()+CG.rows(), lhs_right.cols());
 //    col2<<lhs_right,MatrixXd::Zero(CG.rows(), lhs_right.cols());
 //    MatrixXd col3(CG.cols()+CG.rows()+dEdrr.rows(), CG.rows());
 //    col3<<CG.transpose(),MatrixXd::Zero(CG.rows()+dEdrr.rows(), CG.rows());
 //    MatrixXd KKT_constrains(rhs.rows() + CG.rows(), rhs.cols());
 //    KKT_constrains<<rhs,MatrixXd::Zero(CG.rows(), rhs.cols());
 //    MatrixXd JacKKT(col1.rows(), col1.rows());
 //    JacKKT<<col1, col2, col3;
 //    MatrixXd results = JacKKT.fullPivLu().solve(KKT_constrains).topRows(rhs.rows());
 //    MatrixXd dgds1 = results.topRows(dEdxx.rows());
 //    MatrixXd drds1 = results.bottomRows(dEdrr.rows());
 //    VectorXd fake3 =  dgds1.transpose()*dEdx + drds1.transpose()*dEdr + dEds;
 //    std::cout<<"Energy0: "<<arap.Energy(mesh)<<std::endl;
 //    cout<<mesh.red_s().transpose()<<endl;
	// cout<<mesh.red_r().transpose()<<endl;
	// cout<<mesh.red_x().transpose()<<endl;
	// cout<<mesh.red_w().transpose()<<endl;
    // cout<<fake3.transpose()<<endl<<endl;


	// std::cout<<"DEDs"<<std::endl;
	// std::cout<<DEDs<<std::endl;
	// std::cout<<real<<std::endl;
	// std::cout<<"L2 norm"<<std::endl;
	// std::cout<<(real - DEDs).norm()<<std::endl;
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
        fake[i] = (Eleft - E0)/(0.5*eps);
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
	cout<<"E1 "<<elas.WikipediaEnergy(mesh)<<endl<<endl;



	///////Muscles



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

    // VectorXd& dx = mesh->dx();
    // for(int i=0; i<mov.size(); i++){
    //     dx[3*mov[i]+1] += 3;
    // }

    vector<double> s = {0.996986, 0.730809, 0.99472,   -0.134599,  0.0210262,   0.0216317, 1.0145,
        0.801846,    0.984477,    0.145338,  0.00280612 , -0.0886036,    0.976572 ,
           0.852875,      1.0131 ,   0.175657, -0.00418633,   -0.163312 ,   0.982332  ,
             0.897076,     1.01097 , 0.00616309,   0.0423661,  -0.0619048,    0.992371, 
               0.77626 ,   0.976263 ,  0.0139843 ,  0.0354943,    0.156384,     1.00932 ,  
                0.798815,     1.00102 ,  0.0987563, -0.00658236,   0.0186558,    0.975468, 
                   0.658833,     1.04898 ,  -0.216346,  -0.0343779,     0.10836,     1.00196, 
                      0.749595,     0.99475 ,  0.0158044, -0.00888359,   -0.051811,    0.987621,
                         0.565278,     1.01795 ,  0.0255043,  -0.0394745,  -0.0635043,     1.01052,
                            0.899457,    0.975795 , -0.0599184,    0.015183,   0.0641473,    0.988842 ,
                                0.71731,    0.999449 ,   0.109627,  0.00821576,   0.0452765,     1.00889, 
                                     0.7949,    0.997217,     0.14101,  0.00624776,   0.0176047};
    
   	// mesh->red_s()[0] = s[0];
   	// mesh->red_s()[1] = s[1];
   	// mesh->red_s()[2] = s[2];
    for(int i=0; i<mesh->red_s().size()/6; i++)
   		mesh->red_s()[6*i+1] -= 0.1;
   	// cout<<mesh->red_s().transpose()<<endl;
    mesh->setGlobalF(false, true, false);

    checkARAP(*mesh, *arap);

    // checkElastic(*mesh, *neo);

}