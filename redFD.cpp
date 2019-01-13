#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>
#include <igl/setdiff.h>


#include <json.hpp>

#include "mesh.h"
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
void getMaxTets_Axis_Tolerance(std::vector<int>& ibones, MatrixXd& mV, MatrixXi& mT, double dim, double tolerance = 1-5){
	auto maxX = mV.col(dim).maxCoeff();
	for(int i=0; i< mT.rows(); i++){
		Vector3d centre = (mV.row(mT.row(i)[0])+ mV.row(mT.row(i)[1]) + mV.row(mT.row(i)[2])+ mV.row(mT.row(i)[3]))/4.0;
		if (fabs(centre[dim] - maxX)< tolerance){
			ibones.push_back(i);
		}
	}
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
VectorXd Ex(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
	VectorXd z = mesh.red_x();
	VectorXd fake = VectorXd::Zero(z.size());
	#pragma omp parallel for
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
VectorXd Er(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
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
VectorXd Es(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
	VectorXd fake = VectorXd::Zero(mesh.red_s().size());
	for(int i=0; i<fake.size(); i++){
		mesh.red_s()[i] += 0.5*eps;
		double Eleft = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
		mesh.red_s()[i] -= 0.5*eps;
		
		mesh.red_s()[i] -= 0.5*eps;
		double Eright = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
		mesh.red_s()[i] += 0.5*eps;
		fake[i] = (Eleft - Eright)/eps;
	}
	return fake;
}
//-----------------------

//CHECK Exx--------------
MatrixXd Exx(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
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
MatrixXd Exr(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
	MatrixXd fake = MatrixXd::Zero(mesh.red_x().size(), mesh.red_w().size());
	VectorXd z = mesh.red_x();

	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			mesh.red_w()[j] += eps;
			z[i] += eps;
			double Eij = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_w()[j] -= eps;
			z[i] -= eps;

			mesh.red_w()[j] += eps;
			double Ei = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_w()[j] -= eps;

			z[i] += eps;
			double Ej = arap.Energy(mesh, z, mesh.red_w(), mesh.red_r(), mesh.red_s());
			z[i] -= eps;
			// cout<<Eij<<", "<<Ei<<", "<<Ej<<", "<<E0<<endl;
			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
			// exit(0);
	}
	return fake;
}
//-----------------------

//CHECK Exs-------------
MatrixXd Exs(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
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
MatrixXd Err(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
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
MatrixXd Ers(Mesh& mesh, Reduced_Arap& arap, double E0, double eps){
	MatrixXd fake = MatrixXd::Zero(mesh.red_w().size(), mesh.red_s().size());

	for(int i=0; i<fake.rows(); i++){
		for(int j=0; j<fake.cols(); j++){
			mesh.red_w()[i] += eps;
			mesh.red_s()[j] += eps;
			double Eij = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_s()[j] -= eps;
			mesh.red_w()[i] -= eps;

			mesh.red_w()[i] += eps;
			double Ei = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_w()[i] -= eps;

			mesh.red_s()[j] += eps;
			double Ej = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
			mesh.red_s()[j] -= eps;

			fake(i,j) = ((Eij - Ei - Ej + E0)/(eps*eps));
		}
	}

	// for(int i=0; i<fake.cols(); i++){
	// 	mesh.red_s()[i] += 0.5*eps;
	// 	VectorXd Eleft = arap.constTimeEr(mesh);
	// 	mesh.red_s()[i] -= 0.5*eps;
		
	// 	mesh.red_s()[i] -= 0.5*eps;
	// 	VectorXd Eright = arap.constTimeEr(mesh);
	// 	mesh.red_s()[i] += 0.5*eps;
	// 	fake.col(i) = (Eleft - Eright)/eps;

	// }

	return fake;
}
//-----------------------

int checkRedARAP(Mesh& mesh, Reduced_Arap& arap){
	double eps = j_input["fd_eps"];
	double E0 = arap.Energy(mesh, mesh.red_x(), mesh.red_w(), mesh.red_r(), mesh.red_s());
	cout<<"E0: "<<E0<<endl;

    arap.Gradients(mesh);
    cout<<"Ex"<<endl;
    VectorXd fakeEx = Ex(mesh, arap, E0, eps);
    cout<<(arap.Ex().transpose()-fakeEx.transpose()).norm()<<endl<<endl;
    
    cout<<"Er"<<endl;
    VectorXd fakeEr = Er(mesh, arap, E0, eps);
    cout<<fakeEr.transpose()<<endl;
    cout<<arap.Er().transpose()<<endl;
    cout<<(arap.Er().transpose()-fakeEr.transpose()).norm()<<endl<<endl;
    
    cout<<"Es"<<endl;
	VectorXd fakeEs = Es(mesh, arap,E0, eps);
	cout<<(arap.Es().transpose() - fakeEs.transpose()).norm()<<endl<<endl;

    arap.Hessians(mesh);
    cout<<"Err"<<endl;
    MatrixXd fakeErr = Err(mesh, arap, E0, eps);
    cout<<arap.Err()<<endl;
    cout<<(fakeErr-arap.Err()).norm()<<endl;
    cout<<endl;
   
	MatrixXd fakeExx = Exx(mesh, arap, E0, eps);
	cout<<"Exx"<<endl;
	cout<<(fakeExx-MatrixXd(arap.Exx())).norm()<<endl<<endl;
	cout<<endl<<endl;

    MatrixXd fakeExr = Exr(mesh, arap, E0, eps);
    cout<<"Exr"<<endl;
    cout<<arap.Exr()<<endl;
    cout<<(fakeExr-arap.Exr()).norm()<<endl<<endl;
    cout<<endl<<endl;

    cout<<"Exs"<<endl;
    MatrixXd fakeExs = Exs(mesh, arap, E0, eps);
    cout<<arap.Exs()<<endl;
    cout<<(fakeExs-arap.Exs()).norm()<<endl<<endl;
    cout<<endl;

    cout<<"Ers"<<endl;
    MatrixXd fakeErs = Ers(mesh, arap, E0, eps);
    cout<<arap.Ers()<<endl<<endl;
    cout<<fakeErs<<endl;
    cout<<(fakeErs-arap.Ers()).norm()<<endl<<endl;
    cout<<endl;

    arap.Jacobians(mesh);
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
    cout<<"V size: "<<V.rows()<<endl;
    cout<<"T size: "<<T.rows()<<endl;
    cout<<"F size: "<<F.rows()<<endl;

    std::vector<int> bone1={};
    getMaxTets_Axis_Tolerance(bone1, V, T, 1, 3);
    VectorXi bone1vec = VectorXi::Map(bone1.data(), bone1.size());
    VectorXi all(T.rows());
    MatrixXd Uvec(all.size(), 3);
    for(int i=0; i<T.rows(); i++){
        all[i] = i;
        Uvec.row(i) = Vector3d::UnitY();
    }
    VectorXi muscle1;
    VectorXi shit;
    igl::setdiff(all, bone1vec, muscle1, shit);
   


    std::vector<VectorXi> bones = {bone1vec};
    std::vector<VectorXi> muscles = {muscle1};
    std::vector<int> fix_bones = {0};
    std::vector<int> mov = {};

    std::cout<<"-----Mesh-------"<<std::endl;
    Mesh* mesh = new Mesh(T, V, fix_bones, mov, bones, muscles, Uvec, j_input);

    std::cout<<"-----ARAP-----"<<std::endl;
    Reduced_Arap* redarap = new Reduced_Arap(*mesh);

    std::cout<<"-----Neo-------"<<std::endl;
    Elastic* neo = new Elastic(*mesh);

    cout<<"Done with init, setting values for s, r, x"<<endl;
    std::vector<double> s = {1.22027,    0.495075 ,     1.1781 , 0.00974582 ,
    	0.000871698 , 0.00620995 ,     1.2249 ,   0.467115  ,   1.21814 ,-0.00114247 ,
    	  0.0016028,  -0.0318313 ,    1.16293,    0.514761 ,    1.16153, -0.00925684,
    	  0.0640231, -0.00245551,     1.16368 ,   0.549197,     1.16291,  -0.0314069,
    	  0.0661537,  -0.0237411 ,    1.21513 ,   0.478347 ,    1.22011 ,  0.0469281,
    	  -0.00796292,   0.0371727 ,    1.17835 ,   0.494537 ,    1.22059,   -0.012968,
    	   -0.00408742 ,  0.0147056};

    cout<<s.size()<<", "<<mesh->red_s().size()<<endl;
    for(int i=0; i<s.size(); i++){
    	mesh->red_s()[i] = s[i];
    }

    redarap->minimize(*mesh);

    checkRedARAP(*mesh, *redarap);
    // checkElastic(*mesh, *neo);

}