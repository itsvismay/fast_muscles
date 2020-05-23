#include "linesearch.h"
#include "muscle_energy_gradient.h"
#include "stablenh_energy_gradient.h"

#include <igl/polar_dec.h>

using namespace Eigen;
using namespace std;

void exact::polar_dec(VectorXd& x, const std::vector<Eigen::VectorXi>& bone_tets){
		
	for(int t = 0; t < bone_tets.size(); t++){
		for(int i=0; i < bone_tets[t].size(); i++){
			int b = bone_tets[t][i];

			Eigen::Matrix3d _r, _t;
			Matrix3d dFb = Map<Matrix3d>(x.segment<9>(9*b).data()).transpose();
			igl::polar_dec(dFb, _r, _t);

			x[9*b+0] = _r(0,0);
      		x[9*b+1] = _r(0,1);
      		x[9*b+2] = _r(0,2);
      		x[9*b+3] = _r(1,0);
      		x[9*b+4] = _r(1,1);
      		x[9*b+5] = _r(1,2);
      		x[9*b+6] = _r(2,0);
      		x[9*b+7] = _r(2,1);
      		x[9*b+8] = _r(2,2);
		}
	}
		
	
}

double exact::linesearch(int& tot_ls_its, 
						VectorXd& Fvec,  
						const VectorXd& grad, 
						const VectorXd& drt, 
						double activation,
						VectorXd& q, 
                        const MatrixXi& T, 
                        const VectorXd& eY, 
                        const VectorXd& eP, 
                        const VectorXd& rest_tet_vols, 
                        const MatrixXd& Uvec,
						const Eigen::SparseMatrix<double, Eigen::RowMajor>& Y, 
						const Eigen::SparseMatrix<double, Eigen::RowMajor>& B, 
						const SparseMatrix<double, Eigen::RowMajor>& PF,
						const VectorXd& c,
                        const std::vector<Eigen::VectorXi>& bone_tets){
    
	// Decreasing and increasing factors
	VectorXd x = Fvec;
	VectorXd xp = x;
	double step = 1;
	
    const double dec = 0.5;
    const double inc = 2.1;
    int pmax_linesearch = 100;
    int plinesearch = 1;//1 for armijo, 2 for wolfe
    double pftol = 1e-4;
    double pwolfe = 0.9;
    double pmax_step = 1e8;
    double pmin_step = 1e-20;


    // Check the value of step
    if(step <= double(0))
        std::invalid_argument("'step' must be positive");

    //polar_dec(store, x);
    //exact::acap_solve( q, PF, ACAP, Y, B, x, c);

   	double fx = exact::stablenh::energy(x, T, eY, eP, rest_tet_vols) 
                + activation*exact::muscle::energy(x, T, rest_tet_vols, Uvec);
    // Save the function value at the current x
    const double fx_init = fx;
    // Projection of gradient on the search direction
    const double dg_init = grad.dot(drt);
    // Make sure d points to a descent direction
    if(dg_init > 0)
        std::logic_error("the moving direction increases the objective function value");

    const double dg_test = pftol * dg_init;
    double width;


    int iter;
    for(iter = 0; iter < pmax_linesearch; iter++)
    {
    	
        // x_{k+1} = x_k + step * d_k
        x = xp + step * drt;

        //polar_dec(x, bone_tets);

        // Evaluate this candidate
        // store.printState(iter, "ls", q);
        //exact::acap_solve(q, PF, ACAP, Y, B, x, c);
       	fx = exact::stablenh::energy(x, T, eY, eP, rest_tet_vols) 
                + activation*exact::muscle::energy(x, T, rest_tet_vols, Uvec);

        if(fx > fx_init + step * dg_test)
        {
            width = dec;
        } else {
            // Armijo condition is met
            if(plinesearch == 1)
                break;

            const double dg = grad.dot(drt);
            if(dg < pwolfe * dg_init)
            {
                width = inc;
            } else {
                // Regular Wolfe condition is met
                if(plinesearch == 2)
                    break;

                if(dg > -pwolfe * dg_init)
                {
                    width = dec;
                } else {
                    // Strong Wolfe condition is met
                    break;
                }
            }
        }
        tot_ls_its += iter;
       

        if(iter >= pmax_linesearch)
       		throw std::runtime_error("the line search routine reached the maximum number of iterations");
        


        if(step < pmin_step)
        {
        	return 0; //throw std::runtime_error("the line search step became smaller than the minimum value allowed");
        } 
            

        if(step > pmax_step)
            throw std::runtime_error("the line search step became larger than the maximum value allowed");

        step *= width;
    }
    // cout<<"			ls iters: "<<iter<<endl;
    cout<<"			step: "<<step<<endl;
    return step;
}