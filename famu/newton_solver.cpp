// #include "newton_solver.h"
// #include "store.h"
// #include <Eigen/UmfPackSupport>
// #include <igl/polar_dec.h>
// #include <Eigen/LU>
// #include <Eigen/Cholesky>
// #include <igl/Timer.h>
// #include <omp.h>
// using Store = famu::Store;
// using namespace Eigen;
// using namespace std;



// void famu::polar_dec(Store& store, VectorXd& dFvec){
// 		if(store.jinput["polar_dec"]){
// 			//project bones dF back to rotations
// 			if(store.jinput["reduced"]){
// 				for(int b =0; b < store.bone_tets.size(); b++){
// 					Eigen::Matrix3d _r, _t;
// 					Matrix3d dFb = Map<Matrix3d>(dFvec.segment<9>(9*b).data()).transpose();
// 					igl::polar_dec(dFb, _r, _t);

// 					dFvec[9*b+0] = _r(0,0);
// 		      		dFvec[9*b+1] = _r(0,1);
// 		      		dFvec[9*b+2] = _r(0,2);
// 		      		dFvec[9*b+3] = _r(1,0);
// 		      		dFvec[9*b+4] = _r(1,1);
// 		      		dFvec[9*b+5] = _r(1,2);
// 		      		dFvec[9*b+6] = _r(2,0);
// 		      		dFvec[9*b+7] = _r(2,1);
// 		      		dFvec[9*b+8] = _r(2,2);
				
// 				}

// 			}else{
// 				for(int t = 0; t < store.bone_tets.size(); t++){
// 					for(int i=0; i<store.bone_tets[t].size(); i++){
// 						int b =store.bone_tets[t][i];

// 						Eigen::Matrix3d _r, _t;
// 						Matrix3d dFb = Map<Matrix3d>(dFvec.segment<9>(9*b).data()).transpose();
// 						igl::polar_dec(dFb, _r, _t);

// 						dFvec[9*b+0] = _r(0,0);
// 			      		dFvec[9*b+1] = _r(0,1);
// 			      		dFvec[9*b+2] = _r(0,2);
// 			      		dFvec[9*b+3] = _r(1,0);
// 			      		dFvec[9*b+4] = _r(1,1);
// 			      		dFvec[9*b+5] = _r(1,2);
// 			      		dFvec[9*b+6] = _r(2,0);
// 			      		dFvec[9*b+7] = _r(2,1);
// 			      		dFvec[9*b+8] = _r(2,2);
// 					}
// 				}
// 			}
// 		}
// 	}