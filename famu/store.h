#ifndef STORE 
#define STORE 
#include <json.hpp>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <igl/Timer.h>
#include <Eigen/LU>
#include <Eigen/UmfPackSupport>
#include <iostream>
#include <fstream>

#define EIGEN_USE_MKL_ALL
#ifdef EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#endif
#include <omp.h>

#define NUM_MODES 48
typedef Eigen::Triplet<double> Trip;
typedef Eigen::Triplet<int> iTrip;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;
typedef Eigen::Matrix<double, 9, NUM_MODES> Matrix9xModes;
typedef Eigen::Matrix<double, NUM_MODES, NUM_MODES> MatrixModesxModes;

namespace famu{

	struct Store{
		nlohmann::json jinput;
		igl::Timer timer;

		Eigen::VectorXd muscle_mag;
		Eigen::VectorXi bone_or_muscle;
		Eigen::VectorXd rest_tet_volume;
		
		double alpha_arap = 1e4;
		double alpha_neo = 1;
		double restNeo =0;

		Eigen::MatrixXd V, discV;
		Eigen::MatrixXi T, discT, F, discF;
		Eigen::MatrixXd Uvec;
		std::vector<std::string> fix_bones = {};
		std::vector<Eigen::VectorXi> bone_tets = {};
		std::vector<Eigen::VectorXi> muscle_tets = {};
		std::map<std::string, int> bone_name_index_map;
		std::map<std::string, int> muscle_name_index_map;
		std::vector< std::pair<std::vector<std::string>, Eigen::MatrixXd>> joint_bones_verts;
		Eigen::VectorXd relativeStiffness;
		Eigen::VectorXd eY, eP;
        
        // Young's Modulus per vertex (averaged from incident tets using eY)
		Eigen::VectorXd elogVY;
		std::vector<double> bone_vols;
		std::vector<int> fixverts, movverts;
		std::vector<int> mfix, mmov;

		Eigen::SparseMatrix<double, Eigen::RowMajor> dF;
		Eigen::SparseMatrix<double, Eigen::RowMajor> D, _D;
		Eigen::SparseMatrix<double, Eigen::RowMajor> C;
		Eigen::SparseMatrix<double, Eigen::RowMajor> S;
		Eigen::SparseMatrix<double, Eigen::RowMajor> ProjectF, PickBoneF;
		Eigen::SparseMatrix<double, Eigen::RowMajor> ConstrainProjection, UnconstrainProjection;
		Eigen::SparseMatrix<double, Eigen::RowMajor> JointConstraints, NullJ;
		Eigen::SparseMatrix<double, Eigen::RowMajor> Y, Bx, Bf;
		Eigen::MatrixXd G;

		Eigen::VectorXd dFvec, BfI0;
		Eigen::VectorXd x, dx, x0, lambda2, acap_solve_result, acap_solve_rhs;

		
		// Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> SPLU;
		Eigen::PardisoLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> ACAP_KKT_SPLU;
		Eigen::VectorXd acaptmp_sizex;
		Eigen::VectorXd acaptmp_sizedFvec1, acaptmp_sizedFvec2;
		Eigen::UmfPackLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> NM_SPLU;//TODO: optimize this away
		
		
		//Fast Terms
		double x0tStDtDSx0;
		Eigen::VectorXd x0tStDtDSY;
		Eigen::SparseMatrix<double, Eigen::RowMajor> YtStDtDSY; //ddE/dxdx
		Eigen::VectorXd x0tStDt_dF_DSx0;
		Eigen::SparseMatrix<double, Eigen::RowMajor> YtStDt_dF_DSx0; //ddE/dxdF
		Eigen::SparseMatrix<double, Eigen::RowMajor> x0tStDt_dF_dF_DSx0; //ddE/dFdF

		Eigen::VectorXd DSx0;
		Eigen::SparseMatrix<double, Eigen::RowMajor> DSY;
		Eigen::SparseMatrix<double, Eigen::RowMajor> StDtDS;
		Eigen::SparseMatrix<double, Eigen::RowMajor> DSx0_mat;
		std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> fastMuscles;
		Eigen::SparseMatrix<double, Eigen::RowMajor> JacdxdF;

		//Hessians
		Eigen::SparseMatrix<double, Eigen::RowMajor> acapHess, muscleHess, neoHess;
		Eigen::MatrixXd denseAcapHess, denseMuscleHess, denseNeoHess;

		//woodbury matrices
		//(A + BCD)-1
		//A = Hneo + Hmuscle + P'Z'ZP
		//B = -P'Z'DSYG
		//C = Inv(GtYtStDtDSYG)
		//D = G'Y'S'D'ZP
		std::vector<Eigen::LDLT<Matrix9d>> vecInvA;
		Eigen::MatrixXd WoodB, WoodD, InvC, WoodC;
		Eigen::VectorXd eigenvalues;
		std::vector<int> contract_muscles;


		//External Force matrices
		Eigen::VectorXd YtMg, ContactForce, GravityForce;
		Eigen::SparseMatrix<double, Eigen::RowMajor> ContactP, ContactP1, ContactP2, ContactHess;


		template <typename T, int whatever, typename IND>
		int Serialize(Eigen::SparseMatrix<T, whatever, IND>& m, std::string input_file) {
		    std::vector<iTrip> res;
		    int sz = m.nonZeros();
		    m.makeCompressed();

		    std::fstream writeFile;
		    writeFile.open(input_file, std::ios::binary | std::ios::out);

		    if(writeFile.is_open())
		    {
		        IND rows, cols, nnzs, outS, innS;
		        rows = m.rows()     ;
		        cols = m.cols()     ;
		        nnzs = m.nonZeros() ;
		        outS = m.outerSize();
		        innS = m.innerSize();

		        writeFile.write((const char *)&(rows), sizeof(IND));
		        writeFile.write((const char *)&(cols), sizeof(IND));
		        writeFile.write((const char *)&(nnzs), sizeof(IND));
		        writeFile.write((const char *)&(outS), sizeof(IND));
		        writeFile.write((const char *)&(innS), sizeof(IND));

		        writeFile.write((const char *)(m.valuePtr()),       sizeof(T  ) * m.nonZeros());
		        writeFile.write((const char *)(m.outerIndexPtr()),  sizeof(IND) * m.outerSize());
		        writeFile.write((const char *)(m.innerIndexPtr()),  sizeof(IND) * m.nonZeros());

		        writeFile.close();
		    }
		    return 1;
		}

		template <typename T, int whatever, typename IND>
		int Deserialize(Eigen::SparseMatrix<T, whatever, IND>& m, std::string input_file) {
		    std::fstream readFile;
		    readFile.open(input_file, std::ios::binary | std::ios::in);
		    if(!readFile.good()){
		    	return 0;
		    }
		    if(readFile.is_open())
		    {
		        IND rows, cols, nnz, inSz, outSz;
		        readFile.read((char*)&rows , sizeof(IND));
		        readFile.read((char*)&cols , sizeof(IND));
		        readFile.read((char*)&nnz  , sizeof(IND));
		        readFile.read((char*)&inSz , sizeof(IND));
		        readFile.read((char*)&outSz, sizeof(IND));

		        m.resize(rows, cols);
		        m.makeCompressed();
		        m.resizeNonZeros(nnz);

		        readFile.read((char*)(m.valuePtr())     , sizeof(T  ) * nnz  );
		        readFile.read((char*)(m.outerIndexPtr()), sizeof(IND) * outSz);
		        readFile.read((char*)(m.innerIndexPtr()), sizeof(IND) * nnz );

		        m.finalize();
		        readFile.close();

		    } // file is open
		    else{
		    	return 0;
		    }
		    return 1;
		}
	};
}

#endif
