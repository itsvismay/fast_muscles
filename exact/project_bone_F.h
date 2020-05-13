#ifndef BONE_ELEM_DEF_GRAD_PROJECTION_MATRIX
#define BONE_ELEM_DEF_GRAD_PROJECTION_MATRIX

#include "store.h"
using Store=exact::Store;

namespace exact
{
	void project_bone_F(Store& store, Eigen::VectorXi& bone_or_muscle, Eigen::SparseMatrix<double, Eigen::RowMajor>& mN, Eigen::SparseMatrix<double, Eigen::RowMajor>& mAN);
}

#endif