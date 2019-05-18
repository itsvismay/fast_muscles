#ifndef BONE_ELEM_DEF_GRAD_PROJECTION_MATRIX
#define BONE_ELEM_DEF_GRAD_PROJECTION_MATRIX

#include "store.h"
using Store=famu::Store;

namespace famu
{
	void bone_def_grad_projection_matrix(Store& store, Eigen::SparseMatrix<double, Eigen::RowMajor>& mN, Eigen::SparseMatrix<double, Eigen::RowMajor>& mAN);
}

#endif