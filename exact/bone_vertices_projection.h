#ifndef BONE_VERTS_PROJECTION
#define BONE_VERTS_PROJECTION
#include "store.h"

using Store=exact::Store;
using namespace Eigen;

namespace exact
{
	void bone_vertices_projection(Store& store, Eigen::SparseMatrix<double,Eigen::RowMajor>& mY);

}
#endif