#ifndef DISC_MESH_FUNCTIONS
#define DISC_MESH_FUNCTIONS

#include "store.h"
using Store = famu::Store;

namespace famu
{
	void setDiscontinuousMeshT(Eigen::MatrixXi& mT, Eigen::MatrixXi& discT);
	void discontinuousV(Store& store);

}
#endif