#ifndef MESH_TYPES_H
#define MESH_TYPES_H

#include <Eigen/Core>

namespace muscle_gen {
	struct Mesh {
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
	};


	struct TetMesh {
		Eigen::MatrixXd V;
		Eigen::MatrixXi T;
	};
}

#endif