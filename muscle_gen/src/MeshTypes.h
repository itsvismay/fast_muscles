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

		// Shouldn't be here. Hack.
		Eigen::VectorXi orig_indices; // Unique vert indices in *original* vert mesh
		Eigen::VectorXi I; // Optional mapping of indices back to another mesh
		Eigen::VectorXi J;
	};
}

#endif