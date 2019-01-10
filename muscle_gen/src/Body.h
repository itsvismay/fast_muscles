#ifndef BODY_H
#define BODY_H

#include <Eigen/Core>

#include <vector>
#include <map>
#include <string>

#include "MeshTypes.h"

namespace muscle_gen {

	struct Body {
		// Combined structures
		Mesh surf_mesh; // Simple (no remeshing) union of surface meshes
		
		TetMesh tet_mesh; // Tet mesh for whole muscle + bone system
		
		Eigen::MatrixXd combined_fiber_directions; // |T| x 3 (Zeros for bones)
		Eigen::MatrixXd harmonic_boundary_verts;


		std::vector<int> joint_indices; // All tets belonging to joints

		// Split up and mapped by name
		std::map<std::string, TetMesh> split_tet_meshes;

		std::map<std::string, Mesh> bone_surfs;
		std::map<std::string, Mesh> muscle_surfs;

		std::map<std::string, std::vector<int>> bone_indices;
		std::map<std::string, std::vector<int>> muscle_indices;

		void write(const std::string &out_dir);
		void read(const std::string &in_dir); // TODO not implemented
	};
}

#endif