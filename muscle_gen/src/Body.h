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
		Mesh tendon_regions; // A manifold watertight mesh. If a muscle tet is contained in this volume, then it should be marked as a tendon
		
		TetMesh tet_mesh; // Tet mesh for whole muscle + bone system
		
		Eigen::MatrixXd combined_fiber_directions; // |T| x 3 (Zeros for bones)
		Eigen::MatrixXd harmonic_boundary_verts;
		Eigen::VectorXd combined_relative_stiffness;


		std::vector<int> joint_indices; // All tets belonging to joints

		// Split up and mapped by name
		std::map<std::string, TetMesh> split_tet_meshes;

		std::map<std::string, Mesh> bone_surfs;
		std::map<std::string, Mesh> muscle_surfs;

		std::map<std::string, std::vector<int>> bone_indices;
		std::map<std::string, std::vector<int>> muscle_indices;

		Eigen::VectorXi tet_is_tendon; // Vector of length |T|. is_tendon[i] == 1 iff tet i is a tendon (as well as a muscle), otherwise is_tendon[i] == 0

		void write(const std::string &out_dir);
		void read(const std::string &in_dir); // TODO not implemented
	};
}

#endif