#include "Body.h"

#include <igl/writeDMAT.h>
#include <igl/writeOBJ.h>
#include <igl/list_to_matrix.h>
#include <igl/boundary_facets.h>

#include "lf_utils.h"

using namespace muscle_gen;
using namespace Eigen;
using string = std::string;

void Body::write(const std::string &out_dir) {
	string surf_path = lf::path::join(out_dir, "input_surf_mesh.obj");
	igl::writeOBJ(surf_path, surf_mesh.V, surf_mesh.F);

	string T_path = lf::path::join(out_dir, "tet_mesh_T.dmat");
	igl::writeDMAT(T_path, tet_mesh.T, false);

	string V_path = lf::path::join(out_dir, "tet_mesh_V.dmat");
	igl::writeDMAT(V_path, tet_mesh.V, false);

	string combined_relative_stiffness_path = lf::path::join(out_dir, "combined_relative_stiffness.dmat");
	igl::writeDMAT(combined_relative_stiffness_path, combined_relative_stiffness, false);

	string combined_fiber_directions_path = lf::path::join(out_dir, "combined_fiber_directions.dmat");
	combined_fiber_directions.rowwise().normalize();
	igl::writeDMAT(combined_fiber_directions_path, combined_fiber_directions, false);

	string joint_indices_path = lf::path::join(out_dir, "joint_indices.dmat");
	VectorXi joint_indices_I;
	igl::list_to_matrix(joint_indices, joint_indices_I);
	igl::writeDMAT(joint_indices_path, joint_indices_I, false);

	// for(const auto &el : split_tet_meshes) {
	// 	auto name = el.first;

	// 	T_path = lf::path::join(out_dir, name + "_split_tet_mesh_T.dmat");
	// 	igl::writeDMAT(T_path, split_tet_meshes[name].T, false);

	// 	V_path = lf::path::join(out_dir, name + "_split_tet_mesh_V.dmat");
	// 	igl::writeDMAT(V_path, split_tet_meshes[name].V, false);
	// }

	for(const auto &el : bone_indices) {
		auto name = el.first;

		string bone_indices_path = lf::path::join(out_dir, name + "_bone_indices.dmat");
		VectorXi bone_indices_I;
		igl::list_to_matrix(bone_indices[name], bone_indices_I);
		igl::writeDMAT(bone_indices_path, bone_indices_I, false);
	}

	for(const auto &el : muscle_indices) {
		auto name = el.first;

		string muscle_indices_path = lf::path::join(out_dir, name + "_muscle_indices.dmat");
		VectorXi muscle_indices_I;
		igl::list_to_matrix(muscle_indices[name], muscle_indices_I);
		igl::writeDMAT(muscle_indices_path, muscle_indices_I, false);
	}

	string is_tendon_path = lf::path::join(out_dir, "tet_is_tendon.dmat");
	igl::writeDMAT(is_tendon_path, tet_is_tendon, false);

	MatrixXi surfaceF;
	igl::boundary_facets(tet_mesh.T, surfaceF);
	string tet_surf_path = lf::path::join(out_dir, "tet_surf_mesh.obj");
	igl::writeOBJ(tet_surf_path, tet_mesh.V, surfaceF);
}