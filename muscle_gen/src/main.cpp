#include <iostream>
#include <vector>

#include <tetwild/tetwild.h>
#include <tetwild/Args.h>

#include <imgui/imgui.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include <igl/barycenter.h>
#include <igl/combine.h>
#include <igl/grad.h>
#include <igl/harmonic.h>
#include <igl/list_to_matrix.h>
#include <igl/readDMAT.h>
#include <igl/readOBJ.h>
#include <igl/remove_unreferenced.h>
#include <igl/signed_distance.h>
#include <igl/slice.h>
#include <igl/unique.h>
#include <igl/writeDMAT.h>

#include <nlohmann/json.hpp>

#include "lf_utils.h"

#include "MeshTypes.h"
#include "viewer.h"

using namespace muscle_gen;
using namespace Eigen;
using json = nlohmann::json;

struct Args {
	bool load_existing_tets = false;
	std::string config_path;
}

Args parse_args(int argc, char *argv) {
	Args args;

	if(argc != 2 && argc != 3) {
		std::cout << "Usage: ./muscle_gen <config>.json [optional] --load_existing_tets " << std::endl;
		exit(0);
	}
	
	args.config_path = argv[1];

	if(argc >= 3) {
		if(std::string(argv[2]) == "--load_existing_tets") {
			args.load_existing_tets = true;
		}
	}

	return args;
}

int main(int argc, char *argv[])
{
	Args args = parse_args(argc, argv);
	

	std::string output_dir = argv[1];
	std::vector<std::string> paths = {argv[2], argv[3], argv[4]};
	std::vector<std::string> names;
	std::vector<Mesh> meshes;

	// Load the meshes
	for(const auto &path : paths) {
		std::string name = lf::path::base_name(path, false);
		Mesh mesh;
		igl::readOBJ(path, mesh.V, mesh.F);
		meshes.push_back(mesh);
		names.push_back(name);
	}

	// Combine them into one mesh
	Mesh combined;
	std::vector<MatrixXd> Vs;
	std::vector<MatrixXi> Fs;
	for(const auto &mesh : meshes) {
		Vs.push_back(mesh.V);
		Fs.push_back(mesh.F);
	}
	igl::combine(Vs, Fs, combined.V, combined.F);

	// Tetrahedralize it
	TetMesh combined_tet_mesh;
	if(load_existing_tets) {
		igl::readDMAT((output_dir + "/combined_T.dmat"), combined_tet_mesh.T);
		igl::readDMAT((output_dir + "/combined_V.dmat"), combined_tet_mesh.V);
	} else {
		tetwild::Args tetwild_args;
		tetwild::tetrahedralization(combined.V, combined.F, combined_tet_mesh.V, combined_tet_mesh.T, combined_tet_mesh.A, tetwild_args);
	}

	// Now determine which tets are which
	Eigen::MatrixXd centers;
	igl::barycenter(combined_tet_mesh.V,combined_tet_mesh.T,centers);

	// For each mesh, compute the signed distances from the barycenter of each tet
	std::vector<VectorXd> distances_per_mesh;
	for(const auto &mesh : meshes) {
		VectorXd S;
		VectorXi I;
		MatrixXd C, N;
		igl::signed_distance(centers, mesh.V, mesh.F, igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_WINDING_NUMBER, S, I, C, N);
		distances_per_mesh.push_back(S);
	}

	// For each tet, find the first mesh (in listed order) that contains its center
	std::vector<std::vector<int>> tet_indices_per_mesh(meshes.size(), std::vector<int>());
	for(int i = 0; i < combined_tet_mesh.T.rows(); i++) {
		for(int j = 0; j < meshes.size(); j++) {
			const double dist_to_mesh_j = distances_per_mesh[j](i);
			if(dist_to_mesh_j <= 0) {
				tet_indices_per_mesh[j].push_back(i);
				break;
			}
		}
	}
	
	// Split into separate meshes for debugging
	std::vector<TetMesh> tet_meshes;
	for(const auto &tet_indices : tet_indices_per_mesh) {
		MatrixXi T(tet_indices.size(), 4);
		for(int i = 0; i < tet_indices.size(); i++) {
			T.row(i) = combined_tet_mesh.T.row(tet_indices[i]);
		}
		TetMesh tet_mesh;
		tet_mesh.T = T;
		tet_mesh.V = combined_tet_mesh.V; // TODO shouldn't really store duplicates
		tet_meshes.push_back(tet_mesh);
	}


	// Do the poisson solve to get fiber directions
	// Identify boundary verts
	std::vector<Mesh> bone_surface_meshes = {meshes[0], meshes[1]}; // TODO need better way of specifying this
	TetMesh muscle_mesh_temp = tet_meshes[2];
	TetMesh muscle_mesh;
	MatrixXi temp;
	igl::remove_unreferenced(muscle_mesh_temp.V, muscle_mesh_temp.T, muscle_mesh.V, muscle_mesh.T, temp);
	VectorXi muscle_verts_I;
	igl::unique(muscle_mesh.T, muscle_verts_I); // TODO need better way of specifying which mesh to do this on
	// VectorXd muscle_verts_V;
	// igl::slice(combined_tet_mesh.V, muscle_verts_I, 1, muscle_verts_V);

	// Boundary conditions
	std::vector<VectorXd> muscle_to_bone_dists;
	for(const auto &mesh : bone_surface_meshes) {
		VectorXd S;
		VectorXi I;
		MatrixXd C, N;
		igl::signed_distance(muscle_mesh.V, mesh.V, mesh.F, igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_WINDING_NUMBER, S, I, C, N);
		muscle_to_bone_dists.push_back(S);
	}

	std::vector<int> boundary_vert_Is;
	const double tolerance = 1e-5; // TODO what's a good tolerance?
	for(int i = 0; i < muscle_verts_I.size(); i++) {
		const int muscle_vert_index = muscle_verts_I[i];
		for(const auto &muscle_to_bone_dist : muscle_to_bone_dists) {
			if(std::abs(muscle_to_bone_dist[muscle_vert_index]) <= tolerance) {
				boundary_vert_Is.push_back(muscle_vert_index);
				break;
			}
		}
	}

	VectorXi b;
	igl::list_to_matrix(boundary_vert_Is, b);
	// MatrixXd bc = RowVector3d(0.0, 1.0, 0.0).replicate(b.size(), 1); // TODO How do I specify boundary direction better? Get the normal?
	VectorXd bc(b.size(), 1);

	int top_vert_i, bottom_vert_i;
	Mesh muscle_surface_mesh = meshes[2]; // TODO need a better way
	muscle_surface_mesh.V.col(1).maxCoeff(&top_vert_i);
	muscle_surface_mesh.V.col(1).minCoeff(&bottom_vert_i);

	VectorXd top_vert = muscle_surface_mesh.V.row(top_vert_i);
	VectorXd bottom_vert = muscle_surface_mesh.V.row(bottom_vert_i);
	for(int i = 0; i < b.size(); i++) {
		VectorXd v = muscle_mesh.V.row(b(i));
		VectorXd top_to_v = v - top_vert;
		VectorXd bottom_to_v = v - bottom_vert;
		if(top_to_v.norm() > bottom_to_v.norm()) {
			bc(i) = 1.0;
		} else {
			bc(i) = -1.0;
		}
	}

	MatrixXd W;
	igl::harmonic(muscle_mesh.V, muscle_mesh.T, b, bc, 2, W);

	SparseMatrix<double> G;
	igl::grad(muscle_mesh.V, muscle_mesh.T, G);

	MatrixXd fiber_directions = Map<const MatrixXd>((G*W).eval().data(), muscle_mesh.T.rows(), 3);
	fiber_directions.rowwise().normalize();

	double fiber_scale = 0.1;
	std::vector<MatrixXd> fiber_edges(2);
	MatrixXd muscle_centers;
	igl::barycenter(muscle_mesh.V, muscle_mesh.T, muscle_centers);
	fiber_edges[0] = muscle_centers;
	fiber_edges[1] = muscle_centers +  fiber_directions * fiber_scale;

	// Map them back to the combined mesh
	MatrixXd combined_fiber_directions = MatrixXd::Zero(combined_tet_mesh.T.rows(), 3);
	const std::vector<int> &muscle_to_combined_indices = tet_indices_per_mesh[2]; // TODO improve this
	for(int i = 0; i < fiber_directions.rows(); i++) {
		combined_fiber_directions.row(muscle_to_combined_indices[i]) = fiber_directions.row(i);
	}


	//Save to disk
	//json config; // TODO

	igl::writeDMAT((output_dir + "/combined_T.dmat"), combined_tet_mesh.T, false);
	igl::writeDMAT((output_dir + "/combined_V.dmat"), combined_tet_mesh.V, false);
	igl::writeDMAT((output_dir + "/fiber_directions.dmat"), combined_fiber_directions, false);
	for (int i = 0; i < meshes.size(); ++i) {
		VectorXi indices;
		igl::list_to_matrix(tet_indices_per_mesh[i], indices);
		igl::writeDMAT((output_dir + "/" + names[i] + "_I.dmat"), indices, false);
	}


	// Launch viewer with simple menu
	launch_viewer(tet_meshes, fiber_edges);
}