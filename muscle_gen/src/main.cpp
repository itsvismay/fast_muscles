#include <iostream>
#include <vector>

#include <tetwild/tetwild.h>
#include <tetwild/Args.h>

#include <imgui/imgui.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include <igl/barycenter.h>
#include <igl/boundary_facets.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/combine.h>
#include <igl/grad.h>
#include <igl/harmonic.h>
#include <igl/intersect.h>
#include <igl/list_to_matrix.h>
#include <igl/readDMAT.h>
#include <igl/readOBJ.h>
#include <igl/remove_unreferenced.h>
#include <igl/signed_distance.h>
#include <igl/slice.h>
#include <igl/writeDMAT.h>
#include <igl/unique.h>

#include <nlohmann/json.hpp>

#include "lf_utils.h"

#include "MeshTypes.h"
#include "Body.h"
#include "viewer.h"

using namespace muscle_gen;
using namespace Eigen;
using json = nlohmann::json;

using string = std::string;
using std::map;

struct Args {
	bool load_existing_tets = false;
	string body_dir;
};

Args parse_args(int argc, char *argv[]) {
	Args args;

	if(argc != 2 && argc != 3) {
		std::cout << "Usage: ./muscle_gen <body_dir> [optional] --load_existing_tets " << std::endl;
		exit(0);
	}
	
	args.body_dir = argv[1];

	if(argc >= 3) {
		if(string(argv[2]) == "--load_existing_tets") {
			args.load_existing_tets = true;
		}
	}

	return args;
}


json read_config(const string &body_dir) {
	string config_path = lf::path::join(body_dir, "config.json");
	std::ifstream json_ifstream(config_path);

	json config;
	json_ifstream >> config;
	return config;
}


Mesh load_mesh(const string &obj_path) {
	Mesh mesh;
	igl::readOBJ(obj_path, mesh.V, mesh.F);
	return mesh;
}


void read_surfs(const json &config, const string &surf_dir, map<string, Mesh> &bone_surfs, map<string, Mesh> &muscle_surfs) {	
	for (auto& el : config["muscles"].items()) {
		string surf_path = lf::path::join(surf_dir, el.value()["surface_obj"]);
		muscle_surfs[el.key()] = load_mesh(surf_path);
	}

	for (auto& el : config["bones"].items()) {
		string surf_path = lf::path::join(surf_dir, el.value()["surface_obj"]);
		bone_surfs[el.key()] = load_mesh(surf_path);
	}
}


Mesh combine_surfs(map<string, Mesh> &bone_surfs, map<string, Mesh> &muscle_surfs) {
	Mesh combined;
	std::vector<MatrixXd> Vs;
	std::vector<MatrixXi> Fs;
	for (const auto &el : muscle_surfs) {
		Vs.push_back(el.second.V);
		Fs.push_back(el.second.F);
	}
	for (const auto &el : bone_surfs) {
		Vs.push_back(el.second.V);
		Fs.push_back(el.second.F);
	}
	igl::combine(Vs, Fs, combined.V, combined.F);
	return combined;
}


TetMesh tetrahedralize_mesh(const Mesh &surf_mesh, double eps_rel) {
	TetMesh combined_tet_mesh;
	tetwild::Args tetwild_args;
	tetwild_args.initial_edge_len_rel = tetwild_args.initial_edge_len_rel*1;
	tetwild_args.eps_rel = eps_rel; // Tetwild default is 0.1

	VectorXd A;
	tetwild::tetrahedralization(surf_mesh.V, surf_mesh.F, combined_tet_mesh.V, combined_tet_mesh.T, A, tetwild_args);
	return combined_tet_mesh;
}


TetMesh load_body_tet_mesh(const string &body_dir) {
	string tets_dir = lf::path::join(body_dir, "generated_files/");
	string T_path = lf::path::join(tets_dir, "temp_saved_T.dmat");
	string V_path = lf::path::join(tets_dir, "temp_saved_V.dmat");

	TetMesh tet_mesh;
	std::cout << "Reading " << T_path << std::endl;
	std::cout << "Reading " << V_path << std::endl;
	igl::readDMAT(T_path, tet_mesh.T);
	igl::readDMAT(V_path, tet_mesh.V);	

	return tet_mesh;
}

void save_body_tet_mesh(const string &body_dir, const TetMesh &tet_mesh) {
	string tets_dir = lf::path::join(body_dir, "generated_files/");
	string T_path = lf::path::join(tets_dir, "temp_saved_T.dmat");
	string V_path = lf::path::join(tets_dir, "temp_saved_V.dmat");

	std::cout << "Writing " << T_path << std::endl;
	std::cout << "Writing " << V_path << std::endl;
	igl::writeDMAT(T_path, tet_mesh.T, false);
	igl::writeDMAT(V_path, tet_mesh.V, false);	
}


void compute_indices(
	double eps_absolute, // Computed from tetwild relative epsilon
	const TetMesh &tet_mesh,
	const map<string, Mesh> &bone_surfs,
	const map<string, Mesh> &muscle_surfs,
	map<string, std::vector<int>> &bone_indices,
	map<string, std::vector<int>> &muscle_indices)
{
	Eigen::MatrixXd tet_centers;
	igl::barycenter(tet_mesh.V, tet_mesh.T, tet_centers);

	map<string, VectorXd> bone_dists;
	map<string, VectorXd> muscle_dists;
	for(const auto &el : bone_surfs) {
		VectorXd S; VectorXi I; MatrixXd C, N;
		igl::signed_distance(tet_centers, el.second.V, el.second.F, igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_WINDING_NUMBER, S, I, C, N);
		bone_dists[el.first] = S;
	}
	for(const auto &el : muscle_surfs) {
		VectorXd S; VectorXi I; MatrixXd C, N;
		igl::signed_distance(tet_centers, el.second.V, el.second.F, igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_WINDING_NUMBER, S, I, C, N);
		muscle_dists[el.first] = S;
	}


	// For each tet, find the first mesh (in listed order) that contains its center
	// TODO: This isn't guaranteed to assign all tets. Could have NaNs or other edge cases.
	int unassigned_tets = 0;
	for(int i = 0; i < tet_mesh.T.rows(); i++) {
		bool assigned_tet = false;
		bool found_bone = false;
		// bones
		for(const auto &el : bone_dists) {
			const VectorXd &dists = el.second;
			// We check against eps_absolute because this is what tetwild guarantees as tolerance
			if(dists(i) <= eps_absolute) {
				bone_indices[el.first].push_back(i);
				found_bone = true;
				assigned_tet = true;
				break;
			}
		}
		// muscles
		if(!found_bone) {
			for(const auto &el : muscle_dists) {
				const VectorXd &dists = el.second;
				if(dists(i) <= eps_absolute) {
					muscle_indices[el.first].push_back(i);
					assigned_tet = true;
					break;
				}
			}
		}

		if(!assigned_tet) {
			unassigned_tets++;
			for(const auto &el : muscle_dists) {
				const VectorXd &dists = el.second;
				std::cout << dists(i) << std::endl;
			}
			for(const auto &el : bone_dists) {
				const VectorXd &dists = el.second;
				std::cout << dists(i) << std::endl;
			}
			std::cout << std::endl;
		}
	}
	std::cout << "# Unassigned tets: " << unassigned_tets << std::endl;
	std::cout << "eps_absolute: " << eps_absolute << std::endl;
}


void split_tet_meshes(
	const TetMesh &tet_mesh,
	const map<string, std::vector<int>> &bone_indices,
	const map<string, std::vector<int>> &muscle_indices,
	std::map<string, TetMesh> &split_tet_meshes)
{
	auto both_indices = {bone_indices, muscle_indices};
	for(const auto &indices_map : both_indices) {
		for(const auto &el : indices_map) {
			TetMesh temp_tet_mesh;
			const std::vector<int> &indices = el.second;
			temp_tet_mesh.T.resize(indices.size(), 4);
			for(int i = 0; i < indices.size(); i++) {
				temp_tet_mesh.T.row(i) = tet_mesh.T.row(indices[i]);	
			}
			temp_tet_mesh.V = tet_mesh.V;

			TetMesh split_tet_mesh;
			MatrixXi temp;
			igl::unique(temp_tet_mesh.T, split_tet_mesh.orig_indices);
			igl::remove_unreferenced(temp_tet_mesh.V, temp_tet_mesh.T, split_tet_mesh.V, split_tet_mesh.T, split_tet_mesh.I, split_tet_mesh.J);

			split_tet_meshes[el.first] = split_tet_mesh;
		}
	}
}


void compute_muscle_fibers(const Body &body, MatrixXd &combined_fiber_directions, MatrixXd &harmonic_boundary_verts) {

	// For each muscle
	// 	Identify attachment points
	// 	set boundary conditions
	// 	Do harmonic solve
	// 	Gradient
	// 	Store fiber directions
	combined_fiber_directions = MatrixXd::Zero(body.tet_mesh.T.rows(), 3);
	std::vector<std::vector<double>> harmonic_boundary_verts_vec;
	for(const auto &el : body.muscle_indices) {
		const auto &muscle_name = el.first;
		const TetMesh &muscle_tet_mesh = body.split_tet_meshes.at(muscle_name);

		// Need to compute boundary conditions for harmonic solve
		VectorXi boundary_verts;
		VectorXd boundary_vals;
	

		std::vector<int> boundary_verts_vec;
		std::vector<double> boundary_vals_vec;
		bool first_bone_found = false; // Only works (for sure) for muscles with two attachments
		for(const auto &bone_el : body.bone_indices) {
			const auto &bone_name = el.first;
			const TetMesh &bone_tet_mesh = body.split_tet_meshes.at(bone_el.first);

			VectorXi inBoth;
			igl::intersect(bone_tet_mesh.orig_indices, muscle_tet_mesh.orig_indices, inBoth);
			
			double bval = 0.0;
			if(inBoth.size() > 0) {
				if(!first_bone_found) {
					bval = 1.0;
				} else {
					bval = -1.0;
				}
				first_bone_found = true;
			}

			for(int j = 0; j < inBoth.size(); j++) {
				int muscle_vert_index = muscle_tet_mesh.I(inBoth(j));
				boundary_vals_vec.push_back(bval);
				boundary_verts_vec.push_back(muscle_vert_index);
				harmonic_boundary_verts_vec.push_back({muscle_tet_mesh.V(muscle_vert_index, 0), muscle_tet_mesh.V(muscle_vert_index, 1), muscle_tet_mesh.V(muscle_vert_index, 2)});
			}
		}
		igl::list_to_matrix(boundary_verts_vec, boundary_verts);
		igl::list_to_matrix(boundary_vals_vec, boundary_vals);

		// Now do the harmonic solve and compute the gradient
		MatrixXd W;
		igl::harmonic(muscle_tet_mesh.V, muscle_tet_mesh.T, boundary_verts, boundary_vals, 1, W);

		SparseMatrix<double> G;
		igl::grad(muscle_tet_mesh.V, muscle_tet_mesh.T, G);

		MatrixXd fiber_directions = Map<const MatrixXd>((G*W).eval().data(), muscle_tet_mesh.T.rows(), 3);
		//fiber_directions.rowwise().normalize();

		// Map them back to the combined mesh
		for(int i = 0; i < fiber_directions.rows(); i++) {
			combined_fiber_directions.row(el.second[i]) = fiber_directions.row(i);
		}
	}
	igl::list_to_matrix(harmonic_boundary_verts_vec, harmonic_boundary_verts);
}

void compute_element_stiffness(const Body& body, VectorXd& combined_relative_stiffness){
	combined_relative_stiffness = VectorXd::Ones(body.tet_mesh.T.rows());

	for(const auto &el: body.muscle_indices){
		VectorXd grad_norms = VectorXd::Zero(el.second.size());
		for(int i=0; i<grad_norms.size(); i++){
			grad_norms[i] = body.combined_fiber_directions.row(el.second[i]).norm();
		}
		double min_grad_norm = grad_norms.minCoeff();
		grad_norms = (1/min_grad_norm)*grad_norms;
		for(int i = 0; i < grad_norms.size(); i++){
			combined_relative_stiffness[el.second[i]] = grad_norms[i];
		}
	}
}


void add_joints(const json &config, const string &obj_dir, Body &body) {
	for (auto& el : config["joints"].items()) {
		string name = el.key();
		const auto &joint = el.value();
		string type = joint["type"];
		string obj_name = joint["location_obj"];


		std::vector<string> bones = joint["bones"];
		string obj_path = lf::path::join(obj_dir, joint["location_obj"]);
		Mesh joint_mesh = load_mesh(obj_path);
		const int n_verts_in_joint = joint_mesh.V.rows();

		// TODO For hinges only
		const int nV = body.tet_mesh.V.rows();
		const int nT = body.tet_mesh.T.rows();
		const int new_nV = nV + n_verts_in_joint;
		const int new_nT = nT + 2 * n_verts_in_joint;
		body.tet_mesh.V.conservativeResize(new_nV, 3);
		body.tet_mesh.T.conservativeResize(new_nT, 4); // 2 for ball, 4 for hinge
		body.combined_fiber_directions.conservativeResize(new_nT, 3);
		
		for(int i = 0; i < n_verts_in_joint; i++) {
			body.tet_mesh.V.row(nV + i) = joint_mesh.V.row(i);
		}

	
		for(int i = 0; i < 2; i++) { 
			const string cur_bone = bones[i];
			auto &cur_bone_indices = body.bone_indices[cur_bone];
			RowVector4i to_attach_tet = body.tet_mesh.T.row(cur_bone_indices[0]);

			if(type == "hinge") {
				for(int j = 0; j < 2; j++) {
					const int new_tet_i = nT + 2 * i + j;
					cur_bone_indices.push_back(new_tet_i);
					body.tet_mesh.T.row(new_tet_i) = RowVector4i(nV, nV+1, to_attach_tet(0 + j), to_attach_tet(1 + j));
					body.combined_fiber_directions.row(new_tet_i) = RowVector3d(0.0, 0.0, 0.0);
					body.joint_indices.push_back(new_tet_i);
				}
			} else if (type == "ball") {
				const int new_tet_i = nT + i;
				cur_bone_indices.push_back(new_tet_i);
				body.tet_mesh.T.row(new_tet_i) = RowVector4i(nV, to_attach_tet(0), to_attach_tet(1), to_attach_tet(2));
				body.combined_fiber_directions.row(new_tet_i) = RowVector3d(0.0, 0.0, 0.0);
				body.joint_indices.push_back(new_tet_i);
			} else {
				std::cout << "Unsupported joint type!" << std::endl;
				exit(1);
			}
		}
		
	}
}


// TODO: Make this (and associated fn's) a member of Body
void generate_body_from_config(const string &body_dir, bool load_existing_tets, Body &body) {
	json config = read_config(body_dir);
	string obj_dir = lf::path::join(body_dir, "objs/");

	read_surfs(config, obj_dir, body.bone_surfs, body.muscle_surfs);
	string surf_dir = lf::path::join(body_dir, "objs/");

	body.surf_mesh = combine_surfs(body.bone_surfs, body.muscle_surfs);

	double eps_rel = 0.06; // This is the tetwild epsilon. Default is 0.1
	if(load_existing_tets) {
		// This is just to speed up development, saved files remain unused
		body.tet_mesh = load_body_tet_mesh(body_dir);
	} else {
		body.tet_mesh = tetrahedralize_mesh(body.surf_mesh, eps_rel);
		save_body_tet_mesh(body_dir, body.tet_mesh);
	}

	// Need to use this when assigning tets to correct mesh component
	double eps_absolute = igl::bounding_box_diagonal(body.surf_mesh.V) * eps_rel / 100.0; // Taken from Tetwild State.h
	compute_indices(eps_absolute, body.tet_mesh, body.bone_surfs, body.muscle_surfs, body.bone_indices, body.muscle_indices);

	split_tet_meshes(body.tet_mesh, body.bone_indices, body.muscle_indices, body.split_tet_meshes);

	compute_muscle_fibers(body, body.combined_fiber_directions, body.harmonic_boundary_verts);
	
	compute_element_stiffness(body, body.combined_relative_stiffness);

	// add_joints(config, obj_dir, body);
}


int main(int argc, char *argv[])
{
	Args args = parse_args(argc, argv);
	string output_dir = lf::path::join(args.body_dir, "generated_files/");

	Body body;
	generate_body_from_config(args.body_dir, args.load_existing_tets, body);

	body.write(output_dir);

	launch_viewer(body);

	return 0;
}



