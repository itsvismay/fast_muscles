#ifndef VIEWER_H
#define VIEWER_H

#include <Eigen/Core>

#include <igl/barycenter.h>

#include "MeshTypes.h"

#include <map>

using namespace muscle_gen;
using namespace Eigen;

namespace muscle_gen {
	const std::vector<Vector3d> color_cycle = {
		Vector3d(1.0, 0.0, 0.0),
		Vector3d(0.0, 1.0, 0.0),
		Vector3d(0.0, 0.0, 1.0),
		Vector3d(1.0, 1.0, 0.0),
		Vector3d(1.0, 0.0, 1.0),
		Vector3d(0.8, 0.8, 0.8),
		Vector3d(0.2, 0.2, 0.2),
	};

	bool update_cutaway(igl::opengl::glfw::Viewer &viewer, const TetMesh &tet_mesh, double t, int mesh_index)
	{
		Eigen::MatrixXd B;
		igl::barycenter(tet_mesh.V,tet_mesh.T, B);

		using namespace std;
		VectorXd v = B.col(2).array() - B.col(2).minCoeff();
		v /= v.col(0).maxCoeff();
		vector<int> s;
		for (unsigned i=0; i<v.size();++i)
			if (v(i) < t)
				s.push_back(i);
		MatrixXd V_temp(s.size()*4,3);
		MatrixXi F_temp(s.size()*4,3);
		for (unsigned i=0; i<s.size();++i)
		{
			V_temp.row(i*4+0) = tet_mesh.V.row(tet_mesh.T(s[i],0));
			V_temp.row(i*4+1) = tet_mesh.V.row(tet_mesh.T(s[i],1));
			V_temp.row(i*4+2) = tet_mesh.V.row(tet_mesh.T(s[i],2));
			V_temp.row(i*4+3) = tet_mesh.V.row(tet_mesh.T(s[i],3));
			F_temp.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
			F_temp.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
			F_temp.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
			F_temp.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
		}

		viewer.selected_data_index = mesh_index;
		viewer.data().clear();
		viewer.data().set_mesh(V_temp,F_temp);
		viewer.data().set_face_based(true);

		const auto c = color_cycle[mesh_index % color_cycle.size()];
		viewer.data().uniform_colors(c, Vector3d(1.0,1.0,1.0), Vector3d(1.0,1.0,1.0));
		return false;
	}

	void update(igl::opengl::glfw::Viewer &viewer, const Body &body, const std::map<std::string, int> &mesh_indices, const std::vector<Eigen::MatrixXd> &fiber_edges, double cutaway_offset) {
		for(const auto &tet_mesh_el : body.split_tet_meshes) {
			update_cutaway(viewer, tet_mesh_el.second, cutaway_offset, mesh_indices.at(tet_mesh_el.first));
		}
		if(fiber_edges.size() > 0) {
			viewer.data().add_edges(fiber_edges[0], fiber_edges[1], RowVector3d(1.0, 0.0, 0.0));
		}

		int n_boundary_verts = body.harmonic_boundary_verts.rows();
		if(n_boundary_verts > 0) {
			MatrixXd C = RowVector3d(0,0,1).replicate(n_boundary_verts, 1);
			viewer.data().add_points(body.harmonic_boundary_verts, C);
		}

		Eigen::MatrixXd B;
		igl::barycenter(body.tet_mesh.V,body.tet_mesh.T, B);
		std::vector<RowVector3d> tendon_tet_points_vec;
		for(int i = 0; i < body.tet_mesh.T.rows(); i++) {
			if(body.tet_is_tendon[i]) {
				tendon_tet_points_vec.push_back(B.row(i));
				viewer.data().add_points(B.row(i), RowVector3d(1.0,1.0,1.0));
			}
		}
		// MatrixXd tendon_tet_points;
		// igl::list_to_matrix()

	}

	void launch_viewer(const Body &body) {
		igl::opengl::glfw::Viewer viewer;
		igl::opengl::glfw::imgui::ImGuiMenu menu;
		viewer.plugins.push_back(&menu);
		float cutaway_offset = 0.5f;


		double fiber_scale = 0.1;
		std::vector<Eigen::MatrixXd> fiber_edges(2);
		MatrixXd muscle_centers;
		igl::barycenter(body.tet_mesh.V, body.tet_mesh.T, muscle_centers);
		fiber_edges[0] = muscle_centers;
		fiber_edges[1] = muscle_centers +  body.combined_fiber_directions * fiber_scale;


		std::map<std::string, int> mesh_indices;
		for(const auto &tet_mesh_el : body.split_tet_meshes) {
			const int mesh_index = viewer.append_mesh();
			mesh_indices[tet_mesh_el.first] = mesh_index;
		}

		
		menu.callback_draw_viewer_menu = [&]()
		{
			if(ImGui::InputFloat("Cutaway Plane", &cutaway_offset, 0.1f, 1.0f, 1)) {
				update(viewer, body, mesh_indices, fiber_edges, cutaway_offset);
			}
		};


		update(viewer, body, mesh_indices, fiber_edges, cutaway_offset);
		viewer.launch();
	}

}

#endif