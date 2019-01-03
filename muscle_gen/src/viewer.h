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
		Vector3d(1.0, 1.0, 0.0)
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
			update_cutaway(viewer, tet_mesh_el.second, cutaway_offset, mesh_index);
		}
		
		menu.callback_draw_viewer_menu = [&]()
		{
			if(ImGui::InputFloat("Cutaway Plane", &cutaway_offset, 0.1f, 1.0f, 1)) {
				for(const auto &tet_mesh_el : body.split_tet_meshes) {
					update_cutaway(viewer, tet_mesh_el.second, cutaway_offset, mesh_indices[tet_mesh_el.first]);
				}
			}
		};


		if(fiber_edges.size() > 0) {
			viewer.data().add_edges(fiber_edges[0], fiber_edges[1], RowVector3d(1.0, 0.0, 0.0));
		}
		
		viewer.launch();
	}

}

#endif