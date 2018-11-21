
#include <geogram/basic/common.h>
#include <tetwild/tetwild.h>
#include <tetwild/Args.h>

#include <imgui/imgui.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include <igl/barycenter.h>
#include <igl/combine.h>
#include <igl/readOBJ.h>
#include <igl/signed_distance.h>

#include <iostream>
#include <vector>

#include "lf_utils.h"

using namespace Eigen;

struct Mesh {
	MatrixXd V;
	MatrixXi F;
};

struct TetMesh {
	MatrixXd V;
	MatrixXi T;
	VectorXd A; // Max dihedral angles..
};

bool update_cutaway(igl::opengl::glfw::Viewer &viewer, const TetMesh &tet_mesh, const MatrixXd &B, double t)
{
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
	viewer.data().clear();
	viewer.data().set_mesh(V_temp,F_temp);
	viewer.data().set_face_based(true);

	return false;
}

void readOBJ_to_mesh(const std::string &path, Mesh &mesh) {
	igl::readOBJ(path, mesh.V, mesh.F);
}

int main(int argc, char *argv[])
{

	if(argc != 4) {
		std::cout << "Usage: ./muscle_gen <muscle>.obj <bone1>.obj <bone2>.obj" << std::endl;
		return 1;
	}
	
	std::string muscle_path = argv[1];
	std::string bone1_path = argv[2];
	std::string bone2_path = argv[3];

	Mesh muscle;
	Mesh bone_1;
	Mesh bone_2;

	readOBJ_to_mesh(muscle_path, muscle);
	readOBJ_to_mesh(bone1_path, bone_1);
	readOBJ_to_mesh(bone2_path, bone_2);

	Mesh combined;
	igl::combine(
		std::vector<MatrixXd>({muscle.V, bone_1.V, bone_2.V}),
		std::vector<MatrixXi>({muscle.F, bone_1.F, bone_2.F}),
		combined.V, combined.F
	);

	TetMesh tet_mesh;
	GEO::initialize();
	tetwild::Args tetwild_args;
	tetwild::tetrahedralization(combined.V, combined.F, tet_mesh.V, tet_mesh.T, tet_mesh.A, tetwild_args);

	std::vector<int> muscle_tets;
	std::vector<int> bone_1_tets;
	std::vector<int> bone_2_tets;
	

	// Now determine which tets are which

	// Launch viewer with simple menu
	float cutaway_offset = 0.5f;
	Eigen::MatrixXd B;
	igl::barycenter(tet_mesh.V,tet_mesh.T,B);

	igl::opengl::glfw::Viewer viewer;
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);
	menu.callback_draw_viewer_menu = [&]()
	{
		if(ImGui::InputFloat("Cutaway Plane", &cutaway_offset, 0.1f, 1.0f, 1)) {
			update_cutaway(viewer, tet_mesh, B, cutaway_offset);
		}
	};

	update_cutaway(viewer, tet_mesh, B, cutaway_offset);
	viewer.launch();
}