filename = 'skinned-clustered_V';
max_iter = 5;
extension = '.dmat';
z_rotation = 1;
axisvec = [-3 3 -5.0899 20 -7 7];
R = axisangle2matrix([0,1,0],-pi/2)'*axisangle2matrix([1,0,0],pi/2)';%*...
    %axisangle2matrix([0,0,1],pi/2+.4)';


wireframe_flag = false;
fiber_flag = false;

inputfile = '../data/simple_muscle/T6785/generated_files/';
muscle_index = readDMAT([inputfile 'muscle_muscle_indices.dmat'])+1;
tet_is_tendon = readDMAT([inputfile 'tet_is_tendon.dmat']);
top_bone_index = readDMAT([inputfile 'top_bone_bone_indices.dmat'])+1;
bottom_bone_index = readDMAT([inputfile 'bottom_bone_bone_indices.dmat'])+1;
bone_index = [top_bone_index;bottom_bone_index];
tet_fiber = readDMAT('../data/simple_joint/generated_files/combined_fiber_directions.dmat');

T = readDMAT([inputfile 'tet_mesh_T.dmat'])+1;
[F,J] = boundary_faces(T);
% F = F(:,[1,3,2]); %Vismay's normals are backwards

%https://www.slicer.org/wiki/Slicer3:2010_GenericAnatomyColors
muscle_color = [192,104,88]/255;
bone_color = [255,250,220]/255;
tendon_color = [255, 255, 255]/255;

%set muscles to 1, bones to 0 , tendons to 2
tet_muscle_ones = false(size(T,1),1);
tet_muscle_ones(muscle_index,:) = true;
muscle_faces = tet_muscle_ones(J,:);
bone_faces = logical(1-muscle_faces);
tendon_faces = logical(tet_is_tendon(J, :));

face_color = zeros(size(F,1),3);
face_color(muscle_faces,:) = repmat(muscle_color,sum(muscle_faces),1);
face_color(bone_faces,:) = repmat(bone_color, sum(bone_faces),1);
face_color(tendon_faces,:) = repmat(tendon_color, sum(tendon_faces), 1);

%V = readDMAT([inputfile 'tet_mesh_V.dmat']);
[V, F1] = readOBJ([inputfile '../EMU6785-Alpha:1000000.000000.obj']);
V = V*R;

t = tsurf(F, V, 'FaceVertexCData', face_color);
t.EdgeColor = 'none';  
l = light('Position',[0 -0.5 .5]*10);
axis equal
view([0,0]);
set(t,fsoft);
set(gca,'Visible','off');
set(gcf,'Color',[1,1,1]);
%s = add_shadow(t,l,'Color',[1,1,1]*0.8,'BackgroundColor',[1,1,1],'Fade','infinite');
