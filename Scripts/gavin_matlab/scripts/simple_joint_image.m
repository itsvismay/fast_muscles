filename = 'skinned-clustered_V';
max_iter = 5;
extension = '.dmat';
z_rotation = 1;

wireframe_flag = false;
fiber_flag = false;

muscle_index = readDMAT('../data/simple_joint/generated_files/muscle_muscle_indices.dmat')+1;
top_bone_index = readDMAT('../data/simple_joint/generated_files/top_bone_bone_indices.dmat')+1;
bottom_bone_index = readDMAT('../data/simple_joint/generated_files/bottom_bone_bone_indices.dmat')+1;
bone_index = [top_bone_index;bottom_bone_index];

tet_fiber = readDMAT('../data/simple_joint/generated_files/combined_fiber_directions.dmat');

T = readDMAT('../data/simple_joint/generated_files/tet_mesh_T.dmat')+1;
[F,J] = boundary_faces(T);
% F = F(:,[1,3,2]); %Vismay's normals are backwards

%https://www.slicer.org/wiki/Slicer3:2010_GenericAnatomyColors
muscle_color = [192,104,88]/255;
bone_color = [255,250,220]/255;

% tet_color = zeros(size(T,1),3);
% tet_color(muscle_index,:) = repmat(muscle_color,size(muscle_index,1),1);
% tet_color(bone_index,:) = repmat(bone_color,size(bone_index,1),1);

tet_muscle_ones = false(size(T,1),1);
tet_muscle_ones(muscle_index,:) = true;
muscle_faces = tet_muscle_ones(J,:);
bone_faces = logical(1-muscle_faces);

face_color = zeros(size(F,1),3);
face_color(muscle_faces,:) = repmat(muscle_color,sum(muscle_faces),1);
face_color(bone_faces,:) = repmat(bone_color,sum(bone_faces),1);

face_fiber = fiber(J,:);
muscle_fiber = face_fiber(muscle_faces,:);

% [~,bone_faces,~] = intersect(J,bone_index);
% [~,muscle_faces,~] = intersect(J,muscle_index);



%%

for iter = 0:max_iter
    V = readDMAT(strcat('../data/skinned-clustered-simple_joint/',filename,num2str(iter),extension));
    n = size(V,1);

    %Get mesh to a neutral position
    V = V*axisangle2matrix([0,1,0],-pi/2)';
    V = V*axisangle2matrix([1,0,0],pi/5)';
    V = V*axisangle2matrix([0,0,1],pi/2+z_rotation)';
    
    deformed_fiber = face_fiber*axisangle2matrix([0,1,0],-pi/2)';
    deformed_fiber = deformed_fiber*axisangle2matrix([1,0,0],pi/5)';
    deformed_fiber = deformed_fiber*axisangle2matrix([0,0,1],pi/2+z_rotation)';
    if iter == 0
        h = max(V)-min(V);
        groundz = min(V(:,3));
    end
    V = V + iter*([1,0,0].*h)*(1+h_buffer);
            
    %%
    dataFit.V = V;
    dataFit.T = T(muscle_index,:);
    %1. Initialize Variables
    dataFit.v = fiber(muscle_index,:);
    dataFit.w = zeros(size(muscle_index,1),3);
    dataFit.t = zeros(size(muscle_index,1),3);
    tex_coords = fitTexCoords3D(dataFit,1);
    %%
    
    for i = 0:1
        if i == 0 && iter == 0
            hold off
        else
            hold on
        end
        
        if(i == 0)
            t = tsurf(F(bone_faces,:),V,'FaceVertexCData',repmat(bone_color,sum(bone_faces),1));
            t.EdgeColor = 'none';  
        else
            t = tsurf(F(muscle_faces,:),V,'FaceVertexCData',repmat(muscle_color,sum(muscle_faces),1));
            t.EdgeColor = 'none';  
            if wireframe_flag 
                t.EdgeColor = 'k';
                t.EdgeAlpha = 0.3;
            end
            if fiber_flag
                BC = barycenter(V,F);
                quiver3(BC(:,1),BC(:,2),BC(:,3),deformed_fiber(:,1),deformed_fiber(:,2),deformed_fiber(:,3),'Color',muscle_color*.6,'Linewidth',.05);
            end
        end
        
        if iter == 0 && i == 0
            l = light('Position',[0 -0.5 .5]*10);
        end
        
        axis equal;
        set(t,fsoft);
        set(gca,'Visible','off');
        set(gcf,'Color',[1,1,1]);
        s = add_shadow(t,l,'Ground',[0,0,-1,groundz]);

%         if i == 1
%             apply_ambient_occlusion(t,'AddLights',false,'SoftLighting',false);
%         end

        view([0,15]);
    end
end

