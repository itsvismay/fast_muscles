function [] = make_animation(filename,vertfile,max_iter,T,MI,JI,R,axisvec,wireframe_flag)
  % MAKE ANIMATION 
  %
  % filename - filename for the output gif
  % vertfile - path and filename of the dmats with the vertex positons
  %         eg: '../data/simple_joint_animation/simple_joint_unreduced
  % max_iter - number of meshes to pull for frams
  % T - Tets
  % MI - muscle indices into T
  % JI - joint indices into T
  % {y,x,z}_rotation - rot of the mesh along the {y,x,z} axis, in order
  % axisvec - fixed axis of the figure
  % optional: wireframe - show wireframe on the muscles (default = false)
  
    if nargin < 11
        wireframe_flag = false;
    end
%     if nargin < 8
%         fiber_flag = false;
%     end
    
    [F,FaceI] = boundary_faces(T);
    
    muscle_color = [192,104,88]/255;
    bone_color = [255,250,220]/255;
    
    muscle_color = [172,84,70]/255;
    bone_color = [247,243,220]/255;
    
    tet_muscle_ones = false(size(T,1),1);
    tet_muscle_ones(MI,:) = true;
    muscle_faces = tet_muscle_ones(FaceI,:);
    
    tet_joint_ones = false(size(T,1),1);
    tet_joint_ones(JI,:) = true;
    joint_faces = tet_joint_ones(FaceI,:);
    
    bone_faces = logical(1-clamp(muscle_faces+joint_faces));

%     face_color = zeros(size(F,1),3);
%     face_color(muscle_faces,:) = repmat(muscle_color,sum(muscle_faces),1);
%     face_color(bone_faces,:) = repmat(bone_color,sum(bone_faces),1);
    %%
    fig = figure(1);
    for iter = 0:max_iter
        strcat(vertfile,'animation',num2str(iter, '%03.f'),'.dmat')
        V = readDMAT(strcat(vertfile,'animation',num2str(iter, '%03.f'),'.dmat'));
        n = size(V,1);

        %Get mesh to a neutral position
        V = V*R;

        for i = 0:1        
            if(i == 0)
                hold off
                t = tsurf(F(bone_faces,:),V,'FaceVertexCData',repmat(bone_color,sum(bone_faces),1));
                t.EdgeColor = 'none';  
                l = light('Position',[0 -0.5 .5]*10);
            else
                hold on
                t = tsurf(F(muscle_faces,:),V,'FaceVertexCData',repmat(muscle_color,sum(muscle_faces),1));
                t.EdgeColor = 'none';  
                if wireframe_flag 
                    t.EdgeColor = 'k';
                    t.EdgeAlpha = 0.3;
                end
%                 if fiber_flag
%                     BC = barycenter(V,F);
%                     quiver3(BC(:,1),BC(:,2),BC(:,3),deformed_fiber(:,1),deformed_fiber(:,2),deformed_fiber(:,3),'Color',muscle_color*.6,'Linewidth',.05);
%                 end
            end

            axis equal;
            set(t,fsoft);
            set(gca,'Visible','off');
            set(gcf,'Color',[1,1,1]);
            s = add_shadow(t,l,'Ground',[0,0,-1,axisvec(5)]);

%                 apply_ambient_occlusion(t,'AddLights',false,'SoftLighting',false);

            view([0,15]);
        end
        axis(axisvec);
        
        drawnow;
        frame = getframe(fig); 
        [imind,cm] = rgb2ind(frame.cdata,256);
        if iter == 0
            imwrite(imind,cm,filename,'gif','Loopcount',Inf,'DelayTime',0);
        else  
            imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0); 
        end

    end
end

