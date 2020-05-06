gauss_mesh_dir ='/Users/vismay/recode/fast_muscles/Scripts/../data/alpha_test/T51271/GaussMuscle51271.obj';
my_mesh_dir = '/Users/vismay/recode/fast_muscles/Scripts/../data/alpha_test/alpha1e';
[restV, restF] = readOBJ(my_mesh_dir+string(0)+"/mesh0.obj");
rest_mesh_size = max(restV(:,2)) - min(restV(:,2))
[gV, gF] = readOBJ(gauss_mesh_dir);
f = zeros(6,1);
for c=0:5
    [V,F] = readOBJ(my_mesh_dir+string(c)+"/mesh1.obj");
    [c, hausdorff(gV, gF, V, F)/rest_mesh_size, norm(gV - V)]
    f(c+1) = norm(gV - V)/rest_mesh_size;
end

t = (0:5)'
h = plot(t,f);
fig = gcf;
fig.Units = "inches";
fig.PaperUnits = "inches";
fig.PaperPosition = [0 0 4 2];
fig.Position = [0 0 4 2];
h.LineWidth = 2;
h.Color = [102, 172, 87]/255;
h.Marker = "o";
h.MarkerFaceColor = [102, 172, 87]/255;;
h.MarkerEdgeColor = [0 0 0];