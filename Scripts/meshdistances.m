gauss_mesh_dir ='/Users/vismay/recode/fast_muscles/Scripts/../data/alpha_test/T6785-threads1-activation50000.000000Gauss.obj';
my_mesh_dir = '/Users/vismay/recode/fast_muscles/Scripts/../data/alpha_test/alpha1e';
[restV, restF] = readOBJ(my_mesh_dir+string(0)+"/mesh0.obj");
rest_mesh_size = max(restV(:,2)) - min(restV(:,2))
[gV, gF] = readOBJ(gauss_mesh_dir);
for c=0:7
    [V,F] = readOBJ(my_mesh_dir+string(c)+"/mesh1.obj");
    [hausdorff(gV, gF, V, F), norm(gV - V)]/rest_mesh_size
end