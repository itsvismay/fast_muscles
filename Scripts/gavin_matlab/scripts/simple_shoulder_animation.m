T = readDMAT('../data/simple_shoulder/generated_files/tet_mesh_T.dmat')+1;
MI = readDMAT('../data/simple_shoulder/generated_files/front_deltoid_muscle_indices.dmat')+1;
MI = [MI; readDMAT('../data/simple_shoulder/generated_files/rear_deltoid_muscle_indices.dmat')+1];
MI = [MI; readDMAT('../data/simple_shoulder/generated_files/top_deltoid_muscle_indices.dmat')+1];
JI = readDMAT('../data/simple_shoulder/generated_files/joint_indices.dmat')+1;

R = axisangle2matrix([0,1,0],-pi/2)'*axisangle2matrix([1,0,0],pi/2)'*...
    axisangle2matrix([0,0,1],pi/2+.4)';

make_animation('gifs/simple_shoulder_test.gif','../data/simple_shoulder_test/simple_shoulder_test', ...
    100,T,MI,JI,R,[-3 10 -5.0899 12.864 0 7]);