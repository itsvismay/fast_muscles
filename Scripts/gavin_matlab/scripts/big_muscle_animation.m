
T = readDMAT('../../../data/simple_muscle_bipennate2/generated_files/tet_mesh_T.dmat')+1;
MI = readDMAT('../../../data/simple_muscle_bipennate2/generated_files/muscle_muscle_indices.dmat')+1;
JI = readDMAT('../../../data/simple_muscle_bipennate2/generated_files/joint_indices.dmat')+1;

R = axisangle2matrix([0,1,0],-pi/2)'*axisangle2matrix([1,0,0],pi/2)'*...
    axisangle2matrix([0,0,1],pi/2+.4)';

make_animation('gifs/simple_muscle_bipennate2_test.gif','../../../data/simple_muscle_bipennate2/', ...
    10,T,MI,JI,R,[-3 3 -5.0899 20 -7 7]);