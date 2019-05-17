T = readDMAT('../data/realistic_biceps/generated_files/tet_mesh_T.dmat')+1;
MI = readDMAT('../data/realistic_biceps/generated_files/biceps_muscle_indices.dmat')+1;
JI = readDMAT('../data/realistic_biceps/generated_files/joint_indices.dmat')+1;

R = axisangle2matrix([0,1,0],-pi/2)'*axisangle2matrix([1,0,0],pi/2)'*...
    axisangle2matrix([0,0,1],pi/2)';

make_animation('gifs/realistic_biceps.gif','../data/realistic_biceps_animation/realistic_bicepsanimation', ...
    100,T,MI,JI,R,[-20 10 -9.0899 12.864 -12 7]);