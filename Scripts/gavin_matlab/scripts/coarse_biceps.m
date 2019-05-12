T = readDMAT('../../../data/realistic_upper_coarse/generated_files/tet_mesh_T.dmat')+1;
MI = readDMAT('../../../data/realistic_upper_coarse/generated_files/biceps_muscle_indices.dmat')+1;
MI = [MI; readDMAT('../../../data/realistic_upper_coarse/generated_files/triceps_muscle_indices.dmat')+1];
JI = readDMAT('../../../data/realistic_upper_coarse/generated_files/joint_indices.dmat')+1;
 
R = axisangle2matrix([0,1,0],-pi/2)'*axisangle2matrix([1,0,0], -pi/2)'*...
    axisangle2matrix([0,0,1],pi)';

make_animation('gifs/realistic_upper_coarse.gif','../../../data/realistic_upper_coarse/', ...
    9,T,MI,JI,R, [-20 10 -9.0899 12.864 -12 15]);