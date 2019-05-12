T = readDMAT('../data/simple_joint/generated_files/tet_mesh_T.dmat')+1;
MI = readDMAT('../data/simple_joint/generated_files/muscle_muscle_indices.dmat')+1;
JI = readDMAT('../data/simple_joint/generated_files/joint_indices.dmat')+1;

R = axisangle2matrix([0,1,0],-pi/2)'*axisangle2matrix([1,0,0],pi/5)'*...
    axisangle2matrix([0,0,1],pi/2+.5)';

make_animation('gifs/unreduced_simple_joint.gif','../data/simple_joint_animation/simple_joint_unreduced', ...
    35,T,MI,JI,R,[-5.6 0.4644 -5.0899 12.864 -5.2076 7.7905]);