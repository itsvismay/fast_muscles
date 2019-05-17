import numpy as np
import sys, os
import cProfile
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
from iglhelpers import *
from scipy.spatial.distance import directed_hausdorff


V1 = igl.eigen.MatrixXd()
F1 = igl.eigen.MatrixXi()
m1 = igl.readOBJ(sys.argv[1], V1, F1)

V2 = igl.eigen.MatrixXd()
F2 = igl.eigen.MatrixXi()
m2 = igl.readOBJ(sys.argv[2], V2, F2)

V1 = e2p(V1)
V2 = e2p(V2)

print(directed_hausdorff(V1,V2))