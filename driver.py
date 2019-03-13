import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import fenics as fe

#==================================================================

sys.path.insert(0, './modules')
from timing_module import TimeIt
from fem_module import FEMSimulation

#==================================================================
obj = FEMSimulation()
obj.run()

