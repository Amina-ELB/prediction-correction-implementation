# Add the parent directory to sys.path to allow importing from src
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.geometry_ls import circle, circle_2, star, smooth_star, step, tan_function, circle_trans

# Mesh parameters
N = 64 # h = 1/N
Lx = 1.0
Ly = 1.0
Lz = 0.

# Mesh type
mesh_type = "unstructured" # "structured" or "unstructured"

# Level set function selection
# Options: circle, star, smooth_star, step, tan_function, circle_trans
level_set_function = circle_trans

# Reinitialization solver and parameters
solver_type = "gmres" # "mumps" coarse solver / "gmres" iterative solver / "cg" iterative solver / "petsc" default ksp solver
l_param = 2.0  # Stabilization parameter
projection = True # True or False for L-2 projection of the norm of the gradient of the LS function.

# Iterative corrector parameters
tolerance = 1e-8
max_iterations = 10

# Noise parameters
use_noise = False # True or False
