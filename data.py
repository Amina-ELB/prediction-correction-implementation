from src.geometry_ls import circle, star, smooth_star, step, tan_function

# Mesh parameters
N = 64
Lx = 1.0
Ly = 1.0

# Level set function selection
# Level set function selection
# Options: circle, star, smooth_star, step, tan_function
level_set_function = tan_function

# Reinitialization solver parameters
l_param = 3.0  # Stabilization parameter

# Iterative corrector parameters
tolerance = 1e-8
max_iterations = 50

# Noise parameters
use_noise = True
