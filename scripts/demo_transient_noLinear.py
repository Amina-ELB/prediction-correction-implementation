import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Configuration du cache FEniCSx pour le calcul en parallèle avec MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    os.environ["XDG_CACHE_HOME"] = f"/tmp/fenics_cache_rank_{comm.rank}"
except ImportError:
    pass

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile, VTKFile
import ufl
import os
import sys
import data
# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh_generation import create_mesh_2D, create_unstructured_mesh_2D
from src.reinitialization import Reinitialization
from src.advection import AdvectionSolver
from src.geometry_ls import circle_2, circle_trans
from src.compute_error import compute_fraction_mass, compute_shape_error

# def main():
#     # 0. Parameters
#     u_max = 1.0  # Vitesse max dans le vortex
#     cfl_target = 0.8
#     h_min = data.Lx / data.N
#     dt = cfl_target * h_min / u_max
    
#     T = 3.0
#     num_steps = int(T / dt)
    
#     # 1. Create Mesh
#     if getattr(data, "mesh_type", "structured") == "unstructured":
#         mesh = create_unstructured_mesh_2D(data.Lx, data.Ly, data.Lx/data.N)
#     else:
#         mesh = create_mesh_2D(data.Lx, data.Ly, data.N, data.N)
    
#     if mesh.comm.rank == 0:
#         print(f"CFL adaptation: h={h_min:.4f}, u_max={u_max:.4f} => dt={dt:.4f} ({num_steps} steps)")
    
#     # --- NOUVEAU CHAMP DE VITESSE ---
#     x = ufl.SpatialCoordinate(mesh)
#     t_fenics = fem.Constant(mesh, 0.0) # Variable temporelle FEniCSx
    
#     # Champ de vitesse du Vortex de Rider-Kothe
#     # Le terme cosinus fait que le champ s'inverse à T/2
#     u_expr = -ufl.sin(ufl.pi * x[0])**2 * ufl.sin(2 * ufl.pi * x[1]) * ufl.cos(ufl.pi * t_fenics / (T))
#     v_expr =  ufl.sin(ufl.pi * x[1])**2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(ufl.pi * t_fenics / (T))
#     velocity = ufl.as_vector((u_expr, v_expr))
    
#     # 2. Define Function Space and Level Set
#     V = fem.functionspace(mesh, ("Lagrange", 1))
#     phi = fem.Function(V)
#     phi.interpolate(circle_trans)
#     phi.x.scatter_forward()
#     phi.name = "level_set"
    
#     # Define exact solution function for mass comparison
#     phi_exact = fem.Function(V)
#     phi_exact.interpolate(circle_trans)
#     # 3. Initialize Solvers
#     advection_solver = AdvectionSolver(V, u_velocity=velocity, dt=dt, method="rk4")
#     reinit_solver = Reinitialization(phi, V, l=data.l_param)
    
#     # 4. Results Directory
#     res_dir = "res_transient_noLinear"
#     if mesh.comm.rank == 0:
#         if not os.path.exists(res_dir):
#             os.makedirs(res_dir)
    
#     # 5. Time Stepping Loop
#     mass_evolution = []
    
#     xdmf = XDMFFile(mesh.comm, f"{res_dir}/evolution.xdmf", "w")
#     xdmf.write_mesh(mesh)
    
#     vtk = VTKFile(mesh.comm, f"{res_dir}/evolution.pvd", "w")
    
#     # Initial mass fraction (should be 1.0)
#     # Initial mass fraction (should be 1.0)
#     initial_fraction = compute_fraction_mass(phi, phi_exact, mesh, V)
#     mass_evolution.append((0.0, initial_fraction))
#     if mesh.comm.rank == 0:
#         print(f"Step 0: Mass Fraction = {initial_fraction:.6e}", flush=True)
        
#     from src.compute_error import error_L2_distance_func_form
    
#     xdmf.write_function(phi, 0.0)
#     vtk.write_function(phi, 0.0)
    
#     # 6. Initialize error log file
#     error_log_path = f"{res_dir}/errors.csv"
#     if mesh.comm.rank == 0:
#         with open(error_log_path, "w") as f:
#             f.write("time,mass_err_rel,eikonal_err_l2\n")

#     for i in range(1,int((num_steps + 1))):
#         t = i * dt
#         t_fenics.value = t - dt / 2  # Evaluate velocity at midpoint of time step (t + dt/2)
        
#         # --- Advection Step ---
#         phi_star = advection_solver.solve(phi)
#         phi.x.array[:] = phi_star.x.array[:]
#         phi.x.scatter_forward()
        
#         # --- Reinitialization Step ---
#         phi_pred = reinit_solver.predictor(phi)
#         phi.x.array[:] = phi_pred.x.array[:]
        
#         it = 0
#         phi_current = fem.Function(V)
#         phi_current.x.array[:] = phi.x.array[:]
        
#         # Correction avec critère d'arrêt (Early stopping / convergence de Picard)
#         tol = 1e-4
#         while it < 5:
#             it += 1
#             phi_next = reinit_solver.corrector(phi_current, is_first_iteration=(it == 1))
            
#             # Erreur relative de l'itération de Picard
#             diff = np.linalg.norm(phi_next.x.array - phi_current.x.array) / (np.linalg.norm(phi_next.x.array) + 1e-12)
#             phi_current.x.array[:] = phi_next.x.array[:]
            
#             if diff < tol:
#                 break
        
#         phi.x.array[:] = phi_current.x.array[:]
#         phi.x.scatter_forward()
        
#         # --- Mass and Eikonal Tracking ---
#         current_fraction = compute_fraction_mass(phi, phi_exact, mesh, V)
#         mass_error_rel = current_fraction 
#         eikonal_err = error_L2_distance_func_form(phi, V)
        
#         if mesh.comm.rank == 0:
#             print(f"Step {i}/{num_steps}: t = {t:.2f} | Mass Err Rel = {mass_error_rel:.4e} | Eikonal Err = {eikonal_err:.4e}", flush=True)
#             # Ecriture en temps réel
#             with open(error_log_path, "a") as f:
#                 f.write(f"{t},{mass_error_rel},{eikonal_err}\n")
        
#         # Save to file
#         xdmf.write_function(phi, t)

#     shape_error, relative_shape_error = compute_shape_error(phi, phi_exact, mesh)
#     xdmf.close()

#     if mesh.comm.rank == 0:
#         print(f"Simulation complete. Results saved in '{res_dir}'")
#         print(f"Shape Error = {shape_error:.4e} | Relative Shape Error = {relative_shape_error:.4e}")

# if __name__ == "__main__":
#     main()


import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Configuration du cache FEniCSx pour le calcul en parallèle avec MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    os.environ["XDG_CACHE_HOME"] = f"/tmp/fenics_cache_rank_{comm.rank}"
except ImportError:
    pass

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile, VTKFile
import ufl
import sys
import data
# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh_generation import create_mesh_2D, create_unstructured_mesh_2D
from src.reinitialization import Reinitialization
from src.advection import AdvectionSolver
from src.geometry_ls import circle_2, circle_trans
from src.compute_error import compute_fraction_mass, compute_shape_error

# Fonctions analytiques requises par RK4 (version numpy)
def vortex_velocity_numpy(points, time, period):
    x, y = points[0], points[1]
    factor = np.cos(np.pi * time / period)
    values = np.empty_like(points)
    values[0] = -np.sin(np.pi * x)**2 * np.sin(2.0 * np.pi * y) * factor
    values[1] =  np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)**2 * factor
    return values

def initial_phi_numpy(points):
    # Doit correspondre à la géométrie utilisée dans circle_trans
    center = np.array([0.5, 0.75])
    radius = 0.2
    return np.sqrt(np.sum((points - center[:, None])**2, axis=0)) - radius

def main():
    # 0. Parameters
    u_max = 1.0  
    cfl_target = 0.8
    h_min = data.Lx / data.N
    dt = cfl_target * h_min / u_max
    
    T = 5.0
    num_steps = int(T / dt)
    
    # 1. Create Mesh
    if getattr(data, "mesh_type", "structured") == "unstructured":
        mesh = create_unstructured_mesh_2D(data.Lx, data.Ly, data.Lx/data.N)
    else:
        mesh = create_mesh_2D(data.Lx, data.Ly, data.N, data.N)
    
    # --- Champs pour SUPG ---
    x = ufl.SpatialCoordinate(mesh)
    t_fenics = fem.Constant(mesh, 0.0) 
    u_expr = -ufl.sin(ufl.pi * x[0])**2 * ufl.sin(2 * ufl.pi * x[1]) * ufl.cos(ufl.pi * t_fenics / T)
    v_expr =  ufl.sin(ufl.pi * x[1])**2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(ufl.pi * t_fenics / T)
    velocity = ufl.as_vector((u_expr, v_expr))
    
    # 2. Define Function Space and Level Set
    V = fem.functionspace(mesh, ("Lagrange", 1))
    phi = fem.Function(V)
    phi.interpolate(circle_trans)
    phi.x.scatter_forward()
    
    phi_exact = fem.Function(V)
    phi_exact.interpolate(circle_trans)

    # 3. Initialize Solvers (Supporte SUPG ou RK4)
    advection_solver = AdvectionSolver(
        V, 
        u_velocity=velocity, 
        dt=dt, 
        method="rk4",
        velocity_func=vortex_velocity_numpy,
        initial_phi_func=initial_phi_numpy,
        period=T
    )
    reinit_solver = Reinitialization(phi, V, l=data.l_param, projection=getattr(data, "projection", False))
    
    # 4. Results Directory
    res_dir = "res_transient_noLinear"
    if mesh.comm.rank == 0:
        if not os.path.exists(res_dir): os.makedirs(res_dir)
    
    xdmf = XDMFFile(mesh.comm, f"{res_dir}/evolution.xdmf", "w")
    xdmf.write_mesh(mesh)
    
    from src.compute_error import error_L2_distance_func_form
    
    # Initial mass fraction
    initial_fraction = compute_fraction_mass(phi, phi_exact, mesh, V)
    if mesh.comm.rank == 0:
        print(f"Step 0: Mass Fraction = {initial_fraction:.6e}", flush=True)
        
    xdmf.write_function(phi, 0.0)
    
    # Initialize error log file
    error_log_path = f"{res_dir}/errors.csv"
    if mesh.comm.rank == 0:
        with open(error_log_path, "w") as f:
            f.write("time,mass_err_rel,eikonal_err_l2\n")
    
    # 5. Time Stepping Loop
    for i in range(1, int(num_steps + 1)):
        t = i * dt
        t_fenics.value = t - dt / 2
        
        # --- Advection Step (Passage du temps t pour RK4) ---
        phi_star = advection_solver.solve(phi, current_time=t)
        phi.x.array[:] = phi_star.x.array[:]
        phi.x.scatter_forward()
        
        # --- Reinitialization Step ---
        phi.x.array[:] = reinit_solver.predictor(phi).x.array
        
        phi_current = fem.Function(V)
        phi_current.x.array[:] = phi.x.array[:]
        
        tol = 1e-4
        for it in range(1, 6):
            phi_next = reinit_solver.corrector(phi_current, is_first_iteration=(it == 1))
            diff = np.linalg.norm(phi_next.x.array - phi_current.x.array) / (np.linalg.norm(phi_next.x.array) + 1e-12)
            phi_current.x.array[:] = phi_next.x.array[:]
            if diff < tol: break
        
        phi.x.array[:] = phi_current.x.array[:]
        phi.x.scatter_forward()
        
        # --- Mass and Eikonal Tracking ---
        current_fraction = compute_fraction_mass(phi, phi_exact, mesh, V)
        mass_error_rel = current_fraction 
        eikonal_err = error_L2_distance_func_form(phi, V)
        
        if mesh.comm.rank == 0:
            print(f"Step {i}/{num_steps}: t = {t:.2f} | Mass Err Rel = {mass_error_rel:.4e} | Eikonal Err = {eikonal_err:.4e}", flush=True)
            with open(error_log_path, "a") as f:
                f.write(f"{t},{mass_error_rel},{eikonal_err}\n")
        
        xdmf.write_function(phi, t)

    xdmf.close()

    shape_error, relative_shape_error = compute_shape_error(phi, phi_exact, mesh)
    xdmf.close()

    if mesh.comm.rank == 0:
        print(f"Simulation complete. Results saved in '{res_dir}'")
        print(f"Shape Error = {shape_error:.4e} | Relative Shape Error = {relative_shape_error:.4e}")

    if mesh.comm.rank == 0:
        print("Simulation complete.")

if __name__ == "__main__":
    main()
