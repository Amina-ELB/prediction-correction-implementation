import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
import ufl
import os

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh_generation import create_mesh_2D
from src.reinitialization import Reinitialization
from src.advection import AdvectionSolver
import data
from src.geometry_ls import circle_trans
from src.compute_error import compute_fraction_mass, error_L2_distance_func_form

def main():
    # 0. Parameters
    T = 1.0  # Temps pour une rotation complète (2*pi rad)
    omega = 2.0 * np.pi
    u_max = omega * 0.5 # Vitesse max aux coins
    
    cfl_target = 0.5
    h_min = data.Lx / data.N
    dt = cfl_target * h_min / u_max
    
    num_steps = int(T / dt)
    
    # 1. Create Mesh
    mesh = create_mesh_2D(data.Lx, data.Ly, data.N, data.N)
    
    if mesh.comm.rank == 0:
        print(f"Rotation Test: h={h_min:.4f}, u_max={u_max:.4f} => dt={dt:.4f} ({num_steps} steps)")
    
    # --- CHAMP DE VITESSE DE ROTATION ---
    x = ufl.SpatialCoordinate(mesh)
    # Rotation autour de (0.5, 0.5)
    u_expr = -omega * (x[1] - 0.5)
    v_expr =  omega * (x[0] - 0.5)
    velocity = ufl.as_vector((u_expr, v_expr))
    
    # 2. Define Function Space and Level Set
    V = fem.functionspace(mesh, ("Lagrange", 1))
    phi = fem.Function(V)
    phi.interpolate(circle_trans)
    phi.x.scatter_forward()
    phi.name = "level_set"
    
    # Define exact solution (initial state) for final error comparison
    phi_init = fem.Function(V)
    phi_init.interpolate(circle_trans)
    
    # 3. Initialize Solvers
    advection_solver = AdvectionSolver(V, u_velocity=velocity, dt=dt, method="supg")
    reinit_solver = Reinitialization(phi, V, l=data.l_param)
    
    # 4. Results Directory
    res_dir = "res_rotation"
    if mesh.comm.rank == 0:
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
    
    # 5. Time Stepping Loop
    xdmf = XDMFFile(mesh.comm, f"{res_dir}/evolution.xdmf", "w")
    xdmf.write_mesh(mesh)
    
    # Initial mass fraction (should be 1.0)
    initial_fraction = compute_fraction_mass(phi, phi_init, mesh, V)
    
    # Initialize error log file
    error_log_path = f"{res_dir}/errors.csv"
    if mesh.comm.rank == 0:
        with open(error_log_path, "w") as f:
            f.write("time,mass_err_rel,eikonal_err_l2\n")
        print(f"Step 0: Mass Fraction = {initial_fraction:.6e}", flush=True)

    xdmf.write_function(phi, 0.0)
    
    for i in range(1, num_steps + 1):
        t = i * dt
        
        # --- Advection Step ---
        phi_star = advection_solver.solve(phi)
        phi.x.array[:] = phi_star.x.array[:]
        phi.x.scatter_forward()
        
        # --- Reinitialization Step ---
        # phi_pred = reinit_solver.predictor(phi)
        # phi.x.array[:] = phi_pred.x.array[:]
        
        it = 0
        phi_current = fem.Function(V)
        phi_current.x.array[:] = phi.x.array[:]
        
        # Correction iterations
        # while it < 5:
        #     it += 1
        #     phi_next = reinit_solver.corrector(phi_current)
        #     phi_current.x.array[:] = phi_next.x.array[:]
        
        phi.x.array[:] = phi_current.x.array[:]
        phi.x.scatter_forward()
        
        # --- Mass and Eikonal Tracking ---
        current_fraction = compute_fraction_mass(phi, phi_init, mesh, V)
        eikonal_err = error_L2_distance_func_form(phi, V)
        
        if mesh.comm.rank == 0:
            if i % 10 == 0 or i == num_steps:
                print(f"Step {i}/{num_steps}: t = {t:.2f} | Mass Err Rel = {current_fraction:.4e} | Eikonal Err = {eikonal_err:.4e}", flush=True)
            
            with open(error_log_path, "a") as f:
                f.write(f"{t},{current_fraction},{eikonal_err}\n")
        
        # Save every few steps to avoid huge files
        if i % 5 == 0 or i == num_steps:
            xdmf.write_function(phi, t)
        
    # Final check: compare with initial state after full rotation
    L2_error_final = fem.assemble_scalar(fem.form((phi - phi_init)**2 * ufl.dx))**0.5
    if mesh.comm.rank == 0:
        print(f"\nFinal L2 error with respect to initial state: {L2_error_final:.4e}")
        print(f"Simulation complete. Results saved in '{res_dir}'")
        
    xdmf.close()

if __name__ == "__main__":
    main()
