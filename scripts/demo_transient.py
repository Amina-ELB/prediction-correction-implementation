import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
import ufl
import sys
# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.mesh_generation import create_mesh_2D, create_unstructured_mesh_2D
from src.reinitialization import Reinitialization
from src.advection import AdvectionSolver
import data
from src.geometry_ls import circle_2
from src.compute_error import compute_fraction_mass

def main():
    # 0. Parameters
    v_x, v_y = 0.2, 0.2
    u_max = np.sqrt(v_x**2 + v_y**2)
    
    # CFL condition: dt = CFL * h / u_max
    cfl_target = 0.5
    h_min = data.Lx / data.N
    dt = cfl_target * h_min / u_max
    
    T = 0.5
    num_steps = int(T / dt)
    # 1. Create Mesh
    if getattr(data, "mesh_type", "structured") == "unstructured":
        mesh = create_unstructured_mesh_2D(data.Lx, data.Ly, data.Lx/data.N)
    else:
        mesh = create_mesh_2D(data.Lx, data.Ly, data.N, data.N)
    
    if mesh.comm.rank == 0:
        print(f"CFL adaptation: h={h_min:.4f}, u_max={u_max:.4f} => dt={dt:.4f} ({num_steps} steps)")
        
    velocity = fem.Constant(mesh, np.array([v_x, v_y], dtype=np.float64))
    
    # 2. Define Function Space and Level Set
    V = fem.functionspace(mesh, ("Lagrange", 1))
    phi = fem.Function(V)
    phi.interpolate(circle_2)
    phi.x.scatter_forward()
    phi.name = "level_set"
    
    # Define exact solution function for mass comparison
    phi_exact = fem.Function(V)
    def circle_exact(x, t=0):
        return ((x[0] - (0.5 + v_x * t))**2 + (x[1] - (0.5 + v_y * t))**2)**0.5 - 0.25 + 1e-16
    
    # 3. Initialize Solvers
    advection_solver = AdvectionSolver(V, u_velocity=velocity, dt=dt, method="supg")
    reinit_solver = Reinitialization(phi, V, l=data.l_param)
    
    # 4. Results Directory
    res_dir = "res_transient"
    if mesh.comm.rank == 0:
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
    
    # 5. Time Stepping Loop
    mass_evolution = []
    
    xdmf = XDMFFile(mesh.comm, f"{res_dir}/evolution.xdmf", "w")
    xdmf.write_mesh(mesh)
    
    # Initial mass fraction (should be 1.0)
    phi_exact.interpolate(lambda x: circle_exact(x, 0))
    initial_fraction = compute_fraction_mass(phi, phi_exact, mesh, V)
    mass_evolution.append((0.0, initial_fraction))
    if mesh.comm.rank == 0:
        print(f"Step 0: Mass Fraction = {initial_fraction:.6e}")
    
    xdmf.write_function(phi, 0.0)
    
    for i in range(1, num_steps + 1):
        t = i * dt
        
        # --- Advection Step ---
        phi_star = advection_solver.solve(phi)
        phi.x.array[:] = phi_star.x.array[:]
        phi.x.scatter_forward()
        
        # --- Reinitialization Step ---
        phi_pred = reinit_solver.predictor(phi)
        phi.x.array[:] = phi_pred.x.array[:]
        
        it = 0
        diff_norm = 1.0
        phi_current = fem.Function(V)
        phi_current.x.array[:] = phi.x.array[:]
        
        while diff_norm > data.tolerance and it < 5:
            it += 1
            phi_next = reinit_solver.corrector(phi_current)
            diff_L2 = fem.form(ufl.inner(phi_next - phi_current, phi_next - phi_current) * ufl.dx)
            diff_norm = np.sqrt(mesh.comm.allreduce(fem.assemble_scalar(diff_L2), op=MPI.SUM))
            phi_current.x.array[:] = phi_next.x.array[:]
        
        phi.x.array[:] = phi_current.x.array[:]
        phi.x.scatter_forward()
        
        # --- Mass Tracking ---
        # Update exact solution for current time
        phi_exact.interpolate(lambda x: circle_exact(x, t))
        current_fraction = compute_fraction_mass(phi, phi_exact, mesh, V)
        mass_evolution.append((t, current_fraction))
        
        if mesh.comm.rank == 0:
            print(f"Step {i}/{num_steps}: t = {t:.2f}, Mass Fraction = {current_fraction:.6e} (Error: {abs(current_fraction-1.0):.2%})")
        
        # Save to file
        xdmf.write_function(phi, t)
        
    xdmf.close()
    
    # 6. Save Mass Evolution to CSV
    if mesh.comm.rank == 0:
        with open(f"{res_dir}/mass_evolution.csv", "w") as f:
            f.write("time,mass_fraction\n")
            for t, m in mass_evolution:
                f.write(f"{t},{m}\n")
        print(f"Simulation complete. Results saved in '{res_dir}'")

if __name__ == "__main__":
    main()
