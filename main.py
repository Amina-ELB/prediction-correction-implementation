import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
import ufl

from src.mesh_generation import create_mesh_2D
from src.reinitialization import Reinitialization
import data
from src.geometry_ls import apply_noise

def main():
    # 1. Create Mesh
    mesh = create_mesh_2D(data.Lx, data.Ly, data.N, data.N)
    
    # 2. Define Function Space and Level Set
    V = fem.functionspace(mesh, ("Lagrange", 1))
    level_set = fem.Function(V)
    level_set.interpolate(data.level_set_function)
    level_set.x.scatter_forward()
    
    if data.use_noise:
        print("Applying noise to level set function...")
        level_set = apply_noise(level_set, mesh)
        
    level_set.name = "phi_0"
    
    # Save initial state
    with XDMFFile(mesh.comm, "res/initial_ls.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(level_set)

    # 3. Initialize Reinitialization Solver
    # l parameter for stabilization
    reinit_solver = Reinitialization(level_set, V, l=data.l_param)
    
    # 4. Run Predictor
    print("Running Predictor step...")
    phi_pred = reinit_solver.predictor(level_set)
    phi_pred.name = "phi_predictor"
    
    with XDMFFile(mesh.comm, "res/predictor_ls.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(phi_pred)

    # 5. Run Corrector
    print("Running Corrector step...")
    
    # Iterative corrector loop
    it = 0
    diff_norm = 1.0
    
    phi_current = fem.Function(V)
    phi_current.x.array[:] = phi_pred.x.array[:]
    
    phi_next = fem.Function(V)
    
    while diff_norm > data.tolerance and it < data.max_iterations:
        it += 1
        phi_next = reinit_solver.corrector(phi_current)
        
        # Compute L2 difference
        diff_L2 = fem.form(ufl.inner(phi_next - phi_current, phi_next - phi_current) * ufl.dx)
        diff_L2_local = fem.assemble_scalar(diff_L2)
        diff_L2_global = mesh.comm.allreduce(diff_L2_local, op=MPI.SUM)
        diff_norm = np.sqrt(diff_L2_global)
        
        print(f"Iteration {it}: ||phi_new - phi_old||_L2 = {diff_norm:.4e}")
        
        phi_current.x.array[:] = phi_next.x.array[:]
        
    phi_corr = phi_current
    phi_corr.name = "phi_corrected"
    
    with XDMFFile(mesh.comm, "res/corrected_ls.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(phi_corr)
        
    if it == data.max_iterations:
        print(f"Warning: Corrector did not converge in {data.max_iterations} iterations.")
    else:
        print(f"Corrector converged in {it} iterations.")
        
    print("Reinitialization complete. Results saved in 'res/' directory.")

if __name__ == "__main__":
    main()
