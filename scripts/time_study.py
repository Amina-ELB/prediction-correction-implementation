import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
import ufl

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh_generation import create_mesh_2D, create_mesh_3D
from src.reinitialization import Reinitialization as Reinitialization
# from src.reinitialization import Reinitialization
import data
from src.geometry_ls import apply_noise
from src.compute_error import error_L2_distance_func_form

def main():
    t_start = MPI.Wtime()
    # 1. Create Mesh
    t0 = MPI.Wtime()
    if data.Lz != 0:
        mesh = create_mesh_3D(data.Lx, data.Ly, data.Lz, data.N, data.N, data.N)
    else:
        mesh = create_mesh_2D(data.Lx, data.Ly, data.N, data.N)

    
    # 2. Define Function Space and Level Set
    V = fem.functionspace(mesh, ("Lagrange", 1))
    tdim = mesh.topology.dim
    print(f"Nombre total d'éléments : {mesh.topology.index_map(tdim).size_global}")
    print(f"Nombre total de DOFs : {V.dofmap.index_map.size_global}")
    level_set = fem.Function(V)
    level_set.interpolate(data.level_set_function)
    level_set.x.scatter_forward()


    if data.use_noise:
        print("Applying noise to level set function...", flush=True)
        level_set = apply_noise(level_set, mesh)
        print("Applying noise done", flush=True)
        
    level_set.name = "phi_0"
    
    # Save evolution (Initial)
    evolution_file = XDMFFile(mesh.comm, "res/reinit_evolution.xdmf", "w")
    evolution_file.write_mesh(mesh)
    level_set.name = "phi"
    evolution_file.write_function(level_set, 0.0)

    # Save initial state
    with XDMFFile(mesh.comm, "res/initial_ls.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(level_set)


    # 3. Initialize Reinitialization Solver
    # l parameter for stabilization
    reinit_solver = Reinitialization(level_set, V, l=data.l_param, 
                                     solver_type=data.solver_type, 
                                     preconditioner_type=data.preconditioner_type)

    # 4. Run Predictor
    t0 = MPI.Wtime()
    phi_pred = reinit_solver.predictor(level_set)
    if mesh.comm.rank == 0:
        print(f"Predictor time: {MPI.Wtime() - t0:.4f}s", flush=True)
    phi_pred.name = "phi_predictor"
    
    # Save evolution (Predictor)
    phi_pred.name = "phi"
    evolution_file.write_function(phi_pred, 1.0)
    
    with XDMFFile(mesh.comm, "res/predictor_ls.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(phi_pred)
    print("predictor done", flush=True)

    # 5. Run Corrector
    t0_corr = MPI.Wtime()

    # Iterative corrector loop
    it = 0
    
    phi_current = fem.Function(V)
    phi_current.x.array[:] = level_set.x.array[:]
    
    eik_prev = error_L2_distance_func_form(phi_current, V)
    diff_eik = 1.0
    
    phi_next = fem.Function(V)
    
    diff_func = fem.Function(V)
    diff_L2_form = fem.form(ufl.inner(diff_func, diff_func) * ufl.dx)
    
    t0_corr = MPI.Wtime()
    while it < data.max_iterations and diff_eik > data.tolerance:
        t_it = MPI.Wtime()
        it += 1
        phi_next = reinit_solver.corrector(phi_current)

        # Calculate Eikonal error for stagnation
        eik_current = error_L2_distance_func_form(phi_next, V)
        diff_eik = abs(eik_current - eik_prev)
        eik_prev = eik_current

        # Save evolution (Corrector)
        phi_next.name = "phi"
        evolution_file.write_function(phi_next, float(it + 1))
        
        if mesh.comm.rank == 0:
            print(f"The iteration {it} is done.", flush=True)
        
        # Compute L2 difference
        diff_func.x.array[:] = phi_next.x.array[:] - phi_current.x.array[:]
        diff_L2_local = fem.assemble_scalar(diff_L2_form)
        diff_L2_global = mesh.comm.allreduce(diff_L2_local, op=MPI.SUM)
        diff_norm = np.sqrt(diff_L2_global)
        
        if mesh.comm.rank == 0:
            print(f"Iteration {it}: ||phi_new - phi_old||_L2 = {diff_norm:.4e}, Eikonal Error = {eik_current:.4e}, Diff Eikonal = {diff_eik:.4e} ({MPI.Wtime() - t_it:.4f}s)")
        
        phi_current.x.array[:] = phi_next.x.array[:]
        
    phi_corr = phi_current
    
    with XDMFFile(mesh.comm, "res/corrected_ls.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(phi_corr)
        
    if mesh.comm.rank == 0:
        if it == data.max_iterations:
            print(f"Warning: Corrector did not converge in {data.max_iterations} iterations.")
        else:
            print(f"Corrector converged in {it} iterations.")
        
        print(f"Total Corrector time: {MPI.Wtime() - t0_corr:.4f}s")
        print("Reinitialization complete. Results saved in 'res/' directory.")
        
    # evolution_file.close()
    if mesh.comm.rank == 0:
        print(f"--- Total execution time: {MPI.Wtime() - t_start:.4f}s ---")

if __name__ == "__main__":
    main()
