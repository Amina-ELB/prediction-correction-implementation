import time
import numpy as np
from mpi4py import MPI
from dolfinx import fem
import sys
import os

# Ensure the root directory is accessible so we can import src and data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh_generation import create_mesh_2D
from src.reinitialization import Reinitialization
import data

def main():
    # Setup the mesh and level set outside the timing loop
    mesh = create_mesh_2D(data.Lx, data.Ly, data.N, data.N)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    
    # We will test these solvers
    solvers_to_test = ["mumps", "gmres", "petsc"]
    
    results = []

    for solver_type in solvers_to_test:
        if mesh.comm.rank == 0:
            print(f"\n--- Testing Solver: {solver_type} ---")
            
        try:
            # Recreate level set to start fresh
            level_set = fem.Function(V)
            level_set.interpolate(data.level_set_function)
            level_set.x.scatter_forward()
            
            # Timing setup/assembly
            start_setup = time.perf_counter()
            reinit_solver = Reinitialization(level_set, V, l=data.l_param, solver_type=solver_type)
            end_setup = time.perf_counter()

            # Timing predictor
            start_pred = time.perf_counter()
            phi_pred = reinit_solver.predictor(level_set)
            end_pred = time.perf_counter()

            # Timing corrector
            phi_current = fem.Function(V)
            phi_current.x.array[:] = phi_pred.x.array[:]
            
            it = 0
            start_corr = time.perf_counter()
            while it < data.max_iterations:
                it += 1
                phi_next = reinit_solver.corrector(phi_current)

                phi_current.x.array[:] = phi_next.x.array[:]
            
            end_corr = time.perf_counter()

            time_setup = end_setup - start_setup
            time_pred = end_pred - start_pred
            time_corr = end_corr - start_corr
            total_time = time_setup + time_pred + time_corr
            
            # Gather times on rank 0
            if mesh.comm.rank == 0:
                results.append({
                    "solver": solver_type,
                    "setup": time_setup,
                    "predictor": time_pred,
                    "corrector": time_corr,
                    "total": total_time
                })
                print(f"Setup time     : {time_setup:.4f} s")
                print(f"Predictor time : {time_pred:.4f} s")
                print(f"Corrector time : {time_corr:.4f} s")
                print(f"Total time     : {total_time:.4f} s")
                
        except Exception as e:
            if mesh.comm.rank == 0:
                print(f"Solver {solver_type} failed with error: {e}")

    # Summary
    if mesh.comm.rank == 0:
        print("\n=======================================================")
        print(f"{'Solver':<15} | {'Setup (s)':<10} | {'Pred (s)':<10} | {'Corr (s)':<10} | {'Total (s)':<10}")
        print("-" * 65)
        for res in results:
            print(f"{res['solver']:<15} | {res['setup']:<10.4f} | {res['predictor']:<10.4f} | {res['corrector']:<10.4f} | {res['total']:<10.4f}")
        print("=======================================================\n")

if __name__ == "__main__":
    main()
