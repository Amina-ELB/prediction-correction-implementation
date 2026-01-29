
import sys
import os
import shutil
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reinitialization import Reinitialization
from src.mesh_generation import create_mesh_3D, create_mesh_2D
from src.compute_error import error_L2_distance_func_form

def circle_3D(x):
    r = 0.55
    # Center at (2.5, 2.5, 1)
    return -1 * (((x[0] - 2.5)**2 + (x[1] - 2.5)**2 + (x[2] - 1)**2) - r**2)

def torus(x):
    r = 0.55
    R = 1.8
    # Torus equation: (R - sqrt(x^2 + y^2))^2 + z^2 - r^2
    return (R - np.sqrt(x[0]**2 + x[1]**2))**2 + x[2]**2 - r**2 + 1e-9

def circle_2D(x):
    r = 0.55
    return -1 * (((x[0] - 2.5)**2 + (x[1] - 2.5)**2) - r**2)
    

def main():
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Mesh generation
    # Dimensions: 5x5x5, Resolution: 51x51x51
    msh = create_mesh_3D(5, 5, 5, 51, 51, 51)
    # msh = create_mesh_2D(1, 1, 80, 80)
    
    if rank == 0:
        print("Mesh built")
    
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)

    # Function spaces
    V_ls = fem.functionspace(msh, ("Lagrange", 1))
    Q = fem.functionspace(msh, ("DG", 0))

    # Initialize level set function
    ls_func = fem.Function(V_ls)
    ls_func.interpolate(torus)
    ls_func.name = "ls_func"

    ls_func_init = fem.Function(V_ls)
    ls_func_init.interpolate(torus)

    # Initialize solver
    reinit_solver = Reinitialization(level_set=ls_func, V_ls=V_ls, l=3)

    # Output setup
    compute_error = True
    output_dir = "res"
    xdmf_dir = "res_3D_nKUpdate"
    
    if rank == 0:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        if not os.path.exists(xdmf_dir):
            os.makedirs(xdmf_dir)

    # Wait for directory creation
    comm.Barrier()
    
    if compute_error and rank == 0:
        f_mass = open(f"{output_dir}/cut_erro_mass.txt", "w")
        f_L2_dist = open(f"{output_dir}/cut_error_L2_norm_distance_.txt", "w")
        f_interface = open(f"{output_dir}/error_interface_norm_L2_.txt", "w")

    vect_eikonal_error = [] 
    vect_interface_error = [] 

    time = 0.
    xdmf_ls = io.XDMFFile(msh.comm, f"{xdmf_dir}/level_set_MPI.xdmf", "w")
    xdmf_ls.write_mesh(msh)
    xdmf_ls.write_function(ls_func, time)
    time += 1
    
    num_step = 0
    
    # Initial error
    if compute_error:
        val_error_L2_distance_norm = error_L2_distance_func_form(ls_func, Q)
        collected_error = comm.allreduce(val_error_L2_distance_norm, op=MPI.SUM)
        vect_eikonal_error.append(collected_error)
        
        error_L2interface = reinit_solver.interface_error(ls_func, ls_func_init)
        collected_error_L2 = comm.allreduce(error_L2interface, op=MPI.SUM)
        vect_interface_error.append(collected_error_L2)
        
        if rank == 0:
            f_L2_dist.write(str(collected_error) + "\n")
            f_interface.write(str(collected_error_L2) + "\n")

    if rank == 0:
        print("Reinitialization of the initialized level-set")
        
    # Predictor step
    ls_predict = reinit_solver.predictor(ls_func)
    if rank == 0:
        print("After predictor")
        
    ls_func.x.array[:] = ls_predict.x.array
    xdmf_ls.write_function(ls_func, time)
    time += 1

    # Error after predictor
    if compute_error:
        val_error_L2_distance_norm = error_L2_distance_func_form(ls_func, Q)
        collected_error = comm.allreduce(val_error_L2_distance_norm, op=MPI.SUM)
        vect_eikonal_error.append(collected_error)
        
        error_L2interface = reinit_solver.interface_error(ls_func, ls_func_init)
        collected_error_L2 = comm.allreduce(error_L2interface, op=MPI.SUM)
        vect_interface_error.append(collected_error_L2)
        
        if rank == 0:
            f_L2_dist.write(str(collected_error) + "\n")
            f_interface.write(str(collected_error_L2) + "\n")
            L2_dist_n = collected_error

    # Corrector loop
    error_cv = 1e10
    max_steps = 1 # Was 1 in original code
    
    while (num_step < max_steps) and (error_cv > 1e-8):
        num_step += 1
        ls_correct = reinit_solver.corrector(ls_func)
        ls_func.x.array[:] = ls_correct.x.array
        ls_func.x.scatter_forward() 
        xdmf_ls.write_function(ls_func, time)
        time += 1
        
        if rank == 0:
            print("Iteration ", num_step)
            
        if compute_error:
            val_error_L2_distance_norm = error_L2_distance_func_form(ls_func, Q)
            collected_error = comm.allreduce(val_error_L2_distance_norm, op=MPI.SUM)
            vect_eikonal_error.append(collected_error)
            
            error_L2interface = reinit_solver.interface_error(ls_func, ls_func_init)
            collected_error_L2 = comm.allreduce(error_L2interface, op=MPI.SUM)
            vect_interface_error.append(collected_error_L2)
            
            if rank == 0:
                f_L2_dist.write(str(collected_error) + "\n")
                f_interface.write(str(collected_error_L2) + "\n")
                error_cv = abs(collected_error - L2_dist_n)
                L2_dist_n = collected_error

    if compute_error and rank == 0:
        f_mass.close()
        f_L2_dist.close()
        f_interface.close()
        
    xdmf_ls.close()

if __name__ == "__main__":
    main()
