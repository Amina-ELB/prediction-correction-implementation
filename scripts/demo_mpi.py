
# The modules that will be used are imported:
import numpy as np
import math
import ufl
 
# mathematical language for FEM, auto differentiation, python
from dolfinx import fem, io, mesh
from dolfinx.cpp.mesh import h as mesh_size

# meshes, assembly, c++, ython, pybind
from ufl import ds, dx, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
from typing import TYPE_CHECKING


import sys
import os

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dolfinx.fem import Function
import dolfinx.fem.petsc 
from dolfinx.mesh import locate_entities, meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import ( dx, inner)


from src.reinitialization_mpi import reinitialization
from src.mesh_generation import create_mesh_3D, create_mesh_2D
from src.compute_error import error_L2_distance_func_form

from basix.ufl import element

import shutil


class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'



type_constraint_vm = "PN"

#"UKS" don't work 
i = 0

###################################
#### 	Mesh generation 	   ####
###################################

msh = create_mesh_3D(5,5,5,51,51,51)
#msh = create_mesh_2D(1,1,80,80)
print("Mesh built")
    
msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim-1)

###################################
### Initialization of the spaces###
###################################
V = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim, )))
V_vm = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim, )))
V_ls = fem.functionspace(msh, ("Lagrange", 1))

Q = fem.functionspace(msh, ("DG", 0))
V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, )))

# Initialization of the spacial coordinate
x = ufl.SpatialCoordinate(msh)

def circle_3D(x):
    r = 0.55
    print("***WARNING***")
    print("Error for r=0.2")
    return -1*(((x[0]-2.5)**2+(x[1]-2.5)**2+(x[2]-1)**2)-r**2)

def doughnut(x):
    r = 0.55
    R = 1.8
    return (R-np.sqrt(x[0]**2+x[1]**2))**2+x[2]**2-r**2 +1e-9

def circle_2D(x):
    r = 0.55
    print("***WARNING***")
    print("Error for r=0.2")
    return -1*(((x[0]-2.5)**2+(x[1]-2.5)**2)-r**2)
    
def star(x):
	c = [0.5, 0.5]
	n = 8.
	eps = 0.000000001
	return 0.5 * (ufl.sqrt((x[0] - c[0])*2 + (x[1] - c[1])*2) - 0.1*ufl.cos(n * ufl.atan((x[1]-c[1])/( (x[0]-c[0]) + ufl.sign(x[0]-c[0]+eps)*eps)) +2.)-0.25)
 
ls_func = fem.Function(V_ls)
ls_func.interpolate(doughnut)
dim = msh.topology.dim

ls_func_init = fem.Function(V_ls)
ls_func_init.interpolate(doughnut)


reinit_solver = reinitialization(level_set = ls_func, V_ls=V_ls, l=3)

###################################



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
error = 1

if error == 1:
    if rank == 0:
        if os.path.exists('res'):
            shutil.rmtree('res')
        os.mkdir('res')
        folder_fraction_mass = open("res/cut_erro_mass.txt", "x")
        folder_errorL2_distance_norm = open("res/cut_error_L2_norm_distance_.txt", "x")
        folder_errorinterface_normL2 = open("res/error_interface_norm_L2_.txt", "x")

    vect_eikonal_error = [] 
    vect_interface_error = [] 

time = 0.
if not os.path.exists("res_3D_nKUpdate"):
    os.makedirs("res_3D_nKUpdate")
xdmf_ls = io.XDMFFile(msh.comm, "res_3D_nKUpdate/level_set_MPI.xdmf", "w")
xdmf_ls.write_mesh(msh)
ls_func.name = "ls_func"
xdmf_ls.write_function(ls_func, time)
time += 1

num_step = 0

###################################
####    Reinitialization       ####
###################################
# Reinitialization of the initial level set to have signed ditance function
if error == 1:
    val_error_L2_distance_norm = error_L2_distance_func_form(ls_func,Q)
    collected_error = comm.allreduce(val_error_L2_distance_norm,op=MPI.SUM)
    vect_eikonal_error.append(collected_error)
    error_L2interface = reinit_solver.interface_error(ls_func,ls_func_init)
    collected_error_L2 = comm.allreduce(error_L2interface,op=MPI.SUM)
    vect_interface_error.append(collected_error_L2)
    if rank == 0:
        folder_errorL2_distance_norm.write(str(collected_error)+"\n")
        folder_errorinterface_normL2.write(str(collected_error_L2)+"\n")

print("Reinitialization of the initialized level-set")
ls_predict = reinit_solver.predictor(ls_func)
print("After predictor")
ls_func.x.array[:] = ls_predict.x.array
xdmf_ls.write_function(ls_func, time)
time += 1

if error == 1 :
    val_error_L2_distance_norm = error_L2_distance_func_form(ls_func,Q)
    collected_error = comm.allreduce(val_error_L2_distance_norm,op=MPI.SUM)
    vect_eikonal_error.append(collected_error)
    error_L2interface = reinit_solver.interface_error(ls_func,ls_func_init)
    collected_error_L2 = comm.allreduce(error_L2interface,op=MPI.SUM)
    vect_interface_error.append(collected_error_L2)
    # loss_mass = compute_fraction_mass(ls_func,ls_func_init,msh,V_ls)
    # folder_errorinterface_normL2.write(str(error_L2interface)+"\n")
    # folder_fraction_mass.write(str(loss_mass)+"\n")

    if rank == 0:
        folder_errorL2_distance_norm.write(str(collected_error)+"\n")
        folder_errorinterface_normL2.write(str(collected_error_L2)+"\n")
        L2_dist_n = collected_error
error_cv = 1e10

while (num_step < 1) and (error_cv>1e-8):
    num_step += 1
    ls_correct = reinit_solver.corrector(ls_func)
    ls_func.x.array[:] = ls_correct.x.array
    ls_func.x.scatter_forward() 
    xdmf_ls.write_function(ls_func, time)
    time += 1
    print("Iteration",num_step)
    if error == 1:
        val_error_L2_distance_norm = error_L2_distance_func_form(ls_func,Q)
        collected_error = comm.allreduce(val_error_L2_distance_norm,op=MPI.SUM)
        vect_eikonal_error.append(collected_error)
        error_L2interface = reinit_solver.interface_error(ls_func,ls_func_init)
        collected_error_L2 = comm.allreduce(error_L2interface,op=MPI.SUM)
        vect_interface_error.append(collected_error_L2)
        # loss_mass = compute_fraction_mass(ls_func,ls_func_init,msh,V_ls)
        # folder_errorinterface_normL2.write(str(error_L2interface)+"\n")
        # folder_fraction_mass.write(str(loss_mass)+"\n")
        if rank == 0:
            folder_errorL2_distance_norm.write(str(collected_error)+"\n")
            folder_errorinterface_normL2.write(str(collected_error_L2)+"\n")
            error_cv = abs(collected_error - L2_dist_n)
            L2_dist_n = collected_error