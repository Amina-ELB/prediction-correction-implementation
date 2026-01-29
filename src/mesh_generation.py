
import numpy as np
from mpi4py import MPI
from dolfinx import mesh

from dolfinx.mesh import GhostMode

def create_mesh_3D(Lx, Ly, Lz, nx, ny, nz):
    """
    Creates a 3D box mesh.
    
    Args:
        Lx, Ly, Lz: Dimensions of the box in x, y, z directions.
        nx, ny, nz: Number of elements in x, y, z directions.
        
    Returns:
        dolfinx.mesh.Mesh: The generated 3D mesh.
    """
    msh = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet
    )
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    return msh

def create_mesh_2D(Lx, Ly, nx, ny):
    """
    Creates a 2D rectangle mesh.
    
    Args:
        Lx, Ly: Dimensions of the rectangle in x, y directions.
        nx, ny: Number of elements in x, y directions.
        
    Returns:
        dolfinx.mesh.Mesh: The generated 2D mesh.
    """
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([Lx, Ly])],
        [nx, ny],
        cell_type=mesh.CellType.triangle,
        ghost_mode=GhostMode.shared_facet
    )
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    return msh
