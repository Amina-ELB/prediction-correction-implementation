
import numpy as np
from mpi4py import MPI
from dolfinx import mesh

def create_mesh_3D(Lx, Ly, Lz, nx, ny, nz):
    """
    Creates a 3D box mesh.
    
    Args:
        Lx, Ly, Lz: Dimensions of the box in x, y, z directions.
        nx, ny, nz: Number of elements in x, y, z directions.
        
    Returns:
        dolfinx.mesh.Mesh: The generated 3D mesh.
    """
    return mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron
    )

def create_mesh_2D(Lx, Ly, nx, ny):
    """
    Creates a 2D rectangle mesh.
    
    Args:
        Lx, Ly: Dimensions of the rectangle in x, y directions.
        nx, ny: Number of elements in x, y directions.
        
    Returns:
        dolfinx.mesh.Mesh: The generated 2D mesh.
    """
    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([Lx, Ly])],
        [nx, ny],
        cell_type=mesh.CellType.triangle
    )
