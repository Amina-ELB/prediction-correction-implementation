
import numpy as np
import ufl
from dolfinx import fem
from cutfemx.level_set import locate_entities
from cutfemx.fem import cut_form

def compute_fraction_mass(level_set, sol_exact, mesh, V_ls):
    """
    Computes the fraction of mass conservation.
    
    Args:
        level_set: The current level set function.
        sol_exact: The exact solution level set function.
        mesh: The mesh.
        V_ls: The function space.
        
    Returns:
        float: The ratio of computed mass to exact mass.
    """
    dim = mesh.topology.dim
    
    # locate_entities returns markers
    inside_entities = locate_entities(level_set, dim, "phi<0")
    intersected_entities = locate_entities(level_set, dim, "phi=0")
    
    dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities), (2, intersected_entities)], domain=mesh)

    inside_entities_exact = locate_entities(sol_exact, dim, "phi<0")
    intersected_entities_exact = locate_entities(sol_exact, dim, "phi=0")
    
    dx_exact = ufl.Measure("dx", subdomain_data=[(0, inside_entities_exact), (2, intersected_entities_exact)], domain=mesh)

    # Mass is integral of 1 where phi < 0
    mass_func_form = cut_form(fem.Constant(mesh, 1.0) * dx)
    mass_func = fem.assemble_scalar(mass_func_form)
    
    mass_sol_exact_form = fem.form(fem.Constant(mesh, 1.0) * dx_exact)
    mass_sol_exact = fem.assemble_scalar(mass_sol_exact_form)

    return mass_func / mass_sol_exact


def error_L2_distance_func(func, h, V_ls, xsi=1):
    """
    Computes L2 error of the distance function gradient norm.
    """
    norm = ufl.sqrt(ufl.inner(ufl.grad(func), ufl.grad(func)))
    euclidean_norm_expr = fem.Expression(norm, V_ls.element.interpolation_points())
    euclidean_norm = fem.Function(V_ls)
    euclidean_norm.interpolate(euclidean_norm_expr)
    return (np.sum(np.power(euclidean_norm.x.array[:] - 1, 2)) * (h**2)) * 0.5
     
def error_L2_form(func, sol_exact, V_ls, xsi=1):
    """
    Computes L2 error between func and sol_exact.
    """
    nabla_error = fem.form((xsi * (func - sol_exact))**2 * ufl.dx)
    nabla_error_exp = fem.assemble_scalar(nabla_error)
    return nabla_error_exp**0.5
    
    
def error_L2_distance_func_form(phi, V_ls, xsi=1):
    """
    Computes L2 error of (|grad(phi)| - 1).
    """
    euclidean_norm = ufl.sqrt(ufl.inner(ufl.grad(phi), ufl.grad(phi)))
    nabla_error = fem.form(((xsi * (euclidean_norm - 1))**2) * ufl.dx)
    nabla_error_exp = fem.assemble_scalar(nabla_error)
    return nabla_error_exp**0.5

    
def construct_functions_ls(ls_ef, ls_df, N, map_indexArray_dofs):
    """
    Helper to map DOFs between arrays.
    """
    for i in range(N):
        for j in range(N):
            ls_ef.x.array[round(map_indexArray_dofs[i, j])] = ls_df[j][i]  
    return ls_ef
