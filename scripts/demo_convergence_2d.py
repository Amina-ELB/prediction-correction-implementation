import sys
import os

# Add the parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from mpi4py import MPI

from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import cut_form, cut_function, assemble_scalar as assemble_scalar_cut

from cutfemx.petsc import assemble_vector, assemble_matrix, deactivate, locate_dofs

from dolfinx.fem.assemble import assemble_scalar

from dolfinx import fem, mesh, plot, la

# from dolfinx.fem.assemble import assemble_vector, assemble_matrix
from dolfinx.fem import form
from dolfinx.io import XDMFFile
from dolfinx.mesh import GhostMode
import dolfinx.fem.petsc
import cutfemx
import ufl
from ufl import dS, dx, grad, inner, dc, FacetNormal, CellDiameter, dot, avg, jump, ds
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.compute_error import error_L2_distance_func_form
try:
    from petsc4py import PETSc

    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
except ModuleNotFoundError:
    print("This demo requires petsc4py.")
    exit(0)

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

# Define the function
def circle(x):
    return (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5) - 0.25 * 0.25  + 1e-16

def circle_exact_solution(x):
    return np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) - 0.25 + 1e-16

# Define the function to compute the sign of the level set function
# This function returns -1 for negative values, 1 for positive values, and 0 for zero
def sgn(phi_array):
    sign_array = np.zeros(phi_array.shape)
    for i in range(len(phi_array)):
        if phi_array[i] < 0:
            sign_array[i] = -1
        elif phi_array[i] > 0:
            sign_array[i] = 1
        else:
            sign_array[i] = 0
    return sign_array


def norm(phi):
    return ufl.max_value(ufl.sqrt(ufl.dot(phi, phi)), 1e-15)


def d_1(x):
    return 1.0 / x


def classic(x):
    outarray = np.zeros(x.shape)
    i = 0
    for val in x:
        outarray[i] = d_1(val)
        i += 1
    return outarray


def interface_error(phi_reinit, phi_initial, mesh, dim, xsi=1):
    """Compute L2 error on the interface (phi=0) between two level-set functions.

    Parameters
    - phi_reinit: dolfinx Function for the reinitialized level-set
    - phi_initial: dolfinx Function for the initial level-set to compare against
    - f: level-set Function used to locate the interface (phi=0)
    - mesh: dolfinx.mesh.Mesh where the measurement is taken
    - dim: topological dimension of the mesh (mesh.topology.dim)
    - xsi: unused here kept for API-compatibility

    Returns
    - interface L2 error (float)
    """
    order = 1
    # locate interface and inside cells
    intersected_entities = locate_entities(phi_initial, dim, "phi=0")
    inside_entities = locate_entities(phi_initial, dim, "phi<0")

    # quadrature rules for inside and interface
    inside_quadrature = runtime_quadrature(phi_initial, "phi<0", order)
    interface_quadrature = runtime_quadrature(phi_initial, "phi=0", order)

    quad_domains = [(0, inside_quadrature), (1, interface_quadrature)]

    dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=mesh)

    # pick the interface quadrature measure
    dsq_exact = dx_rt(1)

    # squared difference on the interface
    error_form = cut_form(inner(phi_reinit, phi_reinit ) * dsq_exact)

    # assemble using cutfem assembly for cut measures
    err_val = cutfemx.fem.assemble_scalar(error_form) ** 0.5
    print("Interface error =", err_val)
    return err_val

def error_L2_distance_func_form(phi, V_ls, xsi=1):
    mesh = V_ls.mesh
    euclidean_norm = ufl.sqrt(inner(ufl.grad(phi), ufl.grad(phi)))
    nabla_error = fem.form(((xsi * (euclidean_norm - 1)) ** 2) * ufl.dx)
    nabla_error_exp = fem.assemble_scalar(nabla_error)
    
    vol_form = fem.form(fem.Constant(mesh, 1.0) * ufl.dx)
    volume = fem.assemble_scalar(vol_form)
    
    return (nabla_error_exp / volume)**0.5

import gmsh
from dolfinx.io import gmshio


def create_annulus_mesh(N):
    gmsh.initialize()
    gmsh.model.add("annulus")

    # Geometry parameters
    R_in = 0.15
    R_out = 0.35
    c_x = 0.5
    c_y = 0.5

    # Create geometry
    c1 = gmsh.model.occ.addDisk(c_x, c_y, 0, R_out, R_out)
    c2 = gmsh.model.occ.addDisk(c_x, c_y, 0, R_in, R_in)
    out_dim_tags = gmsh.model.occ.cut([(2, c1)], [(2, c2)])
    gmsh.model.occ.synchronize()

    # Add physical groups
    # Get all curves
    curves = gmsh.model.getEntities(dim=1)
    inner_curves = []
    outer_curves = []

    for dim, tag in curves:
        # Check center of mass to identify curves
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        # Distance from center (0.5, 0.5)
        r_com = np.sqrt((com[0] - c_x) ** 2 + (com[1] - c_y) ** 2)

        if np.isclose(r_com, R_in, atol=1e-1):
            inner_curves.append(tag)
        elif np.isclose(r_com, R_out, atol=1e-1):
            outer_curves.append(tag)

    gmsh.model.addPhysicalGroup(1, inner_curves, 1)  # Inner
    gmsh.model.addPhysicalGroup(1, outer_curves, 2)  # Outer

    surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)  # Domain

    # Mesh size
    # N corresponds to 1/h roughly.
    # Domain size is approx 1 (outer diameter 0.9)
    lc = 1.0 / N
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    gmsh.model.mesh.generate(2)

    msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()

    return msh, facet_markers


def create_square_mesh(N):
    # Domain [0, 1] x [0, 1]
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [N, N],
        cell_type=mesh.CellType.triangle,
        ghost_mode=GhostMode.shared_facet,
    )
    
    # Create facet markers
    # Mark all boundaries as 2 (Outer)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    
    facet_markers = mesh.meshtags(
        msh, fdim, boundary_facets, np.full(len(boundary_facets), 2, dtype=np.int32)
    )
    
    return msh, facet_markers


def solve_problem(N, geometry_type="annulus"):
    if geometry_type == "annulus":
        msh, facet_markers = create_annulus_mesh(N)
    elif geometry_type == "square":
        msh, facet_markers = create_square_mesh(N)
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")

    gdim = 2
    tdim = msh.topology.dim

    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, PETSc.ScalarType(1.0))
    g = fem.Constant(msh, PETSc.ScalarType(0.0))
    gamma = 100
    gamma_g = 1e-1

    n = FacetNormal(msh)
    h = CellDiameter(msh)

    V_ls = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V_ls)
    level_set.interpolate(circle)

    ls_ex = fem.Function(V_ls)
    ls_ex.interpolate(circle_exact_solution)

    # sign = fem.Function(V_ls)
    # sign.x.array[:] = sgn(level_set.x.array)
    eps = 1e-6
    sign = level_set / (ufl.sqrt(level_set**2+eps**2))

    dim = msh.topology.dim

    intersected_entities = locate_entities(level_set, dim, "phi=0")
    inside_entities = locate_entities(level_set, dim, "phi<0")
    outside_entities = locate_entities(level_set, dim, "phi>0")

    V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
    n_K = fem.Function(V_DG)
    compute_normal(n_K, level_set, intersected_entities)
    # n_K = fem.Constant(msh, (PETSc.ScalarType(1.0),PETSc.ScalarType(0.0)))

    DG0 = fem.functionspace(msh, ("DG", 0))
    norm_grad_phi_expr = fem.Expression(norm(grad(level_set)), DG0.element.interpolation_points())
    norm_grad_phi_func = fem.Function(DG0)
    norm_grad_phi_func.interpolate(norm_grad_phi_expr)

    classic_func = fem.Function(DG0)
    classic_func.x.array[:] = classic(norm_grad_phi_func.x.array)

    dof_coordinates = V.tabulate_dof_coordinates()

    interface_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi=0")
    # interface_mesh = create_cut_cells_mesh(msh.comm, interface_cells)
    if len(interface_cells.cut_cells) == 0:
        print("No interface cells found")
        return None, None, None, None, None

    order = 2
    inside_quadrature = runtime_quadrature(level_set, "phi<0", order)
    interface_quadrature = runtime_quadrature(level_set, "phi=0", order)
    outside_quadrature = runtime_quadrature(level_set, "phi>0", order)

    quad_domains = [(0, inside_quadrature), (1, interface_quadrature), (2, outside_quadrature)]

    gp_ids = ghost_penalty_facets(level_set, "phi<0")
    gp_topo = facet_topology(msh, gp_ids)

    dx = ufl.Measure(
        "dx",
        subdomain_data=[(0, inside_entities), (1, intersected_entities), (2, outside_entities)],
        domain=msh,
    )
    dS = ufl.Measure("dS", subdomain_data=[(0, gp_topo)], domain=msh)
    dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

    dxq = dx_rt(0) + dx(0)
    dsq = dx_rt(1)

    # Nitsche BC on boundaries
    # Tag 1: Inner, Tag 2: Outer
    ds_inner = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers, subdomain_id=1)
    ds_outer = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers, subdomain_id=2)

    def aC(u, v):
        a = inner(grad(u), grad(v)) * dx

        # cut cells on surface
        a += -dot(grad(u), n_K) * v * dsq
        a += -dot(grad(v), n_K) * u * dsq
        a += gamma * 1.0 / h * u * v * dsq

        # Nitsche on outer boundary
        a += -dot(grad(u), n) * v * ds_outer
        a += -dot(grad(v), n) * u * ds_outer
        a += gamma * 1.0 / h * u * v * ds_outer

        # Nitsche on inner boundary
        a += -dot(grad(u), n) * v * ds_inner
        a += -dot(grad(v), n) * u * ds_inner
        a += gamma * 1.0 / h * u * v * ds_inner
        return a

    def LC(func):
        L = inner(func * grad(level_set), grad(v)) * dx
        L += -dot(func * grad(level_set), n_K) * v * dsq
        L + -dot(grad(v) / norm(grad(v)), n_K) * level_set * dsq
        L += -dot(grad(v), n_K) * g * dsq
        L += gamma * 1.0 / h * v * g * dsq

        # Nitsche on outer boundary (RHS)
        L += -dot(grad(v), n) * ls_ex * ds_outer
        L += gamma * 1.0 / h * v * ls_ex * ds_outer

        # Nitsche on inner boundary (RHS)
        L += -dot(grad(v), n) * ls_ex * ds_inner
        L += gamma * 1.0 / h * v * ls_ex * ds_inner
        return L

    a = aC(u, v)
    L1 = LC(classic_func)

    a_cut = cut_form(a)
    L1_cut = cut_form(L1)

    A = assemble_matrix(a_cut)
    A.assemble()

    def b_vector(L_cut):
        b = assemble_vector(L_cut)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return b

    b1 = b_vector(L1_cut)

    # Set solver options
    opts = PETSc.Options()  # type: ignore
    opts["ksp_rtol"] = 1.0e-12
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"

    # Create PETSc Krylov solver andv    turn convergence monitoring on
    solver = PETSc.KSP().create(msh.comm)  # type: ignore
    solver.setFromOptions()

    # Set matrix operator
    solver.setOperators(A)

    uh1 = fem.Function(V)
    uh1.name = "uh1"

    # Set a monitor, solve linear system, and display the solver
    # configuration
    # solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
    solver.solve(b1, uh1.x.petsc_vec)
    uh1.x.scatter_forward()

    # Predictor
    uh_predictor = fem.Function(V)
    uh_predictor.name = "uh_predictor"

    a_predictor = inner(grad(u), grad(v)) * dx
    a_predictor += -dot(grad(u), n_K) * v * dsq
    a_predictor += -dot(grad(v), n_K) * u * dsq
    a_predictor += gamma * 1.0 / h * u * v * dsq

    # Nitsche on outer boundary for predictor
    a_predictor += -dot(grad(u), n) * v * ds_outer
    a_predictor += -dot(grad(v), n) * u * ds_outer
    a_predictor += gamma * 1.0 / h * u * v * ds_outer

    # Nitsche on inner boundary for predictor
    a_predictor += -dot(grad(u), n) * v * ds_inner
    a_predictor += -dot(grad(v), n) * u * ds_inner
    a_predictor += gamma * 1.0 / h * u * v * ds_inner

    a_predictor_cut = cut_form(a_predictor)

    L_predictor = inner(sign, v) * dx
    # L_predictor = inner(-1.0, v) * (dx_rt(0) + dx(0))
    # L_predictor += inner(1.0, v) * (dx_rt(2) + dx(2))
    L_predictor += -dot(grad(v), n_K) * g * dsq
    L_predictor += gamma * 1.0 / h * v * g * dsq
    L_predictor += inner(sign, v) * ds

    # Nitsche on outer boundary for predictor RHS
    L_predictor += -dot(grad(v), n) * ls_ex * ds_outer
    L_predictor += gamma * 1.0 / h * v * ls_ex * ds_outer

    # Nitsche on inner boundary for predictor RHS
    L_predictor += -dot(grad(v), n) * ls_ex * ds_inner
    L_predictor += gamma * 1.0 / h * v * ls_ex * ds_inner

    L_predictor_cut = cut_form(L_predictor)

    A_predictor = assemble_matrix(a_predictor_cut)
    A_predictor.assemble()

    b_predictor = assemble_vector(L_predictor_cut)
    b_predictor.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_predictor.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    solver.setOperators(A_predictor)
    solver.solve(b_predictor, uh_predictor.x.petsc_vec)
    uh_predictor.x.scatter_forward()

    # Corrector
    uh_corrector = fem.Function(V)
    uh_corrector.name = "uh_corrector"

    # Initial guess for corrector is the predictor
    uh_corrector.x.array[:] = uh_predictor.x.array[:]

    uh_prev = fem.Function(V)
    uh_prev.x.array[:] = uh_corrector.x.array[:]
    eikonal_norm_pred = error_L2_distance_func_form(uh_prev, V)
    diff_func = fem.Function(V)

    tol = 1e-8
    max_iter = 4000
    it = 0
    diff_norm = 1.0
    diff_eik = 1.0

    while diff_eik > tol and it < max_iter:
        it += 1

        # Update level set for linearization
        level_set.x.array[:] = uh_prev.x.array[:]
        norm_grad_phi_func.interpolate(norm_grad_phi_expr)
        classic_func.x.array[:] = classic(norm_grad_phi_func.x.array)

        b_corrector = assemble_vector(L1_cut)
        b_corrector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_corrector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        solver.setOperators(A)
        solver.solve(b_corrector, uh_corrector.x.petsc_vec)
        uh_corrector.x.scatter_forward()
        eikonal_norm = error_L2_distance_func_form(uh_corrector, V)
        diff_eik = abs(eikonal_norm - eikonal_norm_pred)
        eikonal_norm_pred = eikonal_norm
        # Compute difference
        diff_func.x.array[:] = uh_corrector.x.array - uh_prev.x.array

        # L2 norm of difference
        diff_L2 = fem.form(ufl.inner(diff_func, diff_func) * ufl.dx)
        diff_L2_local = assemble_scalar(diff_L2)
        diff_L2_global = msh.comm.allreduce(diff_L2_local, op=MPI.SUM)
        diff_norm = np.sqrt(diff_L2_global)

        # if it % 10 == 0:
        #     print(f"  Iteration {it}: ||u_new - u_old||_L2 = {diff_norm:.4e}")

        uh_prev.x.array[:] = uh_corrector.x.array[:]

    if it == max_iter:
        print(
            f"  Warning: Corrector did not converge in {max_iter} iterations. Final diff: {diff_norm:.4e}"
        )

    # Error computation
    V3 = fem.functionspace(msh, ("Lagrange", 3))
    ls_ex_3 = fem.Function(V3)
    ls_ex_3.interpolate(circle_exact_solution)

    ls_3 = fem.Function(V3)
    ls_3.interpolate(uh_corrector)

    e_W = fem.Function(V3)
    e_W.x.array[:] = ls_3.x.array - ls_ex_3.x.array

    vol_form = fem.form(fem.Constant(msh, 1.0) * ufl.dx)
    volume = msh.comm.allreduce(assemble_scalar(vol_form), op=MPI.SUM)

    # L2 Error
    error_L2 = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_L2_local = assemble_scalar(error_L2)
    error_L2_global = msh.comm.allreduce(error_L2_local, op=MPI.SUM)
    l2_norm = np.sqrt(error_L2_global / volume)

    # H1 Error
    error_H1 = fem.form(ufl.inner(grad(e_W), grad(e_W)) * ufl.dx)
    error_H1_local = assemble_scalar(error_H1)
    error_H1_global = msh.comm.allreduce(error_H1_local, op=MPI.SUM)
    h1_norm = np.sqrt(error_H1_global / volume)

    # Interface Error (L2 on Gamma)
    interface_norm = interface_error(uh_corrector, ls_ex, msh, msh.topology.dim)

    # Eikonal Error: || |grad(u)| - 1 ||_L2
    # Eikonal Error: || |grad(u)| - 1 ||_L2
    # We use uh_corrector for u
    eikonal_norm = error_L2_distance_func_form(uh_corrector, V)

    # Mesh size (approximate)
    if geometry_type == "annulus":
        h_size = 1.0 / N  # Domain is [-1, 1] x [-1, 1], so length 2.
    else:
        h_size = 1.0 / N  # Domain is [0, 1] x [0, 1], so length 1.

    return h_size, l2_norm, h1_norm, interface_norm, eikonal_norm


if __name__ == "__main__":
    # Geometry selection
    geometry_type = "annulus"
    # geometry_type = "square"

    # Start with N=16 and increase by factor of 1.5
    # 16, 24, 36, 54, 81, 121
    N_values = [40,60,80,120,160,240,360]
    results = []

    output_file = "convergence_results.txt"
    
    with open(output_file, "w") as f:
        header = f"{'N':<5} {'h':<10} {'L2 Error':<15} {'Rate':<10} {'H1 Error':<15} {'Rate':<10} {'Int. Error':<15} {'Rate':<10} {'Eik. Error':<15} {'Rate':<10}"
        print(header)
        f.write(header + "\n")
        
        separator = "-" * 130
        print(separator)
        f.write(separator + "\n")

        prev_h = None
        prev_l2 = None
        prev_h1 = None
        prev_int = None
        prev_eik = None

        h_values = []
        l2_errors = []
        h1_errors = []
        int_errors = []
        eik_errors = []

        for N in N_values:
            h, l2, h1, int_err, eik_err = solve_problem(N, geometry_type=geometry_type)
            if h is None:
                continue

            h_values.append(h)
            l2_errors.append(l2)
            h1_errors.append(h1)
            int_errors.append(int_err)
            eik_errors.append(eik_err)

            rate_l2 = 0.0
            rate_h1 = 0.0
            rate_int = 0.0
            rate_eik = 0.0

            if prev_h is not None:
                rate_l2 = np.log(l2 / prev_l2) / np.log(h / prev_h)
                rate_h1 = np.log(h1 / prev_h1) / np.log(h / prev_h)
                rate_int = np.log(int_err / prev_int) / np.log(h / prev_h)
                rate_eik = np.log(eik_err / prev_eik) / np.log(h / prev_h)

            row = f"{N:<5} {h:<10.4f} {l2:<15.4e} {rate_l2:<10.2f} {h1:<15.4e} {rate_h1:<10.2f} {int_err:<15.4e} {rate_int:<10.2f} {eik_err:<15.4e} {rate_eik:<10.2f}"
            print(row)
            f.write(row + "\n")

            prev_h = h
            prev_l2 = l2
            prev_h1 = h1
            prev_int = int_err
            prev_eik = eik_err

    # Plotting
    plt.figure(figsize=(10, 10))
    
    # Plot errors
    plt.plot(h_values, l2_errors, 'k--o', label=r'$\left\Vert u - u_{ex} \right\Vert_{L^{2}}$')
    plt.plot(h_values, h1_errors, 'k--s', label=r'$\left\Vert \nabla(u - u_{ex}) \right\Vert_{L^{2}}$')
    plt.plot(h_values, int_errors, 'k--^', label=r'$\left\Vert u - u_{ex} \right\Vert_{L^{2}(\Gamma)}$')
    plt.plot(h_values, eik_errors, 'k--d', label=r'$\left\Vert |\nabla u| - 1 \right\Vert_{L^{2}}$')

    # Reference lines
    h_arr = np.array(h_values)
    # O(h^2)
    plt.plot(h_arr, h_arr**2 * (l2_errors[0] / h_arr[0] ** 2), "k-", label="Order 2")
    # O(h)
    plt.plot(h_arr, h_arr * (h1_errors[0] / h_arr[0]), "k-.", label="Order 1")

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('h', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title(f"Convergence Study: Circle Redistance ({geometry_type})", fontsize=16)
    plt.legend(fontsize=14, loc='lower right')
    plt.grid(True, which="both", ls="-")
    plt.savefig("convergence_plot.pdf")
    print("Plot saved to convergence_plot.pdf")
