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
except ModuleNotFoundError:
    print("This demo requires petsc4py and dolfinx.")
    exit(0)

# Define the functions
def circle(x):
    return (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5) - 0.25 * 0.25  + 1e-16

def circle_exact_solution(x):
    return np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) - 0.25 + 1e-16

def norm(phi):
    return ufl.max_value(ufl.sqrt(ufl.dot(phi, phi)), 1e-15)

def interface_error(phi_reinit, phi_initial, mesh, dim, xsi=1):
    order = 1
    intersected_entities = locate_entities(phi_initial, dim, "phi=0")
    inside_entities = locate_entities(phi_initial, dim, "phi<0")

    inside_quadrature = runtime_quadrature(phi_initial, "phi<0", order)
    interface_quadrature = runtime_quadrature(phi_initial, "phi=0", order)

    quad_domains = [(0, inside_quadrature), (1, interface_quadrature)]
    dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=mesh)
    dsq_exact = dx_rt(1)

    error_form = cut_form(inner(phi_reinit, phi_reinit ) * dsq_exact)
    err_val = cutfemx.fem.assemble_scalar(error_form) ** 0.5
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

    R_in = 0.15
    R_out = 0.35
    c_x = 0.5
    c_y = 0.5

    c1 = gmsh.model.occ.addDisk(c_x, c_y, 0, R_out, R_out)
    c2 = gmsh.model.occ.addDisk(c_x, c_y, 0, R_in, R_in)
    out_dim_tags = gmsh.model.occ.cut([(2, c1)], [(2, c2)])
    gmsh.model.occ.synchronize()

    curves = gmsh.model.getEntities(dim=1)
    inner_curves = []
    outer_curves = []

    for dim, tag in curves:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        r_com = np.sqrt((com[0] - c_x) ** 2 + (com[1] - c_y) ** 2)

        if np.isclose(r_com, R_in, atol=1e-1):
            inner_curves.append(tag)
        elif np.isclose(r_com, R_out, atol=1e-1):
            outer_curves.append(tag)

    gmsh.model.addPhysicalGroup(1, inner_curves, 1)
    gmsh.model.addPhysicalGroup(1, outer_curves, 2)

    surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)

    lc = 1.0 / N
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    gmsh.model.mesh.generate(2)

    msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()

    return msh, facet_markers

def create_square_mesh(N):
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [N, N],
        cell_type=mesh.CellType.triangle,
        ghost_mode=GhostMode.shared_facet,
    )
    
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    facet_markers = mesh.meshtags(msh, fdim, boundary_facets, np.full(len(boundary_facets), 2, dtype=np.int32))
    
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

    n = FacetNormal(msh)
    h = CellDiameter(msh)

    V_ls = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V_ls)
    level_set.interpolate(circle)

    ls_ex = fem.Function(V_ls)
    ls_ex.interpolate(circle_exact_solution)

    eps = 1e-6
    sign = level_set / (ufl.sqrt(level_set**2+eps**2))

    dim = msh.topology.dim
    intersected_entities = locate_entities(level_set, dim, "phi=0")
    inside_entities = locate_entities(level_set, dim, "phi<0")
    outside_entities = locate_entities(level_set, dim, "phi>0")

    V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
    n_K = fem.Function(V_DG)
    compute_normal(n_K, level_set, intersected_entities)

    dof_coordinates = V.tabulate_dof_coordinates()
    interface_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi=0")
    if len(interface_cells.cut_cells) == 0:
        return None, None, None, None, None

    order = 2
    inside_quadrature = runtime_quadrature(level_set, "phi<0", order)
    interface_quadrature = runtime_quadrature(level_set, "phi=0", order)
    outside_quadrature = runtime_quadrature(level_set, "phi>0", order)

    quad_domains = [(0, inside_quadrature), (1, interface_quadrature), (2, outside_quadrature)]
    gp_ids = ghost_penalty_facets(level_set, "phi<0")
    gp_topo = facet_topology(msh, gp_ids)

    dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities), (1, intersected_entities), (2, outside_entities)], domain=msh)
    dS = ufl.Measure("dS", subdomain_data=[(0, gp_topo)], domain=msh)
    dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)
    dsq = dx_rt(1)

    ds_inner = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers, subdomain_id=1)
    ds_outer = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers, subdomain_id=2)

    # -----------------------------------------------------
    # PREDICTOR STEP (Linear system based on sign approx)
    # -----------------------------------------------------
    solver = PETSc.KSP().create(msh.comm)
    opts = PETSc.Options()
    opts["ksp_rtol"] = 1.0e-12
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    solver.setFromOptions()

    uh_predictor = fem.Function(V)
    uh_predictor.name = "uh_predictor"

    a_pred = inner(grad(u), grad(v)) * dx
    a_pred += -dot(grad(u), n_K) * v * dsq
    a_pred += -dot(grad(v), n_K) * u * dsq
    a_pred += gamma * 1.0 / h * u * v * dsq
    a_pred += -dot(grad(u), n) * v * ds_outer - dot(grad(v), n) * u * ds_outer + gamma * 1.0 / h * u * v * ds_outer
    a_pred += -dot(grad(u), n) * v * ds_inner - dot(grad(v), n) * u * ds_inner + gamma * 1.0 / h * u * v * ds_inner

    L_pred = inner(sign, v) * dx
    L_pred += -dot(grad(v), n_K) * g * dsq + gamma * 1.0 / h * v * g * dsq
    L_pred += inner(sign, v) * ds
    L_pred += -dot(grad(v), n) * ls_ex * ds_outer + gamma * 1.0 / h * v * ls_ex * ds_outer
    L_pred += -dot(grad(v), n) * ls_ex * ds_inner + gamma * 1.0 / h * v * ls_ex * ds_inner

    A_predictor = assemble_matrix(cut_form(a_pred))
    A_predictor.assemble()
    
    b_predictor = assemble_vector(cut_form(L_pred))
    b_predictor.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_predictor.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    solver.setOperators(A_predictor)
    solver.solve(b_predictor, uh_predictor.x.petsc_vec)
    uh_predictor.x.scatter_forward()

    # -----------------------------------------------------
    # CORRECTOR STEP (Newton Energy Minimization)
    # -----------------------------------------------------
    u_n = level_set
    v_n = ufl.TestFunction(V)
    du_n = ufl.TrialFunction(V)
    
    u_frozen = fem.Function(V)
    
    eps_n = 1e-10
    norm_grad = ufl.sqrt(inner(grad(u_n), grad(u_n)) + eps_n)
    
    # Freeze the geometric normal
    norm_grad_frozen = ufl.sqrt(inner(grad(u_frozen), grad(u_frozen)) + eps_n)
    n_Gamma_frozen = grad(u_frozen) / norm_grad_frozen
    
    # 1. Bulk Eikonal Energy Derivative (F_bulk)
    E_bulk = 0.5 * (norm_grad - 1.0)**2 * dx
    F_bulk = ufl.derivative(E_bulk, u_n, v_n)
    
    # 2. Add Explicit Nitsche Terms to Residual F (Target g=0, ls_ex)
    F_int = (-dot(grad(u_n), n_Gamma_frozen) * v_n - dot(grad(v_n), n_Gamma_frozen) * u_n + gamma * 1.0 / h * u_n * v_n) * dsq
    F_out = (-dot(grad(u_n), n) * v_n - dot(grad(v_n), n) * (u_n - ls_ex) + gamma * 1.0 / h * (u_n - ls_ex) * v_n) * ds_outer
    F_in  = (-dot(grad(u_n), n) * v_n - dot(grad(v_n), n) * (u_n - ls_ex) + gamma * 1.0 / h * (u_n - ls_ex) * v_n) * ds_inner
    
    F_newton = F_bulk + F_int + F_out + F_in
    
    # Jacobian is precisely the formal derivative of the linear Residual F!
    J_newton = ufl.derivative(F_newton, u_n, du_n)
    
    F_cut_newton = cut_form(F_newton)
    J_cut_newton = cut_form(J_newton)

    uh_corrector = fem.Function(V)
    uh_corrector.name = "uh_corrector"
    uh_corrector.x.array[:] = uh_predictor.x.array[:]

    eikonal_norm_pred = error_L2_distance_func_form(uh_corrector, V)

    tol = 1e-8
    max_iter = 50
    it = 0
    diff_eik = 1.0
    alpha = 0.3  # Damping factor

    while diff_eik > tol and it < max_iter:
        it += 1

        # Synchronize level_set for F and J evaluation
        level_set.x.array[:] = uh_corrector.x.array[:]
        # Freeze normal for this Newton step
        u_frozen.x.array[:] = uh_corrector.x.array[:]

        b_newton = assemble_vector(F_cut_newton)
        b_newton.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_newton.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        b_newton.scale(-1.0) # We solve J * du = -F
        
        A_newton = assemble_matrix(J_cut_newton)
        A_newton.assemble()

        solver.setOperators(A_newton)
        du_vec = A_newton.createVecRight()
        
        solver.solve(b_newton, du_vec)
        
        # Damping update
        uh_corrector.x.petsc_vec.axpy(alpha, du_vec)
        uh_corrector.x.scatter_forward()
        
        eikonal_norm = error_L2_distance_func_form(uh_corrector, V)
        diff_eik = abs(eikonal_norm - eikonal_norm_pred)
        eikonal_norm_pred = eikonal_norm


    if it == max_iter:
        print(f"  Warning: Corrector did not converge in {max_iter} iterations.")

    # -----------------------------------------------------
    # Error computations
    # -----------------------------------------------------
    V3 = fem.functionspace(msh, ("Lagrange", 3))
    ls_ex_3 = fem.Function(V3)
    ls_ex_3.interpolate(circle_exact_solution)

    ls_3 = fem.Function(V3)
    ls_3.interpolate(uh_corrector)

    e_W = fem.Function(V3)
    e_W.x.array[:] = ls_3.x.array - ls_ex_3.x.array

    vol_form = fem.form(fem.Constant(msh, 1.0) * ufl.dx)
    volume = msh.comm.allreduce(assemble_scalar(vol_form), op=MPI.SUM)

    error_L2 = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    l2_norm = np.sqrt(msh.comm.allreduce(assemble_scalar(error_L2), op=MPI.SUM) / volume)

    error_H1 = fem.form(ufl.inner(grad(e_W), grad(e_W)) * ufl.dx)
    h1_norm = np.sqrt(msh.comm.allreduce(assemble_scalar(error_H1), op=MPI.SUM) / volume)

    interface_norm = interface_error(uh_corrector, ls_ex, msh, msh.topology.dim)
    eik_norm = error_L2_distance_func_form(uh_corrector, V)

    h_size = 1.0 / N
    return h_size, l2_norm, h1_norm, interface_norm, eik_norm


if __name__ == "__main__":
    geometry_type = "annulus"
    
    N_values = [40, 60, 80, 120, 160]
    output_file = "convergence_results_newton.txt"
    
    with open(output_file, "w") as f:
        header = f"{'N':<5} {'h':<10} {'L2 Error':<15} {'Rate':<10} {'H1 Error':<15} {'Rate':<10} {'Int. Error':<15} {'Rate':<10} {'Eik. Error':<15} {'Rate':<10}"
        print(header)
        f.write(header + "\n")
        
        separator = "-" * 130
        print(separator)
        f.write(separator + "\n")

        prev_h = prev_l2 = prev_h1 = prev_int = prev_eik = None
        h_values, l2_errors, h1_errors, int_errors, eik_errors = [], [], [], [], []

        for N in N_values:
            h, l2, h1, int_err, eik_err = solve_problem(N, geometry_type=geometry_type)
            if h is None:
                continue

            h_values.append(h); l2_errors.append(l2); h1_errors.append(h1)
            int_errors.append(int_err); eik_errors.append(eik_err)

            rate_l2 = rate_h1 = rate_int = rate_eik = 0.0

            if prev_h is not None:
                rate_l2 = np.log(l2 / prev_l2) / np.log(h / prev_h)
                rate_h1 = np.log(h1 / prev_h1) / np.log(h / prev_h)
                rate_int = np.log(int_err / prev_int) / np.log(h / prev_h)
                rate_eik = np.log(eik_err / prev_eik) / np.log(h / prev_h)

            row = f"{N:<5} {h:<10.4f} {l2:<15.4e} {rate_l2:<10.2f} {h1:<15.4e} {rate_h1:<10.2f} {int_err:<15.4e} {rate_int:<10.2f} {eik_err:<15.4e} {rate_eik:<10.2f}"
            print(row)
            f.write(row + "\n")

            prev_h, prev_l2, prev_h1 = h, l2, h1
            prev_int, prev_eik = int_err, eik_err

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.plot(h_values, l2_errors, 'k--o', label=r'$\left\Vert u - u_{ex} \right\Vert_{L^{2}}$')
    plt.plot(h_values, h1_errors, 'k--s', label=r'$\left\Vert \nabla(u - u_{ex}) \right\Vert_{L^{2}}$')
    plt.plot(h_values, int_errors, 'k--^', label=r'$\left\Vert u - u_{ex} \right\Vert_{L^{2}(\Gamma)}$')
    plt.plot(h_values, eik_errors, 'k--d', label=r'$\left\Vert |\nabla u| - 1 \right\Vert_{L^{2}}$')

    h_arr = np.array(h_values)
    plt.plot(h_arr, h_arr**2 * (l2_errors[0] / h_arr[0] ** 2), "k-", label="Order 2")
    plt.plot(h_arr, h_arr * (h1_errors[0] / h_arr[0]), "k-.", label="Order 1")

    plt.yscale('log'); plt.xscale('log')
    plt.xlabel('h', fontsize=14); plt.ylabel('Error', fontsize=14)
    plt.title(f"Newton Convergence Study: Circle Redistance ({geometry_type})", fontsize=16)
    plt.legend(fontsize=14, loc='lower right')
    plt.grid(True, which="both", ls="-")
    plt.savefig("convergence_newton_plot.pdf")
    print("Plot saved to convergence_newton_plot.pdf")
