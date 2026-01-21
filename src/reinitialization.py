
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
from ufl import ds, dx, grad, inner, dot, FacetNormal, CellDiameter
from dolfinx.io import XDMFFile

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature
from cutfemx.fem import cut_form, cut_function
from cutfemx.petsc import assemble_vector, assemble_matrix

class Reinitialization:
    """
    Class for HJ Reinitialization using a predictor-corrector scheme.
    """
    def __init__(self, level_set, V_ls, l=3):
        """
        Initialize the Reinitialization solver.

        Args:
            level_set (dolfinx.fem.Function): The level set function to reinitialize.
            V_ls (dolfinx.fem.FunctionSpace): The function space of the level set.
            l (float): A parameter for the stabilization (default: 3).
        """
        self.level_set = level_set
        self.mesh = level_set.function_space.mesh
        self.l = l
        self.V_ls = V_ls
        
        self.dim = self.mesh.topology.dim
        
        # Locate entities
        self.intersected_entities = locate_entities(self.level_set, self.dim, "phi=0")
        self.inside_entities = locate_entities(self.level_set, self.dim, "phi<0")

        # Compute normal vector n_K
        # Note: This is an approximation based on the gradient of the level set
        self.euclidean_norm = ufl.sqrt(inner(ufl.grad(self.level_set), ufl.grad(self.level_set)))
        # Avoid division by zero
        self.n_K = ufl.grad(self.level_set) / ufl.conditional(ufl.lt(self.euclidean_norm, 1e-10), 1.0, self.euclidean_norm)
        
        self.dof_coordinates = self.V_ls.tabulate_dof_coordinates()
        
        self.order = 1
        self.inside_quadrature = runtime_quadrature(self.level_set, "phi<0", self.order)
        self.interface_quadrature = runtime_quadrature(self.level_set, "phi=0", self.order)

        self.quad_domains = [(0, self.inside_quadrature), (1, self.interface_quadrature)]

        self.dx = ufl.Measure("dx", subdomain_data=[(0, self.inside_entities), (2, self.intersected_entities)], domain=self.mesh)
        self.dx_rt = ufl.Measure("dC", subdomain_data=self.quad_domains, domain=self.mesh)
        
        self.dsq = self.dx_rt(1)
        
        u_r = ufl.TrialFunction(self.V_ls)
        v_r = ufl.TestFunction(self.V_ls)
        self.gamma_r = 1e+2

        self.n_r = FacetNormal(self.mesh)
        self.h_r = CellDiameter(self.mesh)

        # Predictor form
        self.a_predict  = ufl.inner(grad(u_r), grad(v_r)) * self.dx
        self.a_predict += - dot(grad(u_r), self.n_K) * v_r * self.dsq
        self.a_predict += - dot(grad(v_r), self.n_K) * u_r * self.dsq
        self.a_predict += self.gamma_r * 1.0 / self.h_r * u_r * v_r * self.dsq

        self.eps = 1e-6
        self.sign = self.level_set / (ufl.sqrt(self.level_set**2 + self.eps**2))
        self.L_predict = inner(self.l**2 * self.sign, v_r) * self.dx
        
        # Handling potential issues with integration on \D
        # We apply a correction using a DG0 function
        Q = fem.functionspace(self.mesh, ("DG", 0))
        temp = ufl.conditional(ufl.le(self.sign, 0), -1, 0)
        temp = ufl.conditional(ufl.ge(self.sign, 0), 1, temp)

        temp_expr = fem.Expression(temp, Q.element.interpolation_points())
        temp_func = fem.Function(Q)
        temp_func.interpolate(temp_expr)
        
        self.L_predict += temp_func * v_r * ds

        self.a_cut_predict = cut_form(self.a_predict, jit_options={"cache_dir": "ffcx-forms"})
        self.L_cut_predict = cut_form(self.L_predict)
        
        self.A_predict = assemble_matrix(self.a_cut_predict)
        self.A_predict.assemble()

        self.b_predict = assemble_vector(self.L_cut_predict)
        self.b_predict.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_predict.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        self.solver_predict = PETSc.KSP().create(self.mesh.comm)
        self.solver_predict.setOperators(self.A_predict)
        self.solver_predict.setType(PETSc.KSP.Type.PREONLY)
        self.solver_predict.getPC().setType(PETSc.PC.Type.LU)
        self.solver_predict.getPC().setFactorSolverType("mumps")

        # Corrector form
        self.a_correct  = ufl.inner(grad(u_r), grad(v_r)) * self.dx
        self.a_correct += - dot(grad(u_r), self.n_K) * v_r * self.dsq
        self.a_correct += - dot(grad(v_r), self.n_K) * u_r * self.dsq
        self.a_correct += self.gamma_r * 1.0 / self.h_r * u_r * v_r * self.dsq

        self.L_correct = inner(self.n_K, grad(v_r)) * self.dx

        self.a_cut_correct = cut_form(self.a_correct, jit_options={"cache_dir": "ffcx-forms"})
        self.L_cut_correct = cut_form(self.L_correct, jit_options={"cache_dir": "ffcx-forms"})

        self.A_correct = assemble_matrix(self.a_cut_correct)
        self.A_correct.assemble()

        self.b_correct = assemble_vector(self.L_cut_correct)
        self.b_correct.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_correct.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.solver_corrector = PETSc.KSP().create(self.mesh.comm)
        self.solver_corrector.setOperators(self.A_correct)
        self.solver_corrector.setType(PETSc.KSP.Type.PREONLY)
        self.solver_corrector.getPC().setType(PETSc.PC.Type.LU)
        self.solver_corrector.getPC().setFactorSolverType("mumps")

    def predictor(self, level_set):
        """
        Returns the solution of the prediction problem, denoted phi_p.

        Args:
            level_set (dolfinx.fem.Function): The level_set function phi.

        Returns:
            dolfinx.fem.Function: The solution to the prediction problem.
        """
        self.level_set.x.array[:] = level_set.x.array
        self.euclidean_norm = ufl.sqrt(inner(ufl.grad(self.level_set), ufl.grad(self.level_set)))
        self.n_K = ufl.grad(self.level_set) / ufl.conditional(ufl.lt(self.euclidean_norm, 1e-10), 1.0, self.euclidean_norm)

        self.intersected_entities = locate_entities(self.level_set, self.dim, "phi=0")
        self.inside_entities = locate_entities(self.level_set, self.dim, "phi<0")
        self.inside_quadrature = runtime_quadrature(self.level_set, "phi<0", self.order)
        self.interface_quadrature = runtime_quadrature(self.level_set, "phi=0", self.order)

        self.quad_domains = {"cutcell": [(1, self.interface_quadrature)]}

        self.a_cut_predict.update_runtime_domains(self.quad_domains)
        
        self.b_predict = assemble_vector(self.L_cut_predict)
        self.b_predict.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_predict.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        self.A_predict = assemble_matrix(self.a_cut_predict)
        self.A_predict.assemble()

        self.solver_predict.setOperators(self.A_predict)
        self.solver_predict.solve(self.b_predict, self.level_set.x.petsc_vec)
        self.level_set.x.scatter_forward() 
        
        return self.level_set

    def corrector(self, level_set):
        """
        Returns the solution of the correction problem, denoted phi_i.

        Args:
            level_set (dolfinx.fem.Function): The level_set function phi.

        Returns:
            dolfinx.fem.Function: The solution to the correction problem.
        """
        self.level_set.x.array[:] = level_set.x.array
        self.euclidean_norm = ufl.sqrt(inner(ufl.grad(self.level_set), ufl.grad(self.level_set)))
        self.n_K = ufl.grad(self.level_set) / ufl.conditional(ufl.lt(self.euclidean_norm, 1e-10), 1.0, self.euclidean_norm)
        
        self.intersected_entities = locate_entities(self.level_set, self.dim, "phi=0")
        self.inside_entities = locate_entities(self.level_set, self.dim, "phi<0")
        
        # Note: cut_cells and cut_mesh seem to be used for debugging/visualization here
        # I'm keeping the XDMF output but commenting it out by default to avoid clutter
        # cut_cells = cut_entities(self.level_set, self.dof_coordinates, self.intersected_entities, 2, "phi<0")
        # cut_mesh = create_cut_mesh(self.mesh.comm, cut_cells, self.mesh, self.inside_entities)
        # with XDMFFile(self.mesh.comm, "res/uh_cut.xdmf", "w") as file:
        #     file.write_mesh(cut_mesh._mesh)

        self.b_correct = assemble_vector(self.L_cut_correct)
        self.b_correct.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_correct.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        self.A_correct = assemble_matrix(self.a_cut_correct)
        self.A_correct.assemble()

        self.solver_corrector.setOperators(self.A_correct)
        self.solver_corrector.solve(self.b_correct, self.level_set.x.petsc_vec)
        self.level_set.x.scatter_forward() 
        
        return self.level_set
        
    def interface_error(self, phi, f, xsi=1):
        """
        Compute the L2 error on the interface.
        """
        order = 1
        # intersected_entities = locate_entities(f, self.dim, "phi=0") # Unused
        # inside_entities = locate_entities(f, self.dim, "phi<0") # Unused

        inside_quadrature = runtime_quadrature(f, "phi<0", order)
        interface_quadrature = runtime_quadrature(f, "phi=0", order)

        quad_domains = [(0, inside_quadrature), (1, interface_quadrature)]

        dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=self.mesh)

        dsq_exact = dx_rt(1)

        error_L = cut_form(inner(phi, phi) * dsq_exact)

        error_Linterface = fem.assemble_scalar(error_L)**0.5
        # print("error interface =", error_Linterface)
        return error_Linterface
