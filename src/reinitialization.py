import numpy as np
import cutfemx
import ufl
from dolfinx import fem
import dolfinx.fem.petsc
from ufl import ds, dx, grad, inner, dot, CellDiameter
from mpi4py import MPI
from petsc4py import PETSc

from cutfemx.level_set import locate_entities
from cutfemx.quadrature import runtime_quadrature
from cutfemx.fem import cut_form
from cutfemx.petsc import assemble_vector, assemble_matrix

class Reinitialization:
    def __init__(self, level_set, V_ls, l, solver_type="mumps", projection=False):
        self.level_set = level_set
        self.mesh = level_set.function_space.mesh
        self.l = l
        self.V_ls = V_ls
        self.projection = projection

        # --- Espace pour la projection L2 ---
        self.V_vec = fem.functionspace(self.mesh, ("CG", 1, (self.mesh.geometry.dim,)))
        self.grad_projected = fem.Function(self.V_vec)

        u_v = ufl.TrialFunction(self.V_vec)
        v_v = ufl.TestFunction(self.V_vec)
        a_proj = inner(u_v, v_v) * dx
        L_proj = inner(grad(self.level_set), v_v) * dx

        self.a_proj = fem.form(a_proj)
        self.L_proj = fem.form(L_proj)
        self.A_proj = dolfinx.fem.petsc.create_matrix(self.a_proj)
        dolfinx.fem.petsc.assemble_matrix(self.A_proj, self.a_proj)
        self.A_proj.assemble()

        self.solver_proj = PETSc.KSP().create(self.mesh.comm)
        self.solver_proj.setType(PETSc.KSP.Type.CG) 
        self.solver_proj.getPC().setType(PETSc.PC.Type.JACOBI)
        self.solver_proj.setOperators(self.A_proj)
        self.b_proj = dolfinx.fem.petsc.create_vector(self.L_proj)

        self.solver_type = solver_type
        
        self.dim = self.mesh.topology.dim
        self.tdim = self.dim
        self.order = 2
        self.gamma_r = 1e+3
        self.h_r = CellDiameter(self.mesh)
        self.eps = 1e-6
        
        self.solver_predict = PETSc.KSP().create(self.mesh.comm)
        self.solver_corrector = PETSc.KSP().create(self.mesh.comm)
        self._setup_solvers()
        
        self.u_frozen = fem.Function(self.V_ls)
        self._setup_forms()

    def _setup_solvers(self):
        for solver in [self.solver_predict, self.solver_corrector]:
            if self.solver_type.lower() in ["mumps", "umfpack", "superlu_dist", "petsc"]:
                solver.setType(PETSc.KSP.Type.PREONLY)
                solver.getPC().setType(PETSc.PC.Type.LU)
                if self.solver_type.lower() != "petsc":
                    solver.getPC().setFactorSolverType(self.solver_type.lower())
            else:
                solver.setType(self.solver_type.lower())
                solver.getPC().setType(PETSc.PC.Type.ILU)

    def _setup_forms(self):
        self.intersected_entities = locate_entities(self.level_set, self.dim, "phi=0")
        self.interface_quadrature = runtime_quadrature(self.level_set, "phi=0", self.order)
        self.quad_domains = [(1, self.interface_quadrature)]
        
        self.dx_global = ufl.Measure("dx", domain=self.mesh)
        self.dx_rt = ufl.Measure("dC", subdomain_data=self.quad_domains, domain=self.mesh)
        self.dsq = self.dx_rt(1)
        
        # --- L'APPROCHE HYBRIDE GAGNANTE ---
        # 1. Normale LISSÉE (Projected) pour le membre de droite (filtrage du bruit)
        self.euclidean_norm = ufl.sqrt(inner(self.grad_projected, self.grad_projected) + 1e-12)
        self.n_K = self.grad_projected / self.euclidean_norm
        
        # 2. Normale EXACTE P0 (grad de u_frozen) pour la matrice (stabilité de Nitsche)
        norm_grad_frozen = ufl.sqrt(inner(grad(self.u_frozen), grad(self.u_frozen)) + 1e-12)
        self.n_Gamma = grad(self.u_frozen) / norm_grad_frozen
        
        u_r = ufl.TrialFunction(self.V_ls)
        v_r = ufl.TestFunction(self.V_ls)
        
        a_predict = inner(grad(u_r), grad(v_r)) * self.dx_global
        a_predict += - dot(grad(u_r), self.n_K) * v_r * self.dsq
        a_predict += - dot(grad(v_r), self.n_K) * u_r * self.dsq
        a_predict += self.gamma_r * 1.0 / self.h_r * u_r * v_r * self.dsq
        
        self.sign = self.level_set / (ufl.sqrt(self.level_set**2 + self.eps**2))

        Q = fem.functionspace(self.mesh, ("DG", 0))
        temp = ufl.conditional(ufl.le(self.sign,0),-1,0)
        temp = ufl.conditional(ufl.ge(self.sign,0),1,temp)

        self.temp_expr = fem.Expression(temp, Q.element.interpolation_points())
        self.temp = fem.Function(Q)
        self.temp.interpolate(self.temp_expr)
        
        L_predict = inner(self.l**2 * self.sign, v_r) * self.dx_global
        L_predict += self.temp * v_r * ds
        
        jit_opts = {"cache_dir": "ffcx-forms", "cffi_extra_compile_args": ["-O3", "-ffast-math", "-march=native"]}
        self.a_cut_predict = cut_form(a_predict, jit_options=jit_opts)
        self.L_cut_predict = cut_form(L_predict, jit_options=jit_opts)
        
        # MATRICE : utilise la normale EXACTE P0 (n_Gamma) -> Coercivité de Nitsche
        a_correct = inner(grad(u_r), grad(v_r)) * self.dx_global
        a_correct += - dot(grad(u_r), self.n_Gamma) * v_r * self.dsq
        a_correct += - dot(grad(v_r), self.n_Gamma) * u_r * self.dsq
        a_correct += self.gamma_r * 1.0 / self.h_r * u_r * v_r * self.dsq
        
        # VECTEUR : utilise la normale LISSÉE (n_K) -> Cible géométrique sans oscillations
        L_correct = inner(self.n_K, grad(v_r)) * self.dx_global
        L_correct += - inner(self.n_K, self.n_Gamma) * v_r * self.dsq
        
        self.a_cut_correct = cut_form(a_correct, jit_options=jit_opts)
        self.L_cut_correct = cut_form(L_correct, jit_options=jit_opts)

    def update_projected_gradient(self):
        with self.b_proj.localForm() as loc_b:
            loc_b.set(0)
        dolfinx.fem.petsc.assemble_vector(self.b_proj, self.L_proj)
        self.b_proj.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.solver_proj.solve(self.b_proj, self.grad_projected.x.petsc_vec)
        self.grad_projected.x.scatter_forward()
    
    def _update_geometry(self):
        self.intersected_entities = locate_entities(self.level_set, self.dim, "phi=0")
        self.interface_quadrature = runtime_quadrature(self.level_set, "phi=0", self.order)
        new_quad_domains = {"cutcell": [(1, self.interface_quadrature)]}
        self.a_cut_predict.update_runtime_domains(new_quad_domains)
        self.a_cut_correct.update_runtime_domains(new_quad_domains)
        self.L_cut_correct.update_runtime_domains(new_quad_domains)

    def predictor(self, level_set):
        self.level_set.x.array[:] = level_set.x.array[:]
        self.level_set.x.scatter_forward()
        
        self.update_projected_gradient()
        
        self.temp.interpolate(self.temp_expr)
        self.temp.x.scatter_forward()
        
        self._update_geometry()
        
        A = assemble_matrix(self.a_cut_predict)
        A.assemble()
        b = assemble_vector(self.L_cut_predict)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        self.solver_predict.setOperators(A)
        self.solver_predict.solve(b, self.level_set.x.petsc_vec)
        self.level_set.x.scatter_forward() 
        
        return self.level_set

    def corrector(self, level_set, is_first_iteration=True):
        self.level_set.x.array[:] = level_set.x.array[:]
        self.level_set.x.scatter_forward()
        
        # Mise à jour de n_K pour le membre de droite
        self.update_projected_gradient()
        
        if is_first_iteration:
            self.u_frozen.x.array[:] = level_set.x.array[:]
            self.u_frozen.x.scatter_forward()
            
            A = assemble_matrix(self.a_cut_correct)
            A.assemble()
            self.solver_corrector.setOperators(A)
        
        b = assemble_vector(self.L_cut_correct)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        self.solver_corrector.solve(b, self.level_set.x.petsc_vec)
        self.level_set.x.scatter_forward() 
        
        return self.level_set