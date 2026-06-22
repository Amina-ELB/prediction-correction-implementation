import ufl
from dolfinx import fem
import dolfinx.fem.petsc
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI

# class AdvectionSolver:
#     def __init__(self, V, u_velocity, dt):
#         self.V = V
#         self.u_velocity = u_velocity
#         self.dt = dt
#         self.mesh = V.mesh
        
#         phi = ufl.TrialFunction(V)
#         v = ufl.TestFunction(V)
#         self.phi_old = fem.Function(V)
        
#         # 1. Schéma de Crank-Nicolson (Theta = 0.5)
#         theta = 0.5
#         phi_mid = theta * phi + (1.0 - theta) * self.phi_old
        
#         h = ufl.CellDiameter(self.mesh)
#         vel_norm = ufl.sqrt(ufl.dot(self.u_velocity, self.u_velocity))
#         v_norm_safe = ufl.max_value(vel_norm, 1e-12)
        
#         # 2. Paramètre SUPG transitoire (Shakib)
#         tau = 1.0 / ufl.sqrt((2.0 / self.dt)**2 + (2.0 * v_norm_safe / h)**2)
        
#         # 3. Résidu (évalué à t + dt/2)
#         res = (phi - self.phi_old) / self.dt + ufl.dot(self.u_velocity, ufl.grad(phi_mid))
        
#         # Weak form with spatial-only SUPG stabilization (more stable under Crank-Nicolson)
#         self.F = res * v * ufl.dx + tau * ufl.dot(self.u_velocity, ufl.grad(v)) * ufl.dot(self.u_velocity, ufl.grad(phi_mid)) * ufl.dx
        
#         self.a = fem.form(ufl.lhs(self.F))
#         self.L = fem.form(ufl.rhs(self.F))
        
#         # Pre-allocate matrix and vector
#         self.A = dolfinx.fem.petsc.create_matrix(self.a)
#         self.b = dolfinx.fem.petsc.create_vector(self.L)
        
#         self.solver = PETSc.KSP().create(self.mesh.comm)
#         self.solver.setType(PETSc.KSP.Type.PREONLY)
#         self.solver.getPC().setType(PETSc.PC.Type.LU)

#     def solve(self, phi_n):
#         self.phi_old.x.array[:] = phi_n.x.array[:]
#         self.phi_old.x.scatter_forward()
        
#         # Re-assemble A and b (needed if u_velocity is time-dependent)
#         self.A.zeroEntries()
#         dolfinx.fem.petsc.assemble_matrix(self.A, self.a)
#         self.A.assemble()
        
#         with self.b.localForm() as loc_b:
#             loc_b.set(0)
#         dolfinx.fem.petsc.assemble_vector(self.b, self.L)
#         self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
#         self.solver.setOperators(self.A)
        
#         phi_next = fem.Function(self.V)
#         self.solver.solve(self.b, phi_next.x.petsc_vec)
#         phi_next.x.scatter_forward()
        
#         # Debugging
#         # phi_old_l2 = fem.assemble_scalar(fem.form(ufl.inner(self.phi_old, self.phi_old) * ufl.dx))
#         # phi_next_l2 = fem.assemble_scalar(fem.form(ufl.inner(phi_next, phi_next) * ufl.dx))
#         # if self.mesh.comm.rank == 0:
#         #     print(f"DEBUG Advection: ||phi_old||_L2={np.sqrt(self.mesh.comm.allreduce(phi_old_l2, MPI.SUM)):.6e}, ||phi_next||_L2={np.sqrt(self.mesh.comm.allreduce(phi_next_l2, MPI.SUM)):.6e}")
        
#         return phi_next










import ufl
from dolfinx import fem
import dolfinx.fem.petsc
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI

class AdvectionSolver:
    def __init__(self, V, u_velocity, dt, c_crosswind=0.01):
        self.V = V
        self.u_velocity = u_velocity
        self.dt = dt
        self.mesh = V.mesh
        self.step_count = 0
        
        phi = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Mémoire pour BDF2 (besoin de n et n-1)
        self.phi_old = fem.Function(V)   # phi^n
        self.phi_old2 = fem.Function(V)  # phi^{n-1}
        
        h = ufl.CellDiameter(self.mesh)
        vel_norm = ufl.sqrt(ufl.dot(self.u_velocity, self.u_velocity))
        v_norm_safe = ufl.max_value(vel_norm, 1e-12)
        
        # 1. Paramètre SUPG transitoire (Shakib) - Adapté pour l'implicite
        tau = 1.0 / ufl.sqrt((2.0 / self.dt)**2 + (2.0 * v_norm_safe / h)**2)
        
        # 2. Fonction test stabilisée (SUPG consistant)
        v_supg = v + tau * ufl.dot(self.u_velocity, ufl.grad(v))
        
        # 3. Terme optionnel : Légère diffusion croisée (Crosswind diffusion)
        # c_crosswind contrôle l'intensité. Mettre 0.0 pour du SUPG pur.
        nu_art = c_crosswind * h * vel_norm
        crosswind = nu_art * ufl.inner(ufl.grad(phi), ufl.grad(v)) * ufl.dx
        
        # --- FORMULATION BDF1 (Pour le tout premier pas de temps) ---
        res_bdf1 = (phi - self.phi_old) / self.dt + ufl.dot(self.u_velocity, ufl.grad(phi))
        F_bdf1 = res_bdf1 * v_supg * ufl.dx + crosswind
        self.a_bdf1 = fem.form(ufl.lhs(F_bdf1))
        self.L_bdf1 = fem.form(ufl.rhs(F_bdf1))
        
        # --- FORMULATION BDF2 (Pour tous les pas de temps suivants) ---
        res_bdf2 = (3.0 * phi - 4.0 * self.phi_old + self.phi_old2) / (2.0 * self.dt) + ufl.dot(self.u_velocity, ufl.grad(phi))
        F_bdf2 = res_bdf2 * v_supg * ufl.dx + crosswind
        self.a_bdf2 = fem.form(ufl.lhs(F_bdf2))
        self.L_bdf2 = fem.form(ufl.rhs(F_bdf2))
        
        # Pré-allocation (on ne crée qu'une seule matrice A et vecteur b qu'on mettra à jour)
        self.A = dolfinx.fem.petsc.create_matrix(self.a_bdf1)
        self.b = dolfinx.fem.petsc.create_vector(self.L_bdf1)
        
        self.solver = PETSc.KSP().create(self.mesh.comm)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

    def solve(self, phi_n):
        self.step_count += 1
        
        # Mise à jour de l'historique des temps
        self.phi_old2.x.array[:] = self.phi_old.x.array[:]
        self.phi_old.x.array[:] = phi_n.x.array[:]
        self.phi_old2.x.scatter_forward()
        self.phi_old.x.scatter_forward()
        
        # Choix de la forme faible (Euler implicite au démarrage, puis BDF2)
        if self.step_count == 1:
            a_form, L_form = self.a_bdf1, self.L_bdf1
        else:
            a_form, L_form = self.a_bdf2, self.L_bdf2
        
        # Assemblage
        self.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.A, a_form)
        self.A.assemble()
        
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        dolfinx.fem.petsc.assemble_vector(self.b, L_form)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        self.solver.setOperators(self.A)
        
        phi_next = fem.Function(self.V)
        self.solver.solve(self.b, phi_next.x.petsc_vec)
        phi_next.x.scatter_forward()
        
        return phi_next