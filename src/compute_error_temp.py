import numpy as np
import ufl
from dolfinx import fem
from cutfemx.level_set import locate_entities
from mpi4py import MPI


def compute_shape_error(level_set, sol_exact, mesh):
    """
    Calcule l'erreur géométrique de forme (Symmetric Difference Area).
    E_shape = \int | H(phi_curr) - H(phi_exact) | dx
    
    Cette erreur représente l'aire totale des zones qui ne se superposent pas
    entre la solution courante et la solution exacte.
    """
    # Mesure standard avec un degré de quadrature élevé pour les cellules coupées
    dx_high = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})
    
    # 1. Fonctions indicatrices (Heaviside = 1 à l'intérieur, 0 à l'extérieur)
    H_curr = ufl.conditional(ufl.lt(level_set, 0), 1.0, 0.0)
    H_exact = ufl.conditional(ufl.lt(sol_exact, 0), 1.0, 0.0)
    
    # 2. Intégrant de l'erreur (différence symétrique)
    # Comme H vaut 0 ou 1, (H_curr - H_exact)**2 vaut 1 là où les formes diffèrent, et 0 ailleurs.
    error_integrand = (H_curr - H_exact)**2
    
    # 3. Assemblage et réduction MPI
    error_form = fem.form(error_integrand * dx_high)
    shape_error = mesh.comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM)
    
    # Optionnel : Calcul de l'aire exacte pour renvoyer aussi l'erreur relative
    mass_exact_form = fem.form(H_exact * dx_high)
    mass_sol_exact = mesh.comm.allreduce(fem.assemble_scalar(mass_exact_form), op=MPI.SUM)
    
    # On évite la division par zéro si le domaine exact est vide
    relative_shape_error = shape_error / mass_sol_exact if mass_sol_exact > 1e-14 else 0.0
    
    return shape_error, relative_shape_error

def compute_fraction_mass(level_set, sol_exact, mesh, V_ls):
    """
    Computes the fraction of mass conservation.
    Utilise une intégration conditionnelle sur tout le domaine pour inclure précisément
    les cellules coupées (cut-cells), avec une quadrature d'ordre supérieur.
    """
    # Mesure standard avec un degré de quadrature élevé pour bien capturer l'interface
    # dans les cellules coupées.
    dx_high = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 6})
    
    # 1. Masse de la solution courante (1.0 là où phi < 0, sinon 0.0)
    H_curr = ufl.conditional(ufl.lt(level_set, 0), 1.0, 0.0)
    mass_form = fem.form(H_curr * dx_high)
    mass_func = mesh.comm.allreduce(fem.assemble_scalar(mass_form), op=MPI.SUM)
    
    # 2. Masse de la solution exacte (initiale)
    H_exact = ufl.conditional(ufl.lt(sol_exact, 0), 1.0, 0.0)
    mass_exact_form = fem.form(H_exact * dx_high)
    mass_sol_exact = mesh.comm.allreduce(fem.assemble_scalar(mass_exact_form), op=MPI.SUM)

    if abs(mass_sol_exact) < 1e-14:
        return 1.0 # Évite division par zéro au début
        
    return mass_func / mass_sol_exact

def error_L2_distance_func_form(phi, V_ls, xsi=1):
    """
    Calcul direct de l'erreur d'Eikonal (norme L2 du résidu).
    """
    euclidean_norm = ufl.sqrt(ufl.inner(ufl.grad(phi), ufl.grad(phi)))
    # On utilise une mesure standard pour éviter tout conflit
    error_form = fem.form(((xsi * (euclidean_norm - 1))**2) * ufl.dx)
    error_val = fem.assemble_scalar(error_form)
    return np.sqrt(MPI.COMM_WORLD.allreduce(error_val, op=MPI.SUM))
