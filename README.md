# Level-Set Reinitialization Project

This repository contains an implementation of reinitialization methods for the Level-Set method, using FEniCSx and CutFEMx.

## Project Structure

- `main.py`: Main entry point for the reinitialization process.
- `data.py`: Configuration file for simulation parameters.
- `src/`: Contains the main source code (library).
    - `reinitialization.py`: Main class for reinitialization.
    - `reinitialization_mpi.py`: MPI-compatible version.
    - `mesh_generation.py`: Mesh generation utilities.
    - `geometry_ls.py`: Geometry definitions and noise application.
    - `compute_error.py`: Error computation functions.
- `scripts/`: Execution and demonstration scripts.
    - `demo_convergence_2d.py`: Convergence study on a 2D case (circle).
    - `demo_3d.py`: 3D demonstration (torus/sphere).
    - `demo_mpi.py`: Demonstration using MPI.
- `requirements.txt`: List of Python dependencies.

## Quick Start

To run the standard reinitialization process:

1.  Configure your simulation parameters in `data.py` (mesh size, level set function, noise, etc.).
2.  Run the main script:

```bash
python3 main.py
```

Results will be saved in the `res/` directory.

## Installation

It is recommended to use a Conda environment with FEniCSx installed.

```bash
# Example installation with Conda
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista matplotlib
pip install -r requirements.txt

# Install cutfemx
# Please refer to the official repository for installation instructions:
# https://github.com/sclaus2/CutFEMx
```

Make sure the project root folder is in your `PYTHONPATH` or run the scripts from the root as follows.

## Usage

To run the demonstrations, execute the scripts from the project root:

### 2D Convergence
```bash
python3 scripts/demo_convergence_2d.py
```

### 3D Demonstration
```bash
python3 scripts/demo_3d.py
```

### MPI Demonstration
```bash
mpirun -n 4 python3 scripts/demo_mpi.py
```

## Authors
Amina El Bachari
2025 ONERA and MINES Paris - PSL

## Citation

If you use this code in your research, please cite the following paper:

> **A predictor-corrector scheme for approximating signed distances using finite element methods**
> Amina El Bachari, Susanne Claus, Johann Rannou, Vladislav Yastrebov, Pierre Kerfriden
> *Preprint*, 2025
> [Read the paper](https://www.researchgate.net/publication/392941004_A_predictor-corrector_scheme_for_approximating_signed_distances_using_finite_element_methods)
