# B25 ssa metamorphism

This repository contains notebooks and helper code for studying snow specific surface area (SSA) evolution and evaluating the B21 and B25 model formulations against experimental data.

## Notebooks

- [`2.2.grf_microstructure.ipynb`](./2.2.grf_microstructure.ipynb) generates example binary microstructures with `grfsaw` and estimates their SSA from spatial autocorrelation.
- [`3.1.theory_curve.ipynb`](./3.1.theory_curve.ipynb) compares theoretical SSA-evolution curves and explores the sensitivity of the B25 formulation to its parameters.
- [`3.1.optimize_cold_exp.ipynb`](./3.1.optimize_cold_exp.ipynb) prepares cold-room experiments and calibrates model parameters for isothermal and temperature-gradient conditions.
- [`3.1.evaluate_cold_exp_boxplot.ipynb`](./3.1.evaluate_cold_exp_boxplot.ipynb) evaluates model trajectories against the experiments and summarizes errors with metrics and box plots.

## Data and setup

The experiment notebooks read `ssa_bibio.xlsx` and `SSA2014-2018_Hachikubo.xls` from their working directory. Keep these files alongside the notebooks or update the paths in the data-loading cells. The microstructure notebook requires the `grfsaw` package. Model calculations are shared through `function_SSA.py`.

Run notebook cells from top to bottom to create the intermediate experiment data and model results needed by later cells. Some plotting cells save figures to machine-specific paths; change those paths to a writable location before running them elsewhere.
