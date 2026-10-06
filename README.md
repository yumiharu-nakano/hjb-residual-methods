# Residual-based kernel and neural-network methods for HJB equations

Reference implementation for the numerical experiments in the manuscript

> *Convergence of kernel and neural-network methods for Hamilton--Jacobi--Bellman equations on unbounded domains*, Yumiharu Nakano.

The repository compares two residual-based realizations for a manufactured
Hamilton--Jacobi--Bellman (HJB) equation on an expanding space-time cylinder:

- kernel collocation with compactly supported Wendland radial basis functions;
- a tanh physics-informed neural network (PINN) trained with empirical PDE and
  terminal residuals.

The code is intended to reproduce the paper's numerical illustrations. It is
not a general-purpose HJB solver.

## Repository layout

```text
kernel_experiments/       Kernel collocation code, saved results, and plots
pinn_experiments/         PINN code, saved results, plots, and checkpoints
run_all.sh                Full experiment driver used for the paper
requirements.txt          Python dependencies
REPRODUCIBILITY.md        Scope and interpretation of the experiments
```

## Test problem

For spatial dimension `d`, the code considers

```text
v_t + inf_(sigma,beta) {
    0.5 sigma^2 Delta v + (beta/d) sum_i v_{x_i} + f
} = 0,

(sigma,beta) in [0.1,0.4] x [-1,1],
```

with a manufactured source and exact solution

```text
v(t,x) = sin(t + sum_i x_i) exp(-|x|^2/4).
```

The source term and exact derivatives are defined in
`kernel_experiments/hjb_problem.py`. The PINN experiment specializes the same
problem to `d=1`.

## Installation

Python 3.9 or later is recommended. Create an isolated environment and install
the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The reported PINN checkpoints were generated with Python 3.9.2 and PyTorch
2.2.2 on CPU. The kernel scripts also run with newer NumPy and SciPy releases.

## Quick checks

Run one small kernel case:

```bash
cd kernel_experiments
python run_grid.py --d 1 --mode theory --limit 1 --max_iter 100 --out smoke_kernel.csv
```

Run one short PINN training job:

```bash
cd pinn_experiments
python pinn.py --width 16 --M 256 --n_iter_adam 10 --n_iter_lbfgs 0 --device cpu
```

These commands check that the programs execute; they do not reproduce the
reported accuracy.

## Reproducing the reported outputs

The saved CSV files, figures, and PINN checkpoints are included. The
postprocessing scripts regenerate the LaTeX table fragments used by the paper.
To rerun all experiments and regenerate the derived outputs, run

```bash
./run_all.sh
```

The complete run includes optimization jobs and may take hours. Individual
commands are documented in the two experiment directories.

## Important interpretation

The kernel computations penalize violations of the native-space radius and
report the final native-space norm; all saved candidates satisfy the prescribed
radius when checked after optimization. The PINN computations do **not** impose
the hard parameter and Sobolev constraints assumed by the corresponding
convergence theorem. Their layer-norm penalty is only an experimental
regularizer. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full distinction
between the theoretical assumptions and the implemented experiments.

## Citation

Please cite the accompanying manuscript. A `CITATION.cff` file is included so
that GitHub can generate a software citation. The bibliographic entry will be
updated when the manuscript receives a public identifier.

## License

This code is released under the [MIT License](LICENSE).
