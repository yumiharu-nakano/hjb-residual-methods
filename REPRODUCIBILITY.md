# Reproducibility and scope

## What is included

- source code for the kernel and PINN experiments;
- the CSV files used to construct the reported tables and figures;
- generated plots and LaTeX table fragments;
- the retained PINN state dictionaries and their training metadata;
- scripts that regenerate the derived outputs.

All reported errors are evaluated on deterministic grids that are independent
of the kernel collocation set or PINN training samples. The fixed compact core
is `[0,1] x [-1,1]^d`. Full-cylinder errors are also reported on
`[0,1] x [-R,R]^d`.

## Kernel experiments

The default `C6` Wendland profile has the smoothness required by the paper for
`d=1` and `d=2`. The initial coefficient vector is rescaled if necessary, and
the optimization objective penalizes violations of the native-space radius.
The code records the final native-space norm of every candidate. The saved
candidates satisfy the prescribed radius when checked after optimization; the
implementation does not project the final iterate onto the ball.

The theorem-scaled `d=1` sequence uses
`N_t = ceil(R^3)`, `N_x = N_t + 1`, and no extra residual points. It is a finite
illustration of the scaling condition. It does not verify the asymptotic
assumptions, the analytical condition on the exact native-space norm, or every
hypothesis of the convergence theorem.

## PINN experiments

The PINN loss is the empirical squared PDE residual plus the empirical squared
terminal mismatch and a squared product-of-layer-norm penalty. The implementation
does not solve the hard-constrained optimization problem used in the theorem.
Consequently, the reported PINN experiment illustrates the residual-minimization
mechanism but does not by itself verify the theorem's assumptions.

Training uses one fixed held-out sample to retain the best iterate. This is an
oracle-style experimental choice and should not be interpreted as an implementable
a posteriori stopping rule for an unknown solution.

## Numerical evidence

The experiments use one manufactured HJB equation in dimensions one and two.
They support internal consistency checks and show representative behavior of the
two realizations. They are not a broad performance benchmark and do not establish
the analytical convergence results, which are proved separately in the paper.

## Randomness and hardware

The default random seed is zero. The PINN results were produced on CPU; see
`pinn_experiments/checkpoints/metadata.json` for the recorded Python, PyTorch,
and training settings. Results may vary across library versions and hardware.
The current code does not select Apple MPS automatically because second-order
automatic differentiation should be checked carefully before using that backend.
