# PINN experiments

`pinn.py` implements a tanh network for the one-dimensional manufactured HJB
problem. The training objective combines the empirical squared PDE residual,
the empirical squared terminal mismatch, and a product-of-layer-norm penalty.

Files:

- `pinn.py`: model, residual, metrics, and Adam/L-BFGS training;
- `run_grid.py`: preset sweeps over width, sample size, and penalty weight;
- `postprocess.py`: tables and figures from `results.csv`;
- `sobolev_check.py`: numerical postprocessing of retained checkpoints;
- `checkpoints/`: retained state dictionaries and training metadata.

Run the reported small preset and regenerate the derived outputs with

```bash
python run_grid.py --preset small --out results.csv --device cpu --checkpoint_dir checkpoints
python postprocess.py results.csv .
python sobolev_check.py --checkpoint_dir checkpoints --out sobolev_results.csv
```

The penalty is an experimental regularizer. It does not impose the hard
parameter or Sobolev constraints assumed in the paper's convergence theorem.
CPU is the default backend. Apple MPS is not selected automatically because the
second-order derivatives entering the PDE residual require backend-specific
verification.
