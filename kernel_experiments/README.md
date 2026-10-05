# Kernel experiments

`kernel.py` implements compactly supported Wendland kernel collocation for the
manufactured HJB problem in `hjb_problem.py`. The default `C6` profile satisfies
the smoothness requirement used in the accompanying convergence analysis for
`d=1` and `d=2`; `C4` is retained only for comparison.

The main experiment families are:

- `refine`: collocation-grid refinement at fixed truncation radius;
- `eps`: kernel-bandwidth sensitivity;
- `domain`: increasing truncation radius at nearly fixed spatial mesh width;
- `theory`: the theorem-scaled `d=1` sequence, refining space and time while
  increasing the truncation radius.

Representative commands are

```bash
python run_grid.py --d 1 --mode refine --kernel C6 --eps 1 --lam 6 --R 3 --max_iter 20000 --out refine_d1.csv
python run_grid.py --d 1 --mode theory --kernel C6 --eps 1 --lam 6 --max_iter 20000 --out theory_d1.csv
python run_grid.py --d 2 --mode refine --kernel C6 --eps 1.5 --lam 6 --R 3 --max_iter 20000 --out refine_d2.csv
./make_all_outputs.sh
```

Every run reports errors on an independent full-cylinder grid and on the fixed
core `[0,1] x [-1,1]^d`. The CSV column `gamma` is the discrete residual-plus-
terminal quantity used in the paper, and `nrm_H` is the computed native-space
norm.
