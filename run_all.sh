#!/usr/bin/env bash
# Re-run every reported experiment for the Bellman-form test problem.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export MPLCONFIGDIR="$REPO_ROOT/.matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"
cd "$REPO_ROOT/kernel_experiments"
echo "[start] $(date)"
python3 run_grid.py --d 1 --mode refine --kernel C6 --eps 1.0 --lam 6 --R 3 --max_iter 20000 --out refine_d1.csv
python3 run_grid.py --d 1 --mode eps --kernel C6 --lam 6 --R 3 --Nt 32 --Nx 25 --eps_list 0.5,0.75,1.0,1.25,1.5,2.0 --max_iter 20000 --out eps_d1.csv
python3 run_grid.py --d 1 --mode eps --kernel C6 --lam 6 --R 3 --Nt 32 --Nx 25 --eps_list 2.5,3.0,4.0 --max_iter 20000 --out eps_d1_hi.csv
python3 run_grid.py --d 1 --mode domain --kernel C6 --eps 1.0 --lam 6 --max_iter 20000 --out domain_d1.csv
python3 run_grid.py --d 1 --mode theory --kernel C6 --eps 1.0 --lam 6 --max_iter 20000 --out theory_d1.csv
python3 profile_error.py --lam 6 --eps_list 1.0,2.0 --out fig_ker_profile_d1.png
python3 fd_compare.py
echo "[kernel d1 done] $(date)"
python3 run_grid.py --d 2 --mode refine --kernel C6 --eps 1.5 --lam 6 --R 3 --max_iter 20000 --out refine_d2.csv
./make_all_outputs.sh
echo "[kernel d2 done] $(date)"
cd ../pinn_experiments
python3 run_grid.py --preset small --out results.csv --checkpoint_dir checkpoints
python3 postprocess.py results.csv .
python3 sobolev_check.py --checkpoint_dir checkpoints --out sobolev_results.csv
echo "[all done] $(date)"
