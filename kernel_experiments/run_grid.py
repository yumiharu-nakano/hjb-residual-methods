"""Numerical experiments for the manufactured Bellman-form HJB problem.

Modes (--mode):
  legacy    the eight/six ad hoc settings used in earlier drafts
  refine    fill-distance refinement at fixed truncation radius R: N^t fixed,
            N^x refined, so that the error can be plotted against h_x = 2R/(N^x-1)
  eps       a bandwidth sweep at fixed (N^t, N^x)
  domain    truncation study: R_n grows with h_x held (nearly) fixed
  theory    theorem-scaled d=1 sequence: R grows and both N^t and N^x grow

Every run reports the error both on the full truncation cylinder
[0,T] x [-R,R]^d and on the fixed compact core [0,T] x [-1,1]^d, the latter
being the faithful analogue of the locally uniform convergence asserted by the
theory (where R_n -> infinity while the compact set stays fixed).
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

import kernel as K
from kernel import (OptConfig, core_points, eval_points, evaluate_at,
                    native_norm, residual_inf_at, solve_Pn)


SETTINGS_D1 = [
    ("S1", 32, 16, 100), ("S2", 16, 16, 500), ("S3", 16, 32, 100),
    ("S4", 16, 32, 500), ("S5", 32, 32, 100), ("S6", 16, 64, 100),
    ("S7", 16, 64, 500), ("S8", 32, 64, 100),
]

SETTINGS_D2 = [
    ("D2-S1", 8, 8, 200), ("D2-S2", 16, 8, 200), ("D2-S3", 8, 12, 200),
    ("D2-S4", 16, 12, 200), ("D2-S5", 32, 8, 200), ("D2-S6", 16, 16, 200),
]

# Fill-distance refinement at fixed R: N^t fixed, N^x refined geometrically.
REFINE_D1 = [(f"R{i+1}", 32, nx, 100) for i, nx in enumerate([9, 13, 17, 25, 33, 49])]
REFINE_D2 = [(f"R{i+1}", 16, nx, 200) for i, nx in enumerate([7, 9, 11, 13, 15])]

# Truncation study: R grows, h = 2R/(Nx-1) held at 0.375 (d=1) / 0.75 (d=2).
DOMAIN_D1 = [(f"T{i+1}", 32, int(2 * R / 0.375) + 1, 100, float(R))
             for i, R in enumerate([2, 3, 4, 5])]
DOMAIN_D2 = [(f"T{i+1}", 16, int(2 * R / 0.75) + 1, 200, float(R))
             for i, R in enumerate([2, 3, 4, 5])]

# Theorem-scaled sequence for d=1.  With Nt=ceil(R^3) and Nx=Nt+1,
# h_n=sqrt((1/(2Nt))^2+1/(Nx-1)^2)=O(R^{-3}), and hence
# R^{d+1} h_n=R^2 h_n=O(R^{-1}).  Extra points are omitted so that the
# optimized empirical loss is exactly the tensor-grid loss used in the theory.
THEORY_D1 = []
for i, R in enumerate([1.5, 2.0, 2.5, 3.0, 3.5]):
    nt = int(np.ceil(R ** 3))
    THEORY_D1.append((f"H{i+1}", nt, nt + 1, 0, R))

HEADER = ["case", "Nt", "Nx", "N_extra", "N", "M", "d", "R", "eps", "lam", "h",
          "h_fill", "Rdp1_h", "loss_grid", "gamma", "gamma_train", "nrm_H",
          "E_rms", "E_inf", "E_rms_core", "E_inf_core",
          "Res_inf", "n_iter", "cpu_time", "success"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=1, help="spatial dimension")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--max_iter", type=int, default=20000)
    p.add_argument("--limit", type=int, default=0, help="run only first N (0=all)")
    p.add_argument("--start", type=int, default=0, help="start from index")
    p.add_argument("--eps", type=float, default=1.0, help="kernel bandwidth (Wendland scale)")
    p.add_argument("--lam", type=float, default=8.0,
                   help="native-space radius; the theory requires lam > ||v||_H")
    p.add_argument("--kernel", default="C6", choices=["C4", "C6"], help="Wendland profile")
    p.add_argument("--mode", default="refine",
                   choices=["legacy", "refine", "eps", "domain", "theory"])
    p.add_argument("--R", type=float, default=3.0, help="truncation radius (non-domain modes)")
    p.add_argument("--R_core", type=float, default=1.0, help="radius of the fixed compact core")
    p.add_argument("--Nt", type=int, default=32, help="N^t for --mode eps")
    p.add_argument("--Nx", type=int, default=33, help="N^x for --mode eps")
    p.add_argument("--eps_list", default="0.5,0.75,1.0,1.25,1.5,2.0",
                   help="comma-separated bandwidths for --mode eps")
    args = p.parse_args()

    K.set_kernel(args.kernel)

    if args.mode == "refine":
        base = REFINE_D1 if args.d == 1 else REFINE_D2
        settings = [(c, nt, nx, ne, args.R) for (c, nt, nx, ne) in base]
        eps_vals = [args.eps] * len(settings)
    elif args.mode == "domain":
        settings = DOMAIN_D1 if args.d == 1 else DOMAIN_D2
        eps_vals = [args.eps] * len(settings)
    elif args.mode == "eps":
        ev = [float(s) for s in args.eps_list.split(",")]
        settings = [(f"E{i+1}", args.Nt, args.Nx, 100 if args.d == 1 else 200, args.R)
                    for i in range(len(ev))]
        eps_vals = ev
    elif args.mode == "theory":
        if args.d != 1:
            p.error("--mode theory is defined only for d=1")
        settings = THEORY_D1
        eps_vals = [args.eps] * len(settings)
    else:
        base = SETTINGS_D1 if args.d == 1 else SETTINGS_D2
        settings = [(c, nt, nx, ne, args.R) for (c, nt, nx, ne) in base]
        eps_vals = [args.eps] * len(settings)

    if args.start:
        settings, eps_vals = settings[args.start:], eps_vals[args.start:]
    if args.limit:
        settings, eps_vals = settings[:args.limit], eps_vals[:args.limit]

    mode, write_header = "w", True
    if args.start > 0:
        try:
            open(args.out).close()
            mode, write_header = "a", False
        except FileNotFoundError:
            pass

    core = core_points(args.d, 1.0, args.R_core)

    with open(args.out, mode, newline="") as f:
        wr = csv.writer(f)
        if write_header:
            wr.writerow(HEADER)

        for (case, Nt, Nx, N_extra, R), eps in zip(settings, eps_vals):
            cfg = OptConfig(d=args.d, Nt=Nt, Nx=Nx, N_extra=N_extra, R=R, T=1.0,
                            eps=eps, lam=args.lam, max_iter=args.max_iter,
                            seed=0, verbose=False)
            h = 2.0 * R / (Nx - 1) if Nx > 1 else 2.0 * R
            h_fill = np.sqrt((1.0 / (2.0 * Nt)) ** 2
                             + args.d / (Nx - 1) ** 2)
            Rdp1_h = R ** (args.d + 1) * h_fill
            print(f"==> {case}: d={args.d} Nt={Nt} Nx={Nx} R={R} eps={eps} "
                  f"h={h:.4f} ...", flush=True)
            out = solve_Pn(cfg)
            # Report both value errors and residuals on independent grids.
            full = eval_points(args.d, 1.0, R, 51, 101) if args.d == 1 \
                else eval_points(args.d, 1.0, R, 21, 41)
            m = evaluate_at(out["theta"], out["nodes"], eps, full)
            m["Res_inf"] = residual_inf_at(out["theta"], out["nodes"], eps, full)
            mc = evaluate_at(out["theta"], out["nodes"], eps, core)
            nrm = native_norm(out["theta"], out["mats"])
            wr.writerow([case, Nt, Nx, N_extra, out["n_nodes"], out["n_eval"],
                         args.d, R, eps, args.lam, h, h_fill, Rdp1_h,
                         out["loss_grid"], out["gamma"], out["gamma_train"], nrm,
                         m["E_rms"], m["E_inf"], mc["E_rms"], mc["E_inf"],
                         m["Res_inf"], out["n_iter"], out["cpu_time"],
                         out["success"]])
            f.flush()
            print(f"    N={out['n_nodes']} gamma={out['gamma']:.3e} "
                  f"|v_n|_H={nrm:.2f} E_rms={m['E_rms']:.3e} "
                  f"E_rms_core={mc['E_rms']:.3e} E_inf_core={mc['E_inf']:.3e} "
                  f"iters={out['n_iter']} CPU={out['cpu_time']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
