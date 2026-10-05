"""Spatial profile of the kernel-realization error.

For each bandwidth in --eps_list, solve (Pn) at a fixed grid and plot
    x  |-->  max_{t in [0,T]} |v_n(t,x) - v(t,x)|
over [-R, R].  The profile shows where the error of the truncated problem
actually lives, and hence why the theory's locally-uniform statement is tested
on a fixed compact core rather than on all of [-R,R]^d.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import kernel as K
from kernel import OptConfig, native_norm, phi, solve_Pn
import hjb_problem as P

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8.5})


def profile(theta, nodes, eps, R, T, n_t=41, n_x=241):
    tt = np.linspace(0.0, T, n_t)
    xx = np.linspace(-R, R, n_x)
    Tg, Xg = np.meshgrid(tt, xx, indexing="ij")
    pts = np.column_stack([Tg.ravel(), Xg.ravel()])
    diff = pts[:, None, :] - nodes[None, :, :]
    Rm = np.sqrt(np.sum(diff ** 2, axis=-1))
    err = np.abs(phi(Rm, eps) @ theta - P.exact(pts))
    return xx, err.reshape(n_t, n_x).max(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Nt", type=int, default=32)
    p.add_argument("--Nx", type=int, default=25)
    p.add_argument("--R", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=8.0)
    p.add_argument("--max_iter", type=int, default=20000)
    p.add_argument("--eps_list", default="1.0,2.0")
    p.add_argument("--out", default="fig_ker_profile_d1.png")
    args = p.parse_args()

    K.set_kernel("C6")
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for eps in [float(s) for s in args.eps_list.split(",")]:
        cfg = OptConfig(d=1, Nt=args.Nt, Nx=args.Nx, N_extra=100, R=args.R, T=1.0,
                        eps=eps, lam=args.lam, max_iter=args.max_iter, seed=0,
                        verbose=False)
        o = solve_Pn(cfg)
        xx, e = profile(o["theta"], o["nodes"], eps, args.R, 1.0)
        ax.semilogy(xx, e, label=rf"$c={eps:g}$")
        print(f"eps={eps}: gamma={o['gamma']:.3e} "
              f"|v_n|_H={native_norm(o['theta'], o['mats']):.2f} "
              f"max_err={e.max():.3e} core_max={e[np.abs(xx) <= 1].max():.3e}",
              flush=True)
    ax.axvspan(-1, 1, color="0.85", zorder=0)
    ax.text(0.0, ax.get_ylim()[0] * 1.6, r"core $[-1,1]$", ha="center", fontsize=8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\max_{t\in[0,T]}|v_n^{(\lambda)}-v|(t,x)$")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
