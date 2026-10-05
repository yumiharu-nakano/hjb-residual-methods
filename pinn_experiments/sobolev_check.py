"""Compute H^5 norms of the exact networks saved by run_grid.py.

The path-norm penalty is only an experimental proxy. This script loads each
retained checkpoint, promotes its stored coefficients to float64, and evaluates
the actual H^5(Q_n) norm by Gauss--Legendre quadrature and automatic
differentiation. No retraining is performed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import pinn


torch.set_default_dtype(torch.float64)
S = 5


def quad_nodes(R: float, T: float, nt: int = 60, nx: int = 240):
    zt, wt = np.polynomial.legendre.leggauss(nt)
    zx, wx = np.polynomial.legendre.leggauss(nx)
    t = 0.5 * T * (zt + 1)
    wt = 0.5 * T * wt
    x = R * zx
    wx = R * wx
    Tg, Xg = np.meshgrid(t, x, indexing="ij")
    Wg = np.outer(wt, wx)
    return (
        torch.tensor(Tg.ravel()).reshape(-1, 1),
        torch.tensor(Xg.ravel()).reshape(-1, 1),
        torch.tensor(Wg.ravel()),
    )


def sobolev_norm(fun, R: float, T: float, s: int = S, chunk: int = 4000):
    """Return (sum_{|alpha|<=s} int |D^alpha f|^2)^(1/2) on [0,T]x[-R,R]."""
    tq, xq, wq = quad_nodes(R, T)
    total = 0.0
    for k in range(0, tq.shape[0], chunk):
        t = tq[k:k + chunk].clone().requires_grad_(True)
        x = xq[k:k + chunk].clone().requires_grad_(True)
        w = wq[k:k + chunk]
        cur = {(0, 0): fun(t, x)}
        for order in range(s + 1):
            derivatives = [(ij, cur[ij]) for ij in list(cur) if sum(ij) == order]
            for (i, j), value in derivatives:
                total += float((w * value.squeeze(-1).detach() ** 2).sum())
                if order < s:
                    cur[(i + 1, j)] = torch.autograd.grad(
                        value.sum(), t, create_graph=True
                    )[0]
                    cur[(i, j + 1)] = torch.autograd.grad(
                        value.sum(), x, create_graph=True
                    )[0]
    return total ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out", default="sobolev_results.csv")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    paths = sorted(checkpoint_dir.glob("W*_M*_beta*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")

    first = torch.load(paths[0], map_location="cpu", weights_only=True)
    R = float(first["cfg"]["R"])
    T = float(first["cfg"]["T"])
    exact_norm = sobolev_norm(pinn.exact_v, R, T)
    print(f"||v||_H^{S}(Q_n) (exact solution) = {exact_norm:.3f}", flush=True)

    rows = []
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        cfg = pinn.TrainConfig(**saved["cfg"])
        net = pinn.make_net(cfg, torch.device("cpu")).to(dtype=torch.float64)
        net.load_state_dict(saved["state_dict"])
        norm = sobolev_norm(lambda t, x: net(t, x), cfg.R, cfg.T)
        path_norm = float(net.path_norm().detach())
        rows.append((cfg.width, cfg.M, cfg.beta, norm, path_norm, str(path)))
        print(
            f"W={cfg.width:3d} M={cfg.M:5d} beta={cfg.beta:.0e}: "
            f"||v_theta||_H^{S}={norm:9.3f} P(theta)={path_norm:7.2f}",
            flush=True,
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["width", "M", "beta", "H5_norm", "path_norm", "checkpoint"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
