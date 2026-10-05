"""Grid experiments for the d=1 manufactured Bellman-form HJB problem.

Two preset grids:
  --preset small  ~10-15 min on a CPU laptop (5000 Adam iter)
  --preset large  ~hours; uses full Adam + L-BFGS to push errors

Custom grids can be defined in the body of `build_settings`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
import platform
import sys

import torch

from pinn import TrainConfig, train


def build_settings(preset: str) -> tuple[list[dict], dict]:
    """Returns (settings, default_kwargs)."""
    if preset == "small":
        settings = []
        for width in (16, 32, 64):
            for M in (1024, 4096):
                settings.append(dict(width=width, M=M, beta=1e-6))
        for beta in (0.0, 1e-7, 1e-5, 1e-4):
            settings.append(dict(width=32, M=4096, beta=beta))
        defaults = dict(
            depth=3,
            n_iter_adam=5000,
            n_iter_lbfgs=0,
            val_every=1000,
            val_size=2000,
        )
        return settings, defaults

    if preset == "medium":
        settings = []
        for width in (32, 64, 128):
            for M in (4096, 16384):
                settings.append(dict(width=width, M=M, beta=1e-6))
        for beta in (0.0, 1e-7, 1e-5, 1e-4):
            settings.append(dict(width=64, M=4096, beta=beta))
        defaults = dict(
            depth=3,
            n_iter_adam=30000,
            n_iter_lbfgs=200,
            lbfgs_max_iter=20,
            lr_halve_every=8000,
            val_every=2000,
            val_size=3000,
        )
        return settings, defaults

    if preset == "large":
        settings = []
        for width in (32, 64, 128):
            for M in (4096, 16384, 65536):
                settings.append(dict(width=width, M=M, beta=1e-6))
        for beta in (0.0, 1e-8, 1e-7, 1e-5, 1e-4):
            settings.append(dict(width=64, M=16384, beta=beta))
        defaults = dict(
            depth=3,
            n_iter_adam=100000,
            n_iter_lbfgs=500,
            lbfgs_max_iter=20,
            lr_halve_every=20000,
            val_every=5000,
            val_size=5000,
        )
        return settings, defaults

    raise ValueError(f"unknown preset {preset}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["small", "medium", "large"], default="medium")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="run only the first N settings (0 = all)",
    )
    args = p.parse_args()

    settings, defaults = build_settings(args.preset)
    if args.limit:
        settings = settings[: args.limit]

    print(f"# preset={args.preset} device={args.device} settings={len(settings)}", flush=True)
    print(f"# defaults: {defaults}", flush=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metadata = {
        "preset": args.preset,
        "seed": args.seed,
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "settings": settings,
        "defaults": defaults,
    }
    with open(os.path.join(args.checkpoint_dir, "metadata.json"), "w") as fmeta:
        json.dump(metadata, fmeta, indent=2)

    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "width",
                "M",
                "beta",
                "E_rms",
                "E_inf",
                "E_rms_core",
                "E_inf_core",
                "E_T_inf",
                "Res_inf",
                "path_norm",
                "n_params",
                "cpu_time",
                "cpu_time_adam",
                "cpu_time_lbfgs",
            ]
        )

        for s in settings:
            kw = dict(defaults)
            kw.update(s)
            cfg = TrainConfig(
                width=kw["width"],
                depth=kw.get("depth", 3),
                M=kw["M"],
                M_term=max(kw["M"] // 4, 256),
                beta=kw["beta"],
                n_iter_adam=kw["n_iter_adam"],
                n_iter_lbfgs=kw.get("n_iter_lbfgs", 0),
                lbfgs_max_iter=kw.get("lbfgs_max_iter", 20),
                lr0=kw.get("lr0", 1e-3),
                lr_halve_every=kw.get("lr_halve_every", 10000),
                val_every=kw.get("val_every", 2000),
                val_size=kw.get("val_size", 3000),
                seed=args.seed,
                device=args.device,
                verbose=False,
            )
            print(
                f"==> W={cfg.width} M={cfg.M} beta={cfg.beta:.1e} "
                f"adam={cfg.n_iter_adam} lbfgs={cfg.n_iter_lbfgs} ...",
                flush=True,
            )
            res = train(cfg)
            beta_tag = f"{cfg.beta:.0e}"
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"W{cfg.width}_M{cfg.M}_beta{beta_tag}.pt",
            )
            torch.save({"cfg": asdict(cfg), "state_dict": res.state_dict}, checkpoint_path)
            wr.writerow(
                [
                    cfg.width,
                    cfg.M,
                    cfg.beta,
                    res.E_rms,
                    res.E_inf,
                    res.E_rms_core,
                    res.E_inf_core,
                    res.E_T_inf,
                    res.Res_inf,
                    res.path_norm,
                    res.n_params,
                    res.cpu_time,
                    res.cpu_time_adam,
                    res.cpu_time_lbfgs,
                ]
            )
            f.flush()
            print(
                f"    E_rms={res.E_rms:.3e} E_inf={res.E_inf:.3e} "
                f"E_rms_core={res.E_rms_core:.3e} "
                f"Res_inf={res.Res_inf:.3e} pn={res.path_norm:.2e} "
                f"CPU_adam={res.cpu_time_adam:.1f}s "
                f"CPU_lbfgs={res.cpu_time_lbfgs:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
