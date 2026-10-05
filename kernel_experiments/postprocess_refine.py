"""Tables and figures for the kernel refinement / bandwidth / truncation studies.

Usage:
    python postprocess_refine.py refine  refine_d1.csv  _d1 [--slope 1.5]
    python postprocess_refine.py eps     eps_d1.csv     _d1
    python postprocess_refine.py domain  domain_d1.csv  _d1
    python postprocess_refine.py theory  theory_d1.csv  _d1

The refinement figures plot against the spatial grid spacing h_x (not against
the space-time fill distance used in the convergence theorem).

Two error levels are reported throughout:
  * "core"  -- on the fixed compact set [0,T] x [-1,1]^d, the faithful analogue
               of the locally uniform convergence asserted by the theory;
  * "full"  -- on all of [0,T] x [-R,R]^d, which additionally contains the
               truncation error near the artificial boundary |x| = R.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                     "legend.fontsize": 10, "xtick.labelsize": 10,
                     "ytick.labelsize": 10})


def read_rows(path):
    with open(path) as f:
        return [dict(r) for r in csv.DictReader(f)]


def col(rows, key, cast=float):
    return np.array([cast(r[key]) for r in rows])


def fmt(x, digits=3):
    return f"{x:.{digits}e}"


# ---------------------------------------------------------------- figures


def _h_axis(ax, h):
    ax.set_xscale("log")
    ax.set_xticks(h)
    ax.set_xticklabels([f"{v:.3g}" for v in h])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)


def fig_refine(rows, out_path, ref_slope=None, gamma_slope=None):
    rows = sorted(rows, key=lambda r: float(r["h"]))
    h = col(rows, "h")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3))

    ax1.loglog(h, col(rows, "E_rms_core"), "o-", color="C0",
               label=r"$E_{\mathrm{RMS}}$ (core)")
    ax1.loglog(h, col(rows, "E_inf_core"), "s--", color="C1",
               label=r"$E_{\infty}$ (core)")
    ax1.loglog(h, col(rows, "E_rms"), "^:", color="C2",
               label=r"$E_{\mathrm{RMS}}$ (full)")
    if ref_slope is not None:
        e = col(rows, "E_rms_core")
        c = e[0] / h[0] ** ref_slope
        ax1.loglog(h, c * h ** ref_slope, "k-.", lw=1.0,
                   label=rf"$h_x^{{{ref_slope:g}}}$ reference")
    ax1.set_xlabel(r"spatial spacing $h_x$")
    ax1.set_ylabel("error")
    ax1.grid(True, which="major", ls=":", alpha=0.6)
    ax1.legend(loc="lower right")
    _h_axis(ax1, h)

    ax2.loglog(h, col(rows, "gamma"), "o-", color="C3", label=r"$\gamma_n$")
    if gamma_slope is not None:
        gg = col(rows, "gamma")
        c = gg[0] / h[0] ** gamma_slope
        ax2.loglog(h, c * h ** gamma_slope, "k-.", lw=1.0,
                   label=rf"$h_x^{{{gamma_slope:g}}}$ reference")
    ax2.set_xlabel(r"spatial spacing $h_x$")
    ax2.set_ylabel(r"a posteriori residual bound $\gamma_n$")
    ax2.grid(True, which="major", ls=":", alpha=0.6)
    ax2.legend(loc="lower right")
    _h_axis(ax2, h)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def fig_eps(rows, out_path):
    rows = sorted(rows, key=lambda r: float(r["eps"]))
    e = col(rows, "eps")
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    ax.semilogy(e, col(rows, "E_rms_core"), "o-", label=r"$E_{\mathrm{RMS}}$ (core)")
    ax.semilogy(e, col(rows, "E_inf_core"), "s--", label=r"$E_{\infty}$ (core)")
    ax.semilogy(e, col(rows, "gamma"), "^:", label=r"$\gamma_n$")
    ax.set_xlabel(r"kernel bandwidth $c$")
    ax.set_ylabel("error / residual bound")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def fig_domain(rows, out_path):
    rows = sorted(rows, key=lambda r: float(r["R"]))
    R = col(rows, "R")
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    ax.semilogy(R, col(rows, "E_rms_core"), "o-", label=r"$E_{\mathrm{RMS}}$ (core)")
    ax.semilogy(R, col(rows, "E_rms"), "^:", label=r"$E_{\mathrm{RMS}}$ (full)")
    ax.set_xlabel(r"truncation radius $R$")
    ax.set_ylabel("error")
    ax.set_xticks(R)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- tables


def _table(header, body, out_path, label, caption, colspec, placement="tb"):
    L = [f"\\begin{{table}}[{placement}]", r"  \centering", r"  \small",
         f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
         f"  \\begin{{tabular}}{{{colspec}}}", r"    \toprule",
         "    " + header + r" \\", r"    \midrule"]
    L += ["    " + b + r" \\" for b in body]
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")


def table_refine(rows, out_path, label, caption):
    rows = sorted(rows, key=lambda r: float(r["h"]), reverse=True)
    header = (r"$N^t$ & $N^x$ & $N$ & $h_x$ & $\gamma_n$ & "
              r"$\|v_n\|_{\mathcal H}$ & $E_{\mathrm{RMS}}^{\mathrm{core}}$ & "
              r"$E_{\infty}^{\mathrm{core}}$ & $E_{\mathrm{RMS}}$ & CPU [s]")
    body = [
        f"{r['Nt']} & {r['Nx']} & {r['N']} & {float(r['h']):.4f} & "
        f"{fmt(float(r['gamma']))} & {float(r['nrm_H']):.2f} & "
        f"{fmt(float(r['E_rms_core']))} & {fmt(float(r['E_inf_core']))} & "
        f"{fmt(float(r['E_rms']))} & {float(r['cpu_time']):.1f}"
        for r in rows]
    _table(header, body, out_path, label, caption, "cccccccccc")


def table_eps(rows, out_path, label, caption):
    rows = sorted(rows, key=lambda r: float(r["eps"]))
    header = (r"$c$ & $\gamma_n$ & $\|v_n\|_{\mathcal H}$ & "
              r"$E_{\mathrm{RMS}}^{\mathrm{core}}$ & $E_{\infty}^{\mathrm{core}}$ & CPU [s]")
    body = [
        f"{float(r['eps']):.3g} & {fmt(float(r['gamma']))} & "
        f"{float(r['nrm_H']):.2f} & {fmt(float(r['E_rms_core']))} & "
        f"{fmt(float(r['E_inf_core']))} & {float(r['cpu_time']):.1f}"
        for r in rows]
    _table(header, body, out_path, label, caption, "cccccc")


def table_domain(rows, out_path, label, caption):
    rows = sorted(rows, key=lambda r: float(r["R"]))
    header = (r"$R$ & $N^x$ & $N$ & $h_x$ & $\gamma_n$ & "
              r"$E_{\mathrm{RMS}}^{\mathrm{core}}$ & $E_{\mathrm{RMS}}$ & CPU [s]")
    body = [
        f"{float(r['R']):.0f} & {r['Nx']} & {r['N']} & {float(r['h']):.4f} & "
        f"{fmt(float(r['gamma']))} & {fmt(float(r['E_rms_core']))} & "
        f"{fmt(float(r['E_rms']))} & {float(r['cpu_time']):.1f}"
        for r in rows]
    _table(header, body, out_path, label, caption, "cccccccc")


def table_theory(rows, out_path, label, caption):
    rows = sorted(rows, key=lambda r: float(r["R"]))
    header = (r"$R_n$ & $N^t_n$ & $N^x_n$ & $N$ & $R_n^2h_n$ & "
              r"$\widehat{\mathcal L}_n$ & $\gamma_n$ & "
              r"$\|v_n\|_{\mathcal H}$ & $E_{\mathrm{RMS}}^{\mathrm{core}}$ & CPU [s]")
    body = [
        f"{float(r['R']):.1f} & {r['Nt']} & {r['Nx']} & {r['N']} & "
        f"{float(r['Rdp1_h']):.4f} & {fmt(float(r['loss_grid']))} & "
        f"{fmt(float(r['gamma']))} & {float(r['nrm_H']):.2f} & "
        f"{fmt(float(r['E_rms_core']))} & {float(r['cpu_time']):.1f}"
        for r in rows]
    _table(header, body, out_path, label, caption, "cccccccccc", placement="!ht")


def fig_sensitivity(eps_rows, dom_rows, out_path):
    """Two-panel sensitivity figure: bandwidth (left), truncation radius (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3))

    er = sorted(eps_rows, key=lambda r: float(r["eps"]))
    e = col(er, "eps")
    ax1.semilogy(e, col(er, "E_rms_core"), "o-", color="C0",
                 label=r"$E_{\mathrm{RMS}}$ (core)")
    ax1.semilogy(e, col(er, "E_inf_core"), "s--", color="C1",
                 label=r"$E_{\infty}$ (core)")
    ax1.semilogy(e, col(er, "gamma"), "^:", color="C3", label=r"$\gamma_n$")
    ax1.set_xlabel(r"kernel bandwidth $c$")
    ax1.set_ylabel("error / residual bound")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend()

    dr = sorted(dom_rows, key=lambda r: float(r["R"]))
    R = col(dr, "R")
    ax2.semilogy(R, col(dr, "E_rms_core"), "o-", color="C0",
                 label=r"$E_{\mathrm{RMS}}$ (core)")
    ax2.semilogy(R, col(dr, "E_rms"), "^:", color="C2",
                 label=r"$E_{\mathrm{RMS}}$ (full)")
    ax2.set_xlabel(r"truncation radius $R$")
    ax2.set_ylabel("error")
    ax2.set_xticks(R)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


CAPTIONS = {
    "refine": ("Spatial-grid refinement of the kernel realization",
               "tab:ker.refine"),
    "eps": ("Bandwidth sensitivity of the kernel realization", "tab:ker.eps"),
    "domain": ("Truncation-radius study for the kernel realization",
                "tab:ker.domain"),
    "theory": ("Theorem-scaled refinement of the kernel realization",
               "tab:ker.theory"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["refine", "eps", "domain", "theory", "combo"])
    p.add_argument("csv")
    p.add_argument("suffix", nargs="?", default="")
    p.add_argument("--csv2", default=None,
                   help="second csv (domain) for kind=combo")
    p.add_argument("--outdir", default=".")
    p.add_argument("--slope", type=float, default=None)
    p.add_argument("--gamma_slope", type=float, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--caption", default=None)
    args = p.parse_args()

    rows = read_rows(args.csv)
    os.makedirs(args.outdir, exist_ok=True)
    sfx = args.suffix
    if args.kind == "combo":
        fig_sensitivity(rows, read_rows(args.csv2),
                        os.path.join(args.outdir, f"fig_ker_sens{sfx}.png"))
        table_eps(rows, os.path.join(args.outdir, f"table_ker_eps{sfx}.tex"),
                  f"tab:ker.eps{sfx}", CAPTIONS["eps"][0] + " ($d=1$).")
        table_domain(read_rows(args.csv2),
                     os.path.join(args.outdir, f"table_ker_domain{sfx}.tex"),
                     f"tab:ker.domain{sfx}", CAPTIONS["domain"][0] + " ($d=1$).")
        print(f"wrote combo outputs with suffix {sfx!r} into {args.outdir}")
        return
    cap, lab = CAPTIONS[args.kind]
    cap = args.caption or (cap + f" ({'$d=2$' if sfx.endswith('d2') else '$d=1$'}).")
    lab = args.label or (lab + sfx)

    if args.kind == "refine":
        fig_refine(rows, os.path.join(args.outdir, f"fig_ker_refine{sfx}.png"),
                   ref_slope=args.slope, gamma_slope=args.gamma_slope)
        table_refine(rows, os.path.join(args.outdir, f"table_ker_refine{sfx}.tex"),
                     lab, cap)
    elif args.kind == "eps":
        fig_eps(rows, os.path.join(args.outdir, f"fig_ker_eps{sfx}.png"))
        table_eps(rows, os.path.join(args.outdir, f"table_ker_eps{sfx}.tex"), lab, cap)
    elif args.kind == "theory":
        table_theory(rows, os.path.join(args.outdir, f"table_ker_theory{sfx}.tex"),
                     lab, cap)
    else:
        fig_domain(rows, os.path.join(args.outdir, f"fig_ker_domain{sfx}.png"))
        table_domain(rows, os.path.join(args.outdir, f"table_ker_domain{sfx}.tex"),
                     lab, cap)
    print(f"wrote {args.kind} outputs with suffix {sfx!r} into {args.outdir}")


if __name__ == "__main__":
    main()
