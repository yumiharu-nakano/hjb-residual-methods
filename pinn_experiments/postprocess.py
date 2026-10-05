"""Post-process results.csv: produce LaTeX rows and matplotlib plots."""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}e}"


def read_rows(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            row = dict(
                width=int(r["width"]),
                M=int(r["M"]),
                beta=float(r["beta"]),
                E_rms=float(r["E_rms"]),
                E_inf=float(r["E_inf"]),
                E_rms_core=float(r.get("E_rms_core") or "nan"),
                E_inf_core=float(r.get("E_inf_core") or "nan"),
                E_T_inf=float(r["E_T_inf"]),
                Res_inf=float(r["Res_inf"]),
                path_norm=float(r["path_norm"]),
                n_params=int(r["n_params"]),
                cpu_time=float(r["cpu_time"]),
            )
            # Optional columns (medium/large preset)
            for k in ("cpu_time_adam", "cpu_time_lbfgs"):
                if k in r:
                    row[k] = float(r[k])
            out.append(row)
    return out


def deduplicate(rows: list[dict]) -> list[dict]:
    """Keep the first occurrence of each (W, M, beta) setting."""
    seen = set()
    out = []
    for r in rows:
        key = (r["width"], r["M"], r["beta"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def latex_table_main(rows: list[dict], beta_main: float, out_path: str) -> None:
    """Create the width-by-sample-size table at fixed beta."""
    seen: set[tuple[int, int]] = set()
    sel = []
    for r in rows:
        if abs(r["beta"] - beta_main) > 1e-12:
            continue
        key = (r["width"], r["M"])
        if key in seen:
            continue
        seen.add(key)
        sel.append(r)
    sel.sort(key=lambda r: (r["width"], r["M"]))
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    # Format beta_main as 10^k when possible, else %.1e
    import math
    if beta_main > 0:
        k = round(math.log10(beta_main))
        if abs(beta_main - 10 ** k) / beta_main < 1e-6:
            beta_str = f"10^{{{k}}}"
        else:
            beta_str = f"{beta_main:.1e}"
    else:
        beta_str = "0"
    lines.append(
        r"\caption{Errors and computational cost as a function of width $W$ and sample size $M$, "
        r"with $\beta=" + beta_str + r"$ fixed.}"
    )
    lines.append(r"\label{tab:4.main}")
    lines.append(r"\begin{tabular}{cccccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"$W$ & $M$ & $E_{\mathrm{RMS}}^{\mathrm{core}}$ & $E_{\infty}^{\mathrm{core}}$ & "
        r"$E_{\mathrm{RMS}}$ & $E_{\infty}$ & $E_{T,\infty}$ & "
        r"$\mathrm{Res}_{\infty}$ & $\mathcal P(\theta)$ & CPU [s] \\"
    )
    lines.append(r"\midrule")
    for r in sel:
        lines.append(
            f"{r['width']} & {r['M']} & {fmt(r['E_rms_core'])} & "
            f"{fmt(r['E_inf_core'])} & {fmt(r['E_rms'])} & {fmt(r['E_inf'])} & "
            f"{fmt(r['E_T_inf'])} & {fmt(r['Res_inf'])} & {fmt(r['path_norm'], 2)} & "
            f"{r['cpu_time']:.1f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_M_curves(rows: list[dict], beta_main: float, out_path: str) -> None:
    sel = [r for r in deduplicate(rows) if abs(r["beta"] - beta_main) < 1e-12]
    widths = sorted({r["width"] for r in sel})
    fig, ax = plt.subplots(figsize=(4, 3.4))
    for i, w in enumerate(widths):
        rs = sorted([r for r in sel if r["width"] == w], key=lambda r: r["M"])
        Ms = [r["M"] for r in rs]
        c = f"C{i}"
        ax.loglog(Ms, [r["E_rms_core"] for r in rs], "o-", color=c, label=f"$W={w}$")
        ax.loglog(Ms, [r["E_rms"] for r in rs], "s--", color=c, alpha=0.65)
    ax.set_xlabel("$M$ (sample size)")
    ax.set_ylabel("$E_{\\mathrm{RMS}}$")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    h, l = ax.get_legend_handles_labels()
    style = [plt.Line2D([], [], color="0.35", ls="-", marker="o", label="core"),
             plt.Line2D([], [], color="0.35", ls="--", marker="s", label="full")]
    ax.legend(handles=h + style, fontsize=9.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_beta_curves(rows: list[dict], width_main: int, M_main: int, out_path: str) -> None:
    sel = [r for r in deduplicate(rows)
           if r["width"] == width_main and r["M"] == M_main]
    sel.sort(key=lambda r: r["beta"])
    betas = []
    Es = []
    for r in sel:
        # treat beta=0 as 1e-9 for log-axis
        b = r["beta"] if r["beta"] > 0 else 1e-9
        betas.append(b)
        Es.append((r["E_rms_core"], r["E_rms"]))
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.loglog(betas, [e[0] for e in Es], "o-", label="core")
    ax.loglog(betas, [e[1] for e in Es], "s--", alpha=0.6, label="full")
    ax.legend(fontsize=9.5)
    ax.set_xlabel(r"regularization weight $\beta$")
    ax.set_ylabel("$E_{\\mathrm{RMS}}$")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_title(f"$W={width_main}$, $M={M_main}$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)
    rows = read_rows(csv_path)
    print(f"Read {len(rows)} rows.")

    beta_main = 1e-6
    latex_table_main(rows, beta_main, os.path.join(out_dir, "table4_1.tex"))
    plot_M_curves(rows, beta_main, os.path.join(out_dir, "fig_M_curves.png"))
    plot_beta_curves(rows, 32, 4096, os.path.join(out_dir, "fig_beta_curves.png"))
    print("Wrote table4_1.tex, fig_M_curves.png, fig_beta_curves.png in", out_dir)


if __name__ == "__main__":
    main()
