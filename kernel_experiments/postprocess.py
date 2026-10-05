"""Generate a LaTeX table and figures from kernel-collocation results."""

from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}e}"


def read_rows(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            out.append(
                dict(
                    case=r["case"],
                    Nt=int(r["Nt"]),
                    Nx=int(r["Nx"]),
                    N_extra=int(r["N_extra"]),
                    N=int(r["N"]),
                    M=int(r["M"]),
                    gamma=float(r["gamma"]),
                    E_rms=float(r["E_rms"]),
                    E_inf=float(r["E_inf"]),
                    Res_inf=float(r["Res_inf"]),
                    n_iter=int(r["n_iter"]),
                    cpu_time=float(r["cpu_time"]),
                    success=str(r["success"]),
                )
            )
    return out


def latex_main_table(rows: list[dict], out_path: str, label_suffix: str = "") -> None:
    lines = []
    lines.append(r"\begin{table*}[tb]")
    lines.append(r"  \centering")
    if label_suffix:
        d_str = label_suffix.replace("_d", "")
        cap = (
            r"  \caption{Approximation errors, residual bound $\gamma_n$, iteration count, "
            r"and CPU time on the $d=" + d_str + r"$ test problem.}"
        )
    else:
        cap = (
            r"  \caption{Main results for the test problem of Section~\ref{subsec:test_setup}: "
            r"approximation errors, residual bound $\gamma_n$, iteration count, and CPU time.}"
        )
    lines.append(cap)
    lines.append(r"  \label{tab:main_result" + label_suffix + r"}")
    lines.append(r"  \begin{tabular}{ccccccccccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Case & $N^t$ & $N^x$ & $N^{\mathrm{extra}}$ & $N$ & $\gamma_n$ & "
        r"$E_{\mathrm{RMS}}$ & $E_{\infty}$ & $\mathrm{Res}_{\infty}$ & Iter. & CPU [s] \\"
    )
    lines.append(r"    \midrule")
    for r in rows:
        lines.append(
            f"    {r['case']} & {r['Nt']} & {r['Nx']} & {r['N_extra']} & {r['N']} & "
            f"{fmt(r['gamma'])} & {fmt(r['E_rms'])} & {fmt(r['E_inf'])} & "
            f"{fmt(r['Res_inf'])} & {r['n_iter']} & {r['cpu_time']:.1f} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_N_vs_E(rows: list[dict], out_path: str) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["N"])
    fig, ax = plt.subplots(figsize=(5, 4))
    Ns = [r["N"] for r in rows_sorted]
    ax.loglog(Ns, [r["E_rms"] for r in rows_sorted], "o-", label="$E_{\\mathrm{RMS}}$")
    ax.loglog(Ns, [r["E_inf"] for r in rows_sorted], "s--", label="$E_{\\infty}$")
    ax.set_xlabel("$N$ (number of basis functions)")
    ax.set_ylabel("error")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_N_vs_gamma(rows: list[dict], out_path: str) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["N"])
    fig, ax = plt.subplots(figsize=(5, 4))
    Ns = [r["N"] for r in rows_sorted]
    ax.semilogx(Ns, [r["gamma"] for r in rows_sorted], "o-", label="$\\gamma_n$")
    ax.semilogx(
        Ns,
        [r["Res_inf"] for r in rows_sorted],
        "s--",
        label="$\\mathrm{Res}_{\\infty}$",
    )
    ax.set_xlabel("$N$")
    ax.set_ylabel("residual bound")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_N_vs_cpu(rows: list[dict], out_path: str) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["N"])
    fig, ax = plt.subplots(figsize=(5, 4))
    Ns = [r["N"] for r in rows_sorted]
    ax.loglog(Ns, [r["cpu_time"] for r in rows_sorted], "o-")
    ax.set_xlabel("$N$")
    ax.set_ylabel("CPU time [s]")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    suffix = sys.argv[3] if len(sys.argv) > 3 else ""  # e.g. "_d2"
    os.makedirs(out_dir, exist_ok=True)
    rows = read_rows(csv_path)
    print(f"Read {len(rows)} rows from {csv_path}.")

    table_path = os.path.join(out_dir, f"table_main_result{suffix}.tex")
    latex_main_table(rows, table_path, label_suffix=suffix)
    plot_N_vs_E(rows, os.path.join(out_dir, f"fig_N_vs_E{suffix}.png"))
    plot_N_vs_gamma(rows, os.path.join(out_dir, f"fig_N_vs_gamma{suffix}.png"))
    plot_N_vs_cpu(rows, os.path.join(out_dir, f"fig_N_vs_cpu{suffix}.png"))
    print(f"Wrote table{suffix}, figs{suffix} in {out_dir}.")


if __name__ == "__main__":
    main()
