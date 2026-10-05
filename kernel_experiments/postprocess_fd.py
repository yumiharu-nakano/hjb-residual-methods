"""Generate the FD baseline comparison table in LaTeX from results_fd_d1.csv."""

from __future__ import annotations

import csv
import sys


def read_rows(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}e}"


def main():
    fd_path = sys.argv[1] if len(sys.argv) > 1 else "results_fd_d1.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "table_fd_baseline.tex"

    rows = read_rows(fd_path)
    lines = []
    lines.append(r"\begin{table*}[tb]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Finite-difference baseline for the $d=1$ test problem: "
        r"explicit upwind scheme with centered second derivative. "
        r"Boundary values supplied by the analytical solution.}"
    )
    lines.append(r"  \label{tab:fd_baseline}")
    lines.append(r"  \begin{tabular}{ccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Case & $N^x$ & $N^t$ & $h$ & $\Delta t$ & $E_{\mathrm{RMS}}$ & $E_{\infty}$ & CPU [s] \\")
    lines.append(r"    \midrule")
    for r in rows:
        lines.append(
            f"    {r['case']} & {r['Nx']} & {r['Nt']} & "
            f"${fmt(float(r['h']))}$ & ${fmt(float(r['dt']))}$ & "
            f"${fmt(float(r['E_rms']))}$ & ${fmt(float(r['E_inf']))}$ & "
            f"{float(r['cpu_time']):.3f} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    # Fix column count mismatch: we have 8 columns of data, not 7
    # Rewrite the tabular spec
    text = "\n".join(lines)
    text = text.replace("{ccccccc}", "{cccccccc}")

    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
