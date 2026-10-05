"""Write table_fd_compare_d1.tex from results_fd_compare_d1.csv."""
import csv

rows = list(csv.DictReader(open("results_fd_compare_d1.csv")))
e = lambda s: f"{float(s):.3e}"
L = [r"\begin{table}[tb]", r"  \centering", r"  \small",
     r"  \caption{Monotone finite-difference baseline for the $d=1$ test problem (central second"
     r" differences inside the minimum over the diffusion level, upwind first differences, explicit"
     r" Euler in $t$ under the CFL condition), evaluated on the same core and full grids and with the"
     r" same metrics as Table~\ref{tab:ker.refine_d1}. \emph{oracle}: Dirichlet values at $|x|=R$ taken"
     r" from the analytical solution; \emph{extrap}: linear extrapolation at $|x|=R$, i.e.\ no"
     r" knowledge of the exact solution. The core errors coincide to the reported precision for the two"
     r" treatments because the drift has speed at most $1$ and the diffusion coefficient is at most"
     r" $0.08$.}",
     r"  \label{tab:fd.compare_d1}", r"  \begin{tabular}{llccccc}", r"    \toprule",
     r"    BC & $N^x$ & steps & $E_{\mathrm{RMS}}^{\mathrm{core}}$ & $E_{\infty}^{\mathrm{core}}$ & $E_{\mathrm{RMS}}$ & CPU [s] \\",
     r"    \midrule"]
prev = None
for r in rows:
    if prev is not None and r["bc"] != prev:
        L.append(r"    \midrule")
    prev = r["bc"]
    L.append(f"    {r['bc']} & {r['Nx']} & {r['n_steps']} & {e(r['E_rms_core'])} & "
             f"{e(r['E_inf_core'])} & {e(r['E_rms_full'])} & {float(r['cpu_time']):.3f} \\\\")
L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
open("table_fd_compare_d1.tex", "w").write("\n".join(L) + "\n")
print("wrote table_fd_compare_d1.tex")
