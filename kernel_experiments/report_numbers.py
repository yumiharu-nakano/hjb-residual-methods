"""Print the reported kernel numbers directly from the saved CSV files."""
import csv, numpy as np

def rows(p):
    return list(csv.DictReader(open(p)))

def col(rs, k):
    return np.array([float(r[k]) for r in rs])

def slope(h, y, k=0):
    return np.polyfit(np.log(h[k:]), np.log(y[k:]), 1)[0]

for tag, path in (("d=1", "refine_d1.csv"), ("d=2", "refine_d2.csv")):
    rs = sorted(rows(path), key=lambda r: float(r["h"]))
    h, e, ei, ef, g = (col(rs, k) for k in
                       ("h", "E_rms_core", "E_inf_core", "E_rms", "gamma"))
    nrm, cpu = col(rs, "nrm_H"), col(rs, "cpu_time")
    print(f"\n=== refinement {tag} ===")
    print(f"h            : {h[-1]:.4g} -> {h[0]:.4g}")
    print(f"core E_rms   : {e[-1]:.3e} -> {e[0]:.3e}   slope(all)={slope(h,e):.2f}"
          f"  slope(fine)={slope(h,e,0 if len(h)<5 else len(h)-4):.2f}")
    print(f"core E_inf   : {ei[-1]:.3e} -> {ei[0]:.3e}")
    print(f"full E_rms   : {ef[-1]:.3e} -> {ef[0]:.3e}  (min {ef.min():.3e})")
    print(f"gamma        : {g[-1]:.3e} -> {g[0]:.3e}   slope={slope(h,g):.2f}")
    print(f"||v_n||_H    : {nrm.min():.2f} .. {nrm.max():.2f}")
    print(f"CPU [s]      : {cpu.min():.1f} .. {cpu.max():.1f}")
    print(f"success      : {[r['success'] for r in rs]}")
    print(f"n_iter       : {[int(r['n_iter']) for r in rs]}")

rs = sorted(rows("eps_d1_all.csv"), key=lambda r: float(r["eps"]))
print("\n=== bandwidth d=1 ===")
for r in rs:
    print(f"  eps={float(r['eps']):.3g}  gamma={float(r['gamma']):.3e} "
          f"|v_n|_H={float(r['nrm_H']):.2f} core_rms={float(r['E_rms_core']):.3e} "
          f"core_inf={float(r['E_inf_core']):.3e} full_rms={float(r['E_rms']):.3e}")
best = min(rs, key=lambda r: float(r["E_rms_core"]))
print(f"  best eps = {best['eps']}  core_rms={float(best['E_rms_core']):.3e}")

print("\n=== truncation radius d=1 ===")
for r in sorted(rows("domain_d1.csv"), key=lambda r: float(r["R"])):
    print(f"  R={float(r['R']):.0f} N={r['N']} h={float(r['h']):.4f} "
          f"gamma={float(r['gamma']):.3e} core_rms={float(r['E_rms_core']):.3e} "
          f"full_rms={float(r['E_rms']):.3e}")
