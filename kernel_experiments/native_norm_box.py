"""Does enlarging the box change the minimal-norm interpolant? (c=0.5, d=1, tightest case)

||I_n[v]||_H over a box B is the smallest native norm among functions agreeing with v on the
grid in B, so it increases with B and stays below the norm of any global extension of v.
Same mesh spacing, two boxes.
"""
import numpy as np, time
from scipy.spatial.distance import cdist
import kernel as K, hjb_problem as P
K.set_kernel("C6")

def norm_on(tlo, thi, xlo, xhi, ht, hx, c):
    t = np.arange(tlo, thi + 1e-9, ht); x = np.arange(xlo, xhi + 1e-9, hx)
    Tg, Xg = np.meshgrid(t, x, indexing="ij")
    nd = np.column_stack([Tg.ravel(), Xg.ravel()])
    Kmat = K.phi(cdist(nd, nd), c)
    th = np.linalg.solve(Kmat + 1e-12 * np.eye(len(nd)), P.exact(nd))
    return len(nd), float(np.sqrt(max(0.0, th @ Kmat @ th)))

c, ht, hx = 0.5, 1/32, 0.1
for (tlo, thi, xlo, xhi) in ((0, 1, -3, 3), (-0.25, 1.25, -4.5, 4.5), (-0.5, 1.5, -6, 6)):
    t0 = time.perf_counter(); N, nrm = norm_on(tlo, thi, xlo, xhi, ht, hx, c)
    print(f"box t in [{tlo},{thi}], x in [{xlo},{xhi}]: N={N:6d}  ||I_n v||_H = {nrm:.3f}"
          f"   [{time.perf_counter()-t0:.0f}s]", flush=True)
