"""Minimal-norm interpolant over [0,T] x [-X,X] with X growing (t range fixed at [0,T]).

The theorem needs an element of H whose restriction to [0,T] x R^d is v; the minimal norm of
such an element is the limit of ||I_n[v]||_H over grids that become dense in [0,T] x R^d.
Enlarging the t-range instead would interpolate the (non-decaying, hence not in H) continuation
of the formula in t, which is not what the hypothesis asks for.
"""
import numpy as np, time
from scipy.spatial.distance import cdist
import kernel as K, hjb_problem as P
K.set_kernel("C6")

def norm_on(X, ht, hx, c):
    t = np.arange(0.0, 1.0 + 1e-9, ht); x = np.arange(-X, X + 1e-9, hx)
    Tg, Xg = np.meshgrid(t, x, indexing="ij")
    nd = np.column_stack([Tg.ravel(), Xg.ravel()])
    Kmat = K.phi(cdist(nd, nd), c)
    th = np.linalg.solve(Kmat + 1e-12 * np.eye(len(nd)), P.exact(nd))
    return len(nd), float(np.sqrt(max(0.0, th @ Kmat @ th)))

for c in (0.5, 1.0):
    for (X, ht, hx) in ((3, 1/32, 0.1), (4.5, 1/32, 0.1), (6, 1/32, 0.1), (6, 1/48, 0.075)):
        t0 = time.perf_counter(); N, nrm = norm_on(X, ht, hx, c)
        print(f"c={c}  x in [-{X},{X}], ht={ht:.4f}, hx={hx}: N={N:6d}  ||I_n v||_H = {nrm:.3f}"
              f"   [{time.perf_counter()-t0:.0f}s]", flush=True)
