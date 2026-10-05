"""Is lambda > ||v||_H in the experiments?

||I_n[v]||_H is the smallest native-space norm among functions interpolating v on Gamma_n,
hence a lower bound for ||v||_H that increases under refinement and converges to it.
We refine until the sequence saturates.
"""
import numpy as np, time
from scipy.spatial.distance import cdist
import kernel as K, hjb_problem as P

K.set_kernel("C6")

def nodes_grid(d, Nt, Nx, R=3.0, T=1.0):
    ax = [np.linspace(0.0, T, Nt + 1)] + [np.linspace(-R, R, Nx)] * d
    g = np.meshgrid(*ax, indexing="ij")
    return np.column_stack([q.ravel() for q in g])

def interp_norm(nodes, c):
    Kmat = K.phi(cdist(nodes, nodes), c)
    v = P.exact(nodes)
    th = np.linalg.solve(Kmat + 1e-12 * np.eye(len(nodes)), v)
    return float(np.sqrt(max(0.0, th @ Kmat @ th)))

print("d=1 (lambda = 6)")
for c in (0.5, 1.0, 2.0):
    for (Nt, Nx) in ((32, 25), (32, 49), (48, 65), (64, 81)):
        t0 = time.perf_counter(); nd = nodes_grid(1, Nt, Nx)
        print(f"  c={c:<4} Nt={Nt:3d} Nx={Nx:3d} N={len(nd):5d}  ||I_n v||_H = {interp_norm(nd, c):.3f}"
              f"   [{time.perf_counter()-t0:.0f}s]", flush=True)
print("d=2 (lambda = 6)")
for c in (1.5,):
    for (Nt, Nx) in ((16, 13), (16, 15), (16, 17)):
        t0 = time.perf_counter(); nd = nodes_grid(2, Nt, Nx)
        print(f"  c={c:<4} Nt={Nt:3d} Nx={Nx:3d} N={len(nd):5d}  ||I_n v||_H = {interp_norm(nd, c):.3f}"
              f"   [{time.perf_counter()-t0:.0f}s]", flush=True)
