"""Monotone finite-difference baseline for the d=1 manufactured test problem.

The FD solution is evaluated on exactly the same space-time grids and with exactly
the same metrics as the kernel realization (core / full).  Two boundary treatments:
  - "oracle": Dirichlet data taken from the analytical solution;
  - "extrap": linear extrapolation at |x|=R, i.e. no knowledge of the exact solution.

After time reversal tau = T - t, w(tau,x) = v(T-tau,x) solves
    d_tau w = min(a_lo w_xx, a_hi w_xx) + min(b w_x, -b w_x) + f(T-tau, x),
a_lo = SIG_LO^2/2, a_hi = SIG_HI^2/2, b = B_BAR.  The scheme uses central second
differences inside the min and upwind first differences (forward for +b, backward
for -b), explicit Euler in tau, and the CFL condition dt (2 a_hi/h^2 + b/h) <= 1,
which makes it monotone.
"""
from __future__ import annotations
import csv, time
import numpy as np

import hjb_problem as P

T_END = P.T


def march(Nx: int, R: float = 3.0, T: float = T_END, bc: str = "oracle"):
    x = np.linspace(-R, R, Nx)
    h = x[1] - x[0]
    a_lo, a_hi, b = 0.5 * P.SIG_LO ** 2, 0.5 * P.SIG_HI ** 2, P.B_BAR
    dt_cfl = 0.9 * h * h / (2.0 * a_hi + b * h)
    n_steps = int(np.ceil(T / dt_cfl)) + 1
    dt = T / n_steps
    w = P.terminal(x[:, None])
    W = np.empty((n_steps + 1, Nx)); W[0] = w
    t0 = time.perf_counter()
    for k in range(n_steps):
        t_phys = T - k * dt
        f = P.source(np.column_stack([np.full(Nx, t_phys), x]))
        d_xx = np.zeros_like(w); d_p = np.zeros_like(w); d_m = np.zeros_like(w)
        d_xx[1:-1] = (w[2:] - 2.0 * w[1:-1] + w[:-2]) / (h * h)
        d_p[1:-1] = (w[2:] - w[1:-1]) / h
        d_m[1:-1] = (w[1:-1] - w[:-2]) / h
        rhs = np.minimum(a_lo * d_xx, a_hi * d_xx) + np.minimum(b * d_p, -b * d_m) + f
        w_new = w + dt * rhs
        t_new = T - (k + 1) * dt
        if bc == "oracle":
            ends = P.exact(np.array([[t_new, x[0]], [t_new, x[-1]]]))
            w_new[0], w_new[-1] = ends
        elif bc == "extrap":
            w_new[0] = 2.0 * w_new[1] - w_new[2]
            w_new[-1] = 2.0 * w_new[-2] - w_new[-3]
        else:
            raise ValueError(bc)
        w = w_new; W[k + 1] = w
    cpu = time.perf_counter() - t0
    taus = np.linspace(0.0, T, n_steps + 1)
    return x, taus, W, cpu


def bilinear(x, taus, W, t_pts, x_pts, T: float = T_END):
    """Evaluate the FD solution v(t,x)=w(T-t,x) at scattered (t,x)."""
    tau = np.clip(T - t_pts, taus[0], taus[-1])
    xs = np.clip(x_pts, x[0], x[-1])
    dtau = taus[1] - taus[0]; hx = x[1] - x[0]
    i = np.clip(((tau - taus[0]) / dtau).astype(int), 0, len(taus) - 2)
    j = np.clip(((xs - x[0]) / hx).astype(int), 0, len(x) - 2)
    a = (tau - taus[i]) / dtau; bb = (xs - x[j]) / hx
    return ((1 - a) * ((1 - bb) * W[i, j] + bb * W[i, j + 1])
            + a * ((1 - bb) * W[i + 1, j] + bb * W[i + 1, j + 1]))


def metrics(x, taus, W, R_eval, n_t, n_x):
    tt = np.linspace(0.0, T_END, n_t); xx = np.linspace(-R_eval, R_eval, n_x)
    Tg, Xg = np.meshgrid(tt, xx, indexing="ij")
    t_pts = Tg.ravel(); x_pts = Xg.ravel()
    approx = bilinear(x, taus, W, t_pts, x_pts)
    exact = P.exact(np.column_stack([t_pts, x_pts]))
    err = approx - exact
    return float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err)))


def main():
    rows = []
    for bc in ("oracle", "extrap"):
        for Nx in (16, 32, 64, 128, 256, 512):
            x, taus, W, cpu = march(Nx, bc=bc)
            rc, ic = metrics(x, taus, W, 1.0, 21, 41)     # core, same as kernel
            rf, iff = metrics(x, taus, W, 3.0, 51, 101)   # full, same as kernel
            rows.append([bc, Nx, len(taus) - 1, f"{x[1]-x[0]:.4e}",
                         rc, ic, rf, iff, cpu])
            print(f"{bc:7s} Nx={Nx:4d} steps={len(taus)-1:6d} "
                  f"core_rms={rc:.3e} core_inf={ic:.3e} "
                  f"full_rms={rf:.3e} full_inf={iff:.3e} time={cpu:.3f}s", flush=True)
    with open("results_fd_compare_d1.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bc", "Nx", "n_steps", "h", "E_rms_core", "E_inf_core",
                     "E_rms_full", "E_inf_full", "cpu_time"])
        wr.writerows(rows)


if __name__ == "__main__":
    main()
