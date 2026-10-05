"""
Kernel-based collocation for the manufactured problem in hjb_problem.py:

    d_t v + inf_{(s,beta)} { (1/2) s^2 Lap_x v + beta (1/d) sum_i d_{x_i} v + f } = 0,
    (s,beta) in [0.1,0.4] x [-1,1],  exact v = sin(t + sum_i x_i) exp(-|x|^2/4);
see hjb_problem.py.

Approximate solution v_n(y) = sum_j theta_j Phi(y - y^j) with a compactly supported
Wendland radial basis function; see the KERNEL block below.  The default profile is
    phi(r) = (1 - r/eps)_+^8 (32(r/eps)^3 + 25(r/eps)^2 + 8(r/eps) + 1),
which is C^6 and positive definite on R^m for m <= 3, with native space H^tau,
tau = 4.5 (m=2) resp. 5 (m=3), as required by the convergence theory.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

import hjb_problem as P


# ---------------------------------------------------------------------------
# Wendland kernels on R^{d+1} (space-time).
#
# The convergence theory requires the native space to be H^tau(R^{d+1}) with
#     tau > 3 + (d+1)/2,
# so that the Sobolev embedding H^tau -> C^3 holds.  For a compactly supported
# radial profile whose one-dimensional Fourier transform decays like |xi|^{-s},
# the m-dimensional transform decays like |xi|^{-(s+m-1)}, i.e. tau = (s+m-1)/2.
#
#   "C4" : phi_{1,2}(r) = (1-r)_+^5 (8r^2+5r+1)          s = 6
#            m = 2 -> tau = 3.5   (need > 4.0)   INSUFFICIENT
#            m = 3 -> tau = 4.0   (need > 4.5)   INSUFFICIENT
#   "C6" : phi_{3,3}(r) = (1-r)_+^8 (32r^3+25r^2+8r+1)   s = 8
#            m = 2 -> tau = 4.5   (need > 4.0)   OK
#            m = 3 -> tau = 5.0   (need > 4.5)   OK
#
# "C6" is therefore the default; "C4" is retained only for comparison.
# ---------------------------------------------------------------------------

KERNEL = "C6"


def set_kernel(name: str) -> None:
    """Select the Wendland profile globally ("C4" or "C6")."""
    global KERNEL
    if name not in ("C4", "C6"):
        raise ValueError(f"unknown kernel {name!r}; use 'C4' or 'C6'")
    KERNEL = name


def phi(r: np.ndarray, eps: float) -> np.ndarray:
    """Wendland kernel; r is Euclidean distance in R^{d+1}. phi(0) = 1."""
    s = r / eps
    inside = s <= 1.0
    sc = np.minimum(s, 1.0)
    if KERNEL == "C4":
        val = (1.0 - sc) ** 5 * (8.0 * sc ** 2 + 5.0 * sc + 1.0)
    else:  # C6
        val = (1.0 - sc) ** 8 * (32.0 * sc ** 3 + 25.0 * sc ** 2 + 8.0 * sc + 1.0)
    return np.where(inside, val, 0.0)


def phi1(r: np.ndarray, eps: float) -> np.ndarray:
    """phi^{(1)}(r) = phi'(r)/r, extended by its limit at r=0."""
    s = np.minimum(r / eps, 1.0)
    inside = (r / eps) <= 1.0
    if KERNEL == "C4":
        val = -(14.0 / eps ** 2) * (1.0 - s) ** 4 * (4.0 * s + 1.0)
    else:  # C6
        val = -(22.0 / eps ** 2) * (1.0 - s) ** 7 * (16.0 * s ** 2 + 7.0 * s + 1.0)
    return np.where(inside, val, 0.0)


def phi2(r: np.ndarray, eps: float) -> np.ndarray:
    """phi^{(2)}(r) = (1/r) d phi^{(1)}/dr, extended by its limit at r=0."""
    s = np.minimum(r / eps, 1.0)
    inside = (r / eps) <= 1.0
    if KERNEL == "C4":
        val = (280.0 / eps ** 4) * (1.0 - s) ** 3
    else:  # C6
        val = (528.0 / eps ** 4) * (1.0 - s) ** 6 * (6.0 * s + 1.0)
    return np.where(inside, val, 0.0)


# ---------------------------------------------------------------------------
# Build the matrices needed for the residual / native-space norm
# ---------------------------------------------------------------------------


def build_eval_matrices(
    nodes: np.ndarray,
    eval_pts: np.ndarray,
    eps: float,
) -> dict:
    """Build evaluation and derivative matrices, without forming the Gram matrix.

    The first column of nodes/eval_pts is the time coordinate; the remaining
    d columns are the spatial coordinates.

    Returns:
      Phi_eval   (M, N):       Phi(y_eval^l - y^j)
      D_t        (M, N):       partial_t Phi(y_eval^l - y^j)
      Sum_Dx     (M, N):       sum_i partial_{x_i} Phi(y_eval^l - y^j)
      Lap_x      (M, N):       sum_i partial^2_{x_i x_i} Phi(y_eval^l - y^j)
                                = d * phi1 + phi2 * |y_x|^2
    """
    diff = eval_pts[:, None, :] - nodes[None, :, :]   # (M, N, d+1)
    R_MN = np.sqrt(np.sum(diff ** 2, axis=-1))         # (M, N)
    Phi_eval = phi(R_MN, eps)

    # phi^{(1)} and phi^{(2)}
    P1 = phi1(R_MN, eps)
    P2 = phi2(R_MN, eps)

    # partial_t Phi = phi^{(1)} * y_0
    D_t = P1 * diff[:, :, 0]

    # sum_i partial_{x_i} Phi = phi^{(1)} * sum_i y_i_x
    diff_x = diff[:, :, 1:]                            # (M, N, d)
    sum_x = np.sum(diff_x, axis=-1)                    # (M, N)
    Sum_Dx = P1 * sum_x

    # Laplacian over spatial coords:
    # Lap_x Phi = sum_i [phi^{(1)} + phi^{(2)} * y_i_x^2]
    #          = d * phi^{(1)} + phi^{(2)} * |y_x|^2
    d_spatial = diff_x.shape[-1]
    norm_x_sq = np.sum(diff_x ** 2, axis=-1)            # (M, N)
    Lap_x = d_spatial * P1 + P2 * norm_x_sq

    return dict(
        Phi_eval=Phi_eval, D_t=D_t, Sum_Dx=Sum_Dx, Lap_x=Lap_x, R_MN=R_MN, d=d_spatial,
        f=P.source(eval_pts),
    )


def build_matrices(
    nodes: np.ndarray,        # (N, d+1): collocation nodes y^j = (t^j, x^j)
    eval_pts: np.ndarray,     # (M, d+1): residual evaluation points
    eps: float,
) -> dict:
    """Build the Gram, evaluation, and derivative matrices used in optimization."""
    mats = build_eval_matrices(nodes, eval_pts, eps)
    mats["K"] = phi(cdist(nodes, nodes), eps)
    return mats


# ---------------------------------------------------------------------------
# Residual / terminal evaluators
# ---------------------------------------------------------------------------


def residual(theta: np.ndarray, mats: dict) -> np.ndarray:
    """PDE residual r(v_theta) = d_t v + inf_A{...} + f at the M evaluation points."""
    vt = mats["D_t"] @ theta
    vsum_x = mats["Sum_Dx"] @ theta
    vlap = mats["Lap_x"] @ theta
    return vt + P.hamiltonian(vlap, vsum_x, mats["d"]) + mats["f"]


def terminal_mismatch(theta: np.ndarray, Phi_T: np.ndarray, g: np.ndarray) -> np.ndarray:
    return Phi_T @ theta - g


# ---------------------------------------------------------------------------
# Optimization (Pn) via L-BFGS-B
# ---------------------------------------------------------------------------


@dataclass
class OptConfig:
    d: int = 1            # spatial dimension
    Nt: int = 16          # number of time intervals (Nt+1 time grid points)
    Nx: int = 16          # number of grid points per spatial axis
    N_extra: int = 100
    R: float = 3.0
    T: float = 1.0
    eps: float = 0.5
    lam: float = 6.0
    mu_native: float = 1.0e2
    max_iter: int = 5000
    seed: int = 0
    verbose: bool = True


@dataclass
class OptResult:
    theta: np.ndarray
    gamma: float
    iter_count: int
    cpu_time: float
    n_eval_pts: int


def make_grid_with_extra(cfg: OptConfig) -> tuple[np.ndarray, np.ndarray]:
    """Returns (nodes, eval_pts) in dimension (d+1) for spatial dim cfg.d.

    Nodes Gamma_n: (Nt+1) tensor (Nx)^d points on [0,T] x [-R,R]^d.
    Extra eval points Gamma^extra: N_extra Sobol' samples on [0, T-eps0] x [-R,R]^d.
    """
    d = cfg.d
    t_grid = np.linspace(0.0, cfg.T, cfg.Nt + 1)
    x_axes = [np.linspace(-cfg.R, cfg.R, cfg.Nx)] * d
    grids = np.meshgrid(t_grid, *x_axes, indexing="ij")  # list of (Nt+1, Nx, ..., Nx)
    nodes = np.column_stack([g.ravel() for g in grids])  # (N, d+1)

    from scipy.stats import qmc
    sob = qmc.Sobol(d=d + 1, scramble=True, seed=cfg.seed)
    sob.fast_forward(1000)
    raw = sob.random(cfg.N_extra)
    eps0 = 1.0e-8
    cols = [raw[:, 0] * (cfg.T - eps0)]
    for k in range(d):
        cols.append((raw[:, 1 + k] * 2.0 - 1.0) * cfg.R)
    extra = np.column_stack(cols)

    eval_pts = np.vstack([nodes, extra])
    return nodes, eval_pts


def initial_guess(nodes: np.ndarray, mats: dict, lam: float) -> np.ndarray:
    """Solve the collocation system K theta = v_init at the nodes,
    with v_init = g(x) (terminal data) extended constantly in t."""
    v_init = P.terminal(nodes[:, 1:])
    K = mats["K"]
    N = K.shape[0]
    theta = np.linalg.solve(K + 1e-8 * np.eye(N), v_init)
    nrm2 = float(theta @ K @ theta)
    if nrm2 > lam ** 2:
        theta *= lam / math.sqrt(nrm2)
    return theta


def solve_Pn(cfg: OptConfig) -> dict:
    """Solve (Pn) by penalty + L-BFGS-B.

    Reformulation: minimize over theta the piecewise-smooth objective
        J(theta) = (1/M) sum_l |r_l(theta)|^2 + (1/M_T) sum_l |term_l(theta)|^2
                 + mu * max(0, theta^T K theta - lam^2)^2,
    then report gamma = max( max |r|, max |term| ).
    """
    nodes, eval_pts = make_grid_with_extra(cfg)
    N = nodes.shape[0]
    M = eval_pts.shape[0]

    mats = build_matrices(nodes, eval_pts, cfg.eps)
    Kmat = mats["K"]

    # Terminal points: the unique terminal slice of the tensor grid plus the
    # spatial projections of the extra Sobol points.  Projecting every tensor
    # node would repeat each terminal grid point Nt+1 times and distort weights.
    terminal_grid = np.isclose(nodes[:, 0], cfg.T)
    term_grid_pts = nodes[terminal_grid]
    extra_pts = eval_pts[N:]
    extra_term_pts = np.column_stack(
        [np.full(extra_pts.shape[0], cfg.T), extra_pts[:, 1:]]
    )
    term_pts = np.vstack([term_grid_pts, extra_term_pts])
    diff_T = term_pts[:, None, :] - nodes[None, :, :]
    R_T = np.sqrt(np.sum(diff_T ** 2, axis=-1))
    Phi_T = phi(R_T, cfg.eps)
    g_target = P.terminal(term_pts[:, 1:])
    M_T = term_pts.shape[0]

    mu_native = cfg.mu_native

    d = cfg.d
    a_lo, a_hi = 0.5 * P.SIG_LO ** 2, 0.5 * P.SIG_HI ** 2

    def loss_and_grad(theta):
        vt = mats["D_t"] @ theta
        vsum_x = mats["Sum_Dx"] @ theta
        vlap = mats["Lap_x"] @ theta

        r = vt + P.hamiltonian(vlap, vsum_x, d) + mats["f"]
        l_res = float(np.mean(r ** 2))

        term = Phi_T @ theta - g_target
        l_term = float(np.mean(term ** 2))

        nrm2 = float(theta @ Kmat @ theta)
        excess = max(0.0, nrm2 - cfg.lam ** 2)
        l_pen = mu_native * excess ** 2

        loss = l_res + l_term + l_pen

        # Gradient: d/dtheta of (1/M) sum r^2 = (2/M) (Jr^T r)
        coef_lap = np.where(vlap > 0, a_lo, a_hi)[:, None]
        coef_dx = -(P.B_BAR / d) * np.sign(vsum_x)[:, None]
        Jr = mats["D_t"] + coef_lap * mats["Lap_x"] + coef_dx * mats["Sum_Dx"]
        g_res = (2.0 / M) * (Jr.T @ r)

        g_term = (2.0 / M_T) * (Phi_T.T @ term)

        if excess > 0:
            g_pen = mu_native * 2.0 * excess * 2.0 * (Kmat @ theta)
        else:
            g_pen = np.zeros_like(theta)

        grad = g_res + g_term + g_pen
        return loss, grad

    # Initial point
    theta0 = initial_guess(nodes, mats, cfg.lam)

    t0 = time.perf_counter()
    res = minimize(
        loss_and_grad,
        theta0,
        jac=True,
        method="L-BFGS-B",
        # scipy's default maxfun is 15000 and would silently truncate the run
        # long before maxiter; tie it to the iteration budget instead.
        options=dict(maxiter=cfg.max_iter, maxfun=10 * cfg.max_iter,
                     gtol=1e-7, ftol=1e-10),
    )
    cpu_time = time.perf_counter() - t0

    theta_opt = res.x

    # Compute two post-hoc residual bounds.  gamma is exactly the quantity in
    # (3.gamma): tensor-grid residuals and the terminal slice of that grid.
    # gamma_train also includes the Sobol residual points and their projections
    # to t=T, and is retained as an optimization diagnostic.
    r_final = residual(theta_opt, mats)
    term_final = Phi_T @ theta_opt - g_target
    term_grid_mats = build_eval_matrices(nodes, term_grid_pts, cfg.eps)
    term_grid = term_grid_mats["Phi_eval"] @ theta_opt - P.terminal(term_grid_pts[:, 1:])
    gamma = float(max(np.max(np.abs(r_final[:N])), np.max(np.abs(term_grid))))
    gamma_train = float(max(np.max(np.abs(r_final)), np.max(np.abs(term_final))))
    loss_grid = float(np.mean(r_final[:N] ** 2) + np.mean(term_grid ** 2))

    return dict(
        theta=theta_opt,
        gamma=gamma,
        gamma_train=gamma_train,
        loss_grid=loss_grid,
        n_iter=int(res.nit),
        cpu_time=cpu_time,
        success=res.success,
        message=str(res.message),
        n_nodes=N,
        n_eval=M,
        nodes=nodes,
        eval_pts=eval_pts,
        Phi_T=Phi_T,
        g_target=g_target,
        mats=mats,
        final_loss=float(res.fun),
        native_norm=float(np.sqrt(theta_opt @ Kmat @ theta_opt)),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_points(d: int, T: float, R: float, n_t: int, n_x: int) -> np.ndarray:
    """Uniform tensor-product grid on [0,T] x [-R,R]^d."""
    grids = np.meshgrid(np.linspace(0.0, T, n_t),
                        *([np.linspace(-R, R, n_x)] * d), indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


def core_points(d: int, T: float, R_core: float, n_t: int = 21,
                n_x: int = 41) -> np.ndarray:
    """Uniform grid on the fixed compact core [0,T] x [-R_core, R_core]^d."""
    return eval_points(d, T, R_core, n_t, 21 if d >= 2 else n_x)


def evaluate_at(theta: np.ndarray, nodes: np.ndarray, eps: float,
                pts: np.ndarray, chunk: int = 4000) -> dict:
    """Errors of v_n = sum_j theta_j Phi(. - y^j) at arbitrary points.

    Evaluated in chunks so that the (points x nodes x dim) difference array
    stays bounded for the large two-dimensional grids.
    """
    e_max, e_sq, n = 0.0, 0.0, 0
    for k in range(0, pts.shape[0], chunk):
        q = pts[k:k + chunk]
        diff = q[:, None, :] - nodes[None, :, :]
        Rm = np.sqrt(np.sum(diff ** 2, axis=-1))
        err = phi(Rm, eps) @ theta - P.exact(q)
        e_max = max(e_max, float(np.max(np.abs(err))))
        e_sq += float(np.sum(err ** 2))
        n += err.size
    return dict(E_inf=e_max, E_rms=float(np.sqrt(e_sq / n)))


def residual_inf_at(theta: np.ndarray, nodes: np.ndarray, eps: float,
                    pts: np.ndarray, chunk: int = 4000) -> float:
    """Maximum PDE residual on arbitrary points, evaluated in chunks."""
    res_max = 0.0
    for k in range(0, pts.shape[0], chunk):
        q = pts[k:k + chunk]
        mats = build_eval_matrices(nodes, q, eps)
        res_max = max(res_max, float(np.max(np.abs(residual(theta, mats)))))
    return res_max


def native_norm(theta: np.ndarray, mats: dict) -> float:
    return float(np.sqrt(max(0.0, theta @ mats["K"] @ theta)))


def evaluate(theta: np.ndarray, mats: dict, eval_pts: np.ndarray) -> dict:
    v_pred = mats["Phi_eval"] @ theta
    v_true = P.exact(eval_pts)
    err = v_pred - v_true
    E_inf = float(np.max(np.abs(err)))
    E_rms = float(np.sqrt(np.mean(err ** 2)))
    res = residual(theta, mats)
    Res_inf = float(np.max(np.abs(res)))
    return dict(E_inf=E_inf, E_rms=E_rms, Res_inf=Res_inf)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=1, help="spatial dimension")
    p.add_argument("--Nt", type=int, default=16)
    p.add_argument("--Nx", type=int, default=16)
    p.add_argument("--N_extra", type=int, default=100)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=6.0)
    p.add_argument("--mu", type=float, default=1e2)
    p.add_argument("--R", type=float, default=3.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=5000)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = OptConfig(
        d=args.d,
        Nt=args.Nt,
        Nx=args.Nx,
        N_extra=args.N_extra,
        R=args.R,
        T=args.T,
        eps=args.eps,
        lam=args.lam,
        mu_native=args.mu,
        max_iter=args.max_iter,
        verbose=not args.quiet,
    )
    out = solve_Pn(cfg)
    metrics = evaluate(out["theta"], out["mats"], out["eval_pts"])
    print()
    print(f"N={out['n_nodes']} M={out['n_eval']} gamma={out['gamma']:.4e}")
    print(f"E_rms={metrics['E_rms']:.4e} E_inf={metrics['E_inf']:.4e} Res_inf={metrics['Res_inf']:.4e}")
    print(f"iters={out['n_iter']} CPU={out['cpu_time']:.1f}s success={out['success']}")
