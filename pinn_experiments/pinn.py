"""
Sobolev-regularized PINN for the d=1 manufactured HJB test problem:

    d_t v + inf_{(s,beta)} { (1/2) s^2 d_xx v + beta d_x v + f(t,x) } = 0,
    (s,beta) in [SIG_LO,SIG_HI] x [-B_BAR,B_BAR],
    exact v(t,x) = sin(t + x) exp(-x^2/ELL^2),  v(1,x) = g(x),

with the manufactured source f of ../kernel_experiments/hjb_problem.py.  The infimum is
    (1/2) SIG_LO^2 (v_xx)^+ - (1/2) SIG_HI^2 (v_xx)^- - B_BAR |v_x|.

Loss: empirical L^2 residual + L^2 terminal mismatch + beta * (product of Frobenius norms)^2.

Two-phase training: Adam for global descent, then optional L-BFGS for local refinement.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class TanhNet(nn.Module):
    """Feedforward tanh network of fixed width."""

    def __init__(self, d: int, width: int, depth: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Linear] = []
        prev = 1 + d
        for _ in range(depth):
            layers.append(nn.Linear(prev, width))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.linears = nn.ModuleList(layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([t, x], dim=-1)
        for i, lin in enumerate(self.linears):
            z = lin(z)
            if i < len(self.linears) - 1:
                z = torch.tanh(z)
        return z

    def path_norm(self) -> torch.Tensor:
        """Product of Frobenius norms of weight matrices."""
        p = torch.tensor(1.0, device=next(self.parameters()).device)
        for lin in self.linears:
            p = p * torch.linalg.norm(lin.weight, ord="fro")
        return p


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------


SIG_LO, SIG_HI, B_BAR, ELL = 0.1, 0.4, 1.0, 2.0   # keep in sync with hjb_problem.py


def _exact_parts(t: torch.Tensor, x: torch.Tensor):
    """(v, v_t, v_x, v_xx) of the exact solution, d = 1."""
    ph = t + x
    E = torch.exp(-x ** 2 / ELL ** 2)
    s, c = torch.sin(ph), torch.cos(ph)
    v = s * E
    vt = c * E
    vx = E * (c - 2.0 * x / ELL ** 2 * s)
    vxx = E * (-4.0 * x / ELL ** 2 * c + (4.0 * x ** 2 / ELL ** 4 - 1.0 - 2.0 / ELL ** 2) * s)
    return v, vt, vx, vxx


def _hamiltonian(vxx: torch.Tensor, vx: torch.Tensor) -> torch.Tensor:
    return (0.5 * SIG_LO ** 2 * torch.relu(vxx) - 0.5 * SIG_HI ** 2 * torch.relu(-vxx)
            - B_BAR * vx.abs())


def source(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    _, vt, vx, vxx = _exact_parts(t, x)
    return -vt - _hamiltonian(vxx, vx)


def pde_residual(net: TanhNet, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Pointwise PDE residual r(v_theta) at (t, x), shape (M,). Specialized to d=1."""
    assert x.shape[-1] == 1
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)
    v = net(t, x)
    grad_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    grad_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    grad_xx = torch.autograd.grad(grad_x.sum(), x, create_graph=True)[0]
    with torch.no_grad():
        f = source(t.detach(), x.detach())
    res = grad_t + _hamiltonian(grad_xx, grad_x) + f
    return res.squeeze(-1)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_interior(M: int, R: float, T: float, device, generator):
    t = torch.rand(M, 1, generator=generator, device=device) * T
    x = (torch.rand(M, 1, generator=generator, device=device) * 2.0 - 1.0) * R
    return t, x


def sample_terminal(M: int, R: float, device, generator):
    return (torch.rand(M, 1, generator=generator, device=device) * 2.0 - 1.0) * R


def exact_v(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _exact_parts(t, x)[0]


def exact_g(x: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return _exact_parts(torch.full_like(x, T), x)[0]


# ---------------------------------------------------------------------------
# Configuration & Result
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Architecture
    width: int = 64
    depth: int = 3

    # Sampling
    M: int = 4096
    M_term: int = 1024
    R: float = 3.0
    T: float = 1.0
    resample_every: int = 1  # 1 = new sample at every iter; >1 = reuse

    # Loss
    beta: float = 1.0e-6

    # Adam phase
    n_iter_adam: int = 50000
    lr0: float = 1.0e-3
    lr_halve_every: int = 10000

    # L-BFGS phase (optional)
    n_iter_lbfgs: int = 0  # 0 = skip
    lbfgs_max_iter: int = 20  # inner iterations per step

    # Validation
    val_every: int = 1000
    val_size: int = 5000

    # Misc
    seed: int = 0
    device: str = "cpu"  # "cpu", "cuda", "mps". Default is "cpu":
                          # MPS currently has correctness issues with second-order
                          # autograd through torch.relu / torch.minimum.
    verbose: bool = True
    log_every: Optional[int] = None  # if None, equals val_every


@dataclass
class Result:
    cfg: TrainConfig
    E_inf: float
    E_rms: float
    E_inf_core: float
    E_rms_core: float
    E_T_inf: float
    Res_inf: float
    path_norm: float
    final_loss: float
    cpu_time: float
    cpu_time_adam: float
    cpu_time_lbfgs: float
    n_params: int
    history: list = field(default_factory=list)
    state_dict: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def select_device(spec: str) -> torch.device:
    """Resolve a device specification.

    "auto" picks CUDA if available, otherwise CPU. MPS is not selected automatically
    because second-order autograd through torch.relu / torch.minimum is unreliable on
    current PyTorch+MPS; specify "mps" explicitly if you have verified your version
    handles the residual correctly.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(spec)


def make_net(cfg: TrainConfig, device) -> TanhNet:
    torch.manual_seed(cfg.seed)
    net = TanhNet(d=1, width=cfg.width, depth=cfg.depth).to(device)
    return net


def uniform_grid(device, T: float, R: float, n_t: int, n_x: int):
    """Deterministic uniform tensor-product grid on [0,T] x [-R,R]."""
    tt = torch.linspace(0.0, T, n_t, device=device)
    xx = torch.linspace(-R, R, n_x, device=device)
    T_, X_ = torch.meshgrid(tt, xx, indexing="ij")
    return T_.reshape(-1, 1), X_.reshape(-1, 1)


def evaluate_fixed(net: TanhNet, cfg: TrainConfig, device):
    """Final reported metrics on deterministic grids.

    The full-cylinder grid is 51 x 101 on [0,T] x [-R,R]; the core grid is the
    21 x 41 grid on [0,T] x [-1,1] also used for the kernel realization, so that
    the two realizations are compared on identical sets.
    """
    out = {}
    for tag, (R_, n_t, n_x) in (("", (cfg.R, 51, 101)), ("_core", (1.0, 21, 41))):
        t, x = uniform_grid(device, cfg.T, R_, n_t, n_x)
        with torch.no_grad():
            err = (net(t, x).squeeze(-1) - exact_v(t, x).squeeze(-1)).abs()
        out["E_inf" + tag] = err.max().item()
        out["E_rms" + tag] = (err ** 2).mean().sqrt().item()
    x_T = torch.linspace(-cfg.R, cfg.R, 1001, device=device).reshape(-1, 1)
    with torch.no_grad():
        eT = (net(torch.full_like(x_T, cfg.T), x_T).squeeze(-1)
              - exact_g(x_T, cfg.T).squeeze(-1)).abs()
    out["E_T_inf"] = eT.max().item()
    t, x = uniform_grid(device, cfg.T, cfg.R, 51, 101)
    out["Res_inf"] = pde_residual(net, t, x).detach().abs().max().item()
    return out


def core_grid(device, T: float = 1.0, R_core: float = 1.0,
              n_t: int = 21, n_x: int = 41):
    """Deterministic uniform grid on the fixed compact core [0,T] x [-R_core,R_core].

    Identical to the core used for the kernel realization, so that the two
    realizations are compared on the same set.
    """
    tt = torch.linspace(0.0, T, n_t, device=device)
    xx = torch.linspace(-R_core, R_core, n_x, device=device)
    T_, X_ = torch.meshgrid(tt, xx, indexing="ij")
    return T_.reshape(-1, 1), X_.reshape(-1, 1)


def evaluate_core(net: TanhNet, cfg: TrainConfig, device):
    t, x = core_grid(device, cfg.T)
    with torch.no_grad():
        err = (net(t, x).squeeze(-1) - exact_v(t, x).squeeze(-1)).abs()
    return err.max().item(), (err ** 2).mean().sqrt().item()


def make_validation_set(cfg: TrainConfig, device, gen):
    """Generate the fixed held-out sample used for iterate selection."""
    t, x = sample_interior(cfg.val_size, cfg.R, cfg.T, device, gen)
    x_T = sample_terminal(cfg.val_size, cfg.R, device, gen)
    return t, x, x_T


def evaluate(net: TanhNet, cfg: TrainConfig, device, validation_set):
    t, x, x_T = validation_set

    with torch.no_grad():
        v = net(t, x).squeeze(-1)
        v_true = exact_v(t, x).squeeze(-1)
        v_T = net(torch.full_like(x_T, cfg.T), x_T).squeeze(-1)
        g_true = exact_g(x_T, cfg.T).squeeze(-1)
        E_inf = (v - v_true).abs().max().item()
        E_rms = ((v - v_true) ** 2).mean().sqrt().item()
        E_T_inf = (v_T - g_true).abs().max().item()

    res = pde_residual(net, t, x).detach()
    Res_inf = res.abs().max().item()
    E_inf_core, E_rms_core = evaluate_core(net, cfg, device)
    return E_inf, E_rms, E_inf_core, E_rms_core, E_T_inf, Res_inf


def compute_loss(
    net: TanhNet, cfg: TrainConfig, device, gen, t=None, x=None, x_T=None
):
    """Returns (total_loss, (l_res, l_term, l_path), pn)."""
    if t is None:
        t, x = sample_interior(cfg.M, cfg.R, cfg.T, device, gen)
    if x_T is None:
        x_T = sample_terminal(cfg.M_term, cfg.R, device, gen)
    res = pde_residual(net, t, x)
    l_res = (res ** 2).mean()
    v_T = net(torch.full_like(x_T, cfg.T), x_T).squeeze(-1)
    g_t = exact_g(x_T, cfg.T).squeeze(-1)
    l_term = ((v_T - g_t) ** 2).mean()
    pn = net.path_norm()
    l_path = pn ** 2
    return l_res + l_term + cfg.beta * l_path, (l_res, l_term, l_path), pn


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> Result:
    device = select_device(cfg.device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    val_gen = torch.Generator(device=device).manual_seed(cfg.seed + 10000)
    validation_set = make_validation_set(cfg, device, val_gen)

    net = make_net(cfg, device)
    n_params = sum(p.numel() for p in net.parameters())

    log_every = cfg.log_every or cfg.val_every
    history = []
    best = {"rms": math.inf, "state": None}

    # ---- Adam phase ---------------------------------------------------------
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr0)
    t_adam0 = time.perf_counter()
    cached_t = cached_x = cached_xT = None

    for it in range(1, cfg.n_iter_adam + 1):
        if cfg.resample_every <= 1 or (it - 1) % cfg.resample_every == 0:
            cached_t, cached_x = sample_interior(cfg.M, cfg.R, cfg.T, device, gen)
            cached_xT = sample_terminal(cfg.M_term, cfg.R, device, gen)
        loss, parts, pn = compute_loss(
            net, cfg, device, gen, cached_t, cached_x, cached_xT
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if cfg.lr_halve_every and it % cfg.lr_halve_every == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.5

        if it % log_every == 0:
            E_inf, E_rms, E_inf_core, E_rms_core, E_T_inf, Res_inf = evaluate(
                net, cfg, device, validation_set
            )
            history.append(
                dict(
                    phase="adam",
                    it=it,
                    loss=loss.item(),
                    l_res=parts[0].item(),
                    l_term=parts[1].item(),
                    pn=pn.item(),
                    E_rms=E_rms,
                    E_inf_core=E_inf_core,
                    E_rms_core=E_rms_core,
                    E_inf=E_inf,
                    Res_inf=Res_inf,
                )
            )
            if E_rms < best["rms"]:
                best["rms"] = E_rms
                best["state"] = {
                    k: v.detach().clone() for k, v in net.state_dict().items()
                }
            if cfg.verbose:
                print(
                    f"[adam] it={it:7d} loss={loss.item():.3e} l_res={parts[0].item():.3e} "
                    f"l_term={parts[1].item():.3e} pn={pn.item():.2e} "
                    f"E_rms={E_rms:.3e} E_inf={E_inf:.3e}",
                    flush=True,
                )

    cpu_time_adam = time.perf_counter() - t_adam0

    # ---- L-BFGS phase (optional) -------------------------------------------
    cpu_time_lbfgs = 0.0
    if cfg.n_iter_lbfgs > 0:
        t_lbfgs0 = time.perf_counter()
        opt = torch.optim.LBFGS(
            net.parameters(),
            lr=1.0,
            max_iter=cfg.lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss, _, _ = compute_loss(
                net, cfg, device, gen, cached_t, cached_x, cached_xT
            )
            loss.backward()
            return loss

        for it in range(1, cfg.n_iter_lbfgs + 1):
            cached_t, cached_x = sample_interior(cfg.M, cfg.R, cfg.T, device, gen)
            cached_xT = sample_terminal(cfg.M_term, cfg.R, device, gen)
            loss = opt.step(closure)
            if it % max(cfg.n_iter_lbfgs // 10, 1) == 0:
                E_inf, E_rms, E_inf_core, E_rms_core, E_T_inf, Res_inf = evaluate(
                    net, cfg, device, validation_set
                )
                history.append(
                    dict(
                        phase="lbfgs",
                        it=it,
                        loss=loss.item(),
                        E_rms=E_rms,
                    E_inf_core=E_inf_core,
                    E_rms_core=E_rms_core,
                        E_inf=E_inf,
                        Res_inf=Res_inf,
                    )
                )
                if E_rms < best["rms"]:
                    best["rms"] = E_rms
                    best["state"] = {
                        k: v.detach().clone() for k, v in net.state_dict().items()
                    }
                if cfg.verbose:
                    print(
                        f"[lbfgs] it={it:5d} loss={loss.item():.3e} "
                        f"E_rms={E_rms:.3e} E_inf={E_inf:.3e}",
                        flush=True,
                    )

        cpu_time_lbfgs = time.perf_counter() - t_lbfgs0

    cpu_time = cpu_time_adam + cpu_time_lbfgs

    if best["state"] is not None:
        net.load_state_dict(best["state"])

    m = evaluate_fixed(net, cfg, device)
    E_inf, E_rms = m["E_inf"], m["E_rms"]
    E_inf_core, E_rms_core = m["E_inf_core"], m["E_rms_core"]
    E_T_inf, Res_inf = m["E_T_inf"], m["Res_inf"]
    pn = net.path_norm().item()
    final_loss = compute_loss(
        net, cfg, device, gen, cached_t, cached_x, cached_xT
    )[0].item()
    state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    return Result(
        cfg=cfg,
        E_inf=E_inf,
        E_rms=E_rms,
        E_inf_core=E_inf_core,
        E_rms_core=E_rms_core,
        E_T_inf=E_T_inf,
        Res_inf=Res_inf,
        path_norm=pn,
        final_loss=final_loss,
        cpu_time=cpu_time,
        cpu_time_adam=cpu_time_adam,
        cpu_time_lbfgs=cpu_time_lbfgs,
        n_params=n_params,
        history=history,
        state_dict=state_dict,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Sobolev-regularized PINN trainer")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--M_term", type=int, default=1024)
    p.add_argument("--beta", type=float, default=1e-6)
    p.add_argument("--n_iter_adam", type=int, default=50000)
    p.add_argument("--n_iter_lbfgs", type=int, default=0)
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--lr_halve_every", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig(
        width=args.width,
        depth=args.depth,
        M=args.M,
        M_term=args.M_term,
        beta=args.beta,
        n_iter_adam=args.n_iter_adam,
        n_iter_lbfgs=args.n_iter_lbfgs,
        lr0=args.lr0,
        lr_halve_every=args.lr_halve_every,
        seed=args.seed,
        device=args.device,
        val_every=args.val_every,
        val_size=args.val_size,
        verbose=not args.quiet,
    )
    res = train(cfg)
    print()
    print(f"Final E_rms     = {res.E_rms:.4e}")
    print(f"Final E_inf     = {res.E_inf:.4e}")
    print(f"Final E_T_inf   = {res.E_T_inf:.4e}")
    print(f"Final Res_inf   = {res.Res_inf:.4e}")
    print(f"path-norm       = {res.path_norm:.4e}")
    print(f"CPU time (Adam) = {res.cpu_time_adam:.2f}s")
    print(f"CPU time (LBFGS)= {res.cpu_time_lbfgs:.2f}s")
    print(f"params          = {res.n_params}")
