"""Manufactured Hamilton--Jacobi--Bellman test problem.

    d_t v + inf_{(s,beta) in A} { (1/2) s^2 Lap_x v + beta (1/d) sum_i d_{x_i} v + f(t,x) } = 0,
    A = [SIG_LO, SIG_HI] x [-B_BAR, B_BAR],   v(T,x) = g(x) := v(T,x),

with manufactured source f chosen so that the exact solution is

    v(t,x) = sin(t + sum_i x_i) * exp(-|x|^2 / ELL^2).

The infimum is explicit:
    inf = (1/2) SIG_LO^2 (Lap v)^+ - (1/2) SIG_HI^2 (Lap v)^- - (B_BAR/d) |sum_i d_{x_i} v|.
"""
from __future__ import annotations

import numpy as np

T = 1.0
SIG_LO, SIG_HI, B_BAR, ELL = 0.1, 0.4, 1.0, 2.0


def hamiltonian(lap, sum_dx, d):
    """inf over A of the control-dependent part, given Lap_x v and sum_i d_{x_i} v."""
    return (0.5 * SIG_LO ** 2 * np.maximum(lap, 0.0)
            - 0.5 * SIG_HI ** 2 * np.maximum(-lap, 0.0)
            - (B_BAR / d) * np.abs(sum_dx))


def exact_parts(pts: np.ndarray):
    """(v, d_t v, sum_i d_{x_i} v, Lap_x v) at points (t, x_1..x_d)."""
    t, x = pts[:, 0], pts[:, 1:]
    d = x.shape[1]
    ph = t + x.sum(axis=1)
    sx = x.sum(axis=1)
    r2 = (x ** 2).sum(axis=1)
    E = np.exp(-r2 / ELL ** 2)
    s, c = np.sin(ph), np.cos(ph)
    v = s * E
    vt = c * E
    S = E * (d * c - 2.0 * sx / ELL ** 2 * s)
    lap = E * (-4.0 * sx / ELL ** 2 * c + (4.0 * r2 / ELL ** 4 - d - 2.0 * d / ELL ** 2) * s)
    return v, vt, S, lap


def exact(pts: np.ndarray) -> np.ndarray:
    return exact_parts(pts)[0]


def terminal(x: np.ndarray) -> np.ndarray:
    """g(x) = v(T,x); x has shape (n, d)."""
    pts = np.column_stack([np.full(x.shape[0], T), x])
    return exact(pts)


def source(pts: np.ndarray) -> np.ndarray:
    v, vt, S, lap = exact_parts(pts)
    d = pts.shape[1] - 1
    return -vt - hamiltonian(lap, S, d)
