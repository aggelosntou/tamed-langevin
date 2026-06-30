"""Adaptive switched taming maps."""

from __future__ import annotations

import numpy as np


def g_switch(t: np.ndarray | float) -> np.ndarray:
    """Coordinatewise C1 switch used by the adaptive taming map."""
    arr = np.asarray(t, dtype=float)
    out = np.empty_like(arr, dtype=float)

    lo = arr < 1.0
    mid = (arr >= 1.0) & (arr < 2.0)
    hi = arr >= 2.0

    out[lo] = 0.0
    s = arr[mid] - 1.0
    out[mid] = 5.0 * s**2 - 3.0 * s**3
    out[hi] = arr[hi]
    return out


def adaptive_tamed_drift(
    drift: np.ndarray,
    state: np.ndarray,
    step_size: float,
    a_tame: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return adaptive-tamed drift and active-coordinate mask.

    Computes

        h_ad(x) = a*x + (h(x)-a*x)/(1+g(sqrt(lambda)*|h(x)|)).

    The residual h-a*x is divided, while the switch is triggered by the full drift h.
    """
    residual = drift - a_tame * state
    switch = g_switch(np.sqrt(step_size) * np.abs(drift))
    tamed = a_tame * state + residual / (1.0 + switch)
    return tamed, switch > 0.0


def check_g_c1(tol: float = 5.0e-4) -> None:
    """Verify the implemented switch at t=1 and t=2."""
    eps = 1.0e-6

    def scalar_g(x: float) -> float:
        return float(g_switch(np.array([x]))[0])

    gaps = [
        abs(scalar_g(1.0) - 0.0),
        abs(scalar_g(2.0) - 2.0),
        abs((scalar_g(1.0 + eps) - scalar_g(1.0 - eps)) / (2.0 * eps) - 0.0),
        abs((scalar_g(2.0 + eps) - scalar_g(2.0 - eps)) / (2.0 * eps) - 1.0),
    ]

    if max(gaps) > tol:
        raise RuntimeError(f"g-switch failed C1 check; gaps={gaps}")
