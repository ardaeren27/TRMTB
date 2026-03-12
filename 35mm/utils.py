"""
This repo is intentionally "MATLAB-like":
- linear interpolation with extrapolation (interp1 'linear','extrap')
- stable unique for time vectors
- a simple stdout-silencing context manager (MATLAB evalc equivalent)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


EPS = np.finfo(float).eps


def unique_stable(x: np.ndarray, tol: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB-like unique(x,'stable').

    Returns:
        x_u : unique values preserving first occurrence order
        idx : indices into the original x such that x_u = x[idx]
    Notes:
        If tol>0, values within tol are treated as duplicates by a greedy scan.
        This is useful when concatenating time grids that may touch.
    """
    x = np.asarray(x).ravel()
    if x.size == 0:
        return x.copy(), np.array([], dtype=int)

    idx = []
    seen = []
    if tol <= 0.0:
        # Exact match stable unique
        d = {}
        for i, v in enumerate(x):
            if v not in d:
                d[v] = i
                idx.append(i)
        idx = np.array(idx, dtype=int)
        return x[idx], idx

    # Tolerance-based stable unique (greedy)
    last = None
    for i, v in enumerate(x):
        if last is None or abs(v - last) > tol:
            idx.append(i)
            last = v
    idx = np.array(idx, dtype=int)
    return x[idx], idx


def interp1_linear_extrap(x: np.ndarray, y: np.ndarray, xq: np.ndarray | float) -> np.ndarray:
    """
    MATLAB interp1(x,y,xq,'linear','extrap') for y as 1-D.

    Uses numpy.interp for in-range interpolation and explicit linear extrapolation at both ends.
    Preserves the shape of xq.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    xq_arr = np.asarray(xq, dtype=float)
    xq_flat = np.atleast_1d(xq_arr).ravel()

    # Basic linear interpolation inside [x0, xN]
    yq_flat = np.interp(xq_flat, x, y)

    if x.size >= 2:
        mL = (y[1] - y[0]) / max(x[1] - x[0], EPS)
        mR = (y[-1] - y[-2]) / max(x[-1] - x[-2], EPS)

        left = xq_flat < x[0]
        right = xq_flat > x[-1]
        if np.any(left):
            yq_flat[left] = y[0] + mL * (xq_flat[left] - x[0])
        if np.any(right):
            yq_flat[right] = y[-1] + mR * (xq_flat[right] - x[-1])

    yq = yq_flat.reshape(xq_arr.shape) if xq_arr.shape != () else yq_flat[0]
    return yq


def interp1_linear_extrap_matrix(x: np.ndarray, Y: np.ndarray, xq: float) -> np.ndarray:
    """
    MATLAB interp1(x, Y, xq, 'linear','extrap') where Y is [N x M], xq scalar.
    Returns shape (M,).
    """
    x = np.asarray(x, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[0] != x.size:
        raise ValueError("Y must be 2-D with Y.shape[0] == len(x).")

    # Find bracketing indices
    if xq <= x[0]:
        i0, i1 = 0, 1
    elif xq >= x[-1]:
        i0, i1 = x.size - 2, x.size - 1
    else:
        i1 = int(np.searchsorted(x, xq, side="right"))
        i0 = i1 - 1

    dx = max(x[i1] - x[i0], EPS)
    w = (xq - x[i0]) / dx
    return (1.0 - w) * Y[i0, :] + w * Y[i1, :]


def print_solver_banner() -> None:
    print("initiating solver:")


@dataclass
class QuietStdout:
    """
    Context manager to suppress stdout/stderr (MATLAB evalc analogue).
    """
    enabled: bool = True

    def __enter__(self):
        import sys
        import contextlib
        import io
        self._sys = sys
        self._contextlib = contextlib
        self._io = io
        if self.enabled:
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            self._buf = io.StringIO()
            sys.stdout = self._buf
            sys.stderr = self._buf
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self._sys.stdout = self._stdout
            self._sys.stderr = self._stderr
        return False
