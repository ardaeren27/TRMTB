"""
Thermophysical properties for 32CrMoV12‑28 steel + chromium (fallback mode).
Ported from props_32CrMoV1228_ATS.m (fallback branch).

Usage:
    from props_32CrMoV1228 import props
    k = props.cr.k(T)
    rho = props.steel.rho(T)
    cp = props.steel.cp(T)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _linmap(x, x1, x2, y1, y2):
    t = np.clip((x - x1) / (x2 - x1), 0.0, 1.0)
    return y1 + t * (y2 - y1)


def _interp_clamped(x_tab, y_tab, xq):
    xq = np.clip(xq, np.min(x_tab), np.max(x_tab))
    return np.interp(xq, x_tab, y_tab)


def _cp_fallback_sane(T):
    T1, T2 = 293.0, 1273.0
    cp1, cp2 = 420.0, 760.0  # J/(kg·K)
    x = np.clip((T - T1) / (T2 - T1), 0.0, 1.0)
    return cp1 + (cp2 - cp1) * x


@dataclass(frozen=True)
class ChromiumProps:
    k: callable
    rho: callable
    cp: callable


@dataclass(frozen=True)
class SteelProps:
    rho0: float
    cp: callable
    rho: callable
    k: callable


@dataclass(frozen=True)
class Props:
    cr: ChromiumProps
    steel: SteelProps
    meta: str


def build_props() -> Props:
    # Chromium tables (Kelvin)
    Cr_TC = np.array([20, 200, 400, 600, 800, 1000, 1100], dtype=float)
    Cr_k = np.array([98, 92, 82, 72, 63, 57, 55], dtype=float)
    Cr_rho = np.array([7.18, 7.12, 7.05, 7.00, 6.97, 6.94, 6.92], dtype=float) * 1000.0
    Cr_cp = np.array([0.45, 0.49, 0.54, 0.60, 0.66, 0.72, 0.75], dtype=float) * 1000.0
    Cr_TK = Cr_TC + 273.15

    cr = ChromiumProps(
        k=lambda T: _interp_clamped(Cr_TK, Cr_k, T),
        rho=lambda T: _interp_clamped(Cr_TK, Cr_rho, T),
        cp=lambda T: _interp_clamped(Cr_TK, Cr_cp, T),
    )

    # Steel fallback
    rho0 = 7800.0
    alpha = lambda T: np.clip(12e-6 + (18e-6 - 12e-6) * (T - 293.0) / (1273.0 - 293.0), 10e-6, 20e-6)
    epsT = lambda T: alpha(0.5 * (T + 293.0)) * (T - 293.0)
    rho_fun = lambda T: rho0 / (1.0 + epsT(T)) ** 3

    k_low, k_min, k_hi = 35.0, 28.0, 32.0
    T1, Tc, T2 = 293.0, 1038.0, 1273.0
    def k_fun(T):
        T = np.asarray(T, dtype=float)
        return (T <= Tc) * _linmap(T, T1, Tc, k_low, k_min) + (T > Tc) * _linmap(T, Tc, T2, k_min, k_hi)

    steel = SteelProps(
        rho0=rho0,
        cp=lambda T: _cp_fallback_sane(np.asarray(T, dtype=float)),
        rho=lambda T: rho_fun(np.asarray(T, dtype=float)),
        k=lambda T: k_fun(np.asarray(T, dtype=float)),
    )

    return Props(cr=cr, steel=steel, meta="fallback mode with sane cp(T) (420→760 J/kgK)")


props = build_props()

if __name__ == "__main__":
    P = props
    print(f"[props] 32CrMoV12-28 fallback: rho0={P.steel.rho0:.0f} kg/m³ "
          f"cp(293)={P.steel.cp(293):.0f} J/kgK k(293)={P.steel.k(293):.1f} W/mK")
    print(f"[props] Chromium: k(293)={P.cr.k(293):.1f} rho={P.cr.rho(293):.0f} cp={P.cr.cp(293):.0f}")
