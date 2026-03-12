"""
parameters.py

- Declares geometry (piecewise-linear outer radius)
- Provides material property "libraries" as interpolating callables
- Returns a nested dict

All temperatures for material property functions are in °C.
"""
from __future__ import annotations

from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np

from utils import EPS


def _interp_linear_extrap(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray | float], np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    def f(Tc: np.ndarray | float) -> np.ndarray:
        Tc = np.asarray(Tc, dtype=float)
        yq = np.interp(Tc, x, y)
        if x.size >= 2:
            mL = (y[1] - y[0]) / max(x[1] - x[0], EPS)
            mR = (y[-1] - y[-2]) / max(x[-1] - x[-2], EPS)
            left = Tc < x[0]
            right = Tc > x[-1]
            if np.any(left):
                yq[left] = y[0] + mL * (Tc[left] - x[0])
            if np.any(right):
                yq[right] = y[-1] + mR * (Tc[right] - x[-1])
        return yq

    return f


def load_steel(tag: str) -> Tuple[Callable, Callable, Callable]:
    tag = str(tag).upper()

    if tag == "30HN2MFA":
        T_k = np.array([54.2,149.1,250.0,352.0,453.3,553.6,651.1,704.4,723.0,743.3,763.0,783.1,802.8,822.9,842.8,904.9,1004.3])
        k_vals = np.array([35.9,37.3,36.0,33.8,30.9,27.2,19.7,17.1,16.0,15.8,17.1,18.7,19.3,19.5,19.7,20.3,20.6])
        T_cp = np.array([38,70,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,991])
        cp_vals = np.array([0.440,0.462,0.475,0.492,0.505,0.517,0.528,0.539,0.550,0.560,0.569,0.579,0.589,0.598,0.607,0.616,0.625,0.634,0.642,0.658])
        T_rho = np.array([50,100,200,400,600,700,720,725,730,735,740,750,765,770,775,785,800,900,1060])
        rho_vals = np.array([7.77,7.75,7.72,7.65,7.59,7.55,7.55,7.55,7.55,7.55,7.55,7.57,7.58,7.59,7.59,7.59,7.58,7.53,7.46])
    elif tag == "DUPLEX":
        T_k = np.array([52.0,149.1,249.8,351.7,457.0,553.6,654.7,704.2,744.2,762.9,782.6,802.5,811.7,821.7,842.3,904.7,1004.1])
        k_vals = np.array([13.3,15.3,17.0,17.8,18.1,18.7,20.0,20.8,21.4,21.6,21.8,22.2,22.3,22.4,22.7,23.7,25.9])
        T_cp = np.array([38,70,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,991])
        cp_vals = np.array([0.417,0.442,0.462,0.492,0.515,0.534,0.548,0.559,0.567,0.572,0.576,0.579,0.582,0.584,0.588,0.594,0.602,0.614,0.629,0.668])
        T_rho = np.array([50,100,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,1000,1060])
        rho_vals = np.array([7.74,7.72,7.69,7.67,7.65,7.63,7.61,7.60,7.58,7.56,7.54,7.52,7.49,7.47,7.45,7.43,7.40,7.34,7.32])
    elif tag == "38HMJ":
        T_k = np.array([50.9,149.0,250.0,351.3,453.4,553.6,654.7,704.5,741.0,762.6,782.7,802.8,811.9,821.8,842.3,904.7,1004.2])
        k_vals = np.array([30.0,33.6,34.4,33.0,30.7,27.4,22.5,19.4,16.4,19.3,20.9,23.2,24.4,25.4,26.3,27.8,28.9])
        T_cp = np.array([38,70,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,991])
        cp_vals = np.array([0.458,0.485,0.502,0.525,0.543,0.559,0.574,0.587,0.598,0.609,0.618,0.626,0.634,0.640,0.645,0.650,0.653,0.656,0.657,0.658])
        T_rho = np.array([50,100,200,400,600,780,795,800,820,830,840,850,860,870,880,890,900,1000,1060])
        rho_vals = np.array([7.66,7.65,7.62,7.55,7.48,7.42,7.41,7.42,7.43,7.43,7.43,7.43,7.43,7.42,7.42,7.42,7.41,7.37,7.34])
    else:
        raise ValueError(f'Unknown steel type "{tag}"')

    k_fun = _interp_linear_extrap(T_k, k_vals)
    cp_fun = _interp_linear_extrap(T_cp, 1e3 * cp_vals)          # kJ/kgK -> J/kgK
    rho_fun = _interp_linear_extrap(T_rho, 1e3 * rho_vals)       # g/cm3 -> kg/m3
    return rho_fun, cp_fun, k_fun


def load_chromium() -> Tuple[Callable, Callable, Callable]:
    T_ref = np.array([20,100,200,300,400,500,600,700,800,900,1000], dtype=float)
    k_ref = np.array([94,90,85,80,75,70,66,62,59,56,53], dtype=float)
    cp_ref = np.array([450,480,520,560,590,610,630,650,670,690,710], dtype=float)
    rho_ref = np.array([7200,7195,7185,7170,7155,7140,7125,7110,7095,7080,7065], dtype=float)

    k_fun = _interp_linear_extrap(T_ref, k_ref)
    cp_fun = _interp_linear_extrap(T_ref, cp_ref)
    rho_fun = _interp_linear_extrap(T_ref, rho_ref)
    return rho_fun, cp_fun, k_fun


def parameters(*, steel: str = "DUPLEX", tCr_m: float = 0.0, tCr_um: Optional[float] = None, check: bool = True) -> Dict[str, Any]:
    """
    Create parameter dict `p`.

    Args:
        steel: "DUPLEX" | "30HN2MFA" | "38HMJ"
        tCr_m: chromium thickness in meters
        tCr_um: chromium thickness in microns (overrides tCr_m if provided)
        (There are 2 Chromium knobs because I have dementia, I think)
        check: run sanity checks
    """
    if tCr_um is not None:
        tCr_m = float(tCr_um) * 1e-6
    if tCr_m < 0:
        raise ValueError("Chromium thickness must be >= 0.")

    p: Dict[str, Any] = {"geom": {}, "materials": {}, "thermal": {}}

    # geometry (from article piecewise outer radius)
    p["geom"]["Rin_bore_m"] = 17.5e-3
    p["geom"]["z_break_m"] = np.array([0, 0.385, 0.535, 0.880, 2.081, 2.980, 3.150], dtype=float)
    p["geom"]["Rout_break_m"] = 1e-3 * np.array([55.00, 55.00, 57.00, 59.50, 44.07, 31.00, 31.00], dtype=float)
    p["geom"]["L_m"] = float(p["geom"]["z_break_m"][-1])

    # chromium lining
    p["geom"]["tCr_m"] = float(tCr_m)
    p["geom"]["Rin_cr_in_m"] = p["geom"]["Rin_bore_m"]
    p["geom"]["Rout_cr_out_m"] = p["geom"]["Rin_bore_m"] + p["geom"]["tCr_m"]
    p["geom"]["Rin_steel_m"] = p["geom"]["Rout_cr_out_m"]

    def Rout(z: np.ndarray | float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        x = p["geom"]["z_break_m"]
        y = p["geom"]["Rout_break_m"]
        # linear interpolation + extrap
        Rout_val = np.interp(z, x, y)
        if x.size >= 2:
            mL = (y[1] - y[0]) / max(x[1] - x[0], EPS)
            mR = (y[-1] - y[-2]) / max(x[-1] - x[-2], EPS)
            left = z < x[0]
            right = z > x[-1]
            if np.any(left):
                Rout_val[left] = y[0] + mL * (z[left] - x[0])
            if np.any(right):
                Rout_val[right] = y[-1] + mR * (z[right] - x[-1])
        return Rout_val

    p["geom"]["Rout"] = Rout
    zb = p["geom"]["z_break_m"]
    p["geom"]["z_mid_m"] = 0.5 * (zb[:-1] + zb[1:])

    p["geom"]["thickness_total"] = lambda z: Rout(z) - p["geom"]["Rin_bore_m"]
    p["geom"]["thickness_cr"] = lambda z: np.zeros_like(np.asarray(z, dtype=float)) + p["geom"]["tCr_m"]
    p["geom"]["thickness_steel"] = lambda z: Rout(z) - p["geom"]["Rin_steel_m"]

    if check:
        tsteel_edges = p["geom"]["thickness_steel"](zb[:-1])
        if np.any(tsteel_edges <= 0):
            raise ValueError(f'Chromium thickness ({1e6*p["geom"]["tCr_m"]:.0f} µm) leaves no steel in some zones.')

    # materials
    rhoS, cpS, kS = load_steel(steel)
    rhoC, cpC, kC = load_chromium()

    p["materials"]["steel"] = {"tag": str(steel).upper(), "rho": rhoS, "cp": cpS, "k": kS}
    p["materials"]["chromium"] = {"rho": rhoC, "cp": cpC, "k": kC}

    # thermal resistance helpers
    p["thermal"]["Rcyl_per_m"] = lambda k, r1, r2: np.log(np.asarray(r2)/np.asarray(r1)) / (2*np.pi*k)

    def Rlayers_per_m(Tmean, z):
        return (
            p["thermal"]["Rcyl_per_m"](p["materials"]["chromium"]["k"](Tmean), p["geom"]["Rin_bore_m"], p["geom"]["Rout_cr_out_m"])
            + p["thermal"]["Rcyl_per_m"](p["materials"]["steel"]["k"](Tmean), p["geom"]["Rin_steel_m"], Rout(z))
        )

    p["thermal"]["Rlayers_per_m"] = Rlayers_per_m
    return p
