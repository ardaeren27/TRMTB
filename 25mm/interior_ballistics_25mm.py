"""
interior_ballistics_25mm.py

Adapt the existing single-shot interior + transitional ballistics model to the
25×137 mm TP/HEI round using Vallier–Heydenreich data and 25 mm geometry.

- Reuses the exact numerical logic from `initial_python_port/interior_ballistics.py`
  (including debug prints, graphs, and the bunny).
- Pulls geometry/data from `vallier-heydenreich/Geometry25/gun25_geometry.py`.
- Calibrates the unknown propellant impetus scale so the predicted muzzle
  velocity matches the Vallier reference (1100 m/s) while leaving everything
  else unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Make sure we can import the original utilities
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "initial_python_port"))
from utils import EPS  # noqa: E402

sys.path.append(str(REPO_ROOT / "vallier-heydenreich" / "Geometry25"))
from gun25_geometry import gun25_geometry, VALLIER_APDS_25X137_REF  # noqa: E402

# TP/HEI 25×137 mm Vallier-style reference (user-provided)
VALLIER_TPHEI_25X137_REF = {
    "m_proj_kg": 0.185,
    "m_prop_kg": 0.100,
    "V_exit_mps": 1100.0,
    "P_max_MPa": 420.0,
    # NOTE: keep the physical barrel geometry (L0+L = 2.000 m) in gun25_geometry.py.
    # The 1.786 m value is kept as metadata for the rifled portion; the IB travel length
    # remains geom.barrel_len_m unless you explicitly override it.
    "L_rifled_m": 1.786,
    "D_bore_m": 0.025,
    "V0_m3": 106.0e-6,
}

# Select which ammo reference to use for parameterization + optimization targets.
# Switch back to VALLIER_APDS_25X137_REF if you want the old APDS calibration.
ACTIVE_VALLIER_REF = VALLIER_TPHEI_25X137_REF


# ----------------------------- data classes ----------------------------- #
@dataclass
class IBParams:
    mP: float
    mp: float
    s: float
    V0: float
    lm: float
    K: float
    f: float
    eta: float
    gamma: float
    R: float
    rho_p: float
    r1: float
    S1: float
    LAMBDA1: float
    Kappa1: float
    lambda1: float
    p_start: float
    l0: float
    li: np.ndarray
    Di: np.ndarray
    r1_scale: float
    f_scale: float
    T0: float  # initial flame temperature [K]


# ----------------------------- helpers ----------------------------- #
def _scale_li_for_25mm(original_l0: float, original_lm: float, original_li: np.ndarray,
                       new_l0: float, new_lm: float) -> np.ndarray:
    """
    Preserve the relative axial positions of the six HTC zones from the 35 mm
    model and scale them to the 25 mm barrel length.
    """
    frac = (original_li - original_l0) / max(original_lm - original_l0, EPS)
    return new_l0 + frac * (new_lm - new_l0)


def build_25mm_params(f_scale: float = 1.0, p_start_override: float | None = None, l0_override_m: float | None = None,
                      r1_scale: float = 1.0, V0_override: float | None = None) -> IBParams:
    """
    Assemble the 25 mm parameter set, optionally scaling the propellant impetus f.
    """
    geom = gun25_geometry()
    ref = ACTIVE_VALLIER_REF

    # Base constants from the original model (35 mm) to preserve logic
    orig = {
        "mP": 0.380,
        "mp": 0.376,
        "s": 9.98e-4,
        "V0": 373e-6,
        "lm": 2.934,
        "K": 1.37,
        "f": 1.071e6,
        "eta": 1.064e-3,
        "gamma": 1.2,
        "R": 340.0,
        "rho_p": 1600.0,
        "r1": 0.597e-9,
        "S1": 134.4e-6,
        "LAMBDA1": 75.2e-9,
        "Kappa1": 0.755,
        "lambda1": 0.159,
        "p_start": 30e6,
        "l0": 0.216,
        "li": np.array([0.216, 0.385, 0.535, 0.880, 2.081, 2.980], dtype=float),
        "Di": 0.035 * np.ones(6, dtype=float),
    }

    # 25 mm replacements
    mP = ref["m_proj_kg"]
    mp = ref["m_prop_kg"]
    s = geom.area_bore_m2
    V0 = ref["V0_m3"] if V0_override is None else V0_override
    lm = geom.barrel_len_m
    Di = geom.bore_d_m * np.ones(6, dtype=float)

    # Projectile length heuristic: if provided, use it; else scale with caliber
    if l0_override_m is not None:
        l0 = l0_override_m
    else:
        l0 = orig["l0"] * (geom.bore_d_m / 0.035)
    li = _scale_li_for_25mm(orig["l0"], orig["lm"], orig["li"], l0, lm)

    return IBParams(
        mP=mP,
        mp=mp,
        s=s,
        V0=V0,
        lm=lm,
        K=orig["K"],
        f=orig["f"] * f_scale,
        eta=orig["eta"],
        gamma=orig["gamma"],
        R=orig["R"],
        rho_p=orig["rho_p"],
        r1=orig["r1"],
        S1=orig["S1"],
        LAMBDA1=orig["LAMBDA1"],
        Kappa1=orig["Kappa1"],
        lambda1=orig["lambda1"],
        p_start=p_start_override if p_start_override is not None else 18.33e6,
        l0=l0,
        li=li,
        Di=Di,
        r1_scale=r1_scale,
        f_scale=f_scale,
        T0=2000.0,
    )


# ----------------------------- model kernels ----------------------------- #
def make_models(params: IBParams):
    """
    Create parameter-bound thermo_model and thermo_model_v2 closures so we do not
    mutate the original logic.
    """

    def thermo_model(zp: float, Tgas: float, velocityP: float, location: float) -> Tuple[float, float, float, float, float, float]:
        denom = params.V0 + params.s * location - (params.mp / params.rho_p) * (1 - zp) - params.eta * params.mp * zp
        pressureEqn = (params.mp * zp * params.R * Tgas) / denom
        rho = (params.mp * zp) / denom

        m_eff = params.mP
        phi = params.K

        diffLocation = velocityP
        diffVelocity = pressureEqn * params.s / (phi * m_eff)

        diffBurnRate = (params.S1 / params.LAMBDA1) * params.r1 * params.r1_scale * pressureEqn * np.sqrt(1.0 + 4.0 * zp * params.lambda1 / params.Kappa1)
        diffBurnRate = max(diffBurnRate, 0.0)
        if zp >= 1.0:
            diffBurnRate = 0.0

        diffTheta = ((params.f - params.R * Tgas) * params.mp * diffBurnRate - (params.gamma - 1.0) * phi * m_eff * velocityP * diffVelocity) / max(params.mp * zp, EPS)
        return float(rho), float(pressureEqn), float(diffLocation), float(diffVelocity), float(diffBurnRate), float(diffTheta)

    def thermo_model_v2(zp: float, Tgas: float, zeta: float) -> Tuple[float, float, float, float, float]:
        denom = params.V0 + params.s * params.lm - params.mp * (1.0 - zp) / params.rho_p - params.eta * params.mp * (zp - zeta)
        pressureEqn2 = (params.mp * (zp - zeta) * params.R * Tgas) / denom
        rho2 = (params.mp * (zp - zeta)) / denom

        CF_CHOKE = np.sqrt(params.gamma) * (2.0 / (params.gamma + 1.0)) ** ((params.gamma + 1.0) / (2.0 * (params.gamma - 1.0)))
        diffZeta = (params.s * pressureEqn2) / (params.mp * np.sqrt(params.R * Tgas)) * CF_CHOKE

        if zp < 1.0 - 1e-12:
            diffZp = (params.S1 / params.LAMBDA1) * params.r1 * params.r1_scale * pressureEqn2 * np.sqrt(1.0 + 4.0 * zp * params.lambda1 / params.Kappa1)
            diffZp = max(diffZp, 0.0)
        else:
            diffZp = 0.0

        gas_mass_frac = max(zp - zeta, EPS)
        diffTheta2 = ((params.f - params.R * Tgas) * diffZp - (params.gamma - 1.0) * params.R * Tgas * diffZeta) / gas_mass_frac
        return float(rho2), float(pressureEqn2), float(diffZp), float(diffZeta), float(diffTheta2)

    return thermo_model, thermo_model_v2


# ----------------------------- solver (logic preserved) ----------------------------- #
def interior_ballistics_25mm(*, dt: float = 1e-6, t_end: float = 0.02, plot: bool = True, debug: bool = True,
                             params: IBParams | None = None) -> Dict[str, Any]:
    """
    Run the adapted 25 mm single-shot interior + transitional ballistics model.
    Logic matches `initial_python_port/interior_ballistics.py` verbatim.
    """
    params = params or build_25mm_params()
    thermo_model, thermo_model_v2 = make_models(params)

    if debug:
        print("=== interior_ballistics_25mm params ===")
        print(f"mP={params.mP} kg mp={params.mp} kg s={params.s} m^2 V0={params.V0} m^3 lm={params.lm} m")
        print(f"p_start={params.p_start/1e6:.2f} MPa f_scale={params.f_scale:.3f} r1_scale={params.r1_scale:.3f}")
        print(f"f={params.f:.3e} eta={params.eta:.3e} gamma={params.gamma:.3f} R={params.R}")
        print(f"l0={params.l0} m Di={params.Di[0]} m li={params.li}")
        print("=======================================")

    # prealloc
    maxStep = int(np.ceil(t_end / dt) + 1)
    timeVec = np.zeros(maxStep, dtype=float)
    location = np.zeros(maxStep, dtype=float)
    velocityP = np.zeros(maxStep, dtype=float)
    zp = np.zeros(maxStep, dtype=float)
    zetaHist = np.zeros(maxStep, dtype=float)
    theta = np.zeros(maxStep, dtype=float)
    Tgas = np.zeros(maxStep, dtype=float)
    rhoHist = np.zeros(maxStep, dtype=float)
    pressureEqn = np.zeros(maxStep, dtype=float)
    hHist = np.zeros((maxStep, 6), dtype=float)

    velocityP[0] = 0.0
    zp[0] = 0.001
    theta[0] = params.R * params.T0
    Tgas[0] = params.T0

    state = np.array([0.0, 0.0, zp[0], theta[0]], dtype=float)  # [x v zp theta]
    t_on = np.full(6, np.nan, dtype=float)

    released = False
    t_release = np.nan
    t_zp1 = np.nan
    t_exit = np.nan

    n = 0
    t = 0.0

    # -------------------- phase 1 -------------------- #
    while t < t_end:
        prev_state = state.copy()
        prev_t = t

        # k1
        _, P1, dx1, dv1, dzp1, dth1 = thermo_model(prev_state[2], prev_state[3] / params.R, prev_state[1], prev_state[0])
        if (not released) and (P1 < params.p_start):
            dx1 = 0.0
            dv1 = 0.0
            T1 = prev_state[3] / params.R
            z1 = max(prev_state[2], EPS)
            dth1 = ((params.f - params.R * T1) * params.mp * dzp1) / (params.mp * z1)
        k1 = dt * np.array([dx1, dv1, dzp1, dth1], dtype=float)

        # k2
        mid = prev_state + 0.5 * k1
        mid[2] = min(max(mid[2], 0.0), 1.0)
        _, P2, dx2, dv2, dzp2, dth2 = thermo_model(mid[2], mid[3] / params.R, mid[1], mid[0])
        if (not released) and (P2 < params.p_start):
            dx2 = 0.0
            dv2 = 0.0
            T2 = mid[3] / params.R
            z2 = max(mid[2], EPS)
            dth2 = ((params.f - params.R * T2) * params.mp * dzp2) / (params.mp * z2)
        k2 = dt * np.array([dx2, dv2, dzp2, dth2], dtype=float)

        # k3
        mid = prev_state + 0.5 * k2
        mid[2] = min(max(mid[2], 0.0), 1.0)
        _, P3, dx3, dv3, dzp3, dth3 = thermo_model(mid[2], mid[3] / params.R, mid[1], mid[0])
        if (not released) and (P3 < params.p_start):
            dx3 = 0.0
            dv3 = 0.0
            T3 = mid[3] / params.R
            z3 = max(mid[2], EPS)
            dth3 = ((params.f - params.R * T3) * params.mp * dzp3) / (params.mp * z3)
        k3 = dt * np.array([dx3, dv3, dzp3, dth3], dtype=float)

        # k4
        endS = prev_state + k3
        endS[2] = min(max(endS[2], 0.0), 1.0)
        _, P4, dx4, dv4, dzp4, dth4 = thermo_model(endS[2], endS[3] / params.R, endS[1], endS[0])
        if (not released) and (P4 < params.p_start):
            dx4 = 0.0
            dv4 = 0.0
            T4 = endS[3] / params.R
            z4 = max(endS[2], EPS)
            dth4 = ((params.f - params.R * T4) * params.mp * dzp4) / (params.mp * z4)
        k4 = dt * np.array([dx4, dv4, dzp4, dth4], dtype=float)

        new_state = prev_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        new_state[2] = min(max(new_state[2], 0.0), 1.0)
        # cap only above flame temperature to allow natural rise/decay
        new_state[3] = min(new_state[3], params.R * params.T0)

        if not released:
            P_prev = (params.mp * prev_state[2] * params.R * (prev_state[3] / params.R)) / (params.V0 + params.s * prev_state[0] - params.mp * (1 - prev_state[2]) / params.rho_p - params.eta * params.mp * prev_state[2])
            P_new = (params.mp * new_state[2] * params.R * (new_state[3] / params.R)) / (params.V0 + params.s * new_state[0] - params.mp * (1 - new_state[2]) / params.rho_p - params.eta * params.mp * new_state[2])
            if (P_prev < params.p_start) and (P_new >= params.p_start):
                fracR = (params.p_start - P_prev) / max(P_new - P_prev, EPS)
                t_release = prev_t + max(0.0, min(1.0, fracR)) * dt
                released = True
            elif P_new >= params.p_start:
                released = True
                t_release = prev_t + dt

        x_prev = prev_state[0]
        x_new = new_state[0]
        crossed = (x_prev < params.lm) and (x_new >= params.lm)

        if crossed:
            frac = (params.lm - x_prev) / max(x_new - x_prev, EPS)
            exit_state = prev_state + frac * (new_state - prev_state)
            exit_state[2] = min(max(exit_state[2], 0.0), 1.0)
            t = prev_t + frac * dt
            state = exit_state

            location[n] = state[0]
            velocityP[n] = state[1]
            zp[n] = state[2]
            theta[n] = state[3]
            Tgas[n] = theta[n] / params.R

            pressureEqn[n] = (params.mp * zp[n] * params.R * Tgas[n]) / (params.V0 + params.s * location[n] - params.mp * (1 - zp[n]) / params.rho_p - params.eta * params.mp * zp[n])
            rhoHist[n] = (params.mp * zp[n]) / (params.V0 + params.s * location[n] - params.mp * (1 - zp[n]) / params.rho_p - params.eta * params.mp * zp[n])

            v_now = velocityP[n]
            l_now = location[n]
            tail_abs = params.l0 + max(l_now, 0.0)
            wi = (params.li / (params.l0 + l_now)) * v_now
            h_now = (6.1 / (params.Di ** 0.2)) * (rhoHist[n] * np.abs(wi)) ** 0.8

            present = params.li <= tail_abs
            h_now = np.where(present, h_now, 0.0)
            just_on = present & np.isnan(t_on)
            t_on[just_on] = t
            hHist[n, :] = h_now

            timeVec[n] = t
            t_exit = t
            n += 1
            break

        state = new_state

        location[n] = state[0]
        velocityP[n] = state[1]
        zp[n] = state[2]
        theta[n] = state[3]
        Tgas[n] = theta[n] / params.R

        pressureEqn[n] = (params.mp * zp[n] * params.R * Tgas[n]) / (params.V0 + params.s * location[n] - params.mp * (1 - zp[n]) / params.rho_p - params.eta * params.mp * zp[n])
        rhoHist[n] = (params.mp * zp[n]) / (params.V0 + params.s * location[n] - params.mp * (1 - zp[n]) / params.rho_p - params.eta * params.mp * zp[n])

        v_now = velocityP[n]
        l_now = location[n]
        tail_abs = params.l0 + max(l_now, 0.0)
        wi = (params.li / (params.l0 + l_now)) * v_now
        h_now = (6.1 / (params.Di ** 0.2)) * (rhoHist[n] * np.abs(wi)) ** 0.8

        present = params.li <= tail_abs
        h_now = np.where(present, h_now, 0.0)
        just_on = present & np.isnan(t_on)
        t_on[just_on] = t
        hHist[n, :] = h_now

        if debug:
            print(f"\n[phase1] t={t:.8f} s x={location[n]:.6f} m v={velocityP[n]:.6f} m/s zp={zp[n]:.6f}")
            print(f"         Tgas={Tgas[n]:.2f} K P={pressureEqn[n]/1e6:.3f} MPa rho={rhoHist[n]:.3f} kg/m^3")
            print(f"         h_on={np.any(present)} h_sample={h_now[present][0] if np.any(present) else 0.0:.2e}")

        t = prev_t + dt
        timeVec[n] = t
        n += 1

        if n >= maxStep - 2:
            break

    if np.isnan(t_exit):
        t_exit = t

    location[n:] = params.lm
    velocityP[n:] = 0.0

    # -------------------- phase 2 -------------------- #
    zetaCur = 0.0
    zp_cur = zp[max(n - 1, 0)]
    T_phase2 = Tgas[max(n - 1, 0)]

    den2 = params.V0 + params.s * params.lm - params.mp * (1.0 - zp_cur) / params.rho_p - params.eta * params.mp * (zp_cur - zetaCur)
    pressureEqn[max(n - 1, 0)] = (params.mp * (zp_cur - zetaCur) * params.R * T_phase2) / den2
    rhoHist[max(n - 1, 0)] = (params.mp * (zp_cur - zetaCur)) / den2
    zetaHist[max(n - 1, 0)] = zetaCur

    wcr = np.sqrt(params.gamma * params.R * T_phase2) * np.sqrt(2.0 / (params.gamma + 1.0))
    wi2 = (params.li / (params.l0 + params.lm)) * wcr
    hHist[max(n - 1, 0), :] = (6.1 / (params.Di ** 0.2)) * (rhoHist[max(n - 1, 0)] * np.abs(wi2)) ** 0.8

    zeta = zetaCur
    theta_here = theta[max(n - 1, 0)]
    zp_here = zp_cur

    while ((zp_here - zeta) > 1e-6) and (t < t_end) and (n < maxStep - 1):
        _, _, dzp1, dzeta1, dth1 = thermo_model_v2(zp_here, theta_here / params.R, zeta)
        if zp_here >= 1.0 - 1e-12:
            dzp1 = 0.0
        dzp1 = max(dzp1, 0.0)
        k1 = dt * np.array([dzeta1, dth1, dzp1], dtype=float)

        z_mid = min(max(zp_here + 0.5 * k1[2], 0.0), 1.0)
        _, _, dzp2, dzeta2, dth2 = thermo_model_v2(z_mid, (theta_here + 0.5 * k1[1]) / params.R, zeta + 0.5 * k1[0])
        if z_mid >= 1.0 - 1e-12:
            dzp2 = 0.0
        dzp2 = max(dzp2, 0.0)
        k2 = dt * np.array([dzeta2, dth2, dzp2], dtype=float)

        z_mid = min(max(zp_here + 0.5 * k2[2], 0.0), 1.0)
        _, _, dzp3, dzeta3, dth3 = thermo_model_v2(z_mid, (theta_here + 0.5 * k2[1]) / params.R, zeta + 0.5 * k2[0])
        if z_mid >= 1.0 - 1e-12:
            dzp3 = 0.0
        dzp3 = max(dzp3, 0.0)
        k3 = dt * np.array([dzeta3, dth3, dzp3], dtype=float)

        z_end = min(max(zp_here + k3[2], 0.0), 1.0)
        _, _, dzp4, dzeta4, dth4 = thermo_model_v2(z_end, (theta_here + k3[1]) / params.R, zeta + k3[0])
        if z_end >= 1.0 - 1e-12:
            dzp4 = 0.0
        dzp4 = max(dzp4, 0.0)
        k4 = dt * np.array([dzeta4, dth4, dzp4], dtype=float)

        zeta = zeta + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        theta_here = theta_here + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        # cap only above flame temperature
        theta_here = min(theta_here, params.R * params.T0)
        old_zp = zp_here
        zp_here = zp_here + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
        zp_here = min(max(zp_here, 0.0), 1.0)

        if np.isnan(t_zp1) and (old_zp < 1.0) and (zp_here >= 1.0):
            t_zp1 = t + dt

        theta_here = min(theta_here, theta[max(n - 1, 0)], params.R * params.T0)
        theta[n] = theta_here
        Tgas[n] = theta_here / params.R
        zp[n] = zp_here
        zetaHist[n] = zeta

        den2 = params.V0 + params.s * params.lm - params.mp * (1.0 - zp_here) / params.rho_p - params.eta * params.mp * (zp_here - zeta)
        P2 = (params.mp * (zp_here - zeta) * params.R * Tgas[n]) / den2
        pressureEqn[n] = max(P2, 101325.0)
        rhoHist[n] = (params.mp * (zp_here - zeta)) / den2

        wcr = np.sqrt(params.gamma * params.R * Tgas[n]) * np.sqrt(2.0 / (params.gamma + 1.0))
        wi2 = (params.li / (params.l0 + params.lm)) * wcr
        hHist[n, :] = (6.1 / (params.Di ** 0.2)) * (rhoHist[n] * np.abs(wi2)) ** 0.8

        t = t + dt
        timeVec[n] = t
        n += 1
        if debug:
            print(f"\n[phase2] t={t:.8f} s zeta={zeta:.6f} zp={zp_here:.6f}")
            print(f"         Tgas={Tgas[n-1]:.2f} K P={pressureEqn[n-1]/1e6:.3f} MPa rho={rhoHist[n-1]:.3f} kg/m^3")
            print(f"         mass_frac={(zp_here - zeta):.6f} dzp={dzp4:.3e} dzeta={dzeta4:.3e}")

        if pressureEqn[n - 1] <= 0.18e6:
            break

    last = min(n, location.size)
    timeVec = timeVec[:last]
    location = location[:last]
    velocityP = velocityP[:last]
    zp = zp[:last]
    zetaHist = zetaHist[:last]
    Tgas = Tgas[:last]
    pressureEqn = pressureEqn[:last]
    rhoHist = rhoHist[:last]
    hHist = hHist[:last, :]

    if plot:
        _plot_ballistics(timeVec, dt, t_exit, t_zp1, t_release,
                         Tgas, pressureEqn, location, velocityP, zp, zetaHist, rhoHist, hHist, params)

    return {
        "timeVec": timeVec,
        "location": location,
        "velocityP": velocityP,
        "zp": zp,
        "zetaHist": zetaHist,
        "theta": theta[:last],
        "Tgas": Tgas,
        "rhoHist": rhoHist,
        "pressureEqn": pressureEqn,
        "hHist": hHist,
        "t_exit": t_exit,
        "t_release": t_release,
        "t_zp1": t_zp1,
        "params": params,
    }


# ----------------------------- plotting (unchanged except labels) ----------------------------- #
def _plot_ballistics(timeVec, dt, t_exit, t_zp1, t_release,
                     Tgas, pressureEqn, location, velocityP, zp, zetaHist, rhoHist, hHist, params: IBParams):
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Interior Ballistics – 25 mm")
    gs = fig.add_gridspec(3, 3, wspace=0.35, hspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(timeVec, Tgas, linewidth=1.2)
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("T_gas  [K]")
    ax.set_title("Gas Temperature")
    ax.grid(True)

    axL = fig.add_subplot(gs[0, 1])
    axR = axL.twinx()
    axL.plot(timeVec, pressureEqn / 1e6, linewidth=1.2)
    axL.set_ylabel("P [MPa] (linear)")
    axR.semilogy(timeVec, pressureEqn / 1e6, "--", linewidth=1.0)
    axR.set_ylabel("P [MPa] (log)")
    axL.set_xlabel("t [s]")
    axL.set_title("Chamber Pressure (linear + log)")
    axL.grid(True)
    axL.axvline(t_exit, color="k", linestyle="--")
    if not np.isnan(t_zp1):
        axL.axvline(t_zp1, color="r", linestyle="-.")
    if not np.isnan(t_release):
        axL.axvline(t_release, color="g", linestyle="-.")
    try:
        y_top = axL.get_ylim()[1]
        axL.text(t_exit, 0.92 * y_top, "t_exit", rotation=90, va="top", ha="right", fontsize=9)
        if not np.isnan(t_zp1):
            axL.text(t_zp1, 0.92 * y_top, "z_p=1", rotation=90, va="top", ha="right", fontsize=9, color="r")
        if not np.isnan(t_release):
            axL.text(t_release, 0.92 * y_top, "release", rotation=90, va="top", ha="right", fontsize=9, color="g")
    except Exception:
        pass

    ax = fig.add_subplot(gs[0, 2])
    idx = timeVec <= t_exit
    loc = location[idx]
    tv = timeVec[idx]
    if tv.size >= 2:
        v_fd = np.concatenate([[0.0], np.diff(loc)]) / dt
    else:
        v_fd = np.zeros_like(tv)
    ax.plot(tv, v_fd, linewidth=1.2)
    ax.plot(tv, velocityP[idx], linewidth=1.2)
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("v_P  [m/s]")
    ax.set_title("Piston Velocity")
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(timeVec, zp, linewidth=1.2)
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("z_p  [-]")
    ax.set_title("Propellant Burned")
    ax.grid(True)
    if not np.isnan(t_zp1):
        ax.axvline(t_zp1, color="r", linestyle="-.")
        try:
            y_top = ax.get_ylim()[1]
            ax.text(t_zp1, 0.92 * y_top, "z_p=1", rotation=90, va="top", ha="right", fontsize=9, color="r")
        except Exception:
            pass

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(timeVec, zetaHist, linewidth=1.2)
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("zeta  [-]")
    ax.set_title("Gas Expelled")
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(timeVec, rhoHist, linewidth=1.2)
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("rho  [kg/m^3]")
    ax.set_title("Gas Density")
    ax.grid(True)

    h_max = float(np.nanmax(hHist)) if hHist.size else 0.0
    y_h_max = max(2.75e5, 1.10 * h_max)
    y_h_step = 2.5e4
    y_h_ticks = np.arange(0.0, y_h_max + y_h_step, y_h_step)

    axL = fig.add_subplot(gs[2, :])
    axR = axL.twinx()
    colors = ["k", "r", "b", "g", "m", "y"]
    for k in range(6):
        axL.plot(timeVec, hHist[:, k], linewidth=1.1, color=colors[k], label=f"P{k+1}")
    axL.set_xlim(0.0, min(0.007, timeVec[-1]))
    axL.set_ylim(0.0, y_h_max)
    axL.set_yticks(y_h_ticks)
    axL.set_xlabel("t  [s]")
    axL.set_ylabel("h_i  [W/(m^2·K)]")
    axL.grid(True)
    axR.plot(timeVec, Tgas, "r--", linewidth=1.2, label="Tgas")
    axR.set_ylim(0.0, 3.5e3)
    axR.set_ylabel("T_gas  [K]")
    axL.set_title("Convection Coefficients & Gas Temperature")
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="upper left")

    fig2, axL2 = plt.subplots(figsize=(10, 4))
    axR2 = axL2.twinx()
    colors = ["k", "r", "b", "g", "m", "y"]
    for k in range(6):
        axL2.plot(timeVec, hHist[:, k], linewidth=1.1, color=colors[k], label=f"P{k+1}")
    axL2.set_xlim(0.0, min(0.007, timeVec[-1]))
    axL2.set_ylim(0.0, y_h_max)
    axL2.set_yticks(y_h_ticks)
    axL2.set_ylabel("Heat Transfer Coefficient HTC, W·m^{-2}·K^{-1}")
    axL2.grid(True)
    axR2.plot(timeVec, Tgas, "r--", linewidth=1.2, label="Tgas")
    axR2.set_ylim(0.0, 3.5e3)
    yR_ticks = np.arange(0.0, 3.5e3 + 1.0, 250.0)
    axR2.set_yticks(yR_ticks)
    axR2.set_yticklabels([
        "0","2.5×10^2","5.0×10^2","7.5×10^2","1.0×10^3","1.25×10^3","1.50×10^3",
        "1.75×10^3","2.00×10^3","2.25×10^3","2.50×10^3","2.75×10^3",
        "3.00×10^3","3.25×10^3","3.50×10^3"
    ])
    axR2.set_ylabel("Gas Temperature, K")
    axL2.set_xlabel("Time, s")
    axL2.set_title("Convection Coefficients & Gas Temperature")
    hL, lL = axL2.get_legend_handles_labels()
    hR, lR = axR2.get_legend_handles_labels()
    axL2.legend(hL + hR, lL + lR, loc="upper left")

    axL2.minorticks_on()
    axR2.minorticks_on()
    axL2.tick_params(which="both", direction="in")
    axR2.tick_params(which="both", direction="in")
    plt.show()


# ----------------------------- calibration ----------------------------- #
def _simulate_exit_velocity(f_scale: float, r1_scale: float, dt: float, t_end: float,
                           p_start: float, l0_override_m: float | None, V0_override: float | None = None) -> Tuple[float, Dict[str, Any]]:
    params = build_25mm_params(f_scale=f_scale, p_start_override=p_start,
                               l0_override_m=l0_override_m, r1_scale=r1_scale,
                               V0_override=V0_override)
    res = interior_ballistics_25mm(dt=dt, t_end=t_end, plot=False, debug=True, params=params)
    v_exit = float(np.max(res["velocityP"]))
    return v_exit, res


def calibrate_impetus_to_vexit(target_v: float, dt: float = 1e-6, t_end: float = 0.02,
                               r1_scale: float = 1.0, p_start: float = 30e6,
                               l0_override_m: float | None = None,
                               V0_override: float | None = None,
                               f_low: float = 0.3, f_high: float = 4.0,
                               calib_debug: bool = True) -> Tuple[float, Dict[str, Any]]:
    """
    Binary search the impetus scale so that the predicted muzzle velocity matches
    the Vallier reference for a given burn-rate scale.
    """
    low, high = f_low, f_high
    best_res: Dict[str, Any] | None = None
    for _ in range(14):
        mid = 0.5 * (low + high)
        v_mid, res_mid = _simulate_exit_velocity(mid, r1_scale, dt, t_end, p_start, l0_override_m, V0_override=V0_override)
        best_res = res_mid
        if v_mid >= target_v:
            high = mid
        else:
            low = mid
    if calib_debug and best_res is not None:
        print(f"[calib] r1_scale={r1_scale:.3f} f_scale≈{high:.3f} v_exit≈{v_mid:.2f}")
    return high, best_res  # high is the smallest scale that meets/exceeds target


def dual_calibrate(target_v: float, target_pmax: float, dt: float = 1e-6, t_end: float = 0.02,
                  p_start: float = 30e6, l0_override_m: float | None = None,
                  target_texit_s: float | None = None, V0_override: float | None = None,
                  calib_debug: bool = False) -> Tuple[IBParams, Dict[str, Any]]:
    """
    Coarse search over burn-rate scale, inner binary search on impetus so we
    hit muzzle velocity; pick the pair that best matches target peak pressure.
    """
    candidates = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]
    best = None
    best_err = float("inf")
    best_params = None
    best_zp_exit = 0.0
    for r1s in candidates:
        f_scale, res = calibrate_impetus_to_vexit(target_v, dt=dt, t_end=t_end, r1_scale=r1s,
                                                  p_start=p_start, l0_override_m=l0_override_m,
                                                  V0_override=V0_override,
                                                  f_low=0.3, f_high=5.0, calib_debug=calib_debug)
        p_peak = float(np.max(res["pressureEqn"]) / 1e6)
        # estimate burn fraction at exit
        t_exit = res.get("t_exit", res["timeVec"][-1])
        idx_exit = int(np.argmax(res["timeVec"] >= t_exit))
        zp_exit = float(res["zp"][idx_exit]) if idx_exit < res["zp"].size else float(res["zp"][-1])
        err_p = abs(p_peak - target_pmax)
        err_texit = 0.0
        if target_texit_s is not None and not np.isnan(t_exit):
            # weight texit to be comparable to pressure error
            texit_weight = 80.0 * target_pmax / max(target_texit_s, EPS)  # heavier weight on texit
            err_texit = texit_weight * abs(t_exit - target_texit_s)
            # hard penalty if outside 0.05 ms band
            if abs(t_exit - target_texit_s) > 5e-5:
                err_texit *= 5.0
        burn_ok = zp_exit >= 0.99
        score = err_p + err_texit
        if not burn_ok:
            score += 50.0  # penalize incomplete burn
        if score < best_err:
            best_err = score
            best = res
            params = build_25mm_params(f_scale=f_scale, p_start_override=p_start,
                                       l0_override_m=l0_override_m, r1_scale=r1s, V0_override=V0_override)
            best_params = params
            best_zp_exit = zp_exit
        if calib_debug:
            print(f"[calib] r1_scale={r1s:.3f} f_scale={f_scale:.3f} "
                  f"Pmax={p_peak:.1f} MPa t_exit={t_exit*1e3:.3f} ms "
                  f"zp_exit={zp_exit:.3f} score={score:.1f}")
    return best_params, best


# ----------------------------- Satanic Ritual ----------------------------- #


# ----------------------------- 25 mm landmark calibration ----------------------------- #
# Reference landmarks (Vallier–Heydenreich + user-provided peak-location/timing).
# Units: meters, seconds, MPa, m/s.
REF_25MM_LANDMARKS = {
    # --- TP/HEI 25×137 mm (user-provided Vallier outputs) ---
    # Units: meters, seconds, MPa, m/s, kg/m^3
    "x_pmax_m": 0.12522,       # projectile travel at Pmax [m]
    "v_pmax_mps": 416.41,      # projectile velocity at Pmax [m/s]
    "t_pmax_s": 0.00111,       # time to reach Pmax [s]
    "p_muzzle_MPa": 63.43,     # muzzle pressure [MPa]
    "t_exit_s": 0.00302,       # projectile time in bore (travel time) [s]
    "rho_max_kgm3": 654.0,     # max propellant gas density in bore [kg/m^3]

    # Global targets (ammo reference)
    "v_exit_mps": float(ACTIVE_VALLIER_REF["V_exit_mps"]),
    "p_max_MPa": float(ACTIVE_VALLIER_REF["P_max_MPa"]),
}



def extract_metrics(res: Dict[str, Any]) -> Dict[str, float]:
    """Extract peak / exit landmarks from a single-shot IB result dict."""
    t = np.asarray(res["timeVec"], dtype=float)
    x = np.asarray(res["location"], dtype=float)
    v = np.asarray(res["velocityP"], dtype=float)
    p_MPa = np.asarray(res["pressureEqn"], dtype=float) / 1e6
    rho = np.asarray(res.get("rhoHist", np.array([np.nan])), dtype=float)
    rho_max = float(np.nanmax(rho)) if rho.size else float("nan")

    i_pmax = int(np.argmax(p_MPa))
    pmax = float(p_MPa[i_pmax])
    t_pmax = float(t[i_pmax])
    x_pmax = float(x[i_pmax])
    v_pmax = float(v[i_pmax])

    t_exit = float(res.get("t_exit", t[-1]))
    p_muzzle = float(np.interp(t_exit, t, p_MPa))
    v_exit = float(np.max(v))

    return {
        "pmax_MPa": pmax,
        "t_pmax_s": t_pmax,
        "x_pmax_m": x_pmax,
        "v_pmax_mps": v_pmax,
        "t_exit_s": t_exit,
        "p_muzzle_MPa": p_muzzle,
        "v_exit_mps": v_exit,
        "rho_max_kgm3": rho_max,
    }


def score_metrics(m: Dict[str, float], ref: Dict[str, float]) -> float:
    """Weighted objective for matching the landmark set."""
    def rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(b), 1e-12)

    s = 0.0
    s += 4.0 * rel(m["v_exit_mps"], ref["v_exit_mps"])
    s += 2.0 * rel(m["pmax_MPa"], ref["p_max_MPa"])
    s += 6.0 * rel(m["t_exit_s"], ref["t_exit_s"])
    s += 6.0 * rel(m["t_pmax_s"], ref["t_pmax_s"])
    s += 5.0 * rel(m["x_pmax_m"], ref["x_pmax_m"])
    s += 5.0 * rel(m["v_pmax_mps"], ref["v_pmax_mps"])
    s += 3.0 * rel(m["p_muzzle_MPa"], ref["p_muzzle_MPa"])
    if "rho_max_kgm3" in ref and "rho_max_kgm3" in m and np.isfinite(m["rho_max_kgm3"]):
        s += 1.5 * rel(m["rho_max_kgm3"], ref["rho_max_kgm3"])
    return float(s)



def calibrate_25mm_landmarks(
    ref: Dict[str, float] = REF_25MM_LANDMARKS,
    dt: float = 2e-6,
    t_end: float = 0.012,
    p_start: float = 18.33e6,
    l0_override_m: float = 0.100,
    V0_span: Tuple[float, float] = (0.80, 1.20),
    r1_span: Tuple[float, float] = (1.60, 3.20),
    fast: bool = False,
    calib_debug: bool = True,
) -> Tuple[IBParams, Dict[str, Any]]:
    """    Landmark calibration for 25×137 using a *bounded* search.

    We avoid a brute-force 3D grid over (r1, f, V0) by:
      - looping over (r1_scale, V0_override)
      - performing a binary search on f_scale to hit V_exit for each pair
        (re-using `calibrate_impetus_to_vexit`)

    This is dramatically faster and still lets you fit peak timing/location and
    muzzle pressure by leveraging V0_override and r1_scale.

    Args:
      fast: smaller candidate grids for quick iteration.
    """
    target_v = float(ref["v_exit_mps"])
    baseV0 = float(VALLIER_APDS_25X137_REF["V0_m3"])

    if fast:
        r1_grid = np.linspace(r1_span[0], r1_span[1], 9)
        V0_grid = baseV0 * np.linspace(V0_span[0], V0_span[1], 7)
    else:
        r1_grid = np.linspace(r1_span[0], r1_span[1], 13)
        V0_grid = baseV0 * np.linspace(V0_span[0], V0_span[1], 11)

    best_score = float("inf")
    best_params: IBParams | None = None
    best_metrics: Dict[str, float] | None = None

    for r1s in r1_grid:
        for V0o in V0_grid:
            f_scale, res = calibrate_impetus_to_vexit(
                target_v,
                dt=dt,
                t_end=t_end,
                r1_scale=float(r1s),
                p_start=float(p_start),
                l0_override_m=float(l0_override_m),
                V0_override=float(V0o),
                f_low=0.3,
                f_high=8.0,
                calib_debug=False,
            )
            m = extract_metrics(res)
            sc = score_metrics(m, ref)

            if sc < best_score:
                best_score = sc
                best_metrics = m
                # NOTE: params used in the sim are stored inside res["params"]
                best_params = res["params"]

    if best_params is None or best_metrics is None:
        raise RuntimeError("Calibration failed to evaluate any candidate.")

    info = {"score": float(best_score), "metrics": best_metrics}

    if calib_debug:
        m = best_metrics
        print(f"[25mm-calib] score={best_score:.6g}")
        print(f"[25mm-calib] f_scale={best_params.f_scale:.6f}, r1_scale={best_params.r1_scale:.6f}, V0={best_params.V0:.6e} m^3")
        print(f"[25mm-calib] v_exit={m['v_exit_mps']:.2f} m/s, pmax={m['pmax_MPa']:.2f} MPa, t_exit={1e3*m['t_exit_s']:.3f} ms")
        print(f"[25mm-calib] t_pmax={1e3*m['t_pmax_s']:.3f} ms, x_pmax={1e3*m['x_pmax_m']:.2f} mm, v_pmax={m['v_pmax_mps']:.2f} m/s")
        print(f"[25mm-calib] p_muzzle={m['p_muzzle_MPa']:.2f} MPa")

    return best_params, info

if __name__ == "__main__":
    # Landmark calibration (matches peak timing/location + muzzle pressure + exit time)
    assumed_l0 = 0.100  # m
    params, calib_info = calibrate_25mm_landmarks(
        ref=REF_25MM_LANDMARKS,
        dt=2e-6,          # coarse dt for calibration (fast)
        t_end=0.012,      # only need to cover exit + early aftereffect for landmarks
        p_start=18.33e6,
        l0_override_m=assumed_l0,
        fast=False,       # set True for quicker iteration (recommend False)
        calib_debug=True,
    )

    print("Calibration complete.")
    print(f" -> f_scale={params.f_scale:.6f}, r1_scale={params.r1_scale:.6f}, V0={params.V0:.6e} m^3, latch p_start={params.p_start/1e6:.1f} MPa")

    result = interior_ballistics_25mm(dt=1e-6, t_end=0.03, plot=True, debug=True, params=params)

    vmax = float(np.max(result['velocityP']))
    print(f"Predicted muzzle velocity (max): {vmax:.2f} m/s")
    print(f"t_exit: {result['t_exit']:.6f} s, burn completion (t_zp1): {result['t_zp1']}")
