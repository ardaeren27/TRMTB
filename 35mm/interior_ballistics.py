"""
interior_ballistics.py
single-shot, 2-phase interior + transitional model from Zielinski

Exports:
    interior_ballistics(...) -> dict with timeVec, hHist, Tgas, pressureEqn, etc.

If run as a script:
    python interior_ballistics.py
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utils import EPS


def thermo_model(zp: float, Tgas: float, velocityP: float, location: float) -> Tuple[float, float, float, float, float, float]:
    """
    Phase-1 ODE RHS (from article thermoModel).

    Inputs:
        zp         : burn ratio
        Tgas       : gas temperature [K]
        velocityP  : projectile velocity [m/s]
        location   : projectile location x [m] (relative to breech)
    Returns:
        rho, pressureEqn, d(x)/dt, d(v)/dt, d(zp)/dt, d(theta)/dt
    """
# Coefs and Params (self-contained, so no local-global trickery can fool me later on)
    mP = 0.380          # projectile mass [kg]
    mp = 0.376          # propellant mass [kg]
    s = 9.98e-4         # cross-sectional area of barrel bore [m^2]
    V0 = 373e-6         # volume of empty cannon chamber [m^3]
    K = 1.37            # constant of coefficient of secondary works [-]
    f = 1.071e6         # gas impetus [J/kg]
    eta = 1.064e-3      # covolume of propellant gases [m^3/kg]
    gamma = 1.2         # specific heat ratio, c_p/c_v [-]
    R = 340.0           # gas constant of propellant gases [J/kg*K]
    rho_p = 1600.0      # gas density [kg/m^3]

    r1 = 0.597e-9       # Linear burn rate coefficient [m/Pa*s]
    S1 = 134.4e-6       # initial surface grain of propellant [m^2]
    LAMBDA1 = 75.2e-9   # initial volume of the propellant grain [m^3]
    Kappa1 = 0.755      # shape coef [-]
    lambda1 = 0.159     # shape coef [-]

    denom = V0 + s * location - (mp / rho_p) * (1 - zp) - eta * mp * zp
    pressureEqn = (mp * zp * R * Tgas) / denom
    rho = (mp * zp) / denom

    m_eff = mP
    phi = 1.37          # coef of secondary works [-]

    diffLocation = velocityP
    diffVelocity = pressureEqn * s / (phi * m_eff)

    diffBurnRate = (S1 / LAMBDA1) * r1 * pressureEqn * np.sqrt(1.0 + 4.0 * zp * lambda1 / Kappa1)
    diffBurnRate = max(diffBurnRate, 0.0)
    if zp >= 1.0:
        diffBurnRate = 0.0

    diffTheta = ((f - R * Tgas) * mp * diffBurnRate - (gamma - 1.0) * phi * m_eff * velocityP * diffVelocity) / max(mp * zp, EPS)
    return float(rho), float(pressureEqn), float(diffLocation), float(diffVelocity), float(diffBurnRate), float(diffTheta)


def thermo_model_v2(zp: float, Tgas: float, zeta: float) -> Tuple[float, float, float, float, float]:
    # Choked Out-FLow Transitional Ballistics
    """
    Phase-2 ODE RHS (thermoModelv2).

    Inputs:
        zp    : burn ratio (continues evolving)
        Tgas  : gas temperature [K]
        zeta  : expelled fraction
    Returns:
        rho2, pressureEqn2, d(zp)/dt, d(zeta)/dt, d(theta)/dt
    """
    # For parameter comments (units, names etc.) see above
    mp = 0.376
    s = 9.98e-4
    V0 = 373e-6
    lm = 2.934
    f = 1.071e6
    eta = 1.064e-3
    gamma = 1.2
    R = 340.0
    rho_p = 1600.0

    r1 = 0.597e-9
    S1 = 134.4e-6
    LAMBDA1 = 75.2e-9
    Kappa1 = 0.755
    lambda1 = 0.159

    denom = V0 + s * lm - (mp / rho_p) * (1.0 - zp) - eta * mp * (zp - zeta)
    pressureEqn2 = (mp * (zp - zeta) * R * Tgas) / denom
    rho2 = (mp * (zp - zeta)) / denom

    CF_CHOKE = np.sqrt(gamma) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    diffZeta = (s * pressureEqn2) / (mp * np.sqrt(R * Tgas)) * CF_CHOKE

    if zp < 1.0 - 1e-12:
        diffZp = (S1 / LAMBDA1) * r1 * pressureEqn2 * np.sqrt(1.0 + 4.0 * zp * lambda1 / Kappa1)
        diffZp = max(diffZp, 0.0)
    else:
        diffZp = 0.0

    gas_mass_frac = max(zp - zeta, EPS)
    diffTheta2 = ((f - R * Tgas) * diffZp - (gamma - 1.0) * R * Tgas * diffZeta) / gas_mass_frac
    return float(rho2), float(pressureEqn2), float(diffZp), float(diffZeta), float(diffTheta2)


def interior_ballistics(*, dt: float = 1e-6, t_end: float = 0.1, plot: bool = True, debug: bool = False) -> Dict[str, Any]:
    """
    Run the single-shot 2-phase interior + transitional ballistics model.

    Defaults: (Runs quite fast and accurate with default settings)
        dt   = 1e-6
        tEnd = 0.1
    """
    # -------------------- constants (script scope) --------------------
    mP = 0.380
    mp = 0.376
    s = 9.98e-4
    V0 = 373e-6
    lm = 2.934
    K = 1.37
    f = 1.071e6
    eta = 1.064e-3
    gamma = 1.2
    R = 340.0
    rho_p = 1600.0

    p_start = 30e6
    released = False
    t_release = np.nan

    l0 = 0.216
    li = np.array([l0, 0.385, 0.535, 0.880, 2.081, 2.980], dtype=float)
    Di = 0.035 * np.ones(6, dtype=float)

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
    theta[0] = R * 3150.0
    Tgas[0] = 3150.0
    # Tgas[0] = Flame Temperature
    # (combustion begins and volume is constant until projectile release; therefore, initial gas temp should be flame temp by definition)

    state = np.array([0.0, 0.0, zp[0], theta[0]], dtype=float)  # [x v zp theta]
    t_on = np.full(6, np.nan, dtype=float)

    t_zp1 = np.nan
    # BEHOLD! BEAUTIFUL RK LOOPS AHEAD
    # -------------------- phase 1 RK4 loop --------------------
    n = 0  # Python 0-based; corresponds to MATLAB n=1 after first accept
    t = 0.0
    t_exit = np.nan

    while t < t_end:
        prev_state = state.copy()
        prev_t = t

        # k1
        _, P1, dx1, dv1, dzp1, dth1 = thermo_model(prev_state[2], prev_state[3] / R, prev_state[1], prev_state[0])
        if (not released) and (P1 < p_start):
            dx1 = 0.0
            dv1 = 0.0
            T1 = prev_state[3] / R
            z1 = max(prev_state[2], EPS)
            dth1 = ((f - R * T1) * mp * dzp1) / (mp * z1)
        k1 = dt * np.array([dx1, dv1, dzp1, dth1], dtype=float)

        # k2
        mid = prev_state + 0.5 * k1
        mid[2] = min(max(mid[2], 0.0), 1.0)
        _, P2, dx2, dv2, dzp2, dth2 = thermo_model(mid[2], mid[3] / R, mid[1], mid[0])
        if (not released) and (P2 < p_start):
            dx2 = 0.0
            dv2 = 0.0
            T2 = mid[3] / R
            z2 = max(mid[2], EPS)
            dth2 = ((f - R * T2) * mp * dzp2) / (mp * z2)
        k2 = dt * np.array([dx2, dv2, dzp2, dth2], dtype=float)

        # k3
        mid = prev_state + 0.5 * k2
        mid[2] = min(max(mid[2], 0.0), 1.0)
        _, P3, dx3, dv3, dzp3, dth3 = thermo_model(mid[2], mid[3] / R, mid[1], mid[0])
        if (not released) and (P3 < p_start):
            dx3 = 0.0
            dv3 = 0.0
            T3 = mid[3] / R
            z3 = max(mid[2], EPS)
            dth3 = ((f - R * T3) * mp * dzp3) / (mp * z3)
        k3 = dt * np.array([dx3, dv3, dzp3, dth3], dtype=float)

        # k4
        endS = prev_state + k3
        endS[2] = min(max(endS[2], 0.0), 1.0)
        _, P4, dx4, dv4, dzp4, dth4 = thermo_model(endS[2], endS[3] / R, endS[1], endS[0])
        if (not released) and (P4 < p_start):
            dx4 = 0.0
            dv4 = 0.0
            T4 = endS[3] / R
            z4 = max(endS[2], EPS)
            dth4 = ((f - R * T4) * mp * dzp4) / (mp * z4)
        k4 = dt * np.array([dx4, dv4, dzp4, dth4], dtype=float)

        new_state = prev_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        new_state[2] = min(max(new_state[2], 0.0), 1.0)

        # release latch detection (uses EOS at step start/end)
        # (hold the ammunition until pressure reaches some value)
        # (without latching, outputs are inaccurate so quite important)
        if not released:
            P_prev = (mp * prev_state[2] * R * (prev_state[3] / R)) / (V0 + s * prev_state[0] - mp * (1 - prev_state[2]) / rho_p - eta * mp * prev_state[2])
            P_new = (mp * new_state[2] * R * (new_state[3] / R)) / (V0 + s * new_state[0] - mp * (1 - new_state[2]) / rho_p - eta * mp * new_state[2])
            if (P_prev < p_start) and (P_new >= p_start):
                fracR = (p_start - P_prev) / max(P_new - P_prev, EPS)
                t_release = prev_t + max(0.0, min(1.0, fracR)) * dt
                released = True
            elif P_new >= p_start:
                released = True
                t_release = prev_t + dt

        x_prev = prev_state[0]
        x_new = new_state[0]
        crossed = (x_prev < lm) and (x_new >= lm)

        if crossed:
            frac = (lm - x_prev) / max(x_new - x_prev, EPS)
            exit_state = prev_state + frac * (new_state - prev_state)
            exit_state[2] = min(max(exit_state[2], 0.0), 1.0)
            t = prev_t + frac * dt
            state = exit_state

            # bookkeep at exit (index n)
            location[n] = state[0]
            velocityP[n] = state[1]
            zp[n] = state[2]
            theta[n] = state[3]
            Tgas[n] = theta[n] / R

            pressureEqn[n] = (mp * zp[n] * R * Tgas[n]) / (V0 + s * location[n] - mp * (1 - zp[n]) / rho_p - eta * mp * zp[n])
            rhoHist[n] = (mp * zp[n]) / (V0 + s * location[n] - mp * (1 - zp[n]) / rho_p - eta * mp * zp[n])

            # HTC gating (No gas inflow to section until bullet tail clears entrance)
            # This causes the jumps in the HTC graph
            v_now = velocityP[n]
            l_now = location[n]
            tail_abs = l0 + max(l_now, 0.0)
            wi = (li / (l0 + l_now)) * v_now
            h_now = (6.1 / (Di ** 0.2)) * (rhoHist[n] * np.abs(wi)) ** 0.8

            present = li <= tail_abs
            h_now = np.where(present, h_now, 0.0)
            just_on = present & np.isnan(t_on)
            t_on[just_on] = t
            hHist[n, :] = h_now

            timeVec[n] = t
            t_exit = t
            n += 1  # reserve next slot for phase 2
            break

        # no crossing: accept full step at t+dt
        state = new_state

        location[n] = state[0]
        velocityP[n] = state[1]
        zp[n] = state[2]
        theta[n] = state[3]
        Tgas[n] = theta[n] / R

        pressureEqn[n] = (mp * zp[n] * R * Tgas[n]) / (V0 + s * location[n] - mp * (1 - zp[n]) / rho_p - eta * mp * zp[n])
        rhoHist[n] = (mp * zp[n]) / (V0 + s * location[n] - mp * (1 - zp[n]) / rho_p - eta * mp * zp[n])

        v_now = velocityP[n]
        l_now = location[n]
        tail_abs = l0 + max(l_now, 0.0)
        wi = (li / (l0 + l_now)) * v_now
        h_now = (6.1 / (Di ** 0.2)) * (rhoHist[n] * np.abs(wi)) ** 0.8

        present = li <= tail_abs
        h_now = np.where(present, h_now, 0.0)
        just_on = present & np.isnan(t_on)
        t_on[just_on] = t
        hHist[n, :] = h_now

        if debug:
            print(f"\nTime: {t:.8f} s")
            print(f"Location: {location[n]:.6f} m")
            print(f"Velocity: {velocityP[n]:.6f} m/s")
            print(f"Burn Ratio (zp): {zp[n]:.6f}")
            print(f"Gas Temperature: {Tgas[n]:.2f} K")
            print(f"Pressure: {pressureEqn[n]/1000:.2f} kPa")

        t = prev_t + dt
        timeVec[n] = t
        n += 1

        if n >= maxStep - 2:
            # prevent overflow if user t_end is too large for prealloc
            break

    if np.isnan(t_exit):
        # projectile didn't reach muzzle; force exit
        # this means your inputs were so stupid to the point that it's impressive, well done champ
        t_exit = t

    # keep last piston position/velocity for plotting
    location[n:] = lm
    velocityP[n:] = 0.0

    # -------------------- hand-off continuity (seed phase 2) --------------------
    zetaCur = 0.0
    zp_cur = zp[max(n - 1, 0)]
    T_phase2 = Tgas[max(n - 1, 0)]

    den2 = V0 + s * lm - mp * (1.0 - zp_cur) / rho_p - eta * mp * (zp_cur - zetaCur)
    pressureEqn[max(n - 1, 0)] = (mp * (zp_cur - zetaCur) * R * T_phase2) / den2
    rhoHist[max(n - 1, 0)] = (mp * (zp_cur - zetaCur)) / den2
    zetaHist[max(n - 1, 0)] = zetaCur

    # Seed h for phase 2 using critical outflow speed (assumed choked outflow)
    wcr = np.sqrt(gamma * R * T_phase2) * np.sqrt(2.0 / (gamma + 1.0))
    wi2 = (li / (l0 + lm)) * wcr
    hHist[max(n - 1, 0), :] = (6.1 / (Di ** 0.2)) * (rhoHist[max(n - 1, 0)] * np.abs(wi2)) ** 0.8

    # BEHOLD! ANOTHER BEAUTIFUL RK LOOP AHEAD
    # -------------------- phase 2 RK4 loop --------------------
    zeta = zetaCur
    theta_here = theta[max(n - 1, 0)]
    zp_here = zp_cur

    while ((zp_here - zeta) > 1e-6) and (t < t_end) and (n < maxStep - 1):
        # k1
        _, _, dzp1, dzeta1, dth1 = thermo_model_v2(zp_here, theta_here / R, zeta)
        if zp_here >= 1.0 - 1e-12:
            dzp1 = 0.0
        dzp1 = max(dzp1, 0.0)
        k1 = dt * np.array([dzeta1, dth1, dzp1], dtype=float)

        # k2
        z_mid = min(max(zp_here + 0.5 * k1[2], 0.0), 1.0)
        _, _, dzp2, dzeta2, dth2 = thermo_model_v2(z_mid, (theta_here + 0.5 * k1[1]) / R, zeta + 0.5 * k1[0])
        if z_mid >= 1.0 - 1e-12:
            dzp2 = 0.0
        dzp2 = max(dzp2, 0.0)
        k2 = dt * np.array([dzeta2, dth2, dzp2], dtype=float)

        # k3
        z_mid = min(max(zp_here + 0.5 * k2[2], 0.0), 1.0)
        _, _, dzp3, dzeta3, dth3 = thermo_model_v2(z_mid, (theta_here + 0.5 * k2[1]) / R, zeta + 0.5 * k2[0])
        if z_mid >= 1.0 - 1e-12:
            dzp3 = 0.0
        dzp3 = max(dzp3, 0.0)
        k3 = dt * np.array([dzeta3, dth3, dzp3], dtype=float)

        # k4
        z_end = min(max(zp_here + k3[2], 0.0), 1.0)
        _, _, dzp4, dzeta4, dth4 = thermo_model_v2(z_end, (theta_here + k3[1]) / R, zeta + k3[0])
        if z_end >= 1.0 - 1e-12:
            dzp4 = 0.0
        dzp4 = max(dzp4, 0.0)
        k4 = dt * np.array([dzeta4, dth4, dzp4], dtype=float)

        zeta = zeta + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        theta_here = theta_here + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        old_zp = zp_here
        zp_here = zp_here + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
        zp_here = min(max(zp_here, 0.0), 1.0)

        if np.isnan(t_zp1) and (old_zp < 1.0) and (zp_here >= 1.0):
            t_zp1 = t + dt

        theta[n] = theta_here
        Tgas[n] = theta_here / R
        zp[n] = zp_here
        zetaHist[n] = zeta

        den2 = V0 + s * lm - mp * (1.0 - zp_here) / rho_p - eta * mp * (zp_here - zeta)
        P2 = (mp * (zp_here - zeta) * R * Tgas[n]) / den2
        pressureEqn[n] = max(P2, 101325.0)
        rhoHist[n] = (mp * (zp_here - zeta)) / den2

        wcr = np.sqrt(gamma * R * Tgas[n]) * np.sqrt(2.0 / (gamma + 1.0))
        wi2 = (li / (l0 + lm)) * wcr
        hHist[n, :] = (6.1 / (Di ** 0.2)) * (rhoHist[n] * np.abs(wi2)) ** 0.8

        t = t + dt
        timeVec[n] = t
        n += 1

        if debug:
            print(f"\nTime: {t:.8f} s")
            print(f"Zeta: {zeta:.6f}")
            print(f"Gas Temperature: {Tgas[n-1]:.2f} K")
            print(f"Pressure: {pressureEqn[n-1]/1000:.2f} kPa")

        if pressureEqn[n - 1] <= 0.18e6:
            break

    # -------------------- truncate storage --------------------
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

    # -------------------- plots --------------------
    if plot:
        _plot_ballistics(timeVec, dt, t_exit, t_zp1, t_release,
                         Tgas, pressureEqn, location, velocityP, zp, zetaHist, rhoHist, hHist)

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
        "t_on": t_on,
        "constants": {
            "mP": mP, "mp": mp, "s": s, "V0": V0, "lm": lm,
            "K": K, "f": f, "eta": eta, "gamma": gamma, "R": R, "rho_p": rho_p,
            "p_start": p_start, "l0": l0, "li": li, "Di": Di,
        }
    }


def _plot_ballistics(timeVec, dt, t_exit, t_zp1, t_release,
                     Tgas, pressureEqn, location, velocityP, zp, zetaHist, rhoHist, hHist):
    # Main 3x3 tiled figure
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Interior Ballistics")
    gs = fig.add_gridspec(3, 3, wspace=0.35, hspace=0.55)

    # Gas temperature
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(timeVec, Tgas, linewidth=1.2)
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("T_gas  [K]")
    ax.set_title("Gas Temperature")
    ax.grid(True)

    # Chamber pressure with linear + log on twin y
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
    # MATLAB-style xline labels
    try:
        y_top = axL.get_ylim()[1]
        axL.text(t_exit, 0.92*y_top, "t_exit", rotation=90, va="top", ha="right", fontsize=9)
        if not np.isnan(t_zp1):
            axL.text(t_zp1, 0.92*y_top, "z_p=1", rotation=90, va="top", ha="right", fontsize=9, color="r")
        if not np.isnan(t_release):
            axL.text(t_release, 0.92*y_top, "release", rotation=90, va="top", ha="right", fontsize=9, color="g")
    except Exception:
        pass

    # Piston velocity (FD of x vs stored v)
    # Last entry is always off for some reason, but it shouldn't really be a problem)
    ax = fig.add_subplot(gs[0, 2])
    idx = timeVec <= t_exit
    # FD of location for pre-exit visualization
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

    # Propellant burned
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
            ax.text(t_zp1, 0.92*y_top, "z_p=1", rotation=90, va="top", ha="right", fontsize=9, color="r")
        except Exception:
            pass

    # Gas expelled
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(timeVec, zetaHist, linewidth=1.2)
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("zeta  [-]")
    ax.set_title("Gas Expelled")
    ax.grid(True)

    # Gas density
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(timeVec, rhoHist, linewidth=1.2)
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("rho  [kg/m^3]")
    ax.set_title("Gas Density")
    ax.grid(True)

    # Convection coefficients + Tgas overlay (span last row)
    axL = fig.add_subplot(gs[2, :])
    axR = axL.twinx()
    colors = ["k", "r", "b", "g", "m", "y"]
    for k in range(6):
        axL.plot(timeVec, hHist[:, k], linewidth=1.1, color=colors[k], label=f"P{k+1}")
    axL.set_xlim(0.0, 0.007)
    axL.set_ylim(0.0, 2.75e5)
    axL.set_xlabel("t  [s]")
    axL.set_ylabel("h_i  [W/(m^2·K)]")
    axL.grid(True)
    axR.plot(timeVec, Tgas, "r--", linewidth=1.2, label="Tgas")
    axR.set_ylim(0.0, 3.5e3)
    axR.set_ylabel("T_gas  [K]")
    axL.set_title("Convection Coefficients & Gas Temperature")
    # Legend (combine left + right axes)
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="upper left")

    # Separate plot
    fig2, axL2 = plt.subplots(figsize=(10, 4))
    axR2 = axL2.twinx()
    colors = ["k", "r", "b", "g", "m", "y"]
    for k in range(6):
        axL2.plot(timeVec, hHist[:, k], linewidth=1.1, color=colors[k], label=f"P{k+1}")
    axL2.set_xlim(0.0, 0.007)
    axL2.set_ylim(0.0, 2.75e5)
    # MATLAB-style major ticks and labels (0 .. 2.75e5 step 2.5e4)
    yL_ticks = np.arange(0.0, 2.75e5 + 1.0, 2.5e4)
    axL2.set_yticks(yL_ticks)
    axL2.set_yticklabels([
        "0","2.5×10^4","5.0×10^4","7.5×10^4","1.0×10^5","1.25×10^5","1.50×10^5",
        "1.75×10^5","2.00×10^5","2.25×10^5","2.50×10^5","2.75×10^5"
    ])
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

    # Minor ticks
    axL2.minorticks_on()
    axR2.minorticks_on()
    axL2.tick_params(which="both", direction="in")
    axR2.tick_params(which="both", direction="in")
    plt.show()


if __name__ == "__main__":
    res = interior_ballistics(plot=True, debug=True)
    print("interior_ballistics run complete.")
