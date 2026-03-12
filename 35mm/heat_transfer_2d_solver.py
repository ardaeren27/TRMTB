"""
heat_transfer_2d_solver.py
Axisymmetric (r,z) transient heat conduction (finite-volume, fully implicit).
logs inner/outer wall temps for 6 sections + final field.

API:
    out = heat_transfer_2d_solver(**options)

Options:
    steel: 'DUPLEX' | '30HN2MFA' | '38HMJ'
    tCr_m: chromium thickness [m]
    tCr_um: chromium thickness [µm] (overrides tCr_m if given)
    Nr, Nz: mesh counts
    theta: implicitness factor (=1.0 for fully implicit Backwards Euler Scheme)
    dt_fd: dt during heating (Fine advised)
    tEnd_fd: fallback stop time (if no bc provided / user-set)
    dt_tail: coarse dt in tail (default dt_fd) (Coarse advised)
    cool_tail_s: extra cooling after last heating [s]
    plot, debug, debug_stride, flux_check
    bc: dict with {t [s], h [Nt x Nz], Tg [K or °C]} (Tg in K is auto-converted to °C)

Return dict mirrors MATLAB struct
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from utils import EPS, unique_stable, interp1_linear_extrap, interp1_linear_extrap_matrix, print_solver_banner
from interior_ballistics import interior_ballistics


def _arange_inclusive(start: float, step: float, stop: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be > 0")
    if stop < start:
        return np.array([start], dtype=float)
    n = int(np.floor((stop - start) / step + 1e-12))
    arr = start + step * np.arange(n + 1, dtype=float)
    # ensure last isn't beyond stop due to rounding
    arr = arr[arr <= stop + 1e-12]
    if arr.size == 0:
        arr = np.array([start], dtype=float)
    return arr


def face_k(kL: float, kR: float) -> float:
    return float(2.0 * kL * kR / max(kL + kR, EPS))  # harmonic mean for properties


def heat_transfer_2d_solver(
    *,
    steel: str = "DUPLEX",
    tCr_m: float = 0.0,
    tCr_um: Optional[float] = None,
    Nr: int = 240,
    Nz: int = 120,
    theta: float = 1.0,
    dt_fd: float = 1e-5,
    tEnd_fd: float = 8e-3,
    dt_tail: Optional[float] = None,
    cool_tail_s: float = 0.0,
    plot: bool = True,
    debug: bool = False,
    debug_stride: int = 50,
    flux_check: bool = True,
    bc: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print(" {\\__/}")
    print("( • . •)")
    print("/ > 🥕>  Starting 2-D FV heat solver")
    user_set_tEnd = False  # Users can pass tEnd_fd. I'm not quite sure what I was thinking when adding this but keep FALSE, otherwise things go wrong (well, sometimes)

    # Emulate with a heuristic: (So stupid inputs don't break anything, not very sound but works)
    if tEnd_fd != 8e-3:
        user_set_tEnd = True

    if tCr_um is not None:
        tCr_m = float(tCr_um) * 1e-6
    if tCr_m < 0:
        raise ValueError("tCr must be >= 0")

    if dt_tail is None:
        dt_tail = dt_fd  # fallback

    steel = str(steel).upper()
    print_solver_banner()

    # -------------------- geometry --------------------
    Rin = 0.0175
    T0 = 20.0
    hout = 9.2

    sections = {
        "z_mid": np.array([0.2649, 0.4605, 0.7075, 1.4805, 2.5305, 3.0655], dtype=float),
        "R_out": np.array([0.055, 0.0555, 0.0575, 0.052, 0.044, 0.031], dtype=float),
        "names": ["S1", "S3", "S4", "S16", "S25", "S30"],
    }

    L = float(np.max(sections["z_mid"]) + 0.20)

    z_breaks = sections["z_mid"].reshape(-1)
    Rout_breaks = sections["R_out"].reshape(-1)

    def Rout_fun(zz: np.ndarray) -> np.ndarray:
        zz = np.asarray(zz, dtype=float)
        Rout = np.interp(zz, z_breaks, Rout_breaks)
        # extrapolate linearly
        if z_breaks.size >= 2:
            mL = (Rout_breaks[1] - Rout_breaks[0]) / max(z_breaks[1] - z_breaks[0], EPS)
            mR = (Rout_breaks[-1] - Rout_breaks[-2]) / max(z_breaks[-1] - z_breaks[-2], EPS)
            left = zz < z_breaks[0]
            right = zz > z_breaks[-1]
            if np.any(left):
                Rout[left] = Rout_breaks[0] + mL * (zz[left] - z_breaks[0])
            if np.any(right):
                Rout[right] = Rout_breaks[-1] + mR * (zz[right] - z_breaks[-1])
        return Rout

    Rcr = Rin + float(tCr_m)

    # -------------------- mesh --------------------
    z = np.linspace(0.0, L, int(Nz))
    dz = float(z[1] - z[0])
    Rout_z = Rout_fun(z)
    tmax = float(np.max(Rout_z) - Rin)
    r_max = Rin + tmax

    r = np.linspace(Rin, r_max, int(Nr))  # cell centers
    r_face = np.zeros(int(Nr) + 1, dtype=float)
    r_face[0] = Rin
    r_face[-1] = r_max
    for i in range(1, int(Nr)):
        r_face[i] = 0.5 * (r[i - 1] + r[i])

    valid = np.zeros((int(Nr), int(Nz)), dtype=bool)
    i_end = np.zeros(int(Nz), dtype=int)
    for j in range(int(Nz)):
        idx = np.where(r_face[1:] <= Rout_z[j] + 1e-15)[0]
        Ni = int(idx[-1] + 1) if idx.size else 0
        if Ni > 0:
            valid[:Ni, j] = True
        i_end[j] = Ni

    # precompute FV geometry
    # See Patankar's book for further reference

    # BEHOLD! BEAUTIFUL MATHEMATICS BEYOND
    V = np.zeros((int(Nr), int(Nz)), dtype=float)
    A_rW = np.zeros_like(V)
    A_rE = np.zeros_like(V)
    A_zS = np.zeros_like(V)
    A_zN = np.zeros_like(V)
    for j in range(int(Nz)):
        Ni = i_end[j]
        for i in range(Ni):
            rW = r_face[i]
            rE = r_face[i + 1]
            V[i, j] = np.pi * (rE**2 - rW**2) * dz
            A_rW[i, j] = 2.0 * np.pi * rW * dz
            A_rE[i, j] = 2.0 * np.pi * rE * dz
            A_zS[i, j] = np.pi * (rE**2 - rW**2)
            A_zN[i, j] = A_zS[i, j]

    # -------------------- boundary conditions --------------------
    print("Loading boundary conditions...")
    if bc is not None:
        t_ib = np.asarray(bc["t"], dtype=float).reshape(-1)
        h_data = np.asarray(bc["h"], dtype=float)
        Tgas_in = np.asarray(bc["Tg"], dtype=float).reshape(-1)
    else:
        ib = interior_ballistics(plot=False, debug=False)
        t_ib = np.asarray(ib["timeVec"], dtype=float).reshape(-1)
        h_data = np.asarray(ib["hHist"], dtype=float)
        Tgas_in = np.asarray(ib["Tgas"], dtype=float).reshape(-1)

    # Convert K->°C if it is needed (for some reason, idk)
    if np.median(Tgas_in) > 200.0:
        Tgas = Tgas_in - 273.15
    else:
        Tgas = Tgas_in.copy()

    # Force h_data to 6 columns (map 30→6 if needed)
    # ASSIGNING MAPPING IS TRICKY, ALSO NOT WELL EXPLAINED IN THE ARTICLE
    # THEREFORE, MAPPING IN THE ARTICLE IS TAKEN EXACTLY AS IT IS
    # ANY CHANGES MAY OR MAY NOT WORK FOR OTHER GEOMETRIES
    # THEREFORE, STRONG FOE AHEAD
    # (ANY VALUE OTHER THAN 6 OR 30 IS NOT ADVISED)
    if h_data.shape[1] != 6:
        if h_data.shape[1] == 30:
            print("Warning: hHist has 30 cols; downselecting to 6 canonical sections.")
            pick = np.array([1, 5, 10, 16, 25, 30], dtype=int) - 1
            h_data = h_data[:, pick]
        else:
            raise ValueError(f"Expected 6 HTC columns, got {h_data.shape[1]}.")

    # trim lengths to min (minor memory conservation purposes)
    min_len = int(min(t_ib.size, h_data.shape[0], Tgas.size))
    t_ib = t_ib[:min_len]
    h_data = h_data[:min_len, :]
    Tgas = Tgas[:min_len]

    # If user didn't force tEnd, default sim end = BC end
    if (bc is not None) and (not user_set_tEnd):
        tEnd_fd = float(t_ib[-1])

    # -------- build piecewise time vector (fine during heating, coarse during tail) --------
    h_eps = 1e-6
    h_max_over_z = np.max(h_data, axis=1)
    idx_last_heat = np.where(h_max_over_z > h_eps)[0]
    if idx_last_heat.size == 0:
        t_heat_end = 0.0
    else:
        t_heat_end = float(t_ib[idx_last_heat[-1]])

    t_tail_end = t_heat_end + max(0.0, float(cool_tail_s))

    if user_set_tEnd:
        t_stop = float(tEnd_fd)
    else:
        t_stop = float(max(tEnd_fd, t_tail_end))

    # 1) heating zone with fine dt
    t = _arange_inclusive(0.0, float(dt_fd), min(t_heat_end, t_stop))

    # 2) tail cooling zone with coarse dt_tail
    if (t_tail_end > t[-1] + 1e-12) and (cool_tail_s > 0):
        t2_start = max(t[-1], t_heat_end)
        t2 = _arange_inclusive(t2_start + float(dt_tail), float(dt_tail), min(t_tail_end, t_stop))
        t = np.concatenate([t, t2])

    # 3) If t_stop still extends beyond tail_end, fill rest with fine dt_fd again
    if t_stop > t[-1] + 1e-12:
        t3 = _arange_inclusive(t[-1] + float(dt_fd), float(dt_fd), t_stop)
        t = np.concatenate([t, t3])

    t, _ = unique_stable(t)
    Nt = int(t.size)

    # -------------------- map z -> 6 sections --------------------
    zone_id = np.zeros(int(Nz), dtype=int)
    for j in range(int(Nz)):
        zone_id[j] = int(np.argmin(np.abs(z[j] - sections["z_mid"])))  # 0..5

    jstar = np.zeros(6, dtype=int)
    for k2 in range(6):
        jstar[k2] = int(np.argmin(np.abs(z - sections["z_mid"][k2])))

    dz_mm = 1e3 * np.abs(z[jstar] - sections["z_mid"])
    print("Δz to section mids [mm]: " + " ".join([f"{v:.1f}" for v in dz_mm]))

    # -------------------- material props --------------------
    # DEFINE ANY NEW MATERIAL HERE #
    T_props = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)
    k_chrom = np.array([94, 90, 85, 80, 75, 70, 66, 62, 59, 56, 53], dtype=float)
    cp_chrom = np.array([450, 480, 520, 560, 590, 610, 630, 650, 670, 690, 710], dtype=float)
    rho_chrom = np.array([7200, 7195, 7185, 7170, 7155, 7140, 7125, 7110, 7095, 7080, 7065], dtype=float)

    T_D = T_props.copy()
    k_D = np.array([13.3, 15.3, 17.0, 17.8, 18.1, 18.7, 20.0, 20.8, 21.4, 23.7, 25.9], dtype=float)
    cp_D = np.array([417, 462, 515, 548, 567, 576, 582, 588, 602, 629, 668], dtype=float)
    rho_D = np.array([7740, 7720, 7690, 7650, 7610, 7580, 7540, 7490, 7450, 7400, 7340], dtype=float)

    T_k_30 = np.array([54.2, 149.1, 250.0, 352.0, 453.3, 553.6, 651.1, 704.4, 723.0, 743.3, 763.0, 783.1, 802.8, 822.9, 842.8, 904.9, 1004.3], dtype=float)
    k_30 = np.array([35.9, 37.3, 36.0, 33.8, 30.9, 27.2, 19.7, 17.1, 16.0, 15.8, 17.1, 18.7, 19.3, 19.5, 19.7, 20.3, 20.6], dtype=float)
    T_cp_30 = np.array([38, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 991], dtype=float)
    cp_30 = 1000.0 * np.array([0.440, 0.462, 0.475, 0.492, 0.505, 0.517, 0.528, 0.539, 0.550, 0.560, 0.569, 0.579, 0.589, 0.598, 0.607, 0.616, 0.625, 0.634, 0.642, 0.658], dtype=float)
    T_rho_30 = np.array([50, 100, 200, 400, 600, 700, 720, 725, 730, 735, 740, 750, 765, 770, 775, 785, 800, 900, 1060], dtype=float)
    rho_30 = 1000.0 * np.array([7.77, 7.75, 7.72, 7.65, 7.59, 7.55, 7.55, 7.55, 7.55, 7.55, 7.55, 7.57, 7.58, 7.59, 7.59, 7.59, 7.58, 7.53, 7.46], dtype=float)

    T_k_38 = np.array([50.9, 149.0, 250.0, 351.3, 453.4, 553.6, 654.7, 704.5, 741.0, 762.6, 782.7, 802.8, 811.9, 821.8, 842.3, 904.7, 1004.2], dtype=float)
    k_38 = np.array([30.0, 33.6, 34.4, 33.0, 30.7, 27.4, 22.5, 19.4, 16.4, 19.3, 20.9, 23.2, 24.4, 25.4, 26.3, 27.8, 28.9], dtype=float)
    T_cp_38 = np.array([38, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 991], dtype=float)
    cp_38 = 1000.0 * np.array([0.458, 0.485, 0.502, 0.525, 0.543, 0.559, 0.574, 0.587, 0.598, 0.609, 0.618, 0.626, 0.634, 0.640, 0.645, 0.650, 0.653, 0.656, 0.657, 0.658], dtype=float)
    T_rho_38 = np.array([50, 100, 200, 400, 600, 780, 795, 800, 820, 830, 840, 850, 860, 870, 880, 890, 900, 1000, 1060], dtype=float)
    rho_38 = 1000.0 * np.array([7.66, 7.65, 7.62, 7.55, 7.48, 7.42, 7.41, 7.42, 7.43, 7.43, 7.43, 7.43, 7.43, 7.42, 7.42, 7.42, 7.41, 7.37, 7.34], dtype=float)

    if steel == "DUPLEX":
        T_k_S, kS_tab = T_D, k_D
        T_cp_S, cpS_tab = T_D, cp_D
        T_rho_S, rhoS_tab = T_D, rho_D
    elif steel == "30HN2MFA":
        T_k_S, kS_tab = T_k_30, k_30
        T_cp_S, cpS_tab = T_cp_30, cp_30
        T_rho_S, rhoS_tab = T_rho_30, rho_30
    elif steel == "38HMJ":
        T_k_S, kS_tab = T_k_38, k_38
        T_cp_S, cpS_tab = T_cp_38, cp_38
        T_rho_S, rhoS_tab = T_rho_38, rho_38
    else:
        T_k_S, kS_tab = T_D, k_D
        T_cp_S, cpS_tab = T_D, cp_D
        T_rho_S, rhoS_tab = T_D, rho_D

    def kS_fun(Tc): return interp1_linear_extrap(T_k_S, kS_tab, Tc)
    def cpS_fun(Tc): return interp1_linear_extrap(T_cp_S, cpS_tab, Tc)
    def rhoS_fun(Tc): return interp1_linear_extrap(T_rho_S, rhoS_tab, Tc)

    def kC_fun(Tc): return interp1_linear_extrap(T_props, k_chrom, Tc)
    def cpC_fun(Tc): return interp1_linear_extrap(T_props, cp_chrom, Tc)
    def rhoC_fun(Tc): return interp1_linear_extrap(T_props, rho_chrom, Tc)

    # -------------------- storage --------------------
    T = T0 * np.ones((int(Nr), int(Nz)), dtype=float)
    T_inner6 = T0 * np.ones((Nt, 6), dtype=float)
    T_outer6 = T0 * np.ones((Nt, 6), dtype=float)

    first_on_done = np.zeros(6, dtype=bool)
    first_on_pending = np.zeros(6, dtype=bool)

    print(f"FV 2-D: Nr={int(Nr)} Nz={int(Nz)} Nt={Nt} theta={theta:.2f} dt_fd={dt_fd:.3g} s "
          f"dt_tail={dt_tail:.3g} s L={L:.2f} m tCr={1e6*tCr_m:.0f} µm steel={steel}")

    # precompute constant map ordering (column-major due to minor memory optimization trick)
    # everything works tectonically slow but precomputing helps (a bit)
    map_idx = -np.ones((int(Nr), int(Nz)), dtype=int)
    ij_list = []
    ctr = 0
    for j in range(int(Nz)):
        for i in range(i_end[j]):
            map_idx[i, j] = ctr
            ij_list.append((i, j))
            ctr += 1
    Nunk = ctr
    ij_i = np.array([p[0] for p in ij_list], dtype=int)
    ij_j = np.array([p[1] for p in ij_list], dtype=int)

    # -------------------- time marching --------------------
    # BEHOLD! CUMBERSOME CODING LOGIC AHEAD
    for n in range(Nt - 1):
        tN = float(t[n])
        tNp1 = float(t[n + 1])
        dt = tNp1 - tN

        # interpolate BCs at t^{n+1}
        Tg_bc = float(interp1_linear_extrap(t_ib, Tgas, tNp1))  # °C
        h6 = interp1_linear_extrap_matrix(t_ib, h_data, tNp1)   # (6,)
        if np.size(h6) == 1:
            h6 = np.ones(6, dtype=float) * float(h6)
        h_in_col = h6[zone_id]  # length Nz

        # mark first-on flux event
        for k2 in range(6):
            if (not first_on_done[k2]) and (h6[k2] > 0):
                first_on_pending[k2] = True

        # material props @ T^n
        k_n = np.zeros((int(Nr), int(Nz)), dtype=float)
        rhoCp = np.zeros_like(k_n)
        if tCr_m > 0:
            for j in range(int(Nz)):
                Ni = i_end[j]
                for i in range(Ni):
                    Tc = T[i, j]
                    if r[i] <= Rcr + 1e-12:
                        k_n[i, j] = float(kC_fun(Tc))
                        rhoCp[i, j] = float(rhoC_fun(Tc) * cpC_fun(Tc))
                    else:
                        k_n[i, j] = float(kS_fun(Tc))
                        rhoCp[i, j] = float(rhoS_fun(Tc) * cpS_fun(Tc))
        else:
            for j in range(int(Nz)):
                Ni = i_end[j]
                for i in range(Ni):
                    Tc = T[i, j]
                    k_n[i, j] = float(kS_fun(Tc))
                    rhoCp[i, j] = float(rhoS_fun(Tc) * cpS_fun(Tc))

        # assemble linear system (COO) (insert: Avengers assemble joke here)
        rows = []
        cols = []
        vals = []
        rhs = np.zeros(Nunk, dtype=float)
        # Numerical magic ahead
        for j in range(int(Nz)):
            Ni = i_end[j]
            if Ni == 0:
                continue
            for i in range(Ni):
                P = map_idx[i, j]
                if P < 0:
                    continue

                aP_time = rhoCp[i, j] * V[i, j] / max(dt, EPS)
                aP = aP_time
                bP = aP_time * T[i, j]

                # axial conduction south
                if (j > 0) and valid[i, j - 1]:
                    Sidx = map_idx[i, j - 1]
                    kS_f = face_k(k_n[i, j - 1], k_n[i, j])
                    condS = theta * (kS_f * A_zS[i, j] / dz)
                    rows.append(P); cols.append(Sidx); vals.append(-condS)
                    aP += condS

                # axial conduction north
                if (j < int(Nz) - 1) and valid[i, j + 1]:
                    Nidx = map_idx[i, j + 1]
                    kN_f = face_k(k_n[i, j], k_n[i, j + 1])
                    condN = theta * (kN_f * A_zN[i, j] / dz)
                    rows.append(P); cols.append(Nidx); vals.append(-condN)
                    aP += condN

                # radial conduction east (i+1)
                if (i < Ni - 1) and valid[i + 1, j]:
                    Eidx = map_idx[i + 1, j]
                    drE = r[i + 1] - r[i]
                    kE_f = face_k(k_n[i, j], k_n[i + 1, j])
                    condE = theta * (kE_f * A_rE[i, j] / max(drE, EPS))
                    rows.append(P); cols.append(Eidx); vals.append(-condE)
                    aP += condE

                # radial conduction west (i-1)
                if i > 0:
                    Widx = map_idx[i - 1, j]
                    drW = r[i] - r[i - 1]
                    kW_f = face_k(k_n[i - 1, j], k_n[i, j])
                    condW = theta * (kW_f * A_rW[i, j] / max(drW, EPS))
                    rows.append(P); cols.append(Widx); vals.append(-condW)
                    aP += condW

                # inner boundary (i==0)
                if i == 0:
                    hin_now = float(h_in_col[j])
                    drw = max(r[i] - Rin, EPS)
                    knw = k_n[i, j]
                    if hin_now > 0:
                        Uin = 1.0 / (drw / max(knw, EPS) + 1.0 / hin_now)
                        aP += theta * (Uin * A_rW[i, j])
                        bP += theta * (Uin * A_rW[i, j] * Tg_bc)

                # outer boundary (i==Ni-1)
                if i == Ni - 1:
                    dro = max(Rout_z[j] - r[i], EPS)
                    kno = k_n[i, j]
                    Uout = 1.0 / (dro / max(kno, EPS) + 1.0 / hout)
                    Aout = 2.0 * np.pi * Rout_z[j] * dz
                    aP += theta * (Uout * Aout)
                    bP += theta * (Uout * Aout * T0)

                rows.append(P); cols.append(P); vals.append(aP)
                rhs[P] = bP

        A = coo_matrix((vals, (rows, cols)), shape=(Nunk, Nunk)).tocsr()
        sol = spsolve(A, rhs)

        # scatter solution back to field
        T[ij_i, ij_j] = sol

        # log 6-section inner & outer surface temperatures
        for k2 in range(6):
            jj = jstar[k2]
            T_inner6[n + 1, k2] = T[0, jj]
            ii = i_end[jj]
            if ii > 0:
                T_outer6[n + 1, k2] = T[ii - 1, jj]
            else:
                T_outer6[n + 1, k2] = np.nan

        # flux check at first-on
        if flux_check:
            for k2 in range(6):
                if first_on_pending[k2]:
                    jj = jstar[k2]
                    Tw = float(T[0, jj])
                    hin_now = float(h6[k2])
                    qR = hin_now * (Tg_bc - Tw)
                    # FV implied heat flux magnitude through wall-to-center resistance
                    drw = max(r[0] - Rin, EPS)
                    knw = float(k_n[0, jj])
                    if hin_now > 0:
                        Uin = 1.0 / (drw / max(knw, EPS) + 1.0 / hin_now)
                        qFV = Uin * (Tw - Tg_bc)
                    else:
                        qFV = 0.0
                    ratio = qFV / max(1e-16, qR)
                    print(f"S{k2+1} first-on: h={hin_now:0.2e}  Tw={Tw:.1f}  Tg={Tg_bc:.1f}  "
                          f"qR={qR:+.2e}  qFV={qFV:+.2e}  ratio={ratio:.3g}")
                    first_on_done[k2] = True
                    first_on_pending[k2] = False

        # debug prints
        if debug and ((n % int(debug_stride) == 0) or (n == 0) or (n == Nt - 2)):
            Tmask = T.copy()
            Tmask[~valid] = np.nan
            Tmax = np.nanmax(Tmask)
            Tmin = np.nanmin(Tmask)
            Tin = T_inner6[n + 1, :]
            Tout = T_outer6[n + 1, :]
            print(f"{n+1:5d}  t={1e3*tNp1:.3f} ms  dt={1e6*dt:.1f} µs | "
                  f"Tin: {' '.join([f'{v:6.2f}' for v in Tin])} | "
                  f"Tout: {' '.join([f'{v:6.2f}' for v in Tout])} | "
                  f"Tmin/Tmax: {Tmin:6.2f} / {Tmax:6.2f}")

    # last line fill (exact MATLAB behavior)
    if Nt >= 2:
        T_inner6[-1, :] = T_inner6[-2, :]
        T_outer6[-1, :] = T_outer6[-2, :]

    T_final = T.copy()

    # -------------------- plots --------------------
    if plot:
        # inner wall
        fig1, ax1 = plt.subplots(figsize=(9, 4))
        fig1.canvas.manager.set_window_title("Inner-surface T (2-D FV)")
        ax1.plot(1e3 * t, T_inner6, linewidth=1.4)
        ax1.grid(True)
        ax1.set_xlabel("time [ms]")
        ax1.set_ylabel("Inner-surface T [°C]")
        ax1.set_title("Inner-surface temperature at sections")
        ax1.legend(sections["names"], loc="best")

        # outer wall
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        fig2.canvas.manager.set_window_title("Outer-surface T (2-D FV)")
        ax2.plot(1e3 * t, T_outer6, linewidth=1.4)
        ax2.grid(True)
        ax2.set_xlabel("time [ms]")
        ax2.set_ylabel("Outer-surface T [°C]")
        ax2.set_title("Outer-surface temperature at sections")
        ax2.legend(sections["names"], loc="best")

        # final field
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        fig3.canvas.manager.set_window_title("Temperature field (final)")
        Tf = T_final.copy()
        Tf[~valid] = np.nan
        im = ax3.imshow(Tf, origin="lower", aspect="auto",
                        extent=(z[0], z[-1], 1e3 * r[0], 1e3 * r[-1]))
        fig3.colorbar(im, ax=ax3)
        ax3.set_xlabel("z [m]")
        ax3.set_ylabel("r [mm]")
        ax3.set_title("Final T field [°C]")
        ax3.plot(z, 1e3 * Rout_z, "k--", linewidth=1.0)

        plt.show()

    return {
        "t": t,
        "z": z,
        "r": r,
        "valid": valid,
        "T_final": T_final,
        "T_inner6": T_inner6,
        "T_outer6": T_outer6,
        "Rout_z": Rout_z,
        "sections": sections,
        "opts": {
            "steel": steel, "tCr_m": tCr_m, "Nr": int(Nr), "Nz": int(Nz), "theta": float(theta),
            "dt_fd": float(dt_fd), "tEnd_fd": float(tEnd_fd), "dt_tail": float(dt_tail),
            "cool_tail_s": float(cool_tail_s), "plot": bool(plot), "debug": bool(debug),
            "debug_stride": int(debug_stride), "flux_check": bool(flux_check), "bc": bc,
        }
    }


if __name__ == "__main__":
    # quick run using single-shot BC
    heat_transfer_2d_solver(Nr=80, Nz=60, plot=True, debug=True, debug_stride=100, flux_check=False)
