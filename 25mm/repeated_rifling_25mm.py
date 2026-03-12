"""
repeated_rifling.py
Builds repeated-shot boundary conditions (BC schedule) for heat_transfer_2d_solver.

Returns a dict bc with fields:
    t   : (Nt,) seconds
    h   : (Nt, Nz) W/m^2/K
    Tg  : (Nt,) K
    Ttot: float, total time
    meta: dict
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List
import numpy as np
import matplotlib.pyplot as plt

from utils import unique_stable, QuietStdout

from interior_ballistics_25mm import interior_ballistics_25mm as interior_ballistics
from heat_transfer_2d_solver import heat_transfer_2d_solver


def _shot_plan_to_gaps(shot_plan: Sequence[Dict[str, Any]], shot_duration_s: float) -> Dict[str, Any]:
    """
    Convert a user-friendly shot plan into per-shot gaps.
    Each plan entry should include: {"shots": int, "spm": float, "pause_s": float (optional)}.
    pause_s is the cooldown after the *last* shot of that entry.
    """
    gaps: List[float] = []
    plan_meta: List[Dict[str, Any]] = []
    total_shots = 0

    for idx, entry in enumerate(shot_plan):
        if "shots" not in entry or "spm" not in entry:
            raise ValueError("Each shot_plan entry must have 'shots' and 'spm'.")

        shots = int(entry["shots"])
        spm = float(entry["spm"])
        pause_s = float(entry.get("pause_s", entry.get("wait_s", 0.0)))

        if shots < 1:
            raise ValueError(f"shot_plan[{idx}] shots must be >=1.")
        if spm <= 0.0:
            raise ValueError(f"shot_plan[{idx}] spm must be >0.")
        if pause_s < 0.0:
            raise ValueError(f"shot_plan[{idx}] pause_s cannot be negative.")

        cadence_s = 60.0 / spm
        gap_between = max(cadence_s - shot_duration_s, 0.0)

        for i in range(shots):
            is_last = i == shots - 1
            gaps.append(pause_s if is_last else gap_between)
            total_shots += 1

        plan_meta.append(
            {
                "shots": shots,
                "spm": spm,
                "pause_s": pause_s,
                "cadence_s": cadence_s,
                "gap_after_shot_s": gap_between,
            }
        )

    if total_shots == 0:
        raise ValueError("shot_plan must include at least one shot.")

    return {"gaps": np.asarray(gaps, dtype=float), "total_shots": total_shots, "plan_meta": plan_meta}


def rifling_schedule_bc(single: Dict[str, Any], shot_gaps_s: Sequence[float], *, Tamb_C: float = 20.0, zcols: Optional[int] = None, scale: Optional[Sequence[float]] = None, plan_meta: Optional[Sequence[Dict[str, Any]]] = None, tau_purge: float = 5e-3, h_in: float = 9.2) -> Dict[str, Any]:
    """
    single: dict with keys t, h, Tg (Tg in K)
    """
    TambK = float(Tamb_C) + 273.15

    t1 = np.asarray(single["t"], dtype=float).reshape(-1)
    h1 = np.asarray(single["h"], dtype=float)
    Tg1K = np.asarray(single["Tg"], dtype=float).reshape(-1)

    if zcols is None:
        zcols = int(h1.shape[1])

    if scale is None:
        scale_arr = np.ones(h1.shape[1], dtype=float)
    else:
        scale_arr = np.asarray(scale, dtype=float).reshape(-1)
        if scale_arr.size != h1.shape[1]:
            raise ValueError(f"scale must have length {h1.shape[1]} (zones).")

    h1 = h1 * scale_arr.reshape(1, -1)
    Nz = h1.shape[1]

    per_shot_gaps = np.asarray(shot_gaps_s, dtype=float).reshape(-1)
    Nshots = per_shot_gaps.size
    if Nshots == 0:
        raise ValueError("shot_gaps_s must contain at least one gap (one shot).")

    # one block (shot + variable gap)
    t_shot = t1
    h_shot = h1
    Tg_shotK = Tg1K

    t_all = []
    h_all = []
    Tg_all = []
    t0 = 0.0
    for gap_s in per_shot_gaps:
        gap_s = float(gap_s)
        if gap_s > 0.0:
            # After muzzle exit the bore blowdown is fast (order ms). Model this with tau_purge:
            # keep last-shot values for an epsilon, then drop to ambient (Tg=TambK) and a small inner h_in,
            # and hold those for the remainder of the gap.
            t_eps = 1e-9
            tau = float(min(max(tau_purge, 0.0), gap_s))

            if tau <= 0.0:
                # Immediate purge (degenerate)
                t_gap = np.array([t_eps, gap_s], dtype=float)

                h_gap = np.zeros((2, Nz), dtype=float)
                h_gap[0, :] = h_shot[-1, :]
                h_gap[1, :] = float(h_in)

                Tg_gapK = np.array([Tg_shotK[-1], TambK], dtype=float)

            elif tau >= gap_s:
                # Purge occupies full gap
                t_gap = np.array([t_eps, gap_s], dtype=float)

                h_gap = np.zeros((2, Nz), dtype=float)
                h_gap[0, :] = h_shot[-1, :]
                h_gap[1, :] = float(h_in)

                Tg_gapK = np.array([Tg_shotK[-1], TambK], dtype=float)

            else:
                # Purge over tau, then hold ambient
                t_gap = np.array([t_eps, tau, gap_s], dtype=float)

                h_gap = np.zeros((3, Nz), dtype=float)
                h_gap[0, :] = h_shot[-1, :]
                h_gap[1, :] = float(h_in)
                h_gap[2, :] = float(h_in)

                Tg_gapK = np.array([Tg_shotK[-1], TambK, TambK], dtype=float)

        else:
            t_gap = np.array([0.0], dtype=float)
            h_gap = np.zeros((1, Nz), dtype=float)
            Tg_gapK = np.array([TambK], dtype=float)

        block_t = np.concatenate([t_shot, t_shot[-1] + t_gap])
        block_h = np.vstack([h_shot, h_gap])
        block_TgK = np.concatenate([Tg_shotK, Tg_gapK])

        t_all.append(t0 + block_t)
        h_all.append(block_h)
        Tg_all.append(block_TgK)
        t0 = t0 + (t1[-1] + float(gap_s))

    t = np.concatenate(t_all)
    h = np.vstack(h_all)
    TgK = np.concatenate(Tg_all)

    # dedup touching endpoints
    t_u, idx = unique_stable(t)
    h = h[idx, :]
    TgK = TgK[idx]

    return {
        "t": t_u,
        "h": h,
        "Tg": TgK,
        "Ttot": float(t_u[-1]) if t_u.size else 0.0,
        "meta": {
            "Nshots": int(Nshots),
            "Tamb_C": float(Tamb_C),
            "shot_gaps_s": per_shot_gaps.tolist(),
            "plan": plan_meta,
            "tau_purge": float(tau_purge),
            "h_in": float(h_in),
        },
    }


def repeated_rifling(
    *,
    Nshots: int = 7,
    cool_gap: float = 0.5,
    spm: Optional[float] = None,
    shot_plan: Optional[Sequence[Dict[str, Any]]] = None,
    Tamb_C: float = 30.0,
    use_30col: bool = False,
    scale: Optional[Sequence[float]] = None,
    plot: bool = True,
    smoke_test: bool = True,
    ib_plot: bool = False,
    ib_debug: bool = False,
    tau_purge: float = 5e-3,
    h_in: float = 9.2,
) -> Dict[str, Any]:
    """
    Schedule builder for repeated shots.
        - Legacy use: set Nshots + cool_gap for uniform cadence.
        - Constant rate: set spm + Nshots (gap derived from cadence = max(60/spm - t_shot, 0)).
        - Burst plan: pass shot_plan=[{shots, spm, pause_s|wait_s}, ...] for per-burst rates and cooldowns.
          pause_s is applied after the last shot of that burst (kept after the final burst for tail cooling).

    Notes:
        - use_30col: if IB produces 30 zones, keep all 30, else downselect to 6.
        - ib_plot: mirrors MATLAB side effects (interiorBallistics plots). Default False because "reasons".
        - ib_debug: emits the single-shot step-by-step prints. Default False for repeated runs.
    """
    if shot_plan is not None and spm is not None:
        raise ValueError("Provide either shot_plan or spm/cool_gap, not both.")

    if shot_plan is None:
        if int(Nshots) < 1 or int(Nshots) != Nshots:
            raise ValueError("Nshots must be a positive integer.")

    # ---------- 1) single-shot interior ballistics ----------
    ib = interior_ballistics(plot=ib_plot, debug=ib_debug)
    t1 = np.asarray(ib["timeVec"], dtype=float).reshape(-1)
    h1 = np.asarray(ib["hHist"], dtype=float)
    Tg1K = np.asarray(ib["Tgas"], dtype=float).reshape(-1)

    if h1.shape[0] != t1.size or Tg1K.size != t1.size:
        raise ValueError("IB arrays inconsistent: hHist rows and Tgas length must match timeVec.")

    NzIB = h1.shape[1]

    # downselect or keep
    if NzIB == 30 and (not use_30col):
        pick = np.array([1, 5, 10, 16, 25, 30], dtype=int) - 1  # MATLAB -> Python
        h1 = h1[:, pick]
        NzIB = 6
    elif NzIB not in (6, 30):
        raise ValueError(f"Unexpected number of IB zones: {NzIB} (expected 6 or 30).")

    # optional per-zone scale
    if scale is None:
        scale_arr = np.ones(NzIB, dtype=float)
    else:
        scale_arr = np.asarray(scale, dtype=float).reshape(-1)
        if scale_arr.size != NzIB:
            raise ValueError(f"scale must have length {NzIB} (zones).")

    single = {"t": t1, "h": h1, "Tg": Tg1K}

    # ---------- 2) stitch repeated schedule ----------
    plan_meta: Optional[Sequence[Dict[str, Any]]] = None
    if shot_plan is not None:
        plan = _shot_plan_to_gaps(shot_plan, float(t1[-1]))
        shot_gaps = plan["gaps"]
        plan_meta = plan["plan_meta"]
    elif spm is not None:
        cadence_s = 60.0 / float(spm)
        gap_between = max(cadence_s - float(t1[-1]), 0.0)
        shot_gaps = np.full(int(Nshots), gap_between, dtype=float)
        plan_meta = [
            {
                "shots": int(Nshots),
                "spm": float(spm),
                "pause_s": 0.0,
                "cadence_s": cadence_s,
                "gap_after_shot_s": gap_between,
            }
        ]
    else:
        shot_gaps = np.full(int(Nshots), float(cool_gap), dtype=float)

    bc = rifling_schedule_bc(single, shot_gaps, Tamb_C=float(Tamb_C), zcols=NzIB, scale=scale_arr, plan_meta=plan_meta, tau_purge=float(tau_purge), h_in=float(h_in))

    # ---------- 3) quiet smoke test ----------
    if smoke_test:
        try:
            test_bc = {"t": bc["t"][:2], "h": bc["h"][:2, :], "Tg": bc["Tg"][:2]}
            with QuietStdout(True):
                heat_transfer_2d_solver(Nr=16, Nz=8, dt_fd=1e-3, tEnd_fd=1e-3, theta=1.0,
                                        plot=False, debug=False, flux_check=False, bc=test_bc)
        except TypeError as e:
            raise TypeError(
                "Your heat_transfer_2d_solver does not accept the 'bc' option yet. "
                "Use the ported heat_transfer_2d_solver.py in this folder."
            ) from e

    # ---------- 4) mid-run BC plots ----------
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.canvas.manager.set_window_title("BC schedule")

        ax1.plot(bc["t"], bc["Tg"] - 273.15, "k-", linewidth=1.0)
        ax1.grid(True)
        ax1.set_ylabel("T_g [°C]")
        desc = None
        if bc["meta"].get("plan"):
            desc_parts = []
            for entry in bc["meta"]["plan"]:
                part = f"{entry['shots']} @ {entry['spm']:.0f} spm"
                if entry.get("pause_s", 0.0) > 0:
                    part += f" + wait {entry['pause_s']:.1f}s"
                desc_parts.append(part)
            desc = " | ".join(desc_parts)
        ax1.set_title(f"Gas Temperature (N={bc['meta']['Nshots']})" + (f" [{desc}]" if desc else ""))

        ax2.plot(bc["t"], bc["h"][:, 0] / 1e3, "r-", linewidth=1.0)
        idx = min(2, bc["h"].shape[1] - 1)  # MATLAB idx=min(3, size(h,2)) (1-based)
        ax2.plot(bc["t"], bc["h"][:, idx] / 1e3, "b-", linewidth=1.0)
        ax2.grid(True)
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("h [kW/m^2K]")
        ax2.set_title("Sample HTCs (zones 1 & 3)")
        # ---------- Stanton number check (single-shot IB) ----------
        try:
            rho = np.asarray(ib.get("rhoHist"), dtype=float).reshape(-1)
            vP = np.asarray(ib.get("velocityP"), dtype=float).reshape(-1)
            xP = np.asarray(ib.get("location"), dtype=float).reshape(-1)
            params = ib.get("params")
            if params is not None and rho.size == t1.size and vP.size == t1.size and xP.size == t1.size:
                gamma = float(getattr(params, "gamma"))
                R = float(getattr(params, "R"))
                cp = gamma * R / max(gamma - 1.0, 1e-12)  # J/(kg·K)
                li = np.asarray(getattr(params, "li"), dtype=float).reshape(-1)
                l0 = float(getattr(params, "l0"))
                # Zone flow-speed proxy used in IB HTC correlation
                wi = (li.reshape(1, -1) / (l0 + np.maximum(xP, 0.0)).reshape(-1, 1)) * vP.reshape(-1, 1)
                denom = rho.reshape(-1, 1) * cp * np.maximum(np.abs(wi), 0.0)
                St = np.full_like(h1, np.nan, dtype=float)
                ok = denom > 0.0
                St[ok] = h1[ok] / denom[ok]

                figS, axS = plt.subplots(1, 1, figsize=(10, 4))
                figS.canvas.manager.set_window_title("Stanton number (single shot)")
                for k in range(min(6, St.shape[1])):
                    axS.plot(1e3 * t1, St[:, k], linewidth=1.0, label=f"zone {k+1}")
                axS.grid(True)
                axS.set_xlabel("t [ms]")
                axS.set_ylabel("St [-]")
                axS.set_title("Gas-side Stanton number vs time (single-shot IB)")
                axS.legend(loc="best")
        except Exception as _e:
            # Keep behavior identical unless Stanton computation succeeds cleanly
            pass

        plt.show()

    return bc


if __name__ == "__main__":
    repeated_rifling(plot=True, smoke_test=True)