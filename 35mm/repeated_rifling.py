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

from typing import Any, Dict, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt

from utils import unique_stable, QuietStdout

from interior_ballistics import interior_ballistics
from heat_transfer_2d_solver import heat_transfer_2d_solver


def rifling_schedule_bc(single: Dict[str, Any], Nshots: int, gap_s: float, *, Tamb_C: float = 20.0, zcols: Optional[int] = None, scale: Optional[Sequence[float]] = None) -> Dict[str, Any]:
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

    # one block (shot + gap)
    t_shot = t1
    h_shot = h1
    Tg_shotK = Tg1K

    if gap_s > 0:
        t_gap = np.array([0.0, float(gap_s)])
        h_gap = np.zeros((2, Nz), dtype=float)
        Tg_gapK = np.ones(2, dtype=float) * TambK
    else:
        t_gap = np.array([0.0], dtype=float)
        h_gap = np.zeros((1, Nz), dtype=float)
        Tg_gapK = np.array([TambK], dtype=float)

    block_t = np.concatenate([t_shot, t_shot[-1] + t_gap])
    block_h = np.vstack([h_shot, h_gap])
    block_TgK = np.concatenate([Tg_shotK, Tg_gapK])

    # tile N copies
    t_all = []
    h_all = []
    Tg_all = []
    t0 = 0.0
    for _ in range(int(Nshots)):
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
        "meta": {"Nshots": int(Nshots), "gap_s": float(gap_s), "Tamb_C": float(Tamb_C)},
    }


def repeated_rifling(
    *,
    Nshots: int = 7,
    cool_gap: float = 0.5,
    Tamb_C: float = 20.0,
    use_30col: bool = False,
    scale: Optional[Sequence[float]] = None,
    plot: bool = True,
    smoke_test: bool = True,
    ib_plot: bool = False,
    ib_debug: bool = False,
) -> Dict[str, Any]:
    """
    Python equivalent of: (for anyone familiar with the initial MATLAB script)
        bc = repeated_rifling('Nshots',7,'cool_gap',0.5,'Tamb_C',20,'use_30col',false,'plot',true);

    Notes:
        - use_30col: if IB produces 30 zones, keep all 30, else downselect to 6.
        - ib_plot: mirrors MATLAB side effects (interiorBallistics plots). Default False because "reasons".
        - ib_debug: emits the single-shot step-by-step prints. Default False for repeated runs.
    """
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
    bc = rifling_schedule_bc(single, int(Nshots), float(cool_gap), Tamb_C=float(Tamb_C), zcols=NzIB, scale=scale_arr)

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
        ax1.set_title(f"Gas Temperature (N={int(Nshots)}, gap={float(cool_gap):.3f}s)")

        ax2.plot(bc["t"], bc["h"][:, 0] / 1e3, "r-", linewidth=1.0)
        idx = min(2, bc["h"].shape[1] - 1)  # MATLAB idx=min(3, size(h,2)) (1-based)
        ax2.plot(bc["t"], bc["h"][:, idx] / 1e3, "b-", linewidth=1.0)
        ax2.grid(True)
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("h [kW/m^2K]")
        ax2.set_title("Sample HTCs (zones 1 & 3)")
        plt.show()

    return bc


if __name__ == "__main__":
    repeated_rifling(plot=True, smoke_test=True)
