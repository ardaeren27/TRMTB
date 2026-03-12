"""
gun25_geometry.py

25 mm gun geometry + axial zoning, salvaged from:
- BC_inner_25mm.m (geometry + 8 axial zones)  [author: AE workspace]
- 25-VH.py (Vallier-style boundary data assumptions for APDS 25x137)  [author: AE workspace]

Core idea
---------
We keep two axial coordinate systems:

1) Absolute axis z_abs [m]:
   z_abs = 0 at breech face, z_abs = L0 + L at muzzle.

2) Barrel-travel axis x [m]:
   x = 0 at shot start / projectile travel origin in IB tables,
   x increases to L at muzzle (rifling length / barrel length excluding chamber).

Mapping: z_abs = L0 + x

The zoning in BC_inner_25mm.m is specified in absolute coordinates (including chamber).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GunGeometry:
    name: str
    bore_d_m: float          # bore diameter [m]
    chamber_len_m: float     # chamber length L0 [m]
    barrel_len_m: float      # barrel length L  [m] (excluding chamber)
    z_edges_abs_m: np.ndarray  # zone edges along absolute axis [m], size Nz+1

    @property
    def total_len_m(self) -> float:
        return float(self.chamber_len_m + self.barrel_len_m)

    @property
    def area_bore_m2(self) -> float:
        return float(np.pi * (0.5 * self.bore_d_m) ** 2)

    @property
    def Nz(self) -> int:
        return int(self.z_edges_abs_m.size - 1)

    @property
    def z_centers_abs_m(self) -> np.ndarray:
        z = self.z_edges_abs_m
        return 0.5 * (z[:-1] + z[1:])

    @property
    def z_edges_rel_m(self) -> np.ndarray:
        """
        Zone edges measured from barrel start (end of chamber). Anything in chamber => 0.
        Mirrors BC_inner_25mm.m: G.z_edges = max(0, z_abs - L0), and last edge forced to L.
        """
        z_rel = np.maximum(0.0, self.z_edges_abs_m - self.chamber_len_m)
        z_rel[-1] = self.barrel_len_m
        return z_rel

    @property
    def z_centers_rel_m(self) -> np.ndarray:
        zc_rel = np.maximum(0.0, self.z_centers_abs_m - self.chamber_len_m)
        return zc_rel

    @property
    def labels(self) -> list[str]:
        """
        Human-readable labels matching BC_inner_25mm.m:
        S{k}(start-end mm) where start/end are absolute coordinates.
        """
        L0 = self.chamber_len_m
        z_rel = self.z_edges_rel_m
        labs: list[str] = []
        for k in range(self.Nz):
            a_mm = 1e3 * (z_rel[k] + L0)
            b_mm = 1e3 * (z_rel[k + 1] + L0)
            labs.append(f"S{k+1}({a_mm:.0f}-{b_mm:.0f} mm)")
        return labs

    def z_abs_from_x(self, x_travel_m: float | np.ndarray) -> float | np.ndarray:
        """Map barrel travel x -> absolute z axis (breech-origin)."""
        return self.chamber_len_m + np.asarray(x_travel_m)

    def zone_index_from_z_abs(self, z_abs_m: float | np.ndarray) -> np.ndarray:
        """
        Return 0-based zone index for a given absolute coordinate(s).
        Out-of-range values are clipped into [0, Nz-1].
        """
        z = np.asarray(z_abs_m, dtype=float)
        idx = np.searchsorted(self.z_edges_abs_m, z, side="right") - 1
        return np.clip(idx, 0, self.Nz - 1)


def gun25_geometry() -> GunGeometry:
    """
    Geometry and zones copied directly from BC_inner_25mm.m

    BC_inner_25mm.m:
        G.L  = 1.858 m  (barrel length excluding chamber)
        G.L0 = 0.142 m  (chamber length)
        z_abs_edges_mm = [0, 150, 248, 360, 440, 800, 1200, 1600, 2000]
        D_bore = 0.025 m
    """
    L = 1.858
    L0 = 0.142
    bore_d = 0.025

    z_abs_edges_m = np.array([0, 150, 248, 360, 440, 800, 1200, 1600, 2000], dtype=float) * 1e-3

    # sanity: should end at L0+L = 2.000 m
    total = L0 + L
    if abs(z_abs_edges_m[-1] - total) > 1e-9:
        raise ValueError(f"Zone edges end at {z_abs_edges_m[-1]:.6f} m but L0+L = {total:.6f} m")

    return GunGeometry(
        name="25 mm bore – 8 axial zones (S1-S8)",
        bore_d_m=bore_d,
        chamber_len_m=L0,
        barrel_len_m=L,
        z_edges_abs_m=z_abs_edges_m,
    )


# Reference constraints from 25-VH.py (Vallier-style boundary script)
VALLIER_APDS_25X137_REF = {
    "m_proj_kg": 0.134,          # projectile mass
    "m_prop_kg": 0.100,          # propellant mass
    "V_exit_mps": 1345.0,        # muzzle velocity
    "P_max_MPa": 450.0,          # maximum pressure
    "L_barrel_m": 1.858,         # barrel length (rifling) excluding chamber
    "D_bore_m": 0.025,           # bore diameter
    "V0_m3": 106.0e-6,           # chamber free volume (25-VH.py comment: "came from FC")
}


def print_geometry_summary(G: GunGeometry) -> None:
    print("=== 25 mm geometry summary ===")
    print(f"Name           : {G.name}")
    print(f"Bore diameter  : {G.bore_d_m*1e3:.3f} mm")
    print(f"Chamber length : {G.chamber_len_m*1e3:.1f} mm")
    print(f"Barrel length  : {G.barrel_len_m*1e3:.1f} mm")
    print(f"Total length   : {G.total_len_m*1e3:.1f} mm")
    print(f"Zones (Nz)     : {G.Nz}")
    print("")
    print("Zone edges (absolute, mm):")
    print((G.z_edges_abs_m * 1e3).round(3))
    print("Zone centers (absolute, mm):")
    print((G.z_centers_abs_m * 1e3).round(3))
    print("")
    print("Zone edges (relative to barrel start, mm):")
    print((G.z_edges_rel_m * 1e3).round(3))
    print("Labels:")
    for lab in G.labels:
        print("  ", lab)


if __name__ == "__main__":
    G = gun25_geometry()
    print_geometry_summary(G)
