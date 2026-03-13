"""
main_repeated_rifling.py
Builds repeated-shot BC schedule and launches the 2-D solver with staged time steps.
"""
import matplotlib.pyplot as plt

from repeated_rifling_25mm import repeated_rifling
from heat_transfer_2d_solver import heat_transfer_2d_solver


def main():
    # ------------ knobs ------------
    steel = "32CrMoV12-28"
    tCr_um = 60              # Chromium layer thickness [um]

    # Meshing - Radial-Axisymmetric-Structured
    Nr = 100                  # Number of Radial grid points
    Nz = 100                  # Number of Axial grid points

    ''' For most cases, axial conduction is dominated by radial conduction; therefore, mind computational effort when assigning grid'''

# Time March Parameters
    dt_fd = 1e-4             # Time Step [s], ballistics-phase   (Fine advised)
    dt_tail = 5e-2           # Time Step [s], tail-cooling phase (Coarse advised)
    cool_tail_s = 0.00       # cooldown time after last heating [s]

# Other Parameters
    thetaFD = 1.0            # Finite difference solver theta (=1 for Backwards Euler, advised)
    debug = True             # Opt: Debug prints (True => Verbose; False => Silent)
    debug_strd = 10          # Debug print stride

    # repeated schedule (set burst-by-burst rate and cooldown)
    shot_plan = [
        {"shots": 9, "spm": 200, "pause_s": 750.0},          # 9 rounds @200 spm, wait 75 s
        #{"shots": 15, "spm": 600, "pause_s": 20 * 60.0},   # 15 rounds @600 spm, wait 20 min
    ]
    Tamb_C = 20.0            # Ambient temp [C]
    use_30col = False        # Irrelevant, but I'm too lazy to remove

    # ------------ build BC ------------
    bc = repeated_rifling(
        shot_plan=shot_plan,
        Tamb_C=Tamb_C,
        use_30col=use_30col,
        plot=True,
        smoke_test=False,  # smoke_test used in MATLAB, irrelevant here so keep FALSE
        ib_plot=True,     # Opt: Wanna print ballistics plot when initiating HT-solver?, advised FALSE. Run interior_ballistics first to check BC params
    )

    print(f"Launching 2-D FV solver with staged dt (dt_fd={dt_fd:.3g}, dt_tail={dt_tail:.3g}, tail={cool_tail_s:.1f} s)...")

    out = heat_transfer_2d_solver(
        steel=steel,
        tCr_um=tCr_um,
        Nr=Nr,
        Nz=Nz,
        dt_fd=dt_fd,
        dt_tail=dt_tail,
        cool_tail_s=cool_tail_s,
        theta=thetaFD,
        debug=debug,
        debug_stride=debug_strd,
        plot=True,
        bc=bc,
    )

    # ------------ quick post ------------
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    fig1.canvas.manager.set_window_title("Inner-surface T (quick post)")
    ax1.grid(True)
    ax1.plot(1e3 * out["t"], out["T_inner6"], linewidth=1.2)
    ax1.set_xlabel("time [ms]")
    ax1.set_ylabel("T_inner [°C]")
    ax1.set_title("Inner-surface temperature at sections")
    ax1.legend(out["sections"]["names"], loc="best")

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    fig2.canvas.manager.set_window_title("Outer-surface T (quick post)")
    ax2.grid(True)
    ax2.plot(1e3 * out["t"], out["T_outer6"], linewidth=1.2)
    ax2.set_xlabel("time [ms]")
    ax2.set_ylabel("T_outer [°C]")
    ax2.set_title("Outer-surface temperature at sections")
    ax2.legend(out["sections"]["names"], loc="best")

    plt.show()
    return out


if __name__ == "__main__":
    main()
