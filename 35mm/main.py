"""
main.py

Runs the 2-D FV solver using single-shot boundary conditions.
(This script is for debugging purposes only, run repeated_rifling for analysis)
"""
from heat_transfer_2d_solver import heat_transfer_2d_solver


def main():
    out = heat_transfer_2d_solver(
        steel="DUPLEX",
        tCr_um=0,
        Nr=240 * 4,
        Nz=120,
        dt_fd=1e-4,
        tEnd_fd=20e-3,
        debug=True,
        debug_stride=10,
        plot=True,
    )
    return out


if __name__ == "__main__":
    main()