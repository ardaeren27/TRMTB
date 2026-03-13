"""
Vallier-Heydenreich Interior Ballistics Solver — Desktop GUI
==============================================================
Standalone tkinter + matplotlib application.
No browser, no internet, no external files required.

Dependencies: Python 3.8+, numpy, matplotlib (tkinter is bundled with Python).
Compile to .exe:  pyinstaller --onefile --windowed vallier_gui_desktop.py

TR Mekatronik
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import base64
import io
import csv
import json
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

# numpy compat
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# ============================================================================
#  EMBEDDED LOGO (PNG base64)
# ============================================================================
LOGO_B64 = (
    "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"
    "cm8JvgGAAAAAElFTkSuQmCC"
)

# ============================================================================
#  EMBEDDED VALLIER-HEYDENREICH TABLES
# ============================================================================
BALMAT = np.array([
    [0,0,0,0],[.25,.69,.375,.689],[.5,.89,.624,.83],[.75,.97,.828,.924],
    [1,1,1,1],[1.25,.966,1.145,1.063],[1.5,.893,1.26,1.119],[1.75,.828,1.372,1.17],
    [2,.769,1.46,1.218],[2.5,.668,1.609,1.306],[3,.59,1.726,1.387],[3.5,.527,1.824,1.463],
    [4,.472,1.909,1.536],[4.5,.433,1.981,1.606],[5,.397,2.046,1.672],[5.5,.369,2.102,1.737],
    [6,.34,2.158,1.801],[6.5,.319,2.204,1.862],[7,.297,2.25,1.923],[7.5,.28,2.289,1.983],
    [8,.263,2.328,2.042],[8.5,.25,2.362,2.099],[9,.236,2.395,2.156],[9.5,.225,2.424,2.212],
    [10,.214,2.453,2.267],[10.5,.205,2.479,2.322],[11,.195,2.505,2.376],[11.5,.187,2.528,2.43],
    [12,.179,2.551,2.483],[12.5,.173,2.572,2.521],[13,.166,2.592,2.558],[13.5,.16,2.611,2.625],
    [14,.154,2.63,2.692],[14.5,.149,2.648,2.743],[15,.144,2.665,2.794],[15.5,.14,2.682,2.845],
    [16,.135,2.698,2.895],[16.5,.131,2.714,2.945],[17,.127,2.73,2.994],[18,.12,2.76,3.092],
    [19,.114,2.787,3.189],[20,.108,2.812,3.286],[25,.086,2.921,3.758],[30,.071,3.004,4.214],
    [35,.06,3.07,4.659],[40,.052,3.132,5.095],[45,.046,3.182,5.523],[50,.041,3.22,5.946],
    [75,.027,3.373,7.995],[100,.02,3.48,9.996],
])

HEIBALMAT = np.array([
    [.2,.026,.15,.322,.274,.744],[.25,.036,.196,.337,.306,.792],
    [.3,.047,.246,.352,.338,.842],[.35,.06,.3,.367,.368,.893],
    [.4,.074,.358,.383,.4,.946],[.45,.09,.42,.399,.432,1.],
    [.5,.109,.487,.416,.465,1.056],[.55,.132,.56,.435,.501,1.116],
    [.6,.16,.642,.457,.541,1.18],[.65,.192,.734,.482,.585,1.249],
    [.7,.231,.835,.511,.635,1.322],[.75,.283,.958,.546,.697,1.406],
    [.8,.36,1.115,.592,.779,1.507],
])

# ============================================================================
#  AMMUNITION PRESETS
# ============================================================================
AMMO_PRESETS = {
    "TP/HEI 25x137": dict(
        m_proj_g=185, m_prop_g=100, V_exit=1100, P_max=420, L_bar_mm=1786,
        D_bar_mm=25, V0_cm3=106, T_flame=3150, I_xx=0.0000124836,
        I_yy=0.000143482, C_ma=2.5, beta_ae=1.2,
        groove_a=0.0032, groove_b=1.4563, groove_tan_a=0.0046, groove_tan_b=0.4563,
    ),
    "APDS 25x137": dict(
        m_proj_g=134, m_prop_g=100, V_exit=1345, P_max=450, L_bar_mm=1858,
        D_bar_mm=25, V0_cm3=106, T_flame=3150, I_xx=0.0000124836,
        I_yy=0.000143482, C_ma=2.5, beta_ae=1.2,
        groove_a=0.0032, groove_b=1.4563, groove_tan_a=0.0046, groove_tan_b=0.4563,
    ),
    "TP/HEI 30x173": dict(
        m_proj_g=365, m_prop_g=153, V_exit=1080, P_max=404, L_bar_mm=2520.2,
        D_bar_mm=30, V0_cm3=156, T_flame=3300, I_xx=0.000039126798,
        I_yy=0.00047145589, C_ma=3.5, beta_ae=1.2,
        groove_a=0.00179*1.06, groove_b=1.5, groove_tan_a=0.002535*1.045, groove_tan_b=0.5,
    ),
    "TPDS 30x173": dict(
        m_proj_g=140, m_prop_g=150, V_exit=1600, P_max=460, L_bar_mm=2191.5,
        D_bar_mm=30, V0_cm3=156, T_flame=3300, I_xx=0.00000936,
        I_yy=0.00004975, C_ma=3.5, beta_ae=1.2,
        groove_a=0.00179*1.06, groove_b=1.5, groove_tan_a=0.002535*1.045, groove_tan_b=0.5,
    ),
}

PARAM_META = [
    ("m_proj_g",     "Projectile mass",      "g"),
    ("m_prop_g",     "Propellant mass",       "g"),
    ("V_exit",       "Muzzle velocity",       "m/s"),
    ("P_max",        "Max chamber pressure",  "MPa"),
    ("L_bar_mm",     "Barrel length",         "mm"),
    ("D_bar_mm",     "Bore diameter",         "mm"),
    ("V0_cm3",       "Chamber volume",        "cm\u00B3"),
    ("T_flame",      "Flame temperature",     "K"),
    ("I_xx",         "I_xx (axial MoI)",      "kg\u00B7m\u00B2"),
    ("I_yy",         "I_yy (transverse MoI)", "kg\u00B7m\u00B2"),
    ("C_ma",         "C_m\u03B1",             "\u2014"),
    ("beta_ae",      "\u03B2 aftereffect",    "\u2014"),
    ("groove_a",     "Groove coeff a",        "\u2014"),
    ("groove_b",     "Groove exponent b",     "\u2014"),
    ("groove_tan_a", "tan(\u03B1) coeff",     "\u2014"),
    ("groove_tan_b", "tan(\u03B1) exponent",  "\u2014"),
]

# ============================================================================
#  SOLVER
# ============================================================================
def run_solver(a: dict) -> dict:
    """Run the Vallier-Heydenreich interior ballistics computation."""
    pi = np.pi
    m_proj = a["m_proj_g"] / 1000
    m_prop = a["m_prop_g"] / 1000
    V_exit = a["V_exit"]
    P_max  = a["P_max"]
    L_bar  = a["L_bar_mm"] / 1000
    D_bar  = a["D_bar_mm"] / 1000
    V0     = a["V0_cm3"] * 1e-6
    T_flame = a["T_flame"]
    T_air   = 288.15
    rho_atm = 1.225
    P_atm   = 101325
    Beta    = 1.8
    gamma   = 1.4

    A_ref = pi * D_bar**2 / 4
    P0    = 0.5 * (m_proj + m_prop/2) * V_exit**2 / (A_ref * L_bar)
    P0_MPa = P0 / 1e6
    eta = P0_MPa / P_max

    def _interp(xt, yt, xq):
        return float(np.interp(xq, xt, yt))

    sigma = _interp(HEIBALMAT[:,0], HEIBALMAT[:,1], eta)
    theta = _interp(HEIBALMAT[:,0], HEIBALMAT[:,2], eta)
    phi   = _interp(HEIBALMAT[:,0], HEIBALMAT[:,3], eta)
    pii   = _interp(HEIBALMAT[:,0], HEIBALMAT[:,4], eta)
    tau   = _interp(HEIBALMAT[:,0], HEIBALMAT[:,5], eta)

    x_Pmax_m  = L_bar * sigma
    x_Pmax_mm = x_Pmax_m * 1000
    v_Pmax    = phi * V_exit
    t_Pmax_s  = (2*L_bar/V_exit) * theta
    t_Pmax_ms = t_Pmax_s * 1000
    P_muzzle  = pii * P0_MPa
    t_exit_s  = (2*L_bar/V_exit) * tau
    t_exit_ms = t_exit_s * 1000
    lam       = L_bar / x_Pmax_m

    bb = BALMAT[BALMAT[:,0] < lam].copy()
    PSI   = _interp(BALMAT[:,0], BALMAT[:,1], lam)
    PHII  = _interp(BALMAT[:,0], BALMAT[:,2], lam)
    DELTA = _interp(BALMAT[:,0], BALMAT[:,3], lam)
    bb = np.vstack([bb, [lam, PSI, PHII, DELTA]])

    X1   = bb[:,0] * x_Pmax_mm / 1000
    P1   = bb[:,1] * P_max * 1e6
    Vel1 = bb[:,2] * v_Pmax
    t1   = bb[:,3] * t_Pmax_ms
    n    = len(X1)

    Vol1 = V0 + X1 * A_ref
    rho1 = m_prop / Vol1
    P_max_Pa = P_max * 1e6
    loop_end = int(np.argmax(P1))
    for i in range(loop_end):
        if P1[i] <= P_max_Pa:
            rho1[i] *= P1[i] / P_max_Pa
            Vol1[i] = V0 + X1[i] * A_ref * P1[i] / P_max_Pa

    constT = T_flame / Beta * V0**(gamma-1)
    T1 = constT / Vol1**(gamma-1)
    for i in range(loop_end):
        T1[i] = T_air + (T1[loop_end]-T_air)/(t1[loop_end]-t1[0])*(t1[i]-t1[0])

    # Post-muzzle
    rho0 = m_prop / Vol1[-1]
    t_end = (40*1000*m_prop) / (2*A_ref*rho_atm*V_exit + 3*A_ref*rho0*V_exit)
    dt = 1e-2
    t2 = np.arange(0, t_end+dt, dt)

    tt2 = t2[-1]
    AA = np.array([[0,0,1],[2*tt2,1,0],[tt2**2,tt2,1]])
    coeff = np.linalg.solve(AA, np.array([T1[-1], 0, T_air]))

    P2   = (P1[-1]-P_atm)/t_end**2*t2**2 + 2*(P_atm-P1[-1])/t_end*t2 + P1[-1]
    rho2 = (rho0-rho_atm)/t_end**2*t2**2 + 2*(rho_atm-rho0)/t_end*t2 + rho0
    Vel2 = V_exit/t_end**2*t2**2 - 2*V_exit/t_end*t2 + V_exit
    T2   = coeff[0]*t2**2 + coeff[1]*t2 + coeff[2]
    t2  += t1[-1]

    tms  = np.concatenate([t1, t2[1:]])
    Pres = np.concatenate([P1, P2[1:]])
    Velo = np.concatenate([Vel1, Vel2[1:]])
    Temp = np.concatenate([T1, T2[1:]])

    # Rifling
    r_bore = D_bar / 2
    X1_mm  = X1 * 1000
    Y      = a["groove_a"] * X1_mm**a["groove_b"]
    tan_al = a["groove_tan_a"] * X1_mm**a["groove_tan_b"]
    alpha_rad = np.arctan(tan_al)
    alpha_deg = np.degrees(alpha_rad)
    omega = Vel1 * np.tan(alpha_rad) / r_bore

    t1_s  = t1 / 1000
    accel = np.zeros_like(Vel1)
    accel[1:] = np.diff(Vel1) / np.diff(t1_s)

    spin = omega[-1]
    Sg = (a["I_xx"]**2 * spin**2) / (2*rho_atm*a["I_yy"]*A_ref*D_bar*V_exit**2*a["C_ma"])
    k_g = np.sqrt(a["I_xx"] / m_proj)
    d_alpha_dx = np.gradient(alpha_rad, X1)

    theta_ddot = (1/r_bore) * (accel*np.tan(alpha_rad) + Vel1**2*d_alpha_dx)
    Torque_1 = (m_proj*k_g**2/r_bore) * ((A_ref*P1/m_proj)*np.tan(alpha_rad) + Vel1**2*d_alpha_dx)
    Torque_2 = a["I_xx"] * theta_ddot

    Force_all = Pres * A_ref
    idx_exit = int(np.argmin(np.abs(tms - t_exit_ms)))
    Force_exit = Force_all[idx_exit]
    t_n = ((a["beta_ae"]-0.5)*m_prop*V_exit) / (Force_exit/2) * 1000
    t_e = t_n + t_exit_ms
    idx_te = int(np.argmin(np.abs(tms - t_e)))
    impulse = np.trapezoid(Force_all[:idx_te+1], tms[:idx_te+1]/1000)
    blast_force = impulse / (t_e/1000)

    return dict(
        eta=eta, P0_MPa=P0_MPa, x_Pmax_mm=x_Pmax_mm, v_Pmax=v_Pmax,
        t_Pmax_ms=t_Pmax_ms, P_muzzle=P_muzzle, t_exit_ms=t_exit_ms,
        max_torque=float(np.max(Torque_1)), omega_exit=float(spin), Sg=Sg,
        t_n=t_n, t_e=t_e, impulse=impulse, blast_force=blast_force,
        # Arrays for plotting
        tms=tms, Pres=Pres, Velo=Velo, Temp=Temp,
        X1=X1, X1_mm=X1_mm, P1=P1, Vel1=Vel1, t1=t1, omega=omega,
        alpha_deg=alpha_deg, Y=Y, Torque_1=Torque_1, Torque_2=Torque_2,
        theta_ddot=theta_ddot, accel=accel,
        Force_all=Force_all, Force_int=P1*A_ref,
        t_impulse=tms[:idx_te+1], Force_impulse=Force_all[:idx_te+1],
    )



# ============================================================================
#  THEME
# ============================================================================
WINDOW_BG   = "#f5f5f7"
SHELL_BG    = "#ffffff"
SIDEBAR_BG  = "#f7f7fa"
CARD_BG     = "#ffffff"
CARD_BG_2   = "#fafafc"
CARD_BG_3   = "#fcfcfd"
STROKE      = "#e6e6eb"
STROKE_SOFT = "#efeff4"
TEXT        = "#0a0a0f"
TEXT_SOFT   = "#5f5f6b"
TEXT_FAINT  = "#8e8e98"
ACCENT      = "#d63b72"
ACCENT_2    = "#ee5b90"
ACCENT_3    = "#b42c5b"
SUCCESS     = "#1f8f5f"
WARNING     = "#b0742f"
DANGER      = "#c63d4e"
PILL_BG     = "#f9edf2"
PILL_NEUTRAL = "#f2f2f7"

PLOT_BG     = "#ffffff"
PLOT_FACE   = "#ffffff"
PLOT_GRID   = "#ececf2"

CHART_RED    = ACCENT
CHART_BLUE   = "#14151a"
CHART_ORANGE = "#b87b3d"
CHART_GREEN  = "#2b8d65"
CHART_PURPLE = "#6e5bd8"
CHART_TEAL   = "#2f8f8a"
CHART_GOLD   = "#8d6a2f"


PARAM_GROUPS = [
    ("Mass, Geometry, Thermochemistry", [
        "m_proj_g", "m_prop_g", "L_bar_mm", "D_bar_mm", "V0_cm3", "T_flame"
    ]),
    ("Performance Targets", [
        "V_exit", "P_max"
    ]),
    ("Stability / Aerodynamic Terms", [
        "I_xx", "I_yy", "C_ma", "beta_ae"
    ]),
    ("Rifling Law", [
        "groove_a", "groove_b", "groove_tan_a", "groove_tan_b"
    ]),
]


USER_PRESET_PATH = Path.home() / ".vallier_gui_user_presets.json"


class MetricCard(tk.Frame):
    def __init__(self, parent, title, accent=ACCENT):
        super().__init__(
            parent,
            bg=CARD_BG,
            highlightthickness=1,
            highlightbackground=STROKE_SOFT,
            bd=0,
        )
        self.configure(padx=16, pady=14)

        top = tk.Frame(self, bg=CARD_BG)
        top.pack(fill="x")

        dot = tk.Canvas(top, width=10, height=10, bg=CARD_BG, highlightthickness=0)
        dot.create_oval(1, 1, 9, 9, fill=accent, outline=accent)
        dot.pack(side="left", pady=(2, 0))

        self.title_label = tk.Label(
            top,
            text=title,
            bg=CARD_BG,
            fg=TEXT_SOFT,
            font=("Segoe UI", 10),
        )
        self.title_label.pack(side="left", padx=(8, 0))

        self.value_label = tk.Label(
            self,
            text="—",
            bg=CARD_BG,
            fg=TEXT,
            font=("Segoe UI", 22, "bold"),
        )
        self.value_label.pack(anchor="w", pady=(10, 2))

        self.meta_label = tk.Label(
            self,
            text="Waiting for a run",
            bg=CARD_BG,
            fg=TEXT_FAINT,
            font=("Segoe UI", 9),
        )
        self.meta_label.pack(anchor="w")

    def set_value(self, value, meta="", color=None):
        self.value_label.configure(text=value, fg=color or TEXT)
        self.meta_label.configure(text=meta)


class PillTabButton(tk.Canvas):
    def __init__(self, parent, text, command, selected=False):
        self._font = ("Segoe UI", 10, "bold")
        self._text = text
        self._command = command
        width = max(118, len(text) * 8 + 42)
        super().__init__(
            parent,
            width=width,
            height=40,
            bg=PILL_NEUTRAL,
            highlightthickness=0,
            bd=0,
            relief="flat",
            cursor="hand2",
        )
        self._selected = selected
        self._hover = False
        self.bind("<Button-1>", lambda _e: self._command())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self._redraw()

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1, x2, y1 + r,
            x2, y2 - r,
            x2, y2, x2 - r, y2,
            x1 + r, y2,
            x1, y2, x1, y2 - r,
            x1, y1 + r,
            x1, y1, x1 + r, y1,
        ]
        return self.create_polygon(points, smooth=True, splinesteps=36, **kwargs)

    def _palette(self):
        if self._selected:
            return "#111115", "#ffffff", "#111115"
        if self._hover:
            return "#ececf2", TEXT, "#ececf2"
        return PILL_NEUTRAL, TEXT_SOFT, PILL_NEUTRAL

    def _redraw(self):
        self.delete("all")
        fill, fg, outline = self._palette()
        self._round_rect(2, 2, int(self["width"]) - 2, int(self["height"]) - 2, 18, fill=fill, outline=outline)
        self.create_text(
            int(self["width"]) / 2,
            int(self["height"]) / 2,
            text=self._text,
            fill=fg,
            font=self._font,
        )

    def _on_enter(self, _event):
        self._hover = True
        self._redraw()

    def _on_leave(self, _event):
        self._hover = False
        self._redraw()

    def set_selected(self, selected):
        self._selected = selected
        self._redraw()


class VallierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vallier–Heydenreich Solver")
        self.root.configure(bg=WINDOW_BG)
        self.root.geometry("1500x920")
        self.root.minsize(1280, 780)

        self.results = None
        self.last_run_params = None
        self.last_run_preset_name = None
        self.param_vars = {}
        self.param_entries = {}
        self.preset_buttons = {}
        self.metric_cards = {}
        self.plot_tabs = {}
        self.page_frames = {}
        self.tab_buttons = {}
        self.user_presets = self._load_user_presets()
        self.all_presets = {}
        self._rebuild_preset_store()
        self.current_preset_name = next(iter(self.all_presets))
        self.current_tab = None

        self._prepare_logo_assets()
        self._configure_styles()
        self._build_ui()
        self._install_scroll_bindings()
        self._load_preset(self.current_preset_name)
        self._show_empty_state()

    # ------------------------------------------------------------------
    #  ASSETS / STYLES
    # ------------------------------------------------------------------
    def _prepare_logo_assets(self):
        self._logo_img = None
        self._logo_small = None
        self._logo_header = None
        self._logo_sidebar = None
        try:
            logo_bytes = base64.b64decode(LOGO_B64)
            self._logo_img = tk.PhotoImage(data=base64.b64encode(logo_bytes))
            self._logo_small = self._logo_img.subsample(4, 4)
            self._logo_header = self._logo_img.subsample(3, 3)
            self._logo_sidebar = self._logo_img.subsample(2, 2)
            self.root.iconphoto(False, self._logo_img)
        except Exception:
            self._logo_img = None


    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Modern.TNotebook",
            background=WINDOW_BG,
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "Modern.TNotebook.Tab",
            background=PILL_NEUTRAL,
            foreground=TEXT_SOFT,
            borderwidth=0,
            padding=(18, 11),
            font=("Segoe UI", 10, "bold"),
            focuscolor=PILL_NEUTRAL,
        )
        style.map(
            "Modern.TNotebook.Tab",
            background=[("selected", CARD_BG), ("active", CARD_BG_2)],
            foreground=[("selected", TEXT), ("active", TEXT)],
        )
        style.configure(
            "Minimal.Vertical.TScrollbar",
            background="#c8c8d3",
            troughcolor=PILL_NEUTRAL,
            bordercolor=PILL_NEUTRAL,
            lightcolor="#c8c8d3",
            darkcolor="#c8c8d3",
            arrowcolor=TEXT_FAINT,
            arrowsize=10,
            gripcount=0,
            relief="flat",
            borderwidth=0,
        )


    def _card(self, parent, bg=CARD_BG, border=STROKE_SOFT, padx=18, pady=18):
        frame = tk.Frame(
            parent,
            bg=bg,
            highlightthickness=1,
            highlightbackground=border,
            bd=0,
        )
        frame.configure(padx=padx, pady=pady)
        return frame


    def _pill(self, parent, text, bg=PILL_NEUTRAL, fg=TEXT_SOFT, padx=11, pady=5, font=("Segoe UI", 9, "bold")):
        lbl = tk.Label(parent, text=text, bg=bg, fg=fg, font=font)
        lbl.configure(padx=padx, pady=pady)
        return lbl

    def _value_color_for_sg(self, sg):
        return SUCCESS if sg >= 1.2 else WARNING

    # ------------------------------------------------------------------
    #  UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        shell = tk.Frame(self.root, bg=WINDOW_BG)
        shell.pack(fill="both", expand=True, padx=22, pady=22)

        self._build_header(shell)

        body = tk.Frame(shell, bg=WINDOW_BG)
        body.pack(fill="both", expand=True, pady=(18, 0))

        self.sidebar = tk.Frame(body, bg=WINDOW_BG, width=370)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.main = tk.Frame(body, bg=WINDOW_BG)
        self.main.pack(side="left", fill="both", expand=True, padx=(20, 0))

        self._build_sidebar(self.sidebar)
        self._build_main_content(self.main)


    def _build_header(self, parent):
        header = self._card(parent, bg=CARD_BG, border=STROKE, padx=26, pady=24)
        header.pack(fill="x")

        left = tk.Frame(header, bg=CARD_BG)
        left.pack(side="left", fill="x", expand=True)

        brand_row = tk.Frame(left, bg=CARD_BG)
        brand_row.pack(anchor="w")
        if self._logo_small:
            tk.Label(brand_row, image=self._logo_small, bg=CARD_BG).pack(side="left", padx=(0, 10))
        self._pill(brand_row, "TRMekatronik", bg=PILL_NEUTRAL, fg=TEXT_SOFT, padx=10, pady=4).pack(side="left")

        tk.Label(
            left,
            text="Vallier–Heydenreich Solver",
            bg=CARD_BG,
            fg=TEXT,
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor="w", pady=(18, 4))

        tk.Label(
            left,
            text="Interior ballistics workspace with the solver and plots organized in a lighter layout.",
            bg=CARD_BG,
            fg=TEXT_SOFT,
            font=("Segoe UI", 11),
        ).pack(anchor="w")

        right = tk.Frame(header, bg=CARD_BG)
        right.pack(side="right", anchor="n")

        self.status_pill = self._pill(right, "Ready", bg=PILL_NEUTRAL, fg=TEXT_SOFT, padx=14, pady=7)
        self.status_pill.pack(side="right", padx=(12, 0))

        self.export_button = tk.Menubutton(
            right,
            text="Export Results",
            bg=PILL_NEUTRAL,
            fg=TEXT,
            activebackground="#e9e9ef",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 10, "bold"),
            padx=18,
            pady=11,
            cursor="hand2",
            direction="below",
        )
        self.export_button.menu = tk.Menu(self.export_button, tearoff=0, bg="#ffffff", fg=TEXT, activebackground=PILL_BG, activeforeground=TEXT)
        self.export_button["menu"] = self.export_button.menu
        self.export_button.menu.add_command(label="Export as XML", command=self._export_results_xml)
        self.export_button.menu.add_command(label="Export as JSON", command=self._export_results_json)
        self.export_button.menu.add_command(label="Export CSV bundle", command=self._export_results_csv_bundle)
        self.export_button.pack(side="right", padx=(0, 10))

        self.run_button = tk.Button(
            right,
            text="Run Solver",
            command=self._run,
            bg=ACCENT,
            fg="#ffffff",
            activebackground=ACCENT_3,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 11, "bold"),
            padx=22,
            pady=11,
            cursor="hand2",
        )
        self.run_button.pack(side="right", padx=(0, 10))


    def _build_sidebar(self, parent):
        brand = self._card(parent, bg=CARD_BG, border=STROKE, padx=20, pady=20)
        brand.pack(fill="x")

        logo_row = tk.Frame(brand, bg=CARD_BG)
        logo_row.pack(fill="x")
        if self._logo_sidebar:
            tk.Label(logo_row, image=self._logo_sidebar, bg=CARD_BG).pack(side="left")
        tk.Label(
            logo_row,
            text="TRMekatronik",
            bg=CARD_BG,
            fg=TEXT,
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left", padx=(12, 0))

        tk.Label(
            brand,
            text="Presets",
            bg=CARD_BG,
            fg=TEXT_FAINT,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(18, 8))

        self.preset_stack = tk.Frame(brand, bg=CARD_BG)
        self.preset_stack.pack(fill="x")
        self._rebuild_preset_buttons_ui()

        preset_actions = tk.Frame(brand, bg=CARD_BG)
        preset_actions.pack(fill="x", pady=(12, 0))

        self.save_preset_button = tk.Button(
            preset_actions,
            text="Save Current",
            command=self._save_current_as_preset,
            bg=PILL_NEUTRAL,
            fg=TEXT,
            activebackground="#e9e9ef",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=10,
            cursor="hand2",
        )
        self.save_preset_button.pack(side="left", fill="x", expand=True)

        self.delete_preset_button = tk.Button(
            preset_actions,
            text="Delete",
            command=self._delete_current_preset,
            bg="#f8f0f3",
            fg=ACCENT_3,
            activebackground="#f5e6ec",
            activeforeground=ACCENT_3,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=10,
            cursor="hand2",
        )
        self.delete_preset_button.pack(side="left", fill="x", expand=True, padx=(10, 0))

        scroll_shell = self._card(parent, bg=CARD_BG, border=STROKE, padx=0, pady=0)
        scroll_shell.pack(fill="both", expand=True, pady=(18, 0))

        header = tk.Frame(scroll_shell, bg=CARD_BG)
        header.pack(fill="x", padx=20, pady=(18, 10))
        tk.Label(
            header,
            text="Input parameters",
            bg=CARD_BG,
            fg=TEXT,
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Mass, geometry, thermochemistry, stability, and rifling.",
            bg=CARD_BG,
            fg=TEXT_SOFT,
            font=("Segoe UI", 10),
            wraplength=300,
            justify="left",
        ).pack(anchor="w", pady=(5, 0))

        canvas_wrap = tk.Frame(scroll_shell, bg=CARD_BG)
        canvas_wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.param_canvas = tk.Canvas(
            canvas_wrap,
            bg=CARD_BG,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.param_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.param_canvas.yview, style="Minimal.Vertical.TScrollbar")
        scrollbar.pack(side="right", fill="y", padx=(8, 0))
        self.param_canvas.configure(yscrollcommand=scrollbar.set)

        self.param_inner = tk.Frame(self.param_canvas, bg=CARD_BG)
        self.param_canvas_window = self.param_canvas.create_window((0, 0), window=self.param_inner, anchor="nw")

        self.param_inner.bind("<Configure>", self._on_param_inner_configure)
        self.param_canvas.bind("<Configure>", self._on_param_canvas_configure)

        meta_map = {key: (label, unit) for key, label, unit in PARAM_META}

        for group_title, keys in PARAM_GROUPS:
            section = self._card(self.param_inner, bg=CARD_BG_2, border=STROKE_SOFT, padx=16, pady=16)
            section.pack(fill="x", padx=8, pady=8)

            tk.Label(
                section,
                text=group_title,
                bg=CARD_BG_2,
                fg=TEXT,
                font=("Segoe UI", 11, "bold"),
            ).pack(anchor="w")

            for key in keys:
                label, unit = meta_map[key]
                self._build_param_row(section, key, label, unit)

        footer = self._card(parent, bg=CARD_BG_3, border=STROKE, padx=16, pady=14)
        footer.pack(fill="x", pady=(18, 0))
        tk.Label(
            footer,
            text="Core model",
            bg=CARD_BG_3,
            fg=TEXT,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            footer,
            text="Presentation refreshed. Vallier–Heydenreich computation unchanged.",
            bg=CARD_BG_3,
            fg=TEXT_SOFT,
            font=("Segoe UI", 9),
            wraplength=300,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))


    def _build_param_row(self, parent, key, label, unit):
        row = tk.Frame(parent, bg=CARD_BG_2)
        row.pack(fill="x", pady=(12, 0))

        text_block = tk.Frame(row, bg=CARD_BG_2)
        text_block.pack(fill="x")

        tk.Label(
            text_block,
            text=label,
            bg=CARD_BG_2,
            fg=TEXT_SOFT,
            font=("Segoe UI", 9),
        ).pack(side="left", anchor="w")

        unit_lbl = tk.Label(
            text_block,
            text=unit,
            bg=PILL_NEUTRAL,
            fg=TEXT_FAINT,
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=3,
        )
        unit_lbl.pack(side="right")

        var = tk.StringVar()
        self.param_vars[key] = var

        entry = tk.Entry(
            row,
            textvariable=var,
            bg="#ffffff",
            fg=TEXT,
            insertbackground=ACCENT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=STROKE,
            highlightcolor=ACCENT,
            font=("Segoe UI", 11),
            justify="right",
        )
        entry.pack(fill="x", pady=(7, 0), ipady=10)
        self.param_entries[key] = entry


    def _build_main_content(self, parent):
        hero = self._card(parent, bg=CARD_BG, border=STROKE, padx=24, pady=22)
        hero.pack(fill="x")

        hero_top = tk.Frame(hero, bg=CARD_BG)
        hero_top.pack(fill="x")

        left = tk.Frame(hero_top, bg=CARD_BG)
        left.pack(side="left", fill="x", expand=True)

        tk.Label(
            left,
            text="Simulation dashboard",
            bg=CARD_BG,
            fg=TEXT,
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left,
            text="Computation summary, key metrics, and plots in one workspace.",
            bg=CARD_BG,
            fg=TEXT_SOFT,
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(6, 0))

        chip_box = tk.Frame(hero_top, bg=CARD_BG)
        chip_box.pack(side="right", anchor="n")
        self.preset_pill = self._pill(chip_box, self.current_preset_name, bg=PILL_BG, fg=ACCENT, padx=12, pady=6)
        self.preset_pill.pack(anchor="e")
        self._pill(chip_box, "Desktop", bg=PILL_NEUTRAL, fg=TEXT_SOFT, padx=12, pady=6).pack(anchor="e", pady=(8, 0))

        metrics = tk.Frame(parent, bg=WINDOW_BG)
        metrics.pack(fill="x", pady=(18, 0))

        defs = [
            ("Muzzle Pressure", ACCENT),
            ("Dwell Time", CHART_ORANGE),
            ("Exit Spin", CHART_GREEN),
            ("Gyro Stability", CHART_PURPLE),
        ]
        for idx, (title, accent) in enumerate(defs):
            card = MetricCard(metrics, title, accent=accent)
            card.pack(side="left", fill="x", expand=True, padx=(0, 14 if idx < len(defs) - 1 else 0))
            self.metric_cards[title] = card

        workspace_shell = self._card(parent, bg=CARD_BG, border=STROKE, padx=18, pady=18)
        workspace_shell.pack(fill="both", expand=True, pady=(18, 0))

        tab_shell = tk.Frame(workspace_shell, bg=PILL_NEUTRAL, padx=8, pady=8)
        tab_shell.pack(anchor="w")

        tabs = [
            ("Overview", "summary"),
            ("Pressure & Velocity", "pv"),
            ("Temperature", "temp"),
            ("Spin & Torque", "spin"),
            ("Groove & Twist", "groove"),
            ("Force & Impulse", "force"),
        ]
        for idx, (title, key) in enumerate(tabs):
            btn = PillTabButton(tab_shell, title, command=lambda k=key: self._select_tab(k), selected=(idx == 0))
            btn.pack(side="left", padx=(0, 8 if idx < len(tabs) - 1 else 0))
            self.tab_buttons[key] = btn

        self.page_host = tk.Frame(workspace_shell, bg=WINDOW_BG)
        self.page_host.pack(fill="both", expand=True, pady=(16, 0))

        self.summary_frame = tk.Frame(self.page_host, bg=WINDOW_BG)
        self.page_frames["summary"] = self.summary_frame

        self.summary_canvas = tk.Canvas(self.summary_frame, bg=WINDOW_BG, highlightthickness=0, bd=0)
        self.summary_canvas.pack(side="left", fill="both", expand=True)
        self.summary_scroll = ttk.Scrollbar(self.summary_frame, orient="vertical", command=self.summary_canvas.yview, style="Minimal.Vertical.TScrollbar")
        self.summary_scroll.pack(side="right", fill="y", padx=(8, 0))
        self.summary_canvas.configure(yscrollcommand=self.summary_scroll.set)

        self.summary_inner = tk.Frame(self.summary_canvas, bg=WINDOW_BG)
        self.summary_canvas_window = self.summary_canvas.create_window((0, 0), window=self.summary_inner, anchor="nw")
        self.summary_inner.bind("<Configure>", self._on_summary_inner_configure)
        self.summary_canvas.bind("<Configure>", self._on_summary_canvas_configure)

        for title, key in tabs[1:]:
            page = tk.Frame(self.page_host, bg=WINDOW_BG)
            self.page_frames[key] = page
            self.plot_tabs[key] = page

        self._select_tab("summary")

    def _on_param_inner_configure(self, _event=None):
        self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))

    def _on_param_canvas_configure(self, event):
        self.param_canvas.itemconfigure(self.param_canvas_window, width=event.width)

    def _on_summary_inner_configure(self, _event=None):
        self.summary_canvas.configure(scrollregion=self.summary_canvas.bbox("all"))

    def _on_summary_canvas_configure(self, event):
        self.summary_canvas.itemconfigure(self.summary_canvas_window, width=event.width)

    def _install_scroll_bindings(self):
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")

    def _resolve_scroll_canvas(self, widget):
        current = widget
        while current is not None:
            if current is self.param_canvas or current is self.param_inner:
                return self.param_canvas
            if current is self.summary_canvas or current is self.summary_inner:
                return self.summary_canvas
            current = getattr(current, "master", None)
        return None

    def _on_mousewheel(self, event):
        canvas = self._resolve_scroll_canvas(event.widget)
        if canvas is None:
            return

        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        elif sys.platform == "darwin":
            delta = -1 * int(event.delta)
        else:
            delta = -1 * int((event.delta or 0) / 120)

        if delta:
            canvas.yview_scroll(delta, "units")

    def _select_tab(self, key):
        if self.current_tab == key:
            return
        if self.current_tab in self.page_frames:
            self.page_frames[self.current_tab].pack_forget()
        self.page_frames[key].pack(fill="both", expand=True)
        self.current_tab = key
        for tab_key, btn in self.tab_buttons.items():
            btn.set_selected(tab_key == key)

    # ------------------------------------------------------------------
    #  PRESETS / INPUTS
    # ------------------------------------------------------------------

    def _load_user_presets(self):
        if not USER_PRESET_PATH.exists():
            return {}
        try:
            raw = json.loads(USER_PRESET_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

        presets = {}
        if isinstance(raw, dict):
            for name, values in raw.items():
                try:
                    presets[str(name)] = self._normalize_preset(values)
                except Exception:
                    continue
        return presets

    def _normalize_preset(self, values):
        normalized = {}
        for key, _label, _unit in PARAM_META:
            if key not in values:
                raise KeyError(key)
            normalized[key] = float(values[key])
        return normalized

    def _rebuild_preset_store(self):
        self.all_presets = {name: dict(payload) for name, payload in AMMO_PRESETS.items()}
        for name, payload in self.user_presets.items():
            self.all_presets[name] = dict(payload)

    def _write_user_presets(self):
        USER_PRESET_PATH.write_text(json.dumps(self.user_presets, indent=2), encoding="utf-8")

    def _preset_names_in_order(self):
        builtins = list(AMMO_PRESETS.keys())
        customs = sorted(name for name in self.all_presets if name not in AMMO_PRESETS)
        return builtins + customs

    def _rebuild_preset_buttons_ui(self):
        if not hasattr(self, "preset_stack"):
            return
        for child in self.preset_stack.winfo_children():
            child.destroy()
        self.preset_buttons = {}
        for name in self._preset_names_in_order():
            btn = tk.Button(
                self.preset_stack,
                text=name,
                command=lambda n=name: self._load_preset(n),
                bg=CARD_BG_2,
                fg=TEXT,
                activebackground=PILL_NEUTRAL,
                activeforeground=TEXT,
                relief="flat",
                bd=0,
                highlightthickness=1,
                highlightbackground=STROKE_SOFT,
                font=("Segoe UI", 10, "bold"),
                padx=14,
                pady=12,
                cursor="hand2",
                anchor="w",
            )
            btn.pack(fill="x", pady=4)
            self.preset_buttons[name] = btn
        self._refresh_preset_buttons()

    def _save_current_as_preset(self):
        try:
            params = self._get_params()
        except ValueError as exc:
            messagebox.showerror("Preset Error", str(exc))
            return

        name = simpledialog.askstring("Save Preset", "Preset name:", parent=self.root)
        if name is None:
            return
        name = name.strip()
        if not name:
            messagebox.showerror("Preset Error", "Preset name cannot be empty.")
            return
        if name in AMMO_PRESETS:
            messagebox.showerror("Preset Error", "Built-in presets are read-only. Choose a new name.")
            return
        if name in self.user_presets:
            if not messagebox.askyesno("Overwrite Preset", f"Overwrite the existing preset '{name}'?"):
                return

        self.user_presets[name] = self._normalize_preset(params)
        self._write_user_presets()
        self._rebuild_preset_store()
        self.current_preset_name = name
        self._rebuild_preset_buttons_ui()
        self._set_status("PRESET SAVED", tone="success")

    def _delete_current_preset(self):
        if self.current_preset_name not in self.user_presets:
            messagebox.showinfo("Delete Preset", "Only custom presets can be deleted.")
            return
        if not messagebox.askyesno("Delete Preset", f"Delete preset '{self.current_preset_name}'?"):
            return
        del self.user_presets[self.current_preset_name]
        self._write_user_presets()
        self._rebuild_preset_store()
        self.current_preset_name = next(iter(self.all_presets))
        self._rebuild_preset_buttons_ui()
        self._load_preset(self.current_preset_name)
        self._set_status("PRESET DELETED", tone="idle")

    def _refresh_preset_buttons(self):
        for name, btn in self.preset_buttons.items():
            selected = (name == self.current_preset_name)
            is_custom = name in self.user_presets
            btn.configure(
                bg=PILL_BG if selected else CARD_BG_2,
                fg=ACCENT if selected else (ACCENT_3 if is_custom else TEXT),
                activebackground=PILL_BG if selected else PILL_NEUTRAL,
                activeforeground=ACCENT if selected else (ACCENT_3 if is_custom else TEXT),
                highlightbackground=ACCENT if selected else STROKE_SOFT,
            )
        if hasattr(self, "preset_pill"):
            self.preset_pill.configure(text=self.current_preset_name)
        if hasattr(self, "delete_preset_button"):
            is_custom = self.current_preset_name in self.user_presets
            self.delete_preset_button.configure(
                state=("normal" if is_custom else "disabled"),
                fg=(ACCENT_3 if is_custom else TEXT_FAINT),
            )

    def _load_preset(self, name):
        preset = self.all_presets[name]
        self.current_preset_name = name
        for key, var in self.param_vars.items():
            val = preset[key]
            if isinstance(val, float) and abs(val) < 0.01 and val != 0:
                var.set(f"{val:.10g}")
            else:
                var.set(f"{val:g}")
        self._refresh_preset_buttons()
        self._set_status("PRESET LOADED", tone="idle")

    def _get_params(self):
        params = {}
        for key, var in self.param_vars.items():
            try:
                params[key] = float(var.get())
                self.param_entries[key].configure(highlightbackground=STROKE, highlightcolor=ACCENT)
            except ValueError:
                self.param_entries[key].configure(highlightbackground=DANGER, highlightcolor=DANGER)
                raise ValueError(f"Invalid value for {key}: '{var.get()}'")
        return params

    def _results_export_payload(self):
        if self.results is None or self.last_run_params is None:
            messagebox.showinfo("Export Results", "Run the solver once before exporting results.")
            return None
        return {
            "metadata": {
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "preset": self.last_run_preset_name,
            },
            "parameters": dict(self.last_run_params),
            "results": self.results,
        }

    def _serialize_value(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return np.asarray(value).tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def _split_results(self):
        scalars = {}
        arrays = {}
        for name, value in self.results.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    scalars[name] = value.item()
                else:
                    arrays[name] = np.asarray(value)
            elif isinstance(value, (list, tuple)):
                arr = np.asarray(value)
                if arr.ndim == 0:
                    scalars[name] = arr.item()
                else:
                    arrays[name] = arr
            elif isinstance(value, (np.floating, np.integer, float, int)):
                scalars[name] = float(value)
        return scalars, arrays

    def _export_results_xml(self):
        payload = self._results_export_payload()
        if payload is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Results as XML",
            defaultextension=".xml",
            filetypes=[("XML files", "*.xml")],
            initialfile="vallier_results.xml",
        )
        if not path:
            return

        scalars, arrays = self._split_results()
        root = ET.Element("vallier_run")
        meta = ET.SubElement(root, "metadata")
        for key, value in payload["metadata"].items():
            ET.SubElement(meta, key).text = str(value)

        params_node = ET.SubElement(root, "parameters")
        for key, value in payload["parameters"].items():
            ET.SubElement(params_node, "parameter", name=key).text = f"{value:.12g}"

        scalars_node = ET.SubElement(root, "scalar_results")
        for key, value in scalars.items():
            ET.SubElement(scalars_node, "result", name=key).text = f"{value:.12g}"

        arrays_node = ET.SubElement(root, "array_results")
        for key, array in arrays.items():
            arr_node = ET.SubElement(arrays_node, "array", name=key, ndim=str(array.ndim), shape="x".join(map(str, array.shape)))
            if array.ndim == 1:
                for idx, value in enumerate(array.tolist()):
                    ET.SubElement(arr_node, "v", index=str(idx)).text = f"{float(value):.12g}"
            else:
                for idx, row in enumerate(array.tolist()):
                    row_node = ET.SubElement(arr_node, "row", index=str(idx))
                    row_node.text = ",".join(f"{float(v):.12g}" for v in np.ravel(row))

        tree = ET.ElementTree(root)
        try:
            ET.indent(tree, space="  ")
        except Exception:
            pass
        tree.write(path, encoding="utf-8", xml_declaration=True)
        self._set_status("XML EXPORTED", tone="success")

    def _export_results_json(self):
        payload = self._results_export_payload()
        if payload is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Results as JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="vallier_results.json",
        )
        if not path:
            return

        serializable = {
            "metadata": payload["metadata"],
            "parameters": payload["parameters"],
            "results": {k: self._serialize_value(v) for k, v in self.results.items()},
        }
        Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        self._set_status("JSON EXPORTED", tone="success")

    def _export_results_csv_bundle(self):
        payload = self._results_export_payload()
        if payload is None:
            return
        root_dir = filedialog.askdirectory(parent=self.root, title="Choose export folder")
        if not root_dir:
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(root_dir) / f"vallier_export_{stamp}"
        export_dir.mkdir(parents=True, exist_ok=True)

        (export_dir / "metadata.json").write_text(json.dumps(payload["metadata"], indent=2), encoding="utf-8")

        with (export_dir / "parameters.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            for key, value in payload["parameters"].items():
                writer.writerow([key, f"{value:.12g}"])

        scalars, arrays = self._split_results()
        with (export_dir / "scalar_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["result", "value"])
            for key, value in scalars.items():
                writer.writerow([key, f"{value:.12g}"])

        arrays_dir = export_dir / "arrays"
        arrays_dir.mkdir(exist_ok=True)
        for key, array in arrays.items():
            target = arrays_dir / f"{key}.csv"
            with target.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if array.ndim == 1:
                    writer.writerow(["index", key])
                    for idx, value in enumerate(array.tolist()):
                        writer.writerow([idx, f"{float(value):.12g}"])
                else:
                    writer.writerow([f"col_{i}" for i in range(array.shape[1] if array.ndim > 1 else 1)])
                    for row in array.tolist():
                        flat_row = np.ravel(row)
                        writer.writerow([f"{float(v):.12g}" for v in flat_row])

        self._set_status("CSV BUNDLE EXPORTED", tone="success")

    # ------------------------------------------------------------------
    #  STATUS / EMPTY STATE
    # ------------------------------------------------------------------

    def _set_status(self, text, tone="idle"):
        if tone == "success":
            bg, fg = "#edf8f2", SUCCESS
        elif tone == "error":
            bg, fg = "#faecef", DANGER
        elif tone == "busy":
            bg, fg = "#fbf2e8", WARNING
        else:
            bg, fg = PILL_NEUTRAL, TEXT_SOFT
        self.status_pill.configure(text=text, bg=bg, fg=fg)


    def _show_empty_state(self):
        for card in self.metric_cards.values():
            card.set_value("—", "Waiting for the first run")

        for widget in self.summary_inner.winfo_children():
            widget.destroy()

        holder = self._card(self.summary_inner, bg=CARD_BG, border=STROKE, padx=30, pady=30)
        holder.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(
            holder,
            text="Run the solver to populate the dashboard",
            bg=CARD_BG,
            fg=TEXT,
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor="w")

        tk.Label(
            holder,
            text="Select a preset or edit the inputs, then run the computation. Results and plots will appear here.",
            bg=CARD_BG,
            fg=TEXT_SOFT,
            font=("Segoe UI", 11),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(10, 22))

        hint_wrap = tk.Frame(holder, bg=CARD_BG)
        hint_wrap.pack(fill="x")
        for idx, text in enumerate([
            "Select an ammunition preset.",
            "Adjust the parameter set if needed.",
            "Run the solver and inspect the plots.",
        ], start=1):
            tile = self._card(hint_wrap, bg=CARD_BG_2, border=STROKE_SOFT, padx=18, pady=18)
            tile.pack(side="left", fill="both", expand=True, padx=(0, 12 if idx < 3 else 0))
            badge = tk.Label(tile, text=str(idx), bg=PILL_BG, fg=ACCENT, font=("Segoe UI", 10, "bold"), padx=10, pady=4)
            badge.pack(anchor="w")
            tk.Label(tile, text=text, bg=CARD_BG_2, fg=TEXT_SOFT, font=("Segoe UI", 10), wraplength=230, justify="left").pack(anchor="w", pady=(12, 0))

        self._show_plot_placeholders()

    def _show_plot_placeholders(self):
        for title, key in [
            ("Pressure & Velocity", "pv"),
            ("Temperature", "temp"),
            ("Spin & Torque", "spin"),
            ("Groove & Twist", "groove"),
            ("Force & Impulse", "force"),
        ]:
            self._clear_plot_tab(key)
            holder = self._card(self.plot_tabs[key], bg=CARD_BG, border=STROKE, padx=24, pady=24)
            holder.pack(fill="both", expand=True, padx=16, pady=16)
            tk.Label(holder, text=title, bg=CARD_BG, fg=TEXT, font=("Segoe UI", 18, "bold")).pack(anchor="w")
            tk.Label(
                holder,
                text="Plot area will appear here after a successful run.",
                bg=CARD_BG,
                fg=TEXT_SOFT,
                font=("Segoe UI", 10),
            ).pack(anchor="w", pady=(6, 24))
            spacer = tk.Frame(holder, bg=PLOT_FACE, highlightthickness=1, highlightbackground=STROKE_SOFT)
            spacer.pack(fill="both", expand=True)
            tk.Label(
                spacer,
                text="No data yet",
                bg=PLOT_FACE,
                fg=TEXT_FAINT,
                font=("Segoe UI", 16, "bold"),
            ).place(relx=0.5, rely=0.5, anchor="center")

    # ------------------------------------------------------------------
    #  RUN
    # ------------------------------------------------------------------
    def _run(self):
        self._set_status("COMPUTING", tone="busy")
        self.root.update_idletasks()

        try:
            params = self._get_params()
        except ValueError as exc:
            self._set_status("INPUT ERROR", tone="error")
            messagebox.showerror("Parameter Error", str(exc))
            return

        try:
            self.results = run_solver(params)
            self.last_run_params = dict(params)
            self.last_run_preset_name = self.current_preset_name
        except Exception as exc:
            self._set_status("SOLVER ERROR", tone="error")
            messagebox.showerror("Solver Error", str(exc))
            return

        self._set_status("RUN COMPLETE", tone="success")
        self._update_metric_cards()
        self._update_summary()
        self._update_plots()

    # ------------------------------------------------------------------
    #  SUMMARY
    # ------------------------------------------------------------------

    def _summary_section(self, parent, title, rows, subtitle=None, accent=ACCENT):
        card = self._card(parent, bg=CARD_BG, border=STROKE, padx=20, pady=18)
        card.pack(fill="x", padx=16, pady=8)

        head = tk.Frame(card, bg=CARD_BG)
        head.pack(fill="x")

        chip_row = tk.Frame(head, bg=CARD_BG)
        chip_row.pack(anchor="w")
        self._pill(chip_row, title, bg=PILL_BG, fg=accent, padx=12, pady=5).pack(side="left")

        if subtitle:
            tk.Label(
                card,
                text=subtitle,
                bg=CARD_BG,
                fg=TEXT_FAINT,
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=(10, 2))

        for label, value, unit, color in rows:
            row = tk.Frame(card, bg=CARD_BG)
            row.pack(fill="x", pady=(12, 0))

            tk.Label(
                row,
                text=label,
                bg=CARD_BG,
                fg=TEXT_SOFT,
                font=("Segoe UI", 10),
            ).pack(side="left", anchor="w")

            right = tk.Frame(row, bg=CARD_BG)
            right.pack(side="right")

            tk.Label(
                right,
                text=value,
                bg=CARD_BG,
                fg=color,
                font=("Segoe UI", 11, "bold"),
            ).pack(side="left")
            if unit:
                tk.Label(
                    right,
                    text=f"  {unit}",
                    bg=CARD_BG,
                    fg=TEXT_FAINT,
                    font=("Segoe UI", 9),
                ).pack(side="left")

        return card

    def _update_metric_cards(self):
        r = self.results
        self.metric_cards["Muzzle Pressure"].set_value(
            f"{r['P_muzzle']:.2f} MPa",
            meta=f"P₀ = {r['P0_MPa']:.2f} MPa",
            color=CHART_RED,
        )
        self.metric_cards["Dwell Time"].set_value(
            f"{r['t_exit_ms']:.3f} ms",
            meta=f"Pmax at {r['t_Pmax_ms']:.3f} ms",
            color=CHART_ORANGE,
        )
        self.metric_cards["Exit Spin"].set_value(
            f"{r['omega_exit']:.1f} rad/s",
            meta=f"Max torque {r['max_torque']:.1f} Nm",
            color=CHART_GREEN,
        )
        self.metric_cards["Gyro Stability"].set_value(
            f"Sg = {r['Sg']:.3f}",
            meta="Stable" if r["Sg"] >= 1.2 else "Below the 1.2 rule-of-thumb",
            color=self._value_color_for_sg(r["Sg"]),
        )


    def _update_summary(self):
        r = self.results
        for widget in self.summary_inner.winfo_children():
            widget.destroy()

        banner = self._card(self.summary_inner, bg=CARD_BG, border=STROKE, padx=24, pady=22)
        banner.pack(fill="x", padx=16, pady=16)

        top = tk.Frame(banner, bg=CARD_BG)
        top.pack(fill="x")

        left = tk.Frame(top, bg=CARD_BG)
        left.pack(side="left", fill="x", expand=True)

        status_text = "Gyroscopically stable" if r["Sg"] >= 1.2 else "Below the 1.2 stability rule-of-thumb"
        status_color = SUCCESS if r["Sg"] >= 1.2 else WARNING

        tk.Label(
            left,
            text="Overview",
            bg=CARD_BG,
            fg=TEXT,
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left,
            text=self.current_preset_name,
            bg=CARD_BG,
            fg=TEXT_SOFT,
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(4, 0))
        tk.Label(
            left,
            text=status_text,
            bg=CARD_BG,
            fg=status_color,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(12, 0))

        pill_box = tk.Frame(top, bg=CARD_BG)
        pill_box.pack(side="right")
        self._pill(pill_box, self.current_preset_name, bg=PILL_BG, fg=ACCENT).pack(anchor="e")
        self._pill(pill_box, f"Sg {r['Sg']:.3f}", bg="#edf8f2" if r["Sg"] >= 1.2 else "#fbf2e8", fg=status_color).pack(anchor="e", pady=(8, 0))

        self._summary_section(
            self.summary_inner,
            "Peak pressure and motion",
            [
                ("Piezometric efficiency η", f"{r['eta']:.4f}", "", ACCENT),
                ("Position at Pmax", f"{r['x_Pmax_mm']:.3f}", "mm", TEXT),
                ("Velocity at Pmax", f"{r['v_Pmax']:.3f}", "m/s", TEXT),
                ("Time to Pmax", f"{r['t_Pmax_ms']:.3f}", "ms", TEXT),
            ],
            subtitle="Core in-bore milestones",
            accent=ACCENT,
        )

        self._summary_section(
            self.summary_inner,
            "Muzzle, spin, and torque",
            [
                ("Muzzle pressure", f"{r['P_muzzle']:.3f}", "MPa", CHART_RED),
                ("Exit spin rate ω", f"{r['omega_exit']:.3f}", "rad/s", CHART_GREEN),
                ("Maximum torque", f"{r['max_torque']:.3f}", "Nm", CHART_PURPLE),
                ("Average base pressure P₀", f"{r['P0_MPa']:.3f}", "MPa", TEXT),
            ],
            subtitle="Primary outputs",
            accent=CHART_GREEN,
        )

        self._summary_section(
            self.summary_inner,
            "Impulse and aftereffect",
            [
                ("Barrel dwell time", f"{r['t_exit_ms']:.3f}", "ms", CHART_ORANGE),
                ("Aftereffect duration tₙ", f"{r['t_n']:.3f}", "ms", TEXT),
                ("Total impulse time tₑ", f"{r['t_e']:.3f}", "ms", TEXT),
                ("Total impulse", f"{r['impulse']:.3f}", "N·s", CHART_GOLD),
                ("Blast force", f"{r['blast_force']:.3f}", "N", TEXT),
            ],
            subtitle="Exit and post-exit timing",
            accent=CHART_ORANGE,
        )

    def _clear_plot_tab(self, key):
        for widget in self.plot_tabs[key].winfo_children():
            widget.destroy()


    def _make_figure(self, nrows=1, ncols=1, figsize=(11.5, 6.0)):
        fig = Figure(figsize=figsize, dpi=100, facecolor=PLOT_FACE)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.11, wspace=0.24, hspace=0.34)

        axes = []
        for idx in range(nrows * ncols):
            ax = fig.add_subplot(nrows, ncols, idx + 1)
            ax.set_facecolor(PLOT_BG)
            ax.grid(True, color=PLOT_GRID, linewidth=0.8, alpha=1.0)
            ax.tick_params(colors=TEXT_SOFT, labelsize=9)
            ax.xaxis.label.set_color(TEXT_SOFT)
            ax.yaxis.label.set_color(TEXT_SOFT)
            ax.title.set_color(TEXT)
            for spine in ax.spines.values():
                spine.set_color(STROKE)
            axes.append(ax)
        return fig, axes


    def _embed_figure(self, tab_key, fig, title, subtitle):
        self._clear_plot_tab(tab_key)
        shell = self._card(self.plot_tabs[tab_key], bg=CARD_BG, border=STROKE, padx=18, pady=18)
        shell.pack(fill="both", expand=True, padx=16, pady=16)

        head = tk.Frame(shell, bg=CARD_BG)
        head.pack(fill="x")

        left = tk.Frame(head, bg=CARD_BG)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=title, bg=CARD_BG, fg=TEXT, font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(left, text=subtitle, bg=CARD_BG, fg=TEXT_SOFT, font=("Segoe UI", 10)).pack(anchor="w", pady=(5, 0))

        canvas = FigureCanvasTkAgg(fig, master=shell)
        toolbar = NavigationToolbar2Tk(canvas, shell, pack_toolbar=False)

        tool_row = tk.Frame(head, bg=CARD_BG)
        tool_row.pack(side="right", anchor="n")

        def tool_button(label, command):
            btn = tk.Button(
                tool_row,
                text=label,
                command=command,
                bg=PILL_NEUTRAL,
                fg=TEXT,
                activebackground="#e9e9ef",
                activeforeground=TEXT,
                relief="flat",
                bd=0,
                highlightthickness=0,
                font=("Segoe UI", 9, "bold"),
                padx=12,
                pady=8,
                cursor="hand2",
            )
            btn.pack(side="left", padx=(8, 0))
            return btn

        def sync_mode():
            mode = str(getattr(toolbar, "mode", "")).upper()
            pan_active = "PAN" in mode
            zoom_active = "ZOOM" in mode
            pan_btn.configure(bg=(PILL_BG if pan_active else PILL_NEUTRAL), fg=(ACCENT if pan_active else TEXT))
            zoom_btn.configure(bg=(PILL_BG if zoom_active else PILL_NEUTRAL), fg=(ACCENT if zoom_active else TEXT))

        home_btn = tool_button("Home", lambda: (toolbar.home(), sync_mode()))
        back_btn = tool_button("Back", lambda: (toolbar.back(), sync_mode()))
        fwd_btn = tool_button("Forward", lambda: (toolbar.forward(), sync_mode()))
        pan_btn = tool_button("Pan", lambda: (toolbar.pan(), sync_mode()))
        zoom_btn = tool_button("Zoom", lambda: (toolbar.zoom(), sync_mode()))
        save_btn = tool_button("Save PNG", toolbar.save_figure)
        sync_mode()

        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(bg=PLOT_FACE, highlightthickness=0, bd=0)
        widget.pack(fill="both", expand=True, pady=(16, 0))

    def _update_plots(self):
        r = self.results
        tms = r["tms"]
        t_exit = r["t_exit_ms"]
        t_pmax = r["t_Pmax_ms"]
        x_pmax = r["x_Pmax_mm"]

        # Pressure & Velocity
        fig, axes = self._make_figure(2, 2, figsize=(12.6, 7.3))
        ax1, ax2, ax3, ax4 = axes

        p_t = r["Pres"] / 1e6
        v_t = r["Velo"]
        p_x = r["P1"] / 1e6
        v_x = r["Vel1"]
        x_mm = r["X1_mm"]

        ax1.plot(tms, p_t, color=CHART_RED, linewidth=2.2)
        ax1.fill_between(tms, 0, p_t, color=CHART_RED, alpha=0.12)
        ax1.axvline(t_pmax, color=ACCENT_2, linewidth=1.2, linestyle="--")
        ax1.axvline(t_exit, color=CHART_GOLD, linewidth=1.2, linestyle=":")
        ax1.set_title("Pressure vs Time")
        ax1.set_xlabel("Time [ms]")
        ax1.set_ylabel("Pressure [MPa]")

        ax2.plot(tms, v_t, color=CHART_BLUE, linewidth=2.2)
        ax2.fill_between(tms, 0, v_t, color=CHART_BLUE, alpha=0.10)
        ax2.axvline(t_exit, color=CHART_GOLD, linewidth=1.2, linestyle=":")
        ax2.set_title("Velocity vs Time")
        ax2.set_xlabel("Time [ms]")
        ax2.set_ylabel("Velocity [m/s]")

        ax3.plot(x_mm, p_x, color=CHART_RED, linewidth=2.2)
        ax3.fill_between(x_mm, 0, p_x, color=CHART_RED, alpha=0.12)
        ax3.axvline(x_pmax, color=ACCENT_2, linewidth=1.2, linestyle="--")
        ax3.set_title("In-Bore Pressure")
        ax3.set_xlabel("Travel [mm]")
        ax3.set_ylabel("Pressure [MPa]")

        ax4.plot(x_mm, v_x, color=CHART_BLUE, linewidth=2.2)
        ax4.fill_between(x_mm, 0, v_x, color=CHART_BLUE, alpha=0.10)
        ax4.axvline(x_pmax, color=ACCENT_2, linewidth=1.2, linestyle="--")
        ax4.set_title("In-Bore Velocity")
        ax4.set_xlabel("Travel [mm]")
        ax4.set_ylabel("Velocity [m/s]")

        self._embed_figure(
            "pv",
            fig,
            "Pressure & Velocity",
            "Dashed line marks the pressure peak; dotted line marks projectile exit in time-domain charts.",
        )

        # Temperature
        fig, axes = self._make_figure(1, 1, figsize=(12.6, 5.7))
        ax = axes[0]
        temp = r["Temp"]
        ax.plot(tms, temp, color=CHART_ORANGE, linewidth=2.3)
        ax.fill_between(tms, np.min(temp), temp, color=CHART_ORANGE, alpha=0.14)
        ax.axvline(t_exit, color=CHART_GOLD, linewidth=1.2, linestyle=":")
        ax.set_title("Gas Temperature vs Time")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Temperature [K]")

        self._embed_figure(
            "temp",
            fig,
            "Temperature",
            "In-bore temperature evolution with post-muzzle decay.",
        )

        # Spin & Torque
        fig, axes = self._make_figure(1, 3, figsize=(14.8, 5.1))
        ax1, ax2, ax3 = axes

        ax1.plot(x_mm, r["omega"], color=CHART_GREEN, linewidth=2.2)
        ax1.fill_between(x_mm, 0, r["omega"], color=CHART_GREEN, alpha=0.10)
        ax1.set_title("Spin Rate")
        ax1.set_xlabel("Travel [mm]")
        ax1.set_ylabel("ω [rad/s]")

        ax2.plot(r["X1"], r["Torque_1"], color=CHART_RED, linewidth=2.0, label="T₁ full expression")
        ax2.plot(r["X1"], r["Torque_2"], color=CHART_BLUE, linewidth=2.0, label="T₂ = I·α")
        leg = ax2.legend(frameon=True, facecolor=PLOT_BG, edgecolor=STROKE, fontsize=9)
        for txt in leg.get_texts():
            txt.set_color(TEXT_SOFT)
        ax2.set_title("Torque on Rifling Lands")
        ax2.set_xlabel("Travel [m]")
        ax2.set_ylabel("Torque [Nm]")

        ax3.plot(x_mm, r["theta_ddot"], color=CHART_PURPLE, linewidth=2.2)
        ax3.fill_between(x_mm, 0, r["theta_ddot"], color=CHART_PURPLE, alpha=0.09)
        ax3.set_title("Angular Acceleration")
        ax3.set_xlabel("Travel [mm]")
        ax3.set_ylabel("θ̈ [rad/s²]")

        self._embed_figure(
            "spin",
            fig,
            "Spin & Torque",
            "Comparison of the full torque expression and the I·α form.",
        )

        # Groove & Twist
        fig, axes = self._make_figure(1, 2, figsize=(13.4, 5.3))
        ax1, ax2 = axes

        ax1.plot(x_mm, r["Y"], color=CHART_TEAL, linewidth=2.3)
        ax1.fill_between(x_mm, 0, r["Y"], color=CHART_TEAL, alpha=0.10)
        ax1.set_title("Groove Profile")
        ax1.set_xlabel("Travel [mm]")
        ax1.set_ylabel("Y [mm]")

        ax2.plot(x_mm, r["alpha_deg"], color=CHART_GOLD, linewidth=2.3)
        ax2.fill_between(x_mm, 0, r["alpha_deg"], color=CHART_GOLD, alpha=0.11)
        ax2.set_title("Twist Angle")
        ax2.set_xlabel("Travel [mm]")
        ax2.set_ylabel("α [deg]")

        self._embed_figure(
            "groove",
            fig,
            "Groove & Twist",
            "Rifling profile and twist-angle development along the bore.",
        )

        # Force & Impulse
        fig, axes = self._make_figure(1, 2, figsize=(13.4, 5.3))
        ax1, ax2 = axes

        ax1.plot(tms, r["Force_all"], color=CHART_RED, linewidth=2.1)
        ax1.fill_between(tms, 0, r["Force_all"], color=CHART_RED, alpha=0.11)
        ax1.axvline(r["t_e"], color=CHART_GOLD, linewidth=1.2, linestyle="--")
        ax1.set_title("Total Force vs Time")
        ax1.set_xlabel("Time [ms]")
        ax1.set_ylabel("Force [N]")

        ax2.plot(r["t_impulse"], r["Force_impulse"], color=CHART_ORANGE, linewidth=2.2)
        ax2.fill_between(r["t_impulse"], 0, r["Force_impulse"], color=CHART_ORANGE, alpha=0.16)
        ax2.set_title("Impulse Region")
        ax2.set_xlabel("Time [ms]")
        ax2.set_ylabel("Force [N]")

        self._embed_figure(
            "force",
            fig,
            "Force & Impulse",
            f"Impulse is integrated up to tₑ = {r['t_e']:.3f} ms.",
        )


# ============================================================================
#  ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()

    try:
        import ctypes
        root.update()
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(1)),
            ctypes.sizeof(ctypes.c_int),
        )
    except Exception:
        pass

    app = VallierApp(root)
    root.mainloop()
