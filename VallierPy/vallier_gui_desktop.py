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
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import base64
import io

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
BG       = "#0d1117"
BG2      = "#161b22"
BG3      = "#1c2129"
BORDER   = "#30363d"
ACCENT   = "#2b81c1"
ACCENT2  = "#58a6ff"
TEXT     = "#e6edf3"
TEXT2    = "#8b949e"
GREEN    = "#3fb950"
RED      = "#f85149"
ORANGE   = "#d29922"

PLOT_BG   = "#0d1117"
PLOT_FACE = "#161b22"
PLOT_GRID = "#21262d"


# ============================================================================
#  MAIN APPLICATION
# ============================================================================
class VallierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vallier-Heydenreich Solver — TR Mekatronik")
        self.root.configure(bg=BG)
        self.root.geometry("1280x800")
        self.root.minsize(1000, 600)

        # Try to set the window icon from the embedded logo
        try:
            logo_bytes = base64.b64decode(LOGO_B64)
            self._logo_img = tk.PhotoImage(data=base64.b64encode(logo_bytes))
            self.root.iconphoto(False, self._logo_img)
        except Exception:
            self._logo_img = None

        self.results = None
        self.param_vars = {}
        self._build_ui()
        self._load_preset(list(AMMO_PRESETS.keys())[0])

    # ------------------------------------------------------------------
    #  UI CONSTRUCTION
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=BG)
        style.configure("Panel.TFrame", background=BG2)
        style.configure("Dark.TLabel", background=BG2, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=BG2, foreground=TEXT, font=("Segoe UI", 11, "bold"))
        style.configure("Small.TLabel", background=BG2, foreground=TEXT2, font=("Segoe UI", 8))
        style.configure("Unit.TLabel", background=BG2, foreground=TEXT2, font=("Consolas", 8))
        style.configure("Result.TLabel", background=BG2, foreground=ACCENT2, font=("Consolas", 10, "bold"))
        style.configure("Sg.TLabel", background=BG2, foreground=GREEN, font=("Consolas", 10, "bold"))
        style.configure("SgBad.TLabel", background=BG2, foreground=RED, font=("Consolas", 10, "bold"))
        style.configure("Dark.TNotebook", background=BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background=BG3, foreground=TEXT2,
                        font=("Segoe UI", 9), padding=[12, 4])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG2)], foreground=[("selected", TEXT)])
        style.configure("Fire.TButton", font=("Consolas", 13, "bold"),
                        foreground="#ffffff", background=ACCENT, padding=[20, 8])
        style.map("Fire.TButton", background=[("active", "#1a6da0")])
        style.configure("Preset.TButton", font=("Consolas", 9),
                        foreground=TEXT, background=BG3, padding=[8, 3])
        style.map("Preset.TButton", background=[("active", ACCENT)])

        # ---- Header ----
        header = tk.Frame(self.root, bg=BG2, height=48)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        if self._logo_img:
            logo_label = tk.Label(header, image=self._logo_img, bg=BG2)
            logo_label.pack(side="left", padx=(16, 8), pady=6)

        tk.Label(header, text="VALLIER-HEYDENREICH SOLVER", bg=BG2, fg=TEXT2,
                 font=("Consolas", 11)).pack(side="right", padx=16)

        # ---- Main paned ----
        main = tk.PanedWindow(self.root, orient="horizontal", bg=BORDER,
                              sashwidth=2, sashrelief="flat")
        main.pack(fill="both", expand=True)

        # ---- Left panel ----
        left = tk.Frame(main, bg=BG2, width=320)
        main.add(left, minsize=280, width=320)

        # Ammo selector
        sel_frame = tk.Frame(left, bg=BG2)
        sel_frame.pack(fill="x", padx=10, pady=(12, 4))
        tk.Label(sel_frame, text="AMMUNITION", bg=BG2, fg=TEXT2,
                 font=("Consolas", 8)).pack(anchor="w")

        btn_frame = tk.Frame(sel_frame, bg=BG2)
        btn_frame.pack(fill="x", pady=(4, 0))
        for i, name in enumerate(AMMO_PRESETS.keys()):
            b = ttk.Button(btn_frame, text=name, style="Preset.TButton",
                           command=lambda n=name: self._load_preset(n))
            b.grid(row=i//2, column=i%2, padx=2, pady=2, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        # Separator
        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", padx=10, pady=8)

        # Parameters (scrollable)
        tk.Label(left, text="PARAMETERS", bg=BG2, fg=TEXT2,
                 font=("Consolas", 8)).pack(anchor="w", padx=10)

        param_canvas = tk.Canvas(left, bg=BG2, highlightthickness=0)
        param_scrollbar = tk.Scrollbar(left, orient="vertical", command=param_canvas.yview)
        param_inner = tk.Frame(param_canvas, bg=BG2)

        param_inner.bind("<Configure>",
                         lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all")))
        param_canvas.create_window((0, 0), window=param_inner, anchor="nw")
        param_canvas.configure(yscrollcommand=param_scrollbar.set)

        param_canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=4)
        param_scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            param_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        param_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        for key, label, unit in PARAM_META:
            row = tk.Frame(param_inner, bg=BG2)
            row.pack(fill="x", pady=1)

            tk.Label(row, text=label, bg=BG2, fg=TEXT2, font=("Segoe UI", 8),
                     width=20, anchor="w").pack(side="left")

            var = tk.StringVar()
            self.param_vars[key] = var
            ent = tk.Entry(row, textvariable=var, bg=BG, fg=ACCENT2,
                           insertbackground=ACCENT2, font=("Consolas", 9),
                           width=14, relief="flat", bd=0,
                           highlightthickness=1, highlightcolor=ACCENT,
                           highlightbackground=BORDER, justify="right")
            ent.pack(side="left", padx=(4, 2))

            tk.Label(row, text=unit, bg=BG2, fg=TEXT2, font=("Consolas", 7),
                     width=8, anchor="w").pack(side="left")

        # Fire button at bottom of left panel
        fire_frame = tk.Frame(left, bg=BG2)
        fire_frame.pack(fill="x", side="bottom", padx=10, pady=12)
        self.fire_btn = ttk.Button(fire_frame, text="\u26A1  FIRE", style="Fire.TButton",
                                   command=self._run)
        self.fire_btn.pack(fill="x")

        # ---- Right panel (notebook) ----
        right = tk.Frame(main, bg=BG)
        main.add(right, minsize=500)

        self.notebook = ttk.Notebook(right, style="Dark.TNotebook")
        self.notebook.pack(fill="both", expand=True)

        # Summary tab
        self.summary_frame = tk.Frame(self.notebook, bg=BG2)
        self.notebook.add(self.summary_frame, text="  Summary  ")
        self.summary_text = tk.Text(self.summary_frame, bg=BG2, fg=TEXT,
                                     font=("Consolas", 10), relief="flat",
                                     state="disabled", wrap="word",
                                     insertbackground=TEXT, selectbackground=ACCENT)
        self.summary_text.pack(fill="both", expand=True, padx=16, pady=12)

        # Plot tabs
        self.plot_tabs = {}
        plot_pages = [
            ("Pressure & Velocity", "pv"),
            ("Temperature", "temp"),
            ("Spin & Torque", "spin"),
            ("Groove & Twist", "groove"),
            ("Force & Impulse", "force"),
        ]
        for title, key in plot_pages:
            frame = tk.Frame(self.notebook, bg=BG)
            self.notebook.add(frame, text=f"  {title}  ")
            self.plot_tabs[key] = frame

        # Placeholder message
        self._show_summary_placeholder()

    # ------------------------------------------------------------------
    #  PRESET LOADING
    # ------------------------------------------------------------------
    def _load_preset(self, name):
        preset = AMMO_PRESETS[name]
        for key, var in self.param_vars.items():
            val = preset[key]
            # Format nicely
            if isinstance(val, float) and val < 0.01:
                var.set(f"{val:.10g}")
            else:
                var.set(f"{val:g}")

    # ------------------------------------------------------------------
    #  GET PARAMS FROM UI
    # ------------------------------------------------------------------
    def _get_params(self) -> dict:
        p = {}
        for key, var in self.param_vars.items():
            try:
                p[key] = float(var.get())
            except ValueError:
                raise ValueError(f"Invalid value for {key}: '{var.get()}'")
        return p

    # ------------------------------------------------------------------
    #  RUN SOLVER
    # ------------------------------------------------------------------
    def _run(self):
        try:
            params = self._get_params()
        except ValueError as e:
            messagebox.showerror("Parameter Error", str(e))
            return

        try:
            self.results = run_solver(params)
        except Exception as e:
            messagebox.showerror("Solver Error", str(e))
            return

        self._update_summary()
        self._update_plots()

    # ------------------------------------------------------------------
    #  SUMMARY
    # ------------------------------------------------------------------
    def _show_summary_placeholder(self):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", "\n\n\n")
        self.summary_text.insert("end", "         Select ammunition and press FIRE\n", "placeholder")
        self.summary_text.tag_configure("placeholder", foreground=TEXT2,
                                         font=("Consolas", 12), justify="center")
        self.summary_text.configure(state="disabled")

    def _update_summary(self):
        r = self.results
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")

        self.summary_text.tag_configure("h", foreground=TEXT, font=("Consolas", 12, "bold"))
        self.summary_text.tag_configure("label", foreground=TEXT2, font=("Consolas", 9))
        self.summary_text.tag_configure("val", foreground=ACCENT2, font=("Consolas", 10, "bold"))
        self.summary_text.tag_configure("unit", foreground=TEXT2, font=("Consolas", 8))
        self.summary_text.tag_configure("good", foreground=GREEN, font=("Consolas", 10, "bold"))
        self.summary_text.tag_configure("bad", foreground=RED, font=("Consolas", 10, "bold"))
        self.summary_text.tag_configure("sep", foreground=BORDER, font=("Consolas", 9))

        self.summary_text.insert("end", "  Vallier-Heydenreich Results\n", "h")
        self.summary_text.insert("end", "  " + "\u2500"*50 + "\n\n", "sep")

        rows = [
            ("Piezometric efficiency  \u03B7", f"{r['eta']:.4f}", ""),
            ("Average base pressure  P\u2080", f"{r['P0_MPa']:.2f}", "MPa"),
            ("Position at P_max", f"{r['x_Pmax_mm']:.2f}", "mm"),
            ("Velocity at P_max", f"{r['v_Pmax']:.2f}", "m/s"),
            ("Time to P_max", f"{r['t_Pmax_ms']:.2f}", "ms"),
            ("Muzzle pressure", f"{r['P_muzzle']:.2f}", "MPa"),
            ("Barrel dwell time", f"{r['t_exit_ms']:.2f}", "ms"),
            ("Maximum torque", f"{r['max_torque']:.2f}", "Nm"),
            ("Exit spin rate  \u03C9", f"{r['omega_exit']:.2f}", "rad/s"),
        ]

        for label, val, unit in rows:
            self.summary_text.insert("end", f"  {label:<36s}", "label")
            self.summary_text.insert("end", f"{val:>12s}", "val")
            self.summary_text.insert("end", f"  {unit}\n", "unit")

        self.summary_text.insert("end", "\n  " + "\u2500"*50 + "\n\n", "sep")

        # Stability indicator
        sg = r["Sg"]
        if sg >= 1.2:
            self.summary_text.insert("end", f"  Sg = {sg:.3f}   \u2714  Gyroscopically stable\n\n", "good")
        else:
            self.summary_text.insert("end", f"  Sg = {sg:.3f}   \u2718  Gyroscopically unstable (fin-stabilized?)\n\n", "bad")

        rows2 = [
            ("Aftereffect duration  t_n", f"{r['t_n']:.3f}", "ms"),
            ("Total impulse time  t_e", f"{r['t_e']:.3f}", "ms"),
            ("Total impulse", f"{r['impulse']:.3f}", "N\u00B7s"),
            ("Blast force", f"{r['blast_force']:.1f}", "N"),
        ]
        for label, val, unit in rows2:
            self.summary_text.insert("end", f"  {label:<36s}", "label")
            self.summary_text.insert("end", f"{val:>12s}", "val")
            self.summary_text.insert("end", f"  {unit}\n", "unit")

        self.summary_text.configure(state="disabled")

    # ------------------------------------------------------------------
    #  PLOTS
    # ------------------------------------------------------------------
    def _clear_tab(self, key):
        for w in self.plot_tabs[key].winfo_children():
            w.destroy()

    def _embed_figure(self, tab_key, fig):
        frame = self.plot_tabs[tab_key]
        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.configure(bg=BG2)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _make_figure(self, nrows=1, ncols=1):
        fig = Figure(figsize=(10, 5), dpi=100, facecolor=PLOT_FACE)
        fig.subplots_adjust(hspace=0.35, wspace=0.3)
        axes = []
        for i in range(nrows * ncols):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors=TEXT2, labelsize=8)
            ax.xaxis.label.set_color(TEXT2)
            ax.yaxis.label.set_color(TEXT2)
            ax.title.set_color(TEXT)
            ax.title.set_fontsize(10)
            ax.grid(True, color=PLOT_GRID, linewidth=0.5, alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            axes.append(ax)
        return fig, axes

    def _update_plots(self):
        r = self.results

        # ---- Pressure & Velocity ----
        self._clear_tab("pv")
        fig, (ax1, ax2, ax3, ax4) = self._make_figure(2, 2)

        ax1.plot(r["tms"], r["Pres"]/1e6, color="#e74c3c", linewidth=1)
        ax1.set_xlabel("Time [ms]"); ax1.set_ylabel("Pressure [MPa]")
        ax1.set_title("Pressure vs Time")

        ax2.plot(r["tms"], r["Velo"], color="#3498db", linewidth=1)
        ax2.set_xlabel("Time [ms]"); ax2.set_ylabel("Velocity [m/s]")
        ax2.set_title("Velocity vs Time")

        ax3.plot(r["X1_mm"], r["P1"]/1e6, color="#e74c3c", linewidth=1.2)
        ax3.set_xlabel("Travel [mm]"); ax3.set_ylabel("Pressure [MPa]")
        ax3.set_title("In-Bore Pressure")

        ax4.plot(r["X1_mm"], r["Vel1"], color="#3498db", linewidth=1.2)
        ax4.set_xlabel("Travel [mm]"); ax4.set_ylabel("Velocity [m/s]")
        ax4.set_title("In-Bore Velocity")

        fig.tight_layout(pad=1.5)
        self._embed_figure("pv", fig)

        # ---- Temperature ----
        self._clear_tab("temp")
        fig, (ax1,) = self._make_figure(1, 1)
        ax1.plot(r["tms"], r["Temp"], color="#e67e22", linewidth=1)
        ax1.set_xlabel("Time [ms]"); ax1.set_ylabel("Temperature [K]")
        ax1.set_title("Gas Temperature vs Time")
        fig.tight_layout(pad=2)
        self._embed_figure("temp", fig)

        # ---- Spin & Torque ----
        self._clear_tab("spin")
        fig, (ax1, ax2, ax3) = self._make_figure(1, 3)
        fig.set_size_inches(12, 4)

        ax1.plot(r["X1_mm"], r["omega"], color="#2ecc71", linewidth=1.2)
        ax1.set_xlabel("Travel [mm]"); ax1.set_ylabel("\u03C9 [rad/s]")
        ax1.set_title("Spin Rate")

        ax2.plot(r["X1"], r["Torque_1"], color="#e74c3c", linewidth=1.2, label="T1 (full)")
        ax2.plot(r["X1"], r["Torque_2"], color="#3498db", linewidth=1.2, label="T2 (I\u00B7\u03B1)")
        ax2.legend(fontsize=8, facecolor=PLOT_BG, edgecolor=BORDER, labelcolor=TEXT2)
        ax2.set_xlabel("Travel [m]"); ax2.set_ylabel("Torque [Nm]")
        ax2.set_title("Torque on Rifling Lands")

        ax3.plot(r["X1_mm"], r["theta_ddot"], color="#9b59b6", linewidth=1.2)
        ax3.set_xlabel("Travel [mm]"); ax3.set_ylabel("\u03B8\u0308 [rad/s\u00B2]")
        ax3.set_title("Angular Acceleration")

        fig.tight_layout(pad=1.5)
        self._embed_figure("spin", fig)

        # ---- Groove & Twist ----
        self._clear_tab("groove")
        fig, (ax1, ax2) = self._make_figure(1, 2)

        ax1.plot(r["X1_mm"], r["Y"], color="#1abc9c", linewidth=1.2)
        ax1.set_xlabel("Travel [mm]"); ax1.set_ylabel("Y [mm]")
        ax1.set_title("Groove Profile")

        ax2.plot(r["X1_mm"], r["alpha_deg"], color="#f39c12", linewidth=1.2)
        ax2.set_xlabel("Travel [mm]"); ax2.set_ylabel("\u03B1 [\u00B0]")
        ax2.set_title("Twist Angle")

        fig.tight_layout(pad=2)
        self._embed_figure("groove", fig)

        # ---- Force & Impulse ----
        self._clear_tab("force")
        fig, (ax1, ax2) = self._make_figure(1, 2)

        ax1.plot(r["tms"], r["Force_all"], color="#e74c3c", linewidth=0.8)
        ax1.set_xlabel("Time [ms]"); ax1.set_ylabel("Force [N]")
        ax1.set_title("Total Force vs Time")

        ax2.plot(r["t_impulse"], r["Force_impulse"], color="#e67e22", linewidth=1.2)
        ax2.fill_between(r["t_impulse"], 0, r["Force_impulse"],
                         color="#e67e22", alpha=0.15)
        ax2.set_xlabel("Time [ms]"); ax2.set_ylabel("Force [N]")
        ax2.set_title("Impulse Region")

        fig.tight_layout(pad=2)
        self._embed_figure("force", fig)


# ============================================================================
#  ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()

    # Dark title bar on Windows 10/11
    try:
        import ctypes
        root.update()
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int))
    except Exception:
        pass

    app = VallierApp(root)
    root.mainloop()
