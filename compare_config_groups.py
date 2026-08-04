"""Superpose, par configuration d'essai (diametre laser x epaisseur cible
x regime d'intensite), toutes les courbes Radioss (tirs/campagne_HERA_simu.h5)
et les courbes experimentales disponibles (tirs/Resultat exp/VSLXX.txt).

Ne lance aucune simulation Code1D : uniquement Radioss (deja calcule,
disponible pour les 22 tirs) + experience (disponible pour un sous-ensemble).
"""
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (diametre_mm, epaisseur_mm, regime, [tirs]) — Tableaux 4 et 5
GROUPS = [
    (1, 0.2, "I basse", [32, 35]),
    (1, 0.2, "I haute", [38, 39, 47]),
    (1, 1.0, "I basse", [33, 40]),
    (1, 1.0, "I haute", [34, 41, 46]),
    (2, 0.2, "I basse", [0]),
    (2, 0.2, "I haute", [17, 18, 19]),
    (2, 0.5, "I basse", [4, 5]),
    (2, 0.5, "I haute", [15, 16]),
    (2, 1.0, "I basse", [6, 7]),
    (2, 1.0, "I haute", [13]),
    (2, 2.0, "I haute", [20]),
]

EXP_DIR = "tirs/Resultat exp"
H5_PATH = "tirs/campagne_HERA_simu.h5"


def onset_time(t, v, frac=0.1):
    vmax = np.max(v)
    idx = np.argmax(v >= frac * vmax)
    return t[idx]


with h5py.File(H5_PATH, "r") as h5f:
    for diam, epaisseur, regime, shots in GROUPS:
        fig, ax = plt.subplots()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, num in enumerate(shots):
            color = colors[i % len(colors)]
            radioss = h5f[f"Shot_{num}/DataAnalysis/velocity_simu"][:]
            t_rad = radioss[:, 0] * 1e9
            v_rad = np.abs(radioss[:, 1]) * 1000.0
            ax.plot(t_rad, v_rad, color=color, linestyle="-",
                     label=f"Tir {num} — Radioss")

            exp_path = f"{EXP_DIR}/VSL{num}.txt"
            if os.path.exists(exp_path):
                exp = np.loadtxt(exp_path, delimiter=",")
                t_exp = exp[:, 0] * 1e9
                v_exp = exp[:, 1] * 1000.0
                t0_exp = onset_time(t_exp, v_exp)
                t0_rad = onset_time(t_rad, v_rad)
                ax.plot(t_exp - t0_exp + t0_rad, v_exp, color=color,
                         linestyle="--", alpha=0.6, linewidth=1,
                         label=f"Tir {num} — Exp.")

        ax.set_xlabel("t [ns]")
        ax.set_ylabel("Vitesse face arriere [m/s]")
        ax.set_title(f"Diametre {diam}mm — epaisseur {epaisseur}mm — {regime}")
        ax.legend(fontsize=8)

        fname = f"tirs/groupe_d{diam}mm_e{epaisseur}mm_{regime.replace(' ', '')}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"{diam}mm / {epaisseur}mm / {regime} (tirs {shots}) -> {fname}")
