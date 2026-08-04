"""Superpose, pour chaque tir, la vitesse de face arriere Code1D
(tirs/tir_XX_free_surface_velocity.csv), Radioss (tirs/campagne_HERA_simu.h5,
Shot_XX/DataAnalysis/velocity_simu) et la mesure experimentale
(tirs/Resultat exp/VSLXX.txt).

Code1D et Radioss partagent deja le meme repere temporel (t=0 = debut du
tir laser, verifie : pics quasi confondus pour le tir 18) -> seule la
courbe experimentale est recalee, sur l'instant ou le signal depasse 10%
de son pic (onset), le declenchement du diagnostic VISAR n'etant pas
synchronise avec t=0.

Tirs 6/7 (basse intensite, signal experimental tres bruite) : le seuil a
10% du pic est peu fiable (declenchement premature sur du bruit) -> on
recale plutot sur l'instant du pic max.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHOTS = [15, 16, 18, 19, 13, 6, 7]
ALIGN_ON_PEAK = {6, 7}
H5_PATH = "tirs/campagne_HERA_simu.h5"


def onset_time(t, v, frac=0.1):
    vmax = np.max(v)
    idx = np.argmax(v >= frac * vmax)
    return t[idx]


def peak_time(t, v):
    return t[np.argmax(v)]


with h5py.File(H5_PATH, "r") as h5f:
    for num in SHOTS:
        exp = np.loadtxt(f"tirs/Resultat exp/VSL{num}.txt", delimiter=",")
        t_exp = exp[:, 0] * 1e9        # s -> ns
        v_exp = exp[:, 1] * 1000.0     # km/s -> m/s

        sim = np.loadtxt(f"tirs/tir_{num:02d}_free_surface_velocity.csv",
                          delimiter=",", skiprows=1)
        t_sim = sim[:, 0]              # deja en ns
        v_sim = sim[:, 1] * 1000.0     # km/s -> m/s

        radioss = h5f[f"Shot_{num}/DataAnalysis/velocity_simu"][:]
        t_rad = radioss[:, 0] * 1e9    # s -> ns
        v_rad = radioss[:, 1] * 1000.0  # km/s -> m/s

        align_fn = peak_time if num in ALIGN_ON_PEAK else onset_time
        t0_exp = align_fn(t_exp, v_exp)
        t0_ref = align_fn(t_sim, v_sim)  # repere commun Code1D/Radioss

        fig, ax = plt.subplots()
        ax.plot(t_exp - t0_exp + t0_ref, v_exp, label="Experience (VISAR)")
        ax.plot(t_sim, v_sim, label="Code1D")
        ax.plot(t_rad, v_rad, label="Radioss")
        ax.set_xlabel("t [ns] (repere Code1D/Radioss, laser t=0)")
        ax.set_ylabel("Vitesse face arriere [m/s]")
        ax.set_title(f"Tir {num}")
        ax.legend()
        fig.savefig(f"tirs/tir_{num:02d}_comparaison_exp.png", dpi=150)
        plt.close(fig)
        print(f"Tir {num}: -> tirs/tir_{num:02d}_comparaison_exp.png")
