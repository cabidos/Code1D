"""Post-traitement : suivi du front de choc, Us/Up, vitesse de surface
libre synthetique (VISAR), export et graphiques.
"""
import math

import numpy as np


def shock_front_positions(history, threshold_fraction=0.1):
    """Position (centre de maille) la plus a droite ou P depasse une
    fraction du pic de pression courant, par instantane. Renvoie (t,
    x_front), en sautant les instantanes sans compression significative.
    """
    t_list, x_list = [], []
    p_peak_running = 0.0
    for snap in history:
        p_peak_running = max(p_peak_running, np.max(snap["P"]))
        if p_peak_running <= 0.0:
            continue
        xc = 0.5 * (snap["x"][1:] + snap["x"][:-1])
        above = snap["P"] >= threshold_fraction * p_peak_running
        if not np.any(above):
            continue
        t_list.append(snap["t"])
        x_list.append(xc[above][-1])
    return np.array(t_list), np.array(x_list)


def shock_velocity(t_front, x_front, t_window=None):
    """Pente au sens des moindres carres de x_front(t) -> Us.

    t_window=(t0,t1) restreint l'ajustement au plateau quasi-stationnaire
    (exclut le transitoire de demarrage).
    """
    t, x = t_front, x_front
    if t_window is not None:
        mask = (t >= t_window[0]) & (t <= t_window[1])
        t, x = t[mask], x[mask]
    if len(t) < 2:
        raise ValueError("pas assez de points pour ajuster une vitesse de choc")
    Us, _ = np.polyfit(t, x, 1)
    return Us


def auto_shock_velocity(history, length, x_fraction=(0.15, 0.85), threshold_fraction=0.3):
    """Vitesse de choc mesuree sur la phase de propagation en regime
    etabli, sans devoir choisir une fenetre temporelle a la main.

    Un ajustement sur tout l'historique (t=0 -> t_max) est fausse par :
    le transitoire de montee en pression (front pas encore a vitesse
    stationnaire), et par tout ce qui se passe apres l'arrivee en face
    arriere (reflexion, reverberations, cf. discussion sur le "plateau").
    On ne garde donc que les instants ou le front est repere entre
    x_fraction[0]*length et x_fraction[1]*length (loin des deux bords).
    """
    t_front, x_front = shock_front_positions(history, threshold_fraction)
    if len(t_front) < 2:
        raise ValueError("pas assez de points pour mesurer Us")

    # ne garder que la phase de propagation initiale (x_front strictement
    # croissant) : au-dela (reflexion en face arriere, reverberations), le
    # front au sens du seuil peut repartir en arriere puis re-avancer, ce
    # qui casse un ajustement lineaire global (pente ~nulle ou fausse).
    increasing = np.diff(x_front) > 0
    if not np.all(increasing):
        first_break = int(np.argmax(~increasing)) + 1
        t_front, x_front = t_front[:first_break], x_front[:first_break]
    if len(t_front) < 2:
        raise ValueError("pas assez de points en phase de propagation initiale pour mesurer Us")

    mask = (x_front >= x_fraction[0] * length) & (x_front <= x_fraction[1] * length)
    if np.sum(mask) < 2:
        mask = np.ones_like(t_front, dtype=bool)
    return shock_velocity(t_front[mask], x_front[mask])


def breakout_time(history, length, threshold_fraction=0.3):
    """Instant approximatif (ns) ou le front de choc atteint la face
    arriere (x_front proche de `length`), avant reflexion/reverberation.
    Renvoie None si le front n'est jamais detecte.

    Meme logique que auto_shock_velocity pour isoler la phase de
    propagation initiale (x_front strictement croissant) : au-dela, le
    front au sens du seuil repart en arriere (reflexion), donc le dernier
    point de la phase croissante est la meilleure estimation du
    "breakout".
    """
    t_front, x_front = shock_front_positions(history, threshold_fraction)
    if len(t_front) == 0:
        return None
    increasing = np.diff(x_front) > 0
    if not np.all(increasing):
        first_break = int(np.argmax(~increasing)) + 1
        t_front, x_front = t_front[:first_break], x_front[:first_break]
    return t_front[-1]


def particle_velocity_behind_front(history, x_front, t_front, offset=0.0):
    """Vitesse nodale echantillonnee juste derriere le front suivi, a
    chaque instant de t_front (offset en um, vers l'arriere du choc)."""
    up = np.empty_like(t_front)
    snaps = {s["t"]: s for s in history}
    for i, (t, xf) in enumerate(zip(t_front, x_front)):
        snap = snaps.get(t) or min(history, key=lambda s: abs(s["t"] - t))
        idx = np.searchsorted(snap["x"], xf - offset)
        idx = min(max(idx, 0), len(snap["u"]) - 1)
        up[i] = snap["u"][idx]
    return up


def free_surface_velocity(history):
    t = np.array([s["t"] for s in history])
    u_fs = np.array([s["u"][-1] for s in history])
    return t, u_fs


def save_free_surface_velocity(history, path):
    """Export texte (t, u_face_arriere) — trace VISAR synthetique,
    directement exploitable (tableur, script d'ajustement)."""
    t, u_fs = free_surface_velocity(history)
    np.savetxt(
        path, np.column_stack([t, u_fs]),
        header="t_ns,u_free_surface_km_s", delimiter=",", comments="",
    )


def _sci_fmt(x, decimals=5):
    """Notation scientifique 'E' sans zero de tete sur l'exposant
    (ex. 1.27532E-9), format attendu par l'outil externe qui relit data.txt."""
    if x == 0:
        return f"0.{'0' * decimals}E+0"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    exp = math.floor(math.log10(ax))
    mant = ax / (10 ** exp)
    mant_str = f"{mant:.{decimals}f}"
    if float(mant_str) >= 10.0:
        mant, exp = mant / 10, exp + 1
        mant_str = f"{mant:.{decimals}f}"
    exp_sign = "+" if exp >= 0 else "-"
    return f"{sign}{mant_str}E{exp_sign}{abs(exp)}"


def save_free_surface_velocity_txt(history, path, clamp=1e-6):
    """Export texte (t_s, u_face_arriere_m_s) en notation scientifique
    'E' sans en-tete, pour data.txt. Les valeurs de vitesse sous `clamp`
    (m/s, bruit numerique avant l'arrivee du choc) sont mises a 0."""
    t_ns, u_km_s = free_surface_velocity(history)
    t_s = t_ns * 1e-9
    u_m_s = u_km_s * 1000.0
    u_m_s = np.where(np.abs(u_m_s) < clamp, 0.0, u_m_s)

    lines = [f"{_sci_fmt(ti)},{_sci_fmt(ui)}" for ti, ui in zip(t_s, u_m_s)]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_free_surface_velocity_excel(history, path):
    """Export Excel (t_ns, u_face_arriere_m_s) — memes donnees que
    save_free_surface_velocity, vitesse convertie de km/s vers m/s."""
    from openpyxl import Workbook

    t, u_fs = free_surface_velocity(history)
    u_fs_m_s = u_fs * 1000.0  # km/s -> m/s

    wb = Workbook()
    ws = wb.active
    ws.title = "backface_velocity"
    ws.append(["t_ns", "u_free_surface_m_s"])
    for ti, ui in zip(t, u_fs_m_s):
        ws.append([float(ti), float(ui)])
    wb.save(path)


def save_history(history, path):
    np.savez(
        path,
        t=np.array([s["t"] for s in history]),
        x=np.array([s["x"] for s in history]),
        u=np.array([s["u"] for s in history]),
        rho=np.array([s["rho"] for s in history]),
        e=np.array([s["e"] for s in history]),
        P=np.array([s["P"] for s in history]),
        S=np.array([s["S"] for s in history]),
        q=np.array([s["q"] for s in history]),
    )


def plot_results(history, path_prefix, title=None, profile_t_start=None, profile_t_end=None,
                  extra_profile_times=(1.0, 3.5)):
    """profile_t_start (ns, optionnel) : premier instant a tracer sur le
    graphe des profils de pression. Par defaut (None), on saute
    automatiquement les instantanes ou rien ne s'est encore passe
    (P~0 partout, ex. t=0) et on part du premier instant ou une
    compression significative existe quelque part dans le domaine.

    profile_t_end (ns, optionnel) : dernier instant a tracer. Utile pour
    s'arreter juste avant que le front n'atteigne la face arriere (cf.
    breakout_time) et ne pas melanger l'onde incidente propre avec la
    reflexion/reverberation sur le meme graphe.

    extra_profile_times (ns) : instants ajoutes de force a l'echantillonnage
    (en plus des 6 pas automatiques), typiquement pour visualiser le pic de
    pression impose pres de la surface (cf. loading.py, pic vers t=T0 puis
    plateau/decroissance jusqu'a Tpul).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_snap = len(history)
    if profile_t_start is None:
        first_idx = next((i for i, s in enumerate(history) if np.max(s["P"]) > 1e-6), 0)
    else:
        first_idx = next((i for i, s in enumerate(history) if s["t"] >= profile_t_start), 0)

    if profile_t_end is None:
        last_idx = n_snap - 1
    else:
        later = [i for i, s in enumerate(history) if s["t"] > profile_t_end]
        last_idx = (later[0] - 1) if later else n_snap - 1
    last_idx = max(last_idx, first_idx)

    sample_idx = np.linspace(first_idx, last_idx, min(6, last_idx - first_idx + 1)).astype(int)
    if extra_profile_times:
        t_all = np.array([s["t"] for s in history])
        extra_idx = [int(np.argmin(np.abs(t_all - te))) for te in extra_profile_times]
        sample_idx = np.unique(np.concatenate([sample_idx, extra_idx]))

    fig, ax = plt.subplots()
    for i in sample_idx:
        snap = history[i]
        xc = 0.5 * (snap["x"][1:] + snap["x"][:-1])
        ax.plot(xc, snap["P"], label=f"t={snap['t']:.1f} ns")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("P [GPa]")
    ax.legend(fontsize=7)
    if title:
        ax.set_title(title)
    fig.savefig(f"{path_prefix}_pressure_profiles.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for snap in history:
        ax.plot(snap["x"], np.full_like(snap["x"], snap["t"]), ".", color="k", ms=1)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("t [ns]")
    if title:
        ax.set_title(title)
    fig.savefig(f"{path_prefix}_xt_diagram.png", dpi=150)
    plt.close(fig)

    t_pmax = np.array([s["t"] for s in history])
    Pmax_t = np.array([np.max(s["P"]) for s in history])
    fig, ax = plt.subplots()
    ax.plot(t_pmax, Pmax_t)
    ax.set_xlabel("t [ns]")
    ax.set_ylabel("Pmax(t) [GPa]")
    if title:
        ax.set_title(title)
    fig.savefig(f"{path_prefix}_pmax_vs_t.png", dpi=150)
    plt.close(fig)

    t_fs, u_fs = free_surface_velocity(history)
    fig, ax = plt.subplots()
    ax.plot(t_fs, u_fs)
    ax.set_xlabel("t [ns]")
    ax.set_ylabel("u surface libre [km/s]")
    if title:
        ax.set_title(title)
    fig.savefig(f"{path_prefix}_free_surface_velocity.png", dpi=150)
    plt.close(fig)
