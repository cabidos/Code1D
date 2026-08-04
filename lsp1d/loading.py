"""Profil de pression temporel (chargement laser), multi-tirs.

Modele semi-empirique "regime direct" (type Fabbro/Berthe, choc laser
sans confinement) : la pression crete et la forme temporelle (montee,
plateau/decroissance rapide, relaxation lineaire, decroissance en loi de
puissance) sont parametrees par l'intensite laser crete (Imax) et la
duree d'impulsion (Tpul), pas saisies a la main comme un trapeze.

Unites d'entree du modele (litterature) : Imax en GW/cm2, Tpul et T0 en
ns. La sortie est en GPa. Le temps de simulation est aussi en ns (voir
eos.py), donc t et t_start s'utilisent directement, sans conversion.

Les tirs se superposent (somme), ce qui couvre aussi bien un train de
tirs bien separes qu'un recouvrement partiel.
"""
import math
from dataclasses import dataclass


@dataclass
class LaserPulseParams:
    t_start: float   # ns, instant de declenchement du tir (repere simulation)
    Imax: float         # GW/cm2, intensite laser crete
    Tpul: float           # ns, duree d'impulsion laser
    T0: float               # ns, temps de montee de la pression

    def pressure_at(self, t):
        return _single_laser_pulse(t, self)


def _pmax_direct(Imax):
    """Pression crete (GPa), regime direct (sans confinement)."""
    if Imax <= 100:
        return 0.56 + 0.08 * Imax - 3.2e-4 * Imax**2
    return 2.52 + 0.0238 * Imax


def _pressure_direct_scalar(Time, Imax, Tpul, T0):
    """Pression (GPa) au temps Time (ns), regime direct.

    Forme : montee lineaire jusqu'a T0, decroissance en loi de puissance
    jusqu'a Tpul, relaxation lineaire jusqu'a Ti=1.5*Tpul, puis
    decroissance en loi de puissance asymptotique au-dela.
    """
    Ti = 1.5 * Tpul
    I1 = (1.1 - 9.76e-3 * Tpul) * Imax
    I2 = (0.84 + 0.016 * Tpul) * Imax

    Pmax = _pmax_direct(Imax)
    Pm = 0.092 * Imax**0.623 if Imax <= 100 else 0.162 + 0.0166 * Imax

    Pi = (
        (0.11 + 0.011 * I1)
        if I1 <= (240 - 3.4 * Tpul)
        else (2.79 - 0.09 * Tpul + 1.12e-3 * Tpul**2) + 2e-3 * I1
    )

    n = (
        (0.5 + 1.76e-2 * Tpul) * math.exp(-(((I2 - 115) / 100) ** 2))
        + 1.14 - 1.06e-2 * Tpul + 1.37e-4 * Tpul**2
    )

    delta = 0.83 - (
        (0.467 - 2.45e-3 * Tpul + 1.75e-4 * Tpul**2)
        * math.exp(-(((I2 - 100) / 100) ** 2))
    )

    log_arg = Pmax / Pm if Pm > 0 else 1.0
    p_exp = math.log(max(log_arg, 1e-30)) / math.log(max(40.0 / T0, 1e-30))

    if Time <= T0:
        return Pmax * (Time / T0)
    elif Time <= Tpul:
        return Pmax * (T0 / Time) ** p_exp
    elif Time < Ti:
        P_tpul = _pressure_direct_scalar(Tpul, Imax, Tpul, T0)
        P_ti = Pi
        slope = (P_ti - P_tpul) / max(Ti - Tpul, 1e-30)
        return P_tpul + slope * (Time - Tpul)
    else:
        denom = Time - Ti * delta
        if denom <= 1e-30:
            return 0.0
        return Pi * ((Ti - Ti * delta) / denom) ** n


def _single_laser_pulse(t, pulse):
    t_rel = t - pulse.t_start  # deja en ns, meme horloge que la simulation
    if t_rel <= 0.0:
        return 0.0
    return _pressure_direct_scalar(t_rel, pulse.Imax, pulse.Tpul, pulse.T0)


def pressure_profile(t, pulses):
    """Pression exterieure imposee a l'instant t (ns), superposition des tirs.

    Chaque element de `pulses` doit exposer une methode pressure_at(t) ->
    GPa (interface commune, cf. LaserPulseParams) : n'importe quel autre
    type de tir (essai/validation, autre modele physique) peut s'y
    brancher sans toucher a cette fonction ni a boundary.py/solver.py.
    """
    if not pulses:
        return 0.0
    return sum(p.pressure_at(t) for p in pulses)
