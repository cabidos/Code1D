"""Equation d'etat Mie-Gruneisen, forme choc Us-Up (ordre 1 ou 2).

Us = C0 + S1*Up (+ S2*Up^2 si S2 != 0), rho0/C0/S1/S2/Gamma0 dans mat.

Coherence d'unites : avec rho en g/cm3, longueurs en um et temps en ns
(vitesses en km/s = um/ns), on a numeriquement 1 GPa*cm3/g = 1 (km/s)^2.
L'energie interne specifique e est donc exprimee directement en
(km/s)^2, sans facteur de conversion, et P = Gamma0*rho0*e tombe
directement en GPa. Reference : e=0 et P=0 a l'etat initial (rho=rho0).
"""
import numpy as np


def _mu(rho, mat):
    return rho / mat.rho0 - 1.0


def _hugoniot_pressure(mu, mat):
    mu = np.asarray(mu, dtype=float)
    P_H = np.empty_like(mu)
    compress = mu >= 0.0

    mc = mu[compress]
    denom = 1.0 - (mat.S1 - 1.0) * mc - mat.S2 * mc**2 / (1.0 + mc)
    P_H[compress] = mat.rho0 * mat.C0**2 * mc * (1.0 + mc) / denom**2

    # detente : extrapolation lineaire, evite la singularite du denominateur
    me = mu[~compress]
    P_H[~compress] = mat.rho0 * mat.C0**2 * me

    return P_H


def _hugoniot_energy(mu, mat):
    P_H = _hugoniot_pressure(mu, mat)
    return 0.5 * P_H / mat.rho0 * mu / (1.0 + mu)


def energy_hugoniot(rho, mat):
    """Energie interne specifique sur la courbe de Hugoniot, a rho donne."""
    return _hugoniot_energy(_mu(rho, mat), mat)


def pressure(rho, e, mat):
    """P(rho, e) generique (forme Mie-Gruneisen)."""
    mu = _mu(rho, mat)
    P_H = _hugoniot_pressure(mu, mat)
    e_H = _hugoniot_energy(mu, mat)
    return P_H + mat.Gamma0 * mat.rho0 * (e - e_H)


def sound_speed(rho, e, mat, floor_fraction=1e-3):
    """Vitesse du son locale pour le controle CFL.

    Identite thermodynamique c^2 = dP/drho|e + (P/rho^2)*dP/de|rho. Les
    derivees de la branche Hugoniot (dP_H/dmu, de_H/dmu) sont prises par
    difference finie centree : la forme ordre 2 rend la derivee analytique
    lourde, et seule une estimation de stabilite est necessaire ici (pas
    une vitesse caracteristique exacte).
    """
    rho = np.asarray(rho, dtype=float)
    e = np.asarray(e, dtype=float)
    mu = _mu(rho, mat)
    h = 1e-6 * (1.0 + np.abs(mu))

    dPH_dmu = (_hugoniot_pressure(mu + h, mat) - _hugoniot_pressure(mu - h, mat)) / (2 * h)
    deH_dmu = (_hugoniot_energy(mu + h, mat) - _hugoniot_energy(mu - h, mat)) / (2 * h)

    dP_drho_e = (dPH_dmu - mat.Gamma0 * mat.rho0 * deH_dmu) / mat.rho0
    dP_de_rho = mat.Gamma0 * mat.rho0

    P = pressure(rho, e, mat)
    c2 = dP_drho_e + (P / rho**2) * dP_de_rho

    floor = (floor_fraction * mat.C0) ** 2
    return np.sqrt(np.maximum(c2, floor))


def longitudinal_sound_speed(rho, e, mat):
    """Vitesse d'onde elastique longitudinale (bulk + cisaillement) :
    c_L^2 = c_bulk^2 + (4/3)*G/rho. Se reduit a sound_speed() si G=0.

    A utiliser pour le controle CFL et l'impedance des CL des que la
    plasticite (module de cisaillement mat.G) est active : l'onde
    elastique precurseur est plus rapide que le son "bulk" seul, et le
    schema explicite doit etre stable vis-a-vis de CETTE vitesse-la.
    """
    c_bulk = sound_speed(rho, e, mat)
    return np.sqrt(c_bulk**2 + (4.0 / 3.0) * mat.G / rho)


def temperature(e, mat):
    """Temperature approchee par capacite calorifique constante :
    T = T_ref + e/cv (e mesure a partir de l'etat initial, e=0 -> T_ref).

    Approximation caloriquement simple (ignore le couplage EOS complet
    entre T et le terme de Gruneisen) : suffisante pour alimenter le
    terme d'adoucissement thermique de Johnson-Cook sans faire de la
    thermique une source de precision manquante ailleurs dans le modele.
    """
    return mat.T_ref + e / mat.cv
