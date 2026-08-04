"""Solveur lagrangien 1D explicite, grille decalee (schema type Wilkins).

Grille : N mailles / N+1 noeuds. Les noeuds portent position x et
vitesse u (leapfrog : u au demi-pas, x au pas entier). Les mailles
portent rho, e, P, q et l'etat elasto-plastique (S = deviateur axial
S_xx, eps_p = deformation plastique equivalente cumulee). La masse de
maille est figee a t=0 (conservation de masse automatique en lagrangien).

Contrainte totale utilisee dans le bilan de quantite de mouvement :
sigma_xx = -P + S_xx (deformation uniaxiale, cf. plasticity.py), donc le
terme "pression totale" qui pilote le bilan est Ptot = P + q - S.
"""
from dataclasses import dataclass

import numpy as np

from . import boundary, eos, plasticity, viscosity


@dataclass
class Mesh:
    x0: np.ndarray   # (N+1,) positions initiales des noeuds, um
    dx0: np.ndarray  # (N,) largeurs de maille initiales, um
    m: np.ndarray    # (N,) masse de maille par unite d'aire


def make_uniform_mesh(length, n_cells, rho0):
    x0 = np.linspace(0.0, length, n_cells + 1)
    dx0 = np.diff(x0)
    m = rho0 * dx0
    return Mesh(x0=x0, dx0=dx0, m=m)


@dataclass
class State:
    x: np.ndarray
    u: np.ndarray
    rho: np.ndarray
    e: np.ndarray
    P: np.ndarray
    q: np.ndarray
    S: np.ndarray
    eps_p: np.ndarray
    t: float


def initial_state(mesh, mat):
    n = len(mesh.m)
    x = mesh.x0.copy()
    u = np.zeros(n + 1)
    rho = np.full(n, mat.rho0)
    e = np.zeros(n)
    P = eos.pressure(rho, e, mat)
    q = np.zeros(n)
    S = np.zeros(n)
    eps_p = np.zeros(n)
    return State(x=x, u=u, rho=rho, e=e, P=P, q=q, S=S, eps_p=eps_p, t=0.0)


def _cell_widths(x):
    return x[1:] - x[:-1]


def _node_masses(mesh):
    m = mesh.m
    node_mass = np.empty(len(m) + 1)
    node_mass[1:-1] = 0.5 * (m[:-1] + m[1:])
    node_mass[0] = 0.5 * m[0]
    node_mass[-1] = 0.5 * m[-1]
    return node_mass


def _cfl_dt(dx, c, du, cfg):
    c_eff = c + 2.0 * cfg.q2 * np.abs(du)
    return cfg.cfl_safety * np.min(dx / c_eff)


def estimate_dt(mesh, mat, cfg):
    """Pas de temps CFL indicatif (materiau au repos, avant arrivee du
    choc) : le pas interne reel diminue une fois le materiau comprime
    (vitesse du son locale plus elevee). N'est pas un parametre reglable
    directement (cf. cfg.cfl_safety, cfg.n_cells) : c'est une consequence
    de la CFL, pas une entree du fichier de controle."""
    dx = mesh.dx0
    rho = np.full(len(dx), mat.rho0)
    e = np.zeros(len(dx))
    c = eos.longitudinal_sound_speed(rho, e, mat)
    du = np.zeros(len(dx))
    return _cfl_dt(dx, c, du, cfg)


def _damp_checkerboard(u, alpha):
    """Amortit le mode noeud-a-noeud (Nyquist, periode 2 mailles).

    La viscosite artificielle standard (basee sur |du|) ne genere aucune
    force nette sur ce mode : dans un damier u_i alternant +A/-A, |du| est
    quasi constant d'une maille a l'autre, donc q aussi -> Ptot[i-1]-Ptot[i]
    (le terme moteur du bilan de quantite de mouvement) n'en est presque
    pas affecte, meme avec q1 tres grand. Ce filtre laplacien 3 points est
    au contraire maximal sur ce mode precis (signe alterne) et quasi nul
    sur les ondes physiques bien resolues (plusieurs mailles/longueur
    d'onde) : c'est le remede standard, pas la viscosite artificielle.
    """
    if alpha <= 0.0:
        return u
    damped = u.copy()
    damped[1:-1] += alpha * (u[:-2] - 2.0 * u[1:-1] + u[2:])
    return damped


def _snapshot(state):
    return {
        "t": state.t,
        "x": state.x.copy(),
        "u": state.u.copy(),
        "rho": state.rho.copy(),
        "e": state.e.copy(),
        "P": state.P.copy(),
        "q": state.q.copy(),
        "S": state.S.copy(),
        "eps_p": state.eps_p.copy(),
    }


def run(mesh, state, mat, cfg, pulses=None):
    node_mass = _node_masses(mesh)
    history = [_snapshot(state)]
    next_output = cfg.output_dt

    while state.t < cfg.t_max:
        dx = _cell_widths(state.x)
        du = state.u[1:] - state.u[:-1]
        c = eos.longitudinal_sound_speed(state.rho, state.e, mat)
        q = viscosity.artificial_pressure(state.rho, du, c, cfg.q1, cfg.q2)

        dt = _cfl_dt(dx, c, du, cfg)
        dt = min(dt, cfg.t_max - state.t)

        Ptot = state.P + q - state.S
        P_left = boundary.boundary_force(
            cfg.boundary_left, "left", state, mat, state.t, c[0], state.rho[0], pulses
        )
        P_right = boundary.boundary_force(
            cfg.boundary_right, "right", state, mat, state.t, c[-1], state.rho[-1], pulses
        )

        node_force = np.empty_like(state.u)
        node_force[1:-1] = Ptot[:-1] - Ptot[1:]
        node_force[0] = P_left - Ptot[0]
        node_force[-1] = Ptot[-1] - P_right

        u_new = state.u + dt * node_force / node_mass
        u_new = _damp_checkerboard(u_new, cfg.filter_alpha)
        x_new = state.x + dt * u_new

        V_old = dx
        V_new = _cell_widths(x_new)
        dV = V_new - V_old
        rho_new = mesh.m / V_new

        # predicteur-correcteur (Wilkins) : P depend implicitement de e via
        # l'EOS, donc on moyenne P sur le pas pour conserver l'energie a
        # l'ordre 2 plutot qu'un simple Euler explicite du travail P*dV.
        # Le travail du deviateur n'est pas reinjecte ici (cf. plasticity.py).
        e_pred = state.e - Ptot * dV / mesh.m
        P_pred = eos.pressure(rho_new, e_pred, mat)
        e_new = state.e - 0.5 * (Ptot + (P_pred + q)) * dV / mesh.m
        P_new = eos.pressure(rho_new, e_new, mat)

        # deformation uniaxiale : deformation volumique = deformation axiale.
        depsilon_xx = dV / V_old
        S_trial = state.S + (4.0 / 3.0) * mat.G * depsilon_xx
        temperature = eos.temperature(state.e, mat)
        eps_p_rate = np.abs(depsilon_xx) / dt
        S_new, eps_p_new = plasticity.radial_return(
            S_trial, state.eps_p, eps_p_rate, temperature, mat
        )

        state = State(
            x=x_new, u=u_new, rho=rho_new, e=e_new, P=P_new, q=q,
            S=S_new, eps_p=eps_p_new, t=state.t + dt,
        )

        if state.t >= next_output or state.t >= cfg.t_max:
            history.append(_snapshot(state))
            next_output += cfg.output_dt

    return history
