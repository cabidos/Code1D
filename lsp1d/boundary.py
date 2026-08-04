"""Conditions aux limites via l'impedance acoustique locale Z = rho*c.

Chaque condition retourne le terme "pression fantome" utilise dans le
bilan de quantite de mouvement nodal de solver.py :
    node_force[0]  = P_ghost_left  - (P[0]  + q[0])
    node_force[-1] = (P[-1] + q[-1]) - P_ghost_right

- pressure_driven : P_ghost = P_ext(t), depuis loading.py (attaque laser).
- free_surface    : P_ghost = 0 (surface libre, condition VISAR standard).
- non_reflecting  : forme "amortisseur" de l'invariant de Riemann sortant,
  equivalente a raccorder la frontiere a un prolongement semi-infini du
  meme materiau au repos (pas de reflexion parasite). Le signe differe
  selon le cote (gauche/droite) car la normale sortante change de sens.
"""
from . import loading


def boundary_force(kind, side, state, mat, t, sound_speed_edge, rho_edge, pulses=None):
    if kind == "pressure_driven":
        return loading.pressure_profile(t, pulses)

    if kind == "free_surface":
        return 0.0

    if kind == "non_reflecting":
        Z = rho_edge * sound_speed_edge
        if side == "left":
            return -Z * state.u[0]
        return Z * state.u[-1]

    raise ValueError(f"unknown boundary kind: {kind!r}")
