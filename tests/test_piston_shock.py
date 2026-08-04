"""Validation physique : cas piston a pression constante imposee sur une
cible au repos. Le front de choc numerique (Us) et la vitesse particule
derriere le front (Up) doivent verifier la relation de Hugoniot d'entree
Us = C0 + S1*Up, avec P = rho0*Us*Up (materiau au repos en amont).

Materiau force en G=0 (pas de resistance JC) : cette relation Us-Up
"pure EOS" ignore la contribution de la resistance du materiau
(sigma_x = P + (2/3)*sigma_y a l'ecoulement plastique) ; elle ne vaut
donc que pour un materiau hydrodynamique. Le test valide le coeur
EOS/solveur, pas la plasticite (qui a son propre test, voir
test_plasticity.py).
"""
from dataclasses import dataclass, replace

import numpy as np

from control_file import MATERIAL
from lsp1d import postproc, solver


@dataclass
class _ConstantPressurePulse:
    """Plateau de pression constant (rampe courte puis plateau long) —
    utilise uniquement pour isoler la physique EOS/solveur dans ce test,
    independamment du modele de tir laser reel (lsp1d/loading.py). Suit
    la meme interface pressure_at(t) que LaserPulseParams."""

    t_start: float
    t_rise: float
    p_peak: float

    def pressure_at(self, t):
        t_rel = t - self.t_start
        if t_rel <= 0.0:
            return 0.0
        if t_rel < self.t_rise:
            return self.p_peak * t_rel / self.t_rise
        return self.p_peak


def analytic_up(P0, mat):
    """Resout rho0*(C0+S1*Up)*Up = P0 pour Up > 0 (racine physique)."""
    a = mat.S1 * mat.rho0
    b = mat.C0 * mat.rho0
    c = -P0
    return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)


def test_piston_shock_matches_hugoniot_relation():
    assert MATERIAL.S2 == 0.0, "test ecrit pour la forme EOS ordre 1 (solution analytique fermee)"

    from control_file import SimConfig

    material = replace(MATERIAL, G=0.0)  # isole la physique EOS/hydro pure
    P0 = 10.0  # GPa, plateau de pression constant
    cfg = SimConfig(
        length=500.0, n_cells=400, cfl_safety=0.4, q1=0.5, q2=2.0,
        t_max=50.0, output_dt=1.0,
        boundary_left="pressure_driven", boundary_right="non_reflecting",
    )
    # rampe de montee courte (mais pas un vrai echelon) : suffisant pour
    # amortir la reponse transitoire sans generer d'oscillations parasites
    # derriere le front (role de q1, voir viscosity.py).
    pulses = [_ConstantPressurePulse(t_start=0.0, t_rise=5.0, p_peak=P0)]

    mesh = solver.make_uniform_mesh(cfg.length, cfg.n_cells, material.rho0)
    state = solver.initial_state(mesh, material)
    history = solver.run(mesh, state, material, cfg, pulses)

    t_front, x_front = postproc.shock_front_positions(history, threshold_fraction=0.3)
    Us_num = postproc.shock_velocity(t_front, x_front, t_window=(10.0, 40.0))

    dx = cfg.length / cfg.n_cells
    # offset genereux : le front est etale sur quelques mailles par q2, et
    # le filtre anti-damier (solver._damp_checkerboard) lisse legerement
    # les gradients raides sur 1-2 mailles supplementaires. Il faut donc
    # echantillonner bien dans le plateau, pas juste derriere le front.
    Up_num = postproc.particle_velocity_behind_front(history, x_front, t_front, offset=15 * dx)
    window = (t_front >= 10.0) & (t_front <= 40.0)
    Up_num_mean = np.mean(Up_num[window])

    Up_analytic = analytic_up(P0, material)
    Us_analytic = material.C0 + material.S1 * Up_analytic

    assert abs(Us_num - Us_analytic) / Us_analytic < 0.05
    assert abs(Up_num_mean - Up_analytic) / Up_analytic < 0.10
