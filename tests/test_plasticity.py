"""Validation physique du retour radial JC : en dessous de la limite
elastique de Hugoniot (HEL), la reponse doit rester purement elastique
(eps_p ~ 0) ; nettement au-dessus, le materiau doit plastifier
(eps_p > 0) derriere le front.
"""
from dataclasses import dataclass

from control_file import MATERIAL, SimConfig
from lsp1d import solver


@dataclass
class _ConstantPressurePulse:
    """Plateau de pression constant, isole du modele de tir laser reel
    (lsp1d/loading.py) pour ce test physique. Interface pressure_at(t)
    identique a LaserPulseParams."""

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


def _run_constant_pressure(p_peak, t_max=60.0):
    cfg = SimConfig(
        length=500.0, n_cells=200, cfl_safety=0.4, q1=0.5, q2=2.0,
        t_max=t_max, output_dt=5.0,
        boundary_left="pressure_driven", boundary_right="non_reflecting",
    )
    pulses = [_ConstantPressurePulse(t_start=0.0, t_rise=5.0, p_peak=p_peak)]

    mesh = solver.make_uniform_mesh(cfg.length, cfg.n_cells, MATERIAL.rho0)
    state = solver.initial_state(mesh, MATERIAL)
    return solver.run(mesh, state, MATERIAL, cfg, pulses)


def test_low_pressure_stays_elastic():
    # HEL indicative pour ce materiau ~0.7 GPa (P_HEL + (2/3)*sigma_y) ;
    # 0.1 GPa est nettement en dessous.
    history = _run_constant_pressure(p_peak=0.1)
    max_eps_p = max(snap["eps_p"].max() for snap in history)
    assert max_eps_p < 1e-4


def test_strong_shock_plastifies():
    history = _run_constant_pressure(p_peak=5.0)
    max_eps_p = max(snap["eps_p"].max() for snap in history)
    assert max_eps_p > 1e-3
