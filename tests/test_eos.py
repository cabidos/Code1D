from dataclasses import replace

import numpy as np

from control_file import MATERIAL
from lsp1d import eos


def test_pressure_zero_at_reference_state():
    P0 = eos.pressure(np.array([MATERIAL.rho0]), np.array([0.0]), MATERIAL)
    assert abs(P0[0]) < 1e-8


def test_pressure_monotonic_in_compression():
    rho = MATERIAL.rho0 * np.linspace(1.0, 1.3, 20)
    e = eos.energy_hugoniot(rho, MATERIAL)
    P = eos.pressure(rho, e, MATERIAL)
    assert np.all(np.diff(P) > 0)


def test_order2_correction_grows_with_compression():
    mat2 = replace(MATERIAL, S2=0.8)

    rho_small = np.array([MATERIAL.rho0 * 1.01])
    rho_large = np.array([MATERIAL.rho0 * 1.3])
    e_small = eos.energy_hugoniot(rho_small, MATERIAL)
    e_large = eos.energy_hugoniot(rho_large, MATERIAL)

    diff_small = abs(
        eos.pressure(rho_small, e_small, mat2)[0] - eos.pressure(rho_small, e_small, MATERIAL)[0]
    )
    diff_large = abs(
        eos.pressure(rho_large, e_large, mat2)[0] - eos.pressure(rho_large, e_large, MATERIAL)[0]
    )
    assert diff_large > diff_small


def test_sound_speed_positive_and_near_c0_at_rest():
    c = eos.sound_speed(np.array([MATERIAL.rho0]), np.array([0.0]), MATERIAL)
    assert c[0] > 0
    assert abs(c[0] - MATERIAL.C0) / MATERIAL.C0 < 0.05
