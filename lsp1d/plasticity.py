"""Plasticite Johnson-Cook, retour radial implicite (vectorise numpy).

Deformation uniaxiale (hypothese standard pour un choc plan 1D) : seule
la deformation axiale est non nulle, donc le deviateur des contraintes
n'a qu'une composante independante S_xx (S_yy=S_zz=-S_xx/2). Pour ce cas
reduit, la contrainte equivalente de von Mises vaut sigma_vm = 1.5*|S_xx|
(identite geometrique, pas une approximation).

Contrainte totale utilisee par solver.py : sigma_xx = -P + S_xx, donc le
terme "pression totale" qui pilote le bilan de quantite de mouvement est
Ptot = P + q - S_xx (voir solver.py).

Simplification assumee : la vitesse de deformation plastique injectee
dans le terme de vitesse JC est approchee par la vitesse de deformation
totale du pas (|dV/V|/dt), pas par un solve totalement couple sur le
terme logarithmique — la contribution de la deformation elastique a la
vitesse totale est negligeable des que le materiau plastifie franchement,
ce qui est le regime d'interet pour un choc. Le chauffage par dissipation
plastique n'est pas reinjecte dans l'energie interne (eos.temperature
reste caloriquement simple) : a un choc de quelques GPa l'echauffement
correspondant est une petite perturbation vis-a-vis de la compression,
mais c'est une simplification a lever si on pousse vers des chocs forts.
"""
import numpy as np


def _yield_stress(eps_p, rate_factor, thermal_factor, mat):
    return (mat.A + mat.B * np.maximum(eps_p, 0.0) ** mat.n) * rate_factor * thermal_factor


def radial_return(sigma_trial, eps_p, eps_p_rate, temperature, mat):
    """Retour radial implicite JC complet (terme thermique + vitesse).

    sigma_trial : S_xx elastique d'essai (increment elastique applique au
    S_xx du pas precedent, avant correction plastique).
    eps_p, eps_p_rate, temperature : etat au debut du pas (deformation
    plastique cumulee, vitesse de deformation estimee, temperature).

    Retourne (S_xx corrige, deformation plastique cumulee mise a jour).
    """
    G = mat.G
    if G <= 0.0:
        return np.zeros_like(sigma_trial), eps_p

    sigma_vm_trial = 1.5 * np.abs(sigma_trial)

    T_star = np.clip((temperature - mat.T_ref) / (mat.T_melt - mat.T_ref), 0.0, 1.0)
    rate_factor = 1.0 + mat.C * np.log(np.maximum(eps_p_rate, mat.eps_dot_ref) / mat.eps_dot_ref)
    thermal_factor = 1.0 - T_star**mat.m

    sigma_y0 = _yield_stress(eps_p, rate_factor, thermal_factor, mat)
    deps_p = np.maximum((sigma_vm_trial - sigma_y0) / (3.0 * G), 0.0)

    for _ in range(8):
        eps_p_trial = eps_p + deps_p
        sy = _yield_stress(eps_p_trial, rate_factor, thermal_factor, mat)
        g = sigma_vm_trial - 3.0 * G * deps_p - sy
        dsy = (
            mat.B * mat.n * np.maximum(eps_p_trial, 1e-12) ** (mat.n - 1.0)
            * rate_factor * thermal_factor
        )
        dg = -3.0 * G - dsy
        deps_p = np.maximum(deps_p - g / dg, 0.0)

    eps_p_new = eps_p + deps_p
    sigma_y_new = _yield_stress(eps_p_new, rate_factor, thermal_factor, mat)

    sigma_vm_new = np.minimum(sigma_vm_trial, sigma_y_new)
    S_new = np.sign(sigma_trial) * sigma_vm_new / 1.5
    return S_new, eps_p_new
