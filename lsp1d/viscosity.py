"""Viscosite artificielle, forme Von Neumann-Richtmyer / Wilkins.

Formule classique en q = rho*(q1_dx*C)^2*(du/dx)^2 + rho*q1*dx*c*|du/dx| :
le facteur dx s'annule analytiquement une fois exprime avec du "brut"
(difference de vitesse entre noeuds, pas la deformation du/dx) — q ne
doit donc PAS etre multiplie par dx explicitement ici, sous peine de
scaler artificiellement avec la resolution/l'unite de longueur choisie.

- terme quadratique q2*du^2 : uniquement en compression (du<0) — capture
  de choc, etale un choc fort sur quelques mailles pour que le schema
  explicite le resolve sans discontinuite vraie. Ne doit pas agir en
  detente sous peine de fausser la forme de l'onde de relaxation.
- terme lineaire q1*c*|du| : actif dans les deux regimes. En plus de son
  role habituel (amortir les oscillations derriere un choc faible), c'est
  le seul terme dissipatif disponible en detente ; sans lui, un noeud de
  surface libre qui accelere brutalement a l'arrivee du choc (bilan de
  quantite de mouvement sans amortissement) part en oscillation numerique
  non physique (aucune viscosite ne s'applique en detente sinon, faute de
  deviateur/plasticite en phase A pour la dissiper naturellement).
"""
import numpy as np


def artificial_pressure(rho, du, c, q1, q2):
    q_linear = rho * q1 * c * np.abs(du)

    q_quad = np.zeros_like(rho)
    compression = du < 0.0
    q_quad[compression] = rho[compression] * q2 * du[compression] ** 2

    return q_linear + q_quad
