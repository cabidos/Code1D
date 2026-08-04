"""Fichier de controle — parametres materiau/simulation, separe du code
(style EVEREST) : c'est ce fichier qu'on modifie pour changer de cas, pas
les modules de lsp1d/. Toutes les entrees sont des variables simples
listees ci-dessous ; l'assemblage en objets (MATERIAL, SIM_CONFIG,
PULSES) utilises par le solveur se fait en bas de fichier, a ne pas
toucher pour un usage courant.

Systeme d'unites : um, ns, g/cm3, km/s (= um/ns), GPa (voir lsp1d/eos.py
pour le detail de la coherence numerique).
"""
from dataclasses import dataclass

from lsp1d.loading import LaserPulseParams

# =====================================================================
# MATERIAU — EOS Mie-Gruneisen, forme choc Us-Up
# Valeurs indicatives (litterature type LASL/Steinberg pour un aluminium
# proche du 6061-T6) — a remplacer par des valeurs calibrees sur des
# essais reels avant toute exploitation quantitative.
# =====================================================================
material_name = "Al (carte Radioss)"
rho0 = 2.698        # g/cm3, densite de reference (RHO_I = 2698 kg/m3)
C0 = 5.328             # km/s, EOS Us-Up, ordonnee a l'origine (son "bulk") — /EOS/GRUNEISEN/2 Radioss (5328 m/s)
S1 = 1.338               # EOS Us-Up, pente lineaire — idem (Radioss)
S2 = 0.0                  # EOS Us-Up, terme quadratique (0 => forme ordre 1)
Gamma0 = 2.0               # coefficient de Gruneisen a rho0 — idem (Radioss)
# Note : la carte Radioss inclut aussi a=0.48 (correction de Gruneisen avec
# le volume, Gamma = Gamma0 - a*mu/(1+mu)) — non implemente ici (Gamma0
# suppose constant).

# =====================================================================
# MATERIAU — plasticite Johnson-Cook (deformation uniaxiale)
# sigma_y = (JC_A + JC_B*eps_p^JC_n) * (1+JC_C*ln(eps_p_rate/eps_dot_ref))
#           * (1 - T_star^JC_m)
# Carte Radioss (Al) : E=69 GPa, nu=0.33 -> G=E/(2*(1+nu))=25.94 GPa.
# a=324 MPa, b=113 MPa, n=0.42, c=0.013, EPS_DOT_0=597 /s.
# Pas de terme thermique dans la carte Radioss fournie (JC_m/T_melt
# inchanges, hors perimetre de cette mise a jour).
# =====================================================================
G = 25.94                     # GPa, module de cisaillement (E=69GPa, nu=0.33)
JC_A = 0.324                   # GPa, limite elastique (a)
JC_B = 0.113                     # GPa, coefficient d'ecrouissage (b)
JC_n = 0.42                        # exposant d'ecrouissage (n)
JC_C = 0.013                         # coefficient de sensibilite a la vitesse (c)
JC_m = 1.0                             # exposant d'adoucissement thermique
eps_dot_ref = 5.97e-7                     # 1/ns (= 597 /s physique, EPS_DOT_0 Radioss)
T_ref = 293.0                              # K, temperature de reference
T_melt = 775.0                               # K, temperature de fusion
cv = 9.0e-4                                    # (km/s)^2/K, capacite calorifique (~0.9 J/g/K)

# =====================================================================
# CHARGEMENT LASER — modele semi-empirique "regime direct"
# (Fabbro/Berthe, choc laser sans confinement, voir lsp1d/loading.py).
# =====================================================================
pulse_t_start = 0.0   # ns, instant de declenchement du tir
Imax = 87             # GW/cm2, intensite laser crete (-> Pmax ~5.1 GPa impose, formule _pmax_direct ; ~4.6-4.8 GPa reellement simule, cf. viscosite artificielle/maillage)
Tpul = 15             # ns, duree d'impulsion laser
T0 = 0.7                   # ns, temps de montee de la pression

# =====================================================================
# SIMULATION — maillage et parametres numeriques
# =====================================================================
length = 200.0          # um, epaisseur de cible
n_cells = 960              # nombre de mailles
cfl_safety = 0.5              # coefficient de securite CFL (<1)
q1 = 0.5                        # viscosite artificielle, terme lineaire
q2 = 2.0                          # viscosite artificielle, terme quadratique
filter_alpha = 0.15                 # amortissement du mode noeud-a-noeud (0=off, 0.5=max stable)
t_max = 80.0                          # ns, duree simulee
output_dt = 0.5                       # ns, pas d'echantillonnage des sorties
boundary_left = "pressure_driven"         # CL gauche : "pressure_driven"
boundary_right = "free_surface"             # CL droite : "free_surface" / "non_reflecting"

# =====================================================================
# Assemblage (ne pas modifier pour un usage courant)
# =====================================================================


@dataclass
class MaterialParams:
    name: str
    rho0: float
    C0: float
    S1: float
    S2: float = 0.0
    Gamma0: float = 2.0
    G: float = 0.0
    A: float = 0.0
    B: float = 0.0
    n: float = 1.0
    C: float = 0.0
    m: float = 1.0
    eps_dot_ref: float = 1.0e-9
    T_ref: float = 293.0
    T_melt: float = 775.0
    cv: float = 9.0e-4


@dataclass
class SimConfig:
    length: float
    n_cells: int = 200
    cfl_safety: float = 0.5
    q1: float = 0.5
    q2: float = 2.0
    filter_alpha: float = 0.15
    t_max: float = 300.0
    output_dt: float = 2.0
    boundary_left: str = "pressure_driven"
    boundary_right: str = "free_surface"


MATERIAL = MaterialParams(
    name=material_name, rho0=rho0, C0=C0, S1=S1, S2=S2, Gamma0=Gamma0,
    G=G, A=JC_A, B=JC_B, n=JC_n, C=JC_C, m=JC_m,
    eps_dot_ref=eps_dot_ref, T_ref=T_ref, T_melt=T_melt, cv=cv,
)

SIM_CONFIG = SimConfig(
    length=length, n_cells=n_cells, cfl_safety=cfl_safety, q1=q1, q2=q2,
    filter_alpha=filter_alpha, t_max=t_max, output_dt=output_dt,
    boundary_left=boundary_left, boundary_right=boundary_right,
)

# Plusieurs tirs : ajouter d'autres LaserPulseParams(...) a cette liste
# (superposition automatique, voir lsp1d/loading.py).
PULSES = [
    LaserPulseParams(t_start=pulse_t_start, Imax=Imax, Tpul=Tpul, T0=T0),
]
