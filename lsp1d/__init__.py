"""lsp1d — code hydro 1D lagrangien pour la simulation de chocs laser.

Systeme d'unites (auto-coherent, pas de facteur de conversion cache) :
    longueur      um
    temps         ns
    masse volum.  g/cm3   (numeriquement = ng/um3)
    vitesse       km/s    (numeriquement = um/ns)
    pression      GPa
Voir eos.py pour le detail de cette coherence numerique.
"""
