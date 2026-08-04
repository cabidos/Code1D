"""Post-traitement des contraintes residuelles apres decharge — phase B.

Necessite l'historique du deviateur elasto-plastique (plasticity.py). Le
solveur phase A est purement hydrodynamique : il n'y a pas d'historique
deviatorique a post-traiter, donc pas de resultat (meme nul) qui aurait
un sens physique.
"""


def compute_residual(history):
    raise NotImplementedError(
        "residual_stress.compute_residual necessite la plasticite phase B "
        "(historique du deviateur) ; le solveur phase A est purement "
        "hydrodynamique."
    )
