"""Point d'entree : control_file -> Mesh/State -> solver.run -> postproc.

Plusieurs facons de lancer :
- IDE (Spyder, VS Code, ...) : ouvrir ce fichier et l'executer directement
  (F5 dans Spyder). Fonctionne quel que soit le repertoire de travail.
- Ligne de commande, depuis Code1D : python -m lsp1d.main

Les imports et les chemins de sortie sont resolus par rapport a ce
fichier (pas au repertoire de travail courant), donc les deux modes de
lancement sont equivalents.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import control_file as cf
from lsp1d import postproc, solver


def main():
    mesh = solver.make_uniform_mesh(cf.SIM_CONFIG.length, cf.SIM_CONFIG.n_cells, cf.MATERIAL.rho0)
    state = solver.initial_state(mesh, cf.MATERIAL)

    dx = cf.SIM_CONFIG.length / cf.SIM_CONFIG.n_cells
    dt0 = solver.estimate_dt(mesh, cf.MATERIAL, cf.SIM_CONFIG)
    print(f"Maillage : {cf.SIM_CONFIG.n_cells} mailles, dx = {dx:.4f} um")
    print(f"Pas de temps CFL (initial, indicatif, non reglable directement) : dt ~= {dt0:.4f} ns")
    print(f"Pas d'echantillonnage des sorties (control_file.output_dt) : {cf.SIM_CONFIG.output_dt} ns")

    history = solver.run(mesh, state, cf.MATERIAL, cf.SIM_CONFIG, cf.PULSES)

    out_prefix = os.path.join(_ROOT, "run_output")
    postproc.save_history(history, out_prefix + ".npz")
    postproc.save_free_surface_velocity(history, out_prefix + "_free_surface_velocity.csv")
    postproc.save_free_surface_velocity_excel(history, out_prefix + "_free_surface_velocity.xlsx")
    postproc.save_free_surface_velocity_txt(history, os.path.join(_ROOT, "data.txt"))

    # graphe des profils de pression : part de t=0 (montee complete de
    # l'onde depuis la source incluse) et s'arrete avant que le front
    # n'atteigne la face arriere, pour ne pas melanger l'onde incidente
    # avec la reflexion/reverberation sur le meme graphe.
    profile_t_end = postproc.breakout_time(history, cf.SIM_CONFIG.length)
    postproc.plot_results(
        history, out_prefix, title=cf.MATERIAL.name,
        profile_t_start=0.0, profile_t_end=profile_t_end,
    )

    try:
        Us = postproc.auto_shock_velocity(history, cf.SIM_CONFIG.length)
        print(f"Vitesse de choc (regime etabli, hors transitoire/bord) : Us = {Us:.3f} km/s")
    except ValueError:
        print("Vitesse de choc : pas assez de points en regime etabli pour l'estimer.")

    _, u_fs = postproc.free_surface_velocity(history)
    print(f"Vitesse de face arriere (pic) : {u_fs.max():.3f} km/s")
    print(f"-> {out_prefix}_free_surface_velocity.csv")


if __name__ == "__main__":
    main()
