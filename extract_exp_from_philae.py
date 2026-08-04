"""Extrait les courbes VISAR experimentales manquantes depuis
tirs/Philae1.h5 (Shot_N/DataAnalysis/Diagnostic_VSLN) et les ecrit au
meme format que les fichiers deja fournis dans tirs/Resultat exp/
(colonnes t [s], v [km/s], sans en-tete).
"""
import os

import h5py
import numpy as np

EXP_DIR = "tirs/Resultat exp"
NEEDED = [0, 4, 5, 6, 7, 13, 15, 16, 17, 18, 19, 20,
          32, 33, 34, 35, 38, 39, 40, 41, 46, 47]

os.makedirs(EXP_DIR, exist_ok=True)

with h5py.File("tirs/Philae1.h5", "r") as f:
    for num in NEEDED:
        out_path = f"{EXP_DIR}/VSL{num}.txt"
        if os.path.exists(out_path):
            continue
        key = f"Shot_{num}/DataAnalysis/Diagnostic_VSL{num}"
        if key not in f:
            print(f"-- Shot_{num} : pas de donnee experimentale dans Philae1.h5 (skip)")
            continue
        ds = f[key][:]
        data = np.array([[float(a), float(b)] for a, b in ds])
        np.savetxt(out_path, data, delimiter=",")
        print(f"-> {out_path} ({len(data)} lignes)")
