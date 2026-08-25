"""
Fige le comportement des caches TELS QU'ILS ETAIENT dans les scripts AC.

    python tools/golden_caches.py --revision HEAD

Produit `tests/golden/caches_originaux.json`, oracle durable de `_cache/`.
Meme raison que pour les lois : une fois les definitions retirees des scripts
AC, plus rien ne dirait ce que le code faisait avant l'extraction.

Ce que l'on enregistre pour chaque fonction :
  - pour une ecriture : le CONTENU JSON produit ;
  - pour une lecture  : la valeur rendue, y compris les cas ou elle vaut None
    -- ces refus (n0 different, coupe differente, cache incomplet) sont la
    vraie logique de ces fonctions, et c'est ce qui doit etre preserve.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np                                        # noqa: E402
from extraction_temoin import AC_FLEXION, fonction_originale  # noqa: E402

XT = [[0.1, -0.2], [1.5, 0.3], [-2.0, 1.1]]
YT = [[0.5], [-0.1], [0.9]]
AG = [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]
SOL = [{"_u": [0.1, -0.2], "g": 0.5, "dg_fc": 0.01, "dg_fy": 0.02},
       {"_u": [1.5, 0.3], "g": -0.1, "dg_fc": 0.03, "dg_fy": 0.04},
       {"_u": [-2.0, 1.1], "g": 0.9}]          # 3e point sans gradients : defaut 0.0
Z = np.array([[1.0, 2.0], [3.0, 4.0]])
SD = (0, 1, {2: 0.0})


def _lire(chemin):
    with open(chemin, encoding="utf-8") as fh:
        return json.load(fh)


def collecter(revision):
    out = {"revision": revision, "cas": {}}
    tmp = tempfile.mkdtemp(prefix="golden_caches_")
    try:
        F = lambda nom, libres=None: fonction_originale(  # noqa: E731
            AC_FLEXION, nom, libres, revision=revision)

        # ---- DOE ---------------------------------------------------------
        doe = os.path.join(tmp, "doe_cache.json")
        libres = {"_DOE_CACHE_FILE": doe, "n0": 3, "params_names": ["fc", "fy"],
                  "n_var": 2, "modelname": "test_pure_flexion",
                  "config_is_identical": True, "np": np, "json": json, "os": os}

        out["cas"]["doe_cache_sig"] = F("_doe_cache_sig", libres)()

        F("_save_doe_cache", libres)(XT, YT, AG)
        out["cas"]["save_doe_cache"] = _lire(doe)
        r = F("_load_doe_cache", libres)()
        out["cas"]["load_doe_cache"] = [np.asarray(x).tolist() for x in r]

        # refus : n0 different
        libres_n0 = dict(libres, n0=5)
        out["cas"]["load_doe_cache_n0_different"] = F("_load_doe_cache", libres_n0)()
        # refus : configuration declaree differente
        libres_off = dict(libres, config_is_identical=False)
        out["cas"]["load_doe_cache_config_differente"] = F("_load_doe_cache", libres_off)()

        F("_save_doe_cache_incremental", libres)(SOL, 2)
        out["cas"]["save_doe_cache_incremental"] = _lire(doe)
        # refus : cache incomplet
        out["cas"]["load_doe_cache_incomplet"] = F("_load_doe_cache", libres)()
        # absence de fichier
        os.remove(doe)
        out["cas"]["load_doe_cache_absent"] = F("_load_doe_cache", libres)()

        # ---- HF ----------------------------------------------------------
        hf = os.path.join(tmp, "hf_cache.json")
        libres_hf = {"config_is_identical": True, "np": np, "json": json, "os": os}

        F("_save_hf_cache", libres_hf)(Z, 2, hf, SD)
        out["cas"]["save_hf_cache"] = _lire(hf)
        out["cas"]["load_hf_cache"] = F("_load_hf_cache", libres_hf)(2, hf, SD).tolist()
        out["cas"]["load_hf_cache_coupe_differente"] = \
            F("_load_hf_cache", libres_hf)(2, hf, (0, 2, {}))

        F("_save_hf_cache_partial", libres_hf)([1.0, None, 3.0, None], 4, hf, SD)
        out["cas"]["save_hf_cache_partial"] = _lire(hf + ".partial")
        out["cas"]["load_hf_cache_partial"] = F("_load_hf_cache_partial", libres_hf)(hf, SD, 4)
        out["cas"]["load_hf_cache_partial_n_different"] = \
            F("_load_hf_cache_partial", libres_hf)(hf, SD, 9)

        plein = os.path.join(tmp, "hf_full.json")
        libres_plein = dict(libres_hf, _HF_FULL_CACHE_FILE=plein, n_var=2, n_grid_hf=2)
        F("_save_hf_grid_full", libres_plein)(Z)
        out["cas"]["save_hf_grid_full"] = _lire(plein)
        out["cas"]["load_hf_grid_full"] = F("_load_hf_grid_full", libres_plein)().tolist()
        out["cas"]["load_hf_grid_full_dimensions_differentes"] = \
            F("_load_hf_grid_full", dict(libres_plein, n_grid_hf=7))()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--revision", default="HEAD")
    args = ap.parse_args()

    out = collecter(args.revision)
    p = os.path.join(REPO, "tests", "golden", "caches_originaux.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("ecrit %s  (%d cas, revision %s)" % (p, len(out["cas"]), args.revision))
    for nom in sorted(out["cas"]):
        v = out["cas"][nom]
        print("   %-42s %s" % (nom, "None" if v is None else type(v).__name__))


if __name__ == "__main__":
    main()
