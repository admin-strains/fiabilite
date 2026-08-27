"""
Cache de la grille haute fidelite (HF).

Extrait de `AC3_pure_flexion.py` / `AC3_moulinblanc.py`, ou ces fonctions
etaient definies dans `if __name__ == '__main__':` -- donc ni importables ni
testables -- et recopiees a l'identique dans les deux scripts.

PHASE 3 du plan de nettoyage. Les corps sont repris VERBATIM ; seules les
variables libres de `main` deviennent des parametres explicites. C'etait tout
l'enjeu : une fonction qui se ferme sur l'espace de noms de `main` ne peut
etre appelee par aucun test.

Les variables libres devenues parametres : `_HF_FULL_CACHE_FILE` ->
`fichier`, `n_var`, `n_grid_hf`, et le drapeau `config_is_identical`.

Aucune dependance lourde : `json`, `os`, `numpy`. Ni OpenTURNS ni Digital
Structure.
"""

import json
import os

import numpy as np


#: CE QUE LA SIGNATURE PROTEGE -- 26/08/2026
#:
#: Ces caches ne validaient que la COUPE (quels axes) et, pour le partiel, le
#: nombre de points. Ni le solveur, ni le solveur lineaire, ni le maillage, ni
#: les BORNES de la grille, ni meme `n_grid_hf` -- son parametre etait recu et
#: jamais lu.
#:
#: Le cas ne s'est pas seulement produit : il etait sur le disque. En bornant
#: le domaine du Moulin Blanc de +/- 7,5 a +/- 6, le fichier
#: `hf_grid_cache.json.partial` a survecu avec
#:
#:     Z_flat = [-0.8555958883063973, null, null, null]
#:     n_total = 4,  slice_def = [0, 1, {}]
#:
#: soit la valeur calculee en u = [-7,5 ; -7,5] sous CuDss. Le run suivant, a
#: +/- 6 et sous MUMPS, l'aurait relue comme la valeur en u = [-6 ; -6] --
#: fy = 54 MPa au lieu de 8,9. Les deux controles existants passaient.
#:
#: `signature` est comparee a la relecture ; un cache qui n'en porte pas est
#: refuse. Le cout est un recalcul, le prix de l'alternative est une figure
#: fausse que rien ne signale.
def _signature_compatible(d, signature, etiquette):
    """Vrai si le cache peut etre relu. Bavard sur le refus."""
    if signature is None:
        return True
    sig = d.get("signature")
    if sig is None:
        print("[%s] cache sans signature (anterieur au controle) -> recalcul, "
              "par prudence" % etiquette, flush=True)
        return False
    if sig != signature:
        ecarts = ["%s: cache=%r courant=%r" % (k, sig.get(k), v)
                  for k, v in signature.items() if sig.get(k) != v]
        print("[%s] signature differente -> recalcul. Ecarts : %s"
              % (etiquette, "; ".join(ecarts) or "(cles absentes)"), flush=True)
        return False
    return True


def save_hf_cache(Z, n_grid_hf_local, cache_file, sd, signature=None):
    try:
        _sd = sd if sd is not None else (0, 1, {})
        json.dump({'Z': Z.tolist(), 'n_grid_hf': n_grid_hf_local,
                   'signature': signature,
                   'slice_def': [_sd[0], _sd[1], {str(k): v for k, v in _sd[2].items()}]},
                  open(cache_file, 'w'), indent=1)
        print(f"[HF CACHE] sauve dans {cache_file}", flush=True)
    except Exception as e:
        print(f"[HF CACHE] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)


def load_hf_cache(n_grid_hf_local, cache_file, sd, config_is_identical=True,
                  signature=None):
    if not config_is_identical:
        return None
    if not os.path.exists(cache_file):
        print(f"[HF CACHE] aucun cache ({cache_file}) -> calcul grille HF", flush=True)
        return None
    try:
        d = json.load(open(cache_file))
        _sd_cache = tuple(d['slice_def'][:2]) if 'slice_def' in d else None
        _sd_now = (sd[0], sd[1]) if sd is not None else (0, 1)
        if _sd_cache != _sd_now:
            print(f"[HF CACHE] coupe differente (cache={_sd_cache}, courant={_sd_now}) -> recalcul", flush=True)
            return None
        # `n_grid_hf_local` etait recu et JAMAIS lu : une grille 2x2 pouvait
        # etre relue pour une demande 15x15.
        _n_cache = d.get('n_grid_hf')
        if _n_cache is not None and _n_cache != n_grid_hf_local:
            print(f"[HF CACHE] cote different (cache={_n_cache}, courant={n_grid_hf_local}) -> recalcul", flush=True)
            return None
        if not _signature_compatible(d, signature, "HF CACHE"):
            return None
        print(f"[HF CACHE] charge depuis {cache_file} (coupe OK -> 0 SOCP grille)", flush=True)
        return np.array(d['Z'])
    except Exception as e:
        print(f"[HF CACHE] lecture echouee ({type(e).__name__}: {e}) -> recalcul", flush=True)
    return None


def save_hf_cache_partial(Z_flat, n_total, cache_file, sd, signature=None):
    """Sauvegarde incrementale de la grille HF (Z_flat peut contenir des None)."""
    try:
        _sd = sd if sd is not None else (0, 1, {})
        json.dump({'Z_flat': Z_flat, 'n_total': n_total, 'complet': False,
                   'signature': signature,
                   'slice_def': [_sd[0], _sd[1], {str(k): v for k, v in _sd[2].items()}]},
                  open(cache_file + '.partial', 'w'), indent=1)
    except Exception as e:
        print(f"[HF CACHE PARTIAL] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)


def load_hf_cache_partial(cache_file, sd, n_total, config_is_identical=True,
                          signature=None):
    """Charge le cache partiel. Retourne une liste Z_flat (avec None) ou None."""
    if not config_is_identical:
        return None
    partial_file = cache_file + '.partial'
    if not os.path.exists(partial_file):
        return None
    try:
        d = json.load(open(partial_file))
        _sd_cache = tuple(d['slice_def'][:2]) if 'slice_def' in d else None
        _sd_now = (sd[0], sd[1]) if sd is not None else (0, 1)
        if _sd_cache != _sd_now or d.get('n_total') != n_total:
            return None
        if not _signature_compatible(d, signature, "HF CACHE PARTIAL"):
            return None
        z = d['Z_flat']
        n_done = sum(1 for v in z if v is not None)
        print(f"[HF CACHE PARTIAL] reprise : {n_done}/{n_total} points deja calcules", flush=True)
        return z
    except Exception:
        return None


def save_hf_grid_full(fichier, Z_full, n_var, n_grid_hf, signature=None):
    try:
        json.dump({'Z': Z_full.tolist(), 'n_var': n_var, 'n_grid': n_grid_hf,
                   'signature': signature},
                  open(fichier, 'w'), indent=1)
        print(f"[HF FULL CACHE] sauve dans {fichier}", flush=True)
    except Exception as e:
        print(f"[HF FULL CACHE] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)


def load_hf_grid_full(fichier, n_var, n_grid_hf, config_is_identical=True,
                      signature=None):
    if not config_is_identical:
        return None
    if not os.path.exists(fichier):
        return None
    try:
        d = json.load(open(fichier))
        if d.get('n_var') != n_var or d.get('n_grid') != n_grid_hf:
            print(f"[HF FULL CACHE] dimensions differentes -> recalcul", flush=True)
            return None
        if not _signature_compatible(d, signature, "HF FULL CACHE"):
            return None
        print(f"[HF FULL CACHE] charge depuis {fichier} (0 SOCP)", flush=True)
        return np.array(d['Z'])
    except Exception as e:
        print(f"[HF FULL CACHE] lecture echouee ({type(e).__name__}: {e}) -> recalcul", flush=True)
    return None
