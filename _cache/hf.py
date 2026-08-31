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

import ecriture as _ecriture


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



#: Deux valeurs figees sont « les memes » en dessous de cet ecart. Elles
#: viennent de `u*`, donc de flottants : une egalite stricte refuserait des
#: caches legitimes pour un dernier bit.
TOL_FIGEES = 1e-9


def _meme_coupe(d, sd, etiquette, tol=TOL_FIGEES):
    """La coupe cachee est-elle CELLE QU'ON DEMANDE ?

    Une coupe, c'est TROIS choses : deux axes, et les valeurs auxquelles les
    autres variables sont figees. Seuls les deux axes etaient compares --
    alors que les valeurs figees etaient bel et bien ecrites dans le fichier.

    `(0, 2, {1: 7.5})` et `(0, 2, {1: -3.0})` ont les memes axes et sont deux
    surfaces sans rapport. Le cache servait l'une pour l'autre, en annoncant
    « coupe OK ». Sans effet a deux variables, ou le dictionnaire est
    toujours vide ; faux au-dela -- et c'est justement au-dela que la coupe
    finale fige les variables secondaires a `u*`.

    CE QUE CE CONTROLE COUTE. `u*` bouge d'un run a l'autre : a trois
    variables ou plus, la grille de la coupe finale sera souvent recalculee.
    C'est le prix a payer pour ne pas tracer une surface calculee ailleurs.
    """
    axes_cachees = tuple(d['slice_def'][:2]) if 'slice_def' in d else None
    axes_demandees = (sd[0], sd[1]) if sd is not None else (0, 1)
    if axes_cachees != axes_demandees:
        print("[%s] coupe differente (cache=%s, courant=%s) -> recalcul"
              % (etiquette, axes_cachees, axes_demandees), flush=True)
        return False

    figees_cachees = d['slice_def'][2] if len(d.get('slice_def', [])) > 2 else None
    figees_demandees = sd[2] if sd is not None and len(sd) > 2 else {}
    if figees_cachees is None:
        if figees_demandees:
            print("[%s] cache sans valeurs figees (anterieur au controle) -> "
                  "recalcul" % etiquette, flush=True)
            return False
        return True

    # les clefs passent par JSON, donc en chaines
    cachees = {int(k): float(v) for k, v in figees_cachees.items()}
    demandees = {int(k): float(v) for k, v in figees_demandees.items()}
    if set(cachees) != set(demandees):
        print("[%s] variables figees differentes (cache=%s, courant=%s) -> "
              "recalcul" % (etiquette, sorted(cachees), sorted(demandees)),
              flush=True)
        return False
    ecarts = {k: abs(cachees[k] - demandees[k]) for k in cachees}
    pires = [k for k, e in ecarts.items() if e > tol]
    if pires:
        print("[%s] valeurs figees differentes -> recalcul. %s"
              % (etiquette,
                 ", ".join("u%d : cache=%.6g courant=%.6g"
                           % (k, cachees[k], demandees[k]) for k in sorted(pires))),
              flush=True)
        return False
    return True


def save_hf_cache(Z, n_grid_hf_local, cache_file, sd, signature=None):
    try:
        _sd = sd if sd is not None else (0, 1, {})
        _ecriture.ecrire_json({'Z': Z.tolist(), 'n_grid_hf': n_grid_hf_local,
                   'signature': signature,
                   'slice_def': [_sd[0], _sd[1], {str(k): v for k, v in _sd[2].items()}]},
                  cache_file, indent=1)
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
        if not _meme_coupe(d, sd, "HF CACHE"):
            return None
        # `n_grid_hf_local` etait recu et JAMAIS lu : une grille 2x2 pouvait
        # etre relue pour une demande 15x15.
        _n_cache = d.get('n_grid_hf')
        if _n_cache is not None and _n_cache != n_grid_hf_local:
            print(f"[HF CACHE] cote different (cache={_n_cache}, courant={n_grid_hf_local}) -> recalcul", flush=True)
            return None
        if not _signature_compatible(d, signature, "HF CACHE"):
            return None
        print(f"[HF CACHE] charge depuis {cache_file} "
              f"(meme coupe, memes valeurs figees -> 0 SOCP grille)", flush=True)
        return np.array(d['Z'])
    except Exception as e:
        print(f"[HF CACHE] lecture echouee ({type(e).__name__}: {e}) -> recalcul", flush=True)
    return None


def save_hf_cache_partial(Z_flat, n_total, cache_file, sd, signature=None):
    """Sauvegarde incrementale de la grille HF (Z_flat peut contenir des None)."""
    try:
        _sd = sd if sd is not None else (0, 1, {})
        _ecriture.ecrire_json({'Z_flat': Z_flat, 'n_total': n_total, 'complet': False,
                   'signature': signature,
                   'slice_def': [_sd[0], _sd[1], {str(k): v for k, v in _sd[2].items()}]},
                  cache_file + '.partial', indent=1)
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
        if d.get('n_total') != n_total:
            return None
        if not _meme_coupe(d, sd, "HF CACHE PARTIAL"):
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
        _ecriture.ecrire_json({'Z': Z_full.tolist(), 'n_var': n_var, 'n_grid': n_grid_hf,
                   'signature': signature},
                  fichier, indent=1)
        print(f"[HF FULL CACHE] sauve dans {fichier}", flush=True)
    except Exception as e:
        print(f"[HF FULL CACHE] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)


def save_hf_grid_full_partial(fichier, Z_flat, n_total, n_var, n_grid_hf,
                              signature=None):
    """Sauvegarde incrementale de la grille COMPLETE. `Z_flat` porte des None.

    LE TROU QUE CECI BOUCHE -- 29/08/2026
    --------------------------------------
    Toutes les boucles qui paient du solveur ecrivent apres CHAQUE point : la
    coupe 2D, les points libres, le plan d'experiences, l'enrichissement.
    `calculer_complete` etait la seule a n'ecrire qu'a la fin -- et c'est
    elle dont le docstring dit : « la seule action du programme dont le budget
    peut depasser la semaine ». A trois variables et un cote de 15, c'est
    3 375 appels ; une interruption au 3 374e les perdait tous.

    Les clefs sont celles du cache complet -- `n_var`, `n_grid`, la signature
    -- plus `n_total` : ce sont elles qui determinent la grille, et rien
    d'autre. Il n'y a pas de coupe ici, donc rien de plus a verifier.
    """
    try:
        _ecriture.ecrire_json({'Z_flat': Z_flat, 'n_total': n_total, 'complet': False,
                   'n_var': n_var, 'n_grid': n_grid_hf, 'signature': signature},
                  fichier + '.partial', indent=1)
    except Exception as e:
        # PAS EN SILENCE : sans ce fichier, une interruption repaie tout, et
        # c'est justement la situation ou personne ne relit le journal apres.
        print(f"[HF FULL PARTIAL] sauvegarde echouee ({type(e).__name__}: {e}) "
              f"-- une interruption perdra les points deja payes", flush=True)


def load_hf_grid_full_partial(fichier, n_total, n_var, n_grid_hf,
                              config_is_identical=True, signature=None):
    """Les points d'une grille complete interrompue, ou None.

    Retourne une liste `Z_flat` de longueur `n_total`, avec `None` aux places
    qui restent a payer.
    """
    if not config_is_identical:
        return None
    partial_file = fichier + '.partial'
    if not os.path.exists(partial_file):
        return None
    try:
        d = json.load(open(partial_file))
        if (d.get('n_total') != n_total or d.get('n_var') != n_var
                or d.get('n_grid') != n_grid_hf):
            print("[HF FULL PARTIAL] dimensions differentes -> recalcul complet",
                  flush=True)
            return None
        if not _signature_compatible(d, signature, "HF FULL PARTIAL"):
            return None
        z = d['Z_flat']
        if len(z) != n_total:
            return None
        n_done = sum(1 for v in z if v is not None)
        print(f"[HF FULL PARTIAL] reprise : {n_done}/{n_total} points deja "
              f"calcules", flush=True)
        return z
    except Exception as e:
        print(f"[HF FULL PARTIAL] lecture echouee ({type(e).__name__}: {e}) -> "
              f"recalcul complet", flush=True)
        return None


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
