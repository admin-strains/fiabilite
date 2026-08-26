"""
Cache du plan d'experiences (DOE).

Extrait de `AC3_pure_flexion.py` / `AC3_moulinblanc.py`, ou ces fonctions
etaient definies dans `if __name__ == '__main__':` -- donc ni importables ni
testables -- et recopiees a l'identique dans les deux scripts.

PHASE 3 du plan de nettoyage. Les corps sont repris VERBATIM ; seules les
variables libres de `main` deviennent des parametres explicites. C'etait tout
l'enjeu : une fonction qui se ferme sur l'espace de noms de `main` ne peut
etre appelee par aucun test.

Les variables libres devenues parametres : `_DOE_CACHE_FILE` -> `fichier`,
`n0`, `params_names`, `n_var`, `modelname`, et le drapeau
`config_is_identical` qui commande la relecture.

Aucune dependance lourde : `json`, `os`, `numpy`. Ni OpenTURNS ni Digital
Structure.
"""

import json
import os

import numpy as np


def doe_cache_sig(n0, params_names, n_var, modelname):
    """Signature de configuration du DOE : ce qui doit coincider pour
    qu'un cache soit reutilisable.

    ATTENTION : cette fonction ne sert QU'A ESTAMPILLER le dump de reprise
    (`restart_state.json`). Elle ne validait rien, et son nom trompait -- voir
    `load_doe_cache`, qui accepte desormais une signature reellement comparee.
    Conservee inchangee : les suites d'origine s'en servent de temoin.
    """
    return {"n0": n0, "params": list(params_names), "n_var": n_var, "modelname": modelname}


def save_doe_cache(fichier, n0, xt, yt, all_grad, signature=None):
    """Ecrit un DOE complet, avec la signature sous laquelle il a ete calcule."""
    try:
        json.dump({"n0": n0, "complet": True,
                   "signature": signature,
                   "xt": np.asarray(xt).tolist(),
                   "yt": np.asarray(yt).tolist(),
                   "all_grad": np.asarray(all_grad).tolist()},
                  open(fichier, "w"), indent=1)
        print(f"[DOE CACHE] sauve (complet) dans {fichier}", flush=True)
    except Exception as e:
        print(f"[DOE CACHE] sauvegarde echouee ({type(e).__name__}: {e})", flush=True)


def load_doe_cache(fichier, n0, config_is_identical=True, signature=None):
    """Relit un DOE complet, ou None si le cache ne correspond pas.

    LE TROU QUE `signature` FERME -- constat du 26/08/2026
    ------------------------------------------------------
    Cette fonction ne validait QUE `n0` et le drapeau `complet`. Elle
    reutilisait donc sans broncher un plan d'experiences calcule avec un autre
    solveur lineaire, une autre taille de maille ou un autre modele. Le cas
    s'est presente le jour meme : en basculant CuDss -> MUMPS sur le Moulin
    Blanc, un cache complet aurait rendu des points issus de l'autre backend,
    et le run aurait melange les deux sans qu'aucune trace ne le dise.

    Un cache sans signature est REFUSE, pas accepte par defaut : le cout est
    un recalcul du plan initial, le prix de l'alternative est un resultat faux
    et silencieux.
    """
    if not config_is_identical:
        return None
    if not os.path.exists(fichier):
        print(f"[DOE CACHE] aucun cache ({fichier}) -> calcul DOE", flush=True)
        return None
    try:
        d = json.load(open(fichier))
        _n0_cache = d.get('n0', len(d.get('xt', [])))
        if _n0_cache != n0:
            print(f"[DOE CACHE] n0 different (cache={_n0_cache}, courant={n0}) -> recalcul DOE", flush=True)
            return None
        if not d.get('complet', False):
            print(f"[DOE CACHE] cache incomplet (complet=False) -> recalcul DOE", flush=True)
            return None
        if signature is not None:
            _sig_cache = d.get('signature')
            if _sig_cache is None:
                print("[DOE CACHE] cache sans signature (anterieur au controle) "
                      "-> recalcul DOE, par prudence", flush=True)
                return None
            if _sig_cache != signature:
                _ecarts = [f"{k}: cache={_sig_cache.get(k)!r} courant={v!r}"
                           for k, v in signature.items() if _sig_cache.get(k) != v]
                print("[DOE CACHE] signature differente -> recalcul DOE. Ecarts : "
                      + "; ".join(_ecarts or ["(cles absentes du cache)"]), flush=True)
                return None
        print(f"[DOE CACHE] charge depuis {fichier} (n0={n0}, complet -> 0 SOCP DOE)", flush=True)
        return np.array(d["xt"]), np.array(d["yt"]), np.array(d["all_grad"])
    except Exception as e:
        print(f"[DOE CACHE] lecture echouee ({type(e).__name__}: {e}) -> recalcul DOE", flush=True)
    return None


def save_doe_cache_incremental(fichier, n0, params_names, SOL, n_done, signature=None):
    """Sauvegarde incrementale : ecrit les n_done premiers points de SOL.
    Gradients deja en U (converties par run_one_SOL). Pas de cle 'complet'."""
    try:
        _xt = [SOL[i]['_u'] for i in range(n_done)]
        _yt = [[SOL[i]['g']] for i in range(n_done)]
        _ag = [[SOL[i].get(f'dg_{p}', 0.0) for p in params_names] for i in range(n_done)]
        json.dump({"n0": n0, "complet": False, "n_completed": n_done,
                   "signature": signature,
                   "xt": _xt, "yt": _yt, "all_grad": _ag},
                  open(fichier, "w"), indent=1)
        print(f"[DOE CACHE INCR] {n_done}/{len(SOL)} pts sauves", flush=True)
    except Exception as e:
        print(f"[DOE CACHE INCR] echoue ({type(e).__name__}: {e})", flush=True)
