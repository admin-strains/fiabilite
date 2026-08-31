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

import ecriture as _ecriture


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
        _ecriture.ecrire_json({"n0": n0, "complet": True,
                   "signature": signature,
                   "xt": np.asarray(xt).tolist(),
                   "yt": np.asarray(yt).tolist(),
                   "all_grad": np.asarray(all_grad).tolist()},
                  fichier, indent=1)
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
        # `.get(f'dg_{p}')` SANS defaut : un gradient absent doit ressortir
        # `null`, jamais 0.0. Un zero fabrique affirmerait que l'etat limite
        # est plat en ce point, et le metamodele l'ajusterait. Le defaut 0.0
        # qui figurait ici ne s'est jamais declenche -- la clef existe
        # toujours, avec la valeur None -- mais il n'attendait qu'une clef
        # manquante pour mentir.
        _ag = [[SOL[i].get(f'dg_{p}') for p in params_names] for i in range(n_done)]
        _ecriture.ecrire_json({"n0": n0, "complet": False, "n_completed": n_done,
                   "signature": signature,
                   "xt": _xt, "yt": _yt, "all_grad": _ag},
                  fichier, indent=1)
        print(f"[DOE CACHE INCR] {n_done}/{len(SOL)} pts sauves", flush=True)
    except Exception as e:
        print(f"[DOE CACHE INCR] echoue ({type(e).__name__}: {e})", flush=True)


def charger_doe_partiel(fichier, n0, signature=None, xt_attendu=None, tol=1e-9):
    """Les points DEJA calcules d'un plan interrompu, ou None.

    LE TROU QUE CETTE FONCTION BOUCHE -- 26/08/2026
    ------------------------------------------------
    `save_doe_cache_incremental` ecrit le plan APRES CHAQUE POINT, avec
    `complet: False`. L'intention de l'auteur est limpide : survivre a une
    interruption. Mais `load_doe_cache` refuse tout cache incomplet, et jetait
    donc systematiquement ce filet.

    Le prix, mesure : trois interruptions dans la meme journee, chaque fois
    pendant le plan, chaque fois ~75 minutes de solveur perdues (5 points a
    ~15 min sur le Moulin Blanc). Le plan est justement la phase ou la casse
    arrive -- c'est la que les points les plus extremes sont evalues en
    premier.

    L'enrichissement, lui, EST protege (`restart_state.json` apres chaque
    point ajoute), et la grille HF aussi (cache partiel). Le plan etait la
    seule phase dont le filet etait ecrit puis ignore.

    POURQUOI `xt_attendu` -- on VERIFIE, on ne SUPPOSE PAS
    ------------------------------------------------------
    Reutiliser k points suppose que le tirage LHS redonne EXACTEMENT les
    memes. C'est vrai aujourd'hui : deux process distincts ont produit des
    coordonnees bit-identiques. Mais cela tient a la graine par defaut
    d'OpenTURNS, que rien ne garantit dans le temps.

    Passer le tirage courant fait donc CONTROLER la coincidence point par
    point. Si elle ne tient plus, on recalcule tout et on le DIT -- au lieu de
    melanger deux plans differents, ce qui ne se verrait nulle part.
    """
    if not os.path.exists(fichier):
        return None
    try:
        d = json.load(open(fichier))
    except Exception as e:
        print("[DOE PARTIEL] lecture echouee (%s: %s)" % (type(e).__name__, e), flush=True)
        return None

    if d.get("n0") != n0:
        print("[DOE PARTIEL] n0 different (cache=%s, courant=%s) -> recalcul complet"
              % (d.get("n0"), n0), flush=True)
        return None

    if signature is not None:
        sig = d.get("signature")
        if sig is None or sig != signature:
            ecarts = ([] if sig is None else
                      ["%s: cache=%r courant=%r" % (k, sig.get(k), v)
                       for k, v in signature.items() if sig.get(k) != v])
            print("[DOE PARTIEL] signature %s -> recalcul complet. %s"
                  % ("absente" if sig is None else "differente",
                     "; ".join(ecarts) or "(cache anterieur au controle)"), flush=True)
            return None

    xt = np.asarray(d.get("xt", []), dtype=float)
    yt = np.asarray(d.get("yt", []), dtype=float)
    ag = np.asarray(d.get("all_grad", []), dtype=float)
    n_faits = int(d.get("n_completed", len(xt)))
    n_faits = max(0, min(n_faits, len(xt), len(yt), len(ag)))
    if n_faits == 0:
        return None

    # Un gradient absent est ecrit `null`, et numpy le convertit en NaN SANS
    # RIEN DIRE (`np.asarray([[None]], dtype=float)` -> `[[nan]]`). Reprendre
    # tel quel injecterait des NaN dans le metamodele -- exactement la
    # corruption silencieuse qu'on cherche a eviter.
    #
    # On tronque donc au PREMIER point incomplet, pour garder la propriete de
    # prefixe dont depend la reprise. Le point tronque sera re-evalue, ce qui
    # rendra aussi son diagnostic `sain` -- et `exclure_points_sans_gradient`
    # decidera de son sort.
    fini = np.all(np.isfinite(ag[:n_faits]), axis=1) if n_faits else np.array([], bool)
    if not np.all(fini):
        premier = int(np.argmin(fini))
        print("[DOE PARTIEL] point %d sans gradient dans le cache -> reprise "
              "tronquee a %d point(s). Les suivants seront re-evalues."
              % (premier, premier), flush=True)
        n_faits = premier
        if n_faits == 0:
            return None

    if xt_attendu is not None:
        attendu = np.asarray(xt_attendu, dtype=float)
        if attendu.shape[1:] != xt.shape[1:]:
            print("[DOE PARTIEL] dimensions incompatibles -> recalcul complet", flush=True)
            return None
        n_faits = min(n_faits, len(attendu))
        ecart = np.max(np.abs(xt[:n_faits] - attendu[:n_faits])) if n_faits else 0.0
        if ecart > tol:
            print("[DOE PARTIEL] le tirage a CHANGE (ecart max %.3e > %.0e) -> "
                  "recalcul complet. Les points caches ne sont pas ceux qu'on "
                  "evaluerait." % (ecart, tol), flush=True)
            return None

    print("[DOE PARTIEL] %d/%d points repris du cache -> autant de SOCP evites"
          % (n_faits, n0), flush=True)
    return xt[:n_faits], yt[:n_faits], ag[:n_faits], n_faits


def greffer_reprise(SOL, repris, params_names):
    """Verse dans `SOL` les points d'un plan interrompu, deja calcules.

    Le cache incremental est ecrit apres CHAQUE point ; jusqu'au 26/08/2026
    il n'etait jamais relu, et chaque interruption coutait tout le plan --
    trois fois dans la meme journee, environ 75 min de solveur a chaque fois.

    `charger_doe_partiel` a deja VERIFIE que le tirage redonne les memes
    points : cette fonction ne suppose rien, elle recopie.

    `repris = None` -- rien a reprendre -- n'est pas une erreur : c'est le
    cas courant d'un plan qui demarre. Retourne le nombre de points greffes.
    """
    if repris is None:
        return 0
    xt_r, yt_r, ag_r, n_faits = repris
    for i in range(n_faits):
        SOL[i]["g"] = float(yt_r[i][0])
        SOL[i]["_u"] = [float(v) for v in xt_r[i]]
        for j, p in enumerate(params_names):
            SOL[i]["dg_%s" % p] = float(ag_r[i][j])
    return n_faits
