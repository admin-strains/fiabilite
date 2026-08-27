# -*- coding: utf-8 -*-
"""
L'etat de reprise : ce qu'un run laisse derriere lui pour qu'un autre
continue son enrichissement au lieu de le refaire.

Sur le Moulin Blanc un point coute 466 s. Un dump de reprise peut porter
quatre-vingt-dix heures de calcul. C'est le fichier le plus cher du depot,
et il etait ecrit et relu par cent quinze lignes recopiees a l'identique
dans les deux etudes, a l'interieur de `if __name__ == '__main__':` --
donc hors de portee de tout test.

Le corps est repris VERBATIM. Seules les variables libres de `main`
deviennent des parametres.

Aucune dependance lourde : `json`, `os`, `numpy`. Ni OpenTURNS ni Digital
Structure -- ce module se teste sans licence et sans solveur.
"""

import json
import os

import numpy as np


NOM_PAR_DEFAUT = "restart_state.json"


def fichier_de(path_ds, nom=NOM_PAR_DEFAUT):
    """Le dump vit dans le `.ds` du modele, PAS dans le depot.

    C'est voulu : il decrit un modele, pas un code. Mais c'est aussi
    pourquoi reprendre une etude sur un autre poste echoue -- voir le
    message de `charger`.
    """
    return os.path.join(path_ds, nom)


def _u_beta(r):
    """Un resultat FORM reduit a ce qui se serialise : le point de
    conception et l'indice de fiabilite.

    Un `ot.FORMResult` ne passe pas par JSON, et n'a pas besoin d'y passer :
    a la reprise on ne relit ces deux valeurs que pour les afficher.
    """
    try:
        return {"u_star": [float(v) for v in np.array(r.getStandardSpaceDesignPoint())],
                "beta": float(r.getHasoferReliabilityIndex())}
    except Exception:
        return None


def _flottants(valeurs):
    """Une liste d'historique, ou `None` reste `None`.

    Les criteres d'arret laissent des trous : en mode `BB`, `hist_BS` n'est
    jamais alimente, et un round peut ne produire aucun ratio. `float(None)`
    leverait ; ecraser en 0.0 mentirait sur la courbe de convergence.
    """
    return [None if v is None else float(v) for v in valeurs]


def construire_etat(*, signature, signature_solveur, modele, timestamp,
                    max_degree, n0, xt, yt, all_grad, xt_eff,
                    enrich_round, round_sizes_prev, historiques, coupe_hf,
                    best_result, best_sp, modes, result_IS):
    """Le contenu du dump, sans ecriture ni journal.

    Separe de `enregistrer` pour qu'un test puisse verifier ce qui est ecrit
    sans toucher au disque.
    """
    st = {}
    # `signature` est la signature FAIBLE (n0, params, n_var, modelname) :
    # elle ignore le solveur, le solveur lineaire, le maillage et les
    # bornes. Elle est conservee telle quelle -- des outils la lisent --
    # mais elle ne protege rien.
    st["signature"] = signature
    # Celle-ci, si. Elle porte 90 heures d'enrichissement.
    st["signature_solveur"] = signature_solveur
    st["modele"]    = modele
    st["timestamp"] = timestamp
    try:    st["max_degree"] = int(max_degree)
    except Exception: st["max_degree"] = None
    st["xt"]       = np.asarray(xt).tolist()       if xt       is not None else None
    st["yt"]       = np.asarray(yt).tolist()       if yt       is not None else None
    st["all_grad"] = np.asarray(all_grad).tolist() if all_grad is not None else None
    st["xt_eff"]   = [np.asarray(p).tolist() for p in xt_eff] if xt_eff else []
    st["n_doe"]    = n0
    st["n_total"]  = int(len(xt)) if xt is not None else 0
    _prev_tot = sum(round_sizes_prev) if round_sizes_prev else 0
    if enrich_round > 0:
        st["round_sizes"] = list(round_sizes_prev) + [int(len(xt)) - _prev_tot]
    else:
        st["round_sizes"] = [int(len(xt))] if xt is not None else []
    st["enrich_round"]     = int(enrich_round)
    st["round_boundaries"] = list(np.cumsum([0] + st["round_sizes"]).astype(int).tolist())
    st["hist_EFF"]     = [float(v) for v in historiques["EFF"]]
    st["hist_BB"]      = _flottants(historiques["BB"])
    st["hist_BS"]      = _flottants(historiques["BS"])
    st["hist_theta"]   = [[float(x) for x in t] for t in historiques["theta"]]
    st["hist_beta_IS"] = _flottants(historiques["beta_IS"])
    st["hf_2d_grid"]   = coupe_hf
    st["best_sp"]     = [float(v) for v in np.array(best_sp)] if best_sp is not None else None
    st["best_result"] = _u_beta(best_result) if best_result is not None else None
    st["modes"]       = [_u_beta(m) for m in modes] if modes else []
    try:
        st["IS"] = {"Pf": float(result_IS.getProbabilityEstimate())} if result_IS is not None else None
    except Exception:
        st["IS"] = None
    return st


def enregistrer(fichier, **champs):
    """Ecrit le dump. Un echec est signale, jamais fatal.

    Le dump est ecrit apres chaque round et a la fin d'un run : une erreur
    de serialisation ne doit pas emporter le calcul qui vient de se
    terminer. Le run continue, le journal porte la raison.
    """
    try:
        st = construire_etat(**champs)
        json.dump(st, open(fichier, "w"), indent=1)
        print("[RESTART DUMP] etat sauve dans %s "
              "(n_total=%s, n_eff=%d, hist_EFF=%d, modes=%d)"
              % (fichier, st["n_total"], len(st["xt_eff"]),
                 len(st["hist_EFF"]), len(st["modes"])), flush=True)
        return st
    except Exception as e:
        print("[RESTART DUMP] sauvegarde echouee (%s: %s)"
              % (type(e).__name__, e), flush=True)
        return None


def charger(fichier, signature_attendue):
    """Relit un dump, et refuse de reprendre sous une autre configuration.

    Deux controles, deux defauts constates :

    1. Sans le controle d'existence, l'absence du dump donnait un
       `FileNotFoundError` brut APRES plusieurs minutes de construction du
       modele CAD. Le cas se produit des qu'on reprend une etude sur un
       autre poste : le dump vit dans le `.ds` du modele, il n'est pas dans
       le depot.

    2. LE DUMP PORTAIT UNE SIGNATURE QUE PERSONNE NE LISAIT (26/08/2026).
       Reprendre un enrichissement apres avoir change de solveur lineaire,
       de maillage ou de bornes melangeait tout, sans une ligne de journal.
       C'est le meme defaut que les caches DOE et HF -- mais ici il porte
       jusqu'a 90 heures de calcul, et le melange serait indetectable dans
       le resultat.

    Leve `SystemExit` : ce n'est pas une erreur de programmation, c'est une
    configuration qu'on refuse d'honorer, et le message s'adresse a qui a
    lance le run.
    """
    if not os.path.isfile(fichier):
        raise SystemExit(
            "restart_enrich_only = true, mais aucun etat a reprendre :\n"
            "  %s\n\n"
            "Ce fichier est produit par un run precedent, dans le .ds du\n"
            "modele. Pour repartir de zero, mettre\n"
            "  restart_enrich_only = false\n"
            "dans le fichier d'etude." % fichier)
    rs = json.load(open(fichier))

    sig_dump = rs.get("signature_solveur")
    if sig_dump != signature_attendue:
        ecarts = ["  %s : dump=%r  courant=%r" % (k, (sig_dump or {}).get(k), v)
                  for k, v in signature_attendue.items()
                  if (sig_dump or {}).get(k) != v]
        raise SystemExit(
            "restart_enrich_only = true, mais le dump n'a PAS ete produit "
            "sous la configuration courante :\n"
            + ("\n".join(ecarts) if sig_dump is not None else
               "  (dump anterieur au controle : aucune signature)")
            + "\n\nReprendre melangerait des points calcules autrement.\n"
              "Soit on retablit la configuration du dump, soit on repart "
              "de zero avec `restart_enrich_only = false`.")
    return rs


def historiques_de(rs):
    """Les cinq historiques du dump, sous les clefs de `construire_etat`.

    Chaque `get` porte son defaut : un dump anterieur a l'ajout d'un
    historique se relit sans lever.
    """
    return {"EFF":     list(rs.get("hist_EFF", [])),
            "BB":      list(rs.get("hist_BB", [])),
            "BS":      list(rs.get("hist_BS", [])),
            "theta":   list(rs.get("hist_theta", [])),
            "beta_IS": list(rs.get("hist_beta_IS", []))}
