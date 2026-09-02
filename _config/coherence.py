r"""Ce que l'etude AFFIRME du modele, verifie contre le modele -- avant le
premier appel solveur.

POURQUOI EN TETE DE RUN ET PAS SEULEMENT DANS LES TESTS
--------------------------------------------------------
Une etude declare ses variables aleatoires dans `PARAM_CONFIG` et designe,
pour chacune, les elements du modele sur lesquels le solveur doit calculer
une sensibilite : des armatures, des solides, un cas de charge. Rien ne
verifiait que ces noms existent dans le `.ds`.

Les consequences etaient silencieuses ou tardives :

* un parametre absent du modele n'etait PAS ecrit par `patch_params` -- le
  solveur evaluait un point qui n'etait pas celui demande, et rendait un
  resultat plausible. Ferme le 29/08/2026 par un refus explicite, mais ce
  refus arrive au PREMIER APPEL SOLVEUR : 466 s sur le Moulin Blanc, et
  cinq heures si le defaut porte sur un point du plan d'experiences ;
* une armature designee qui n'existe pas ne provoque rien du tout : la
  region de sensibilite est simplement vide, et le gradient correspondant
  vaut zero. Un zero de gradient ne se distingue pas d'une insensibilite
  physique.

`tests/test_128_etudes_contre_modeles.py` verifiait deja tout cela, mais
sur l'ARBRE SYNTAXIQUE des etudes -- il ne voit donc que les noms ecrits
en clair, jamais ceux qu'une etude calcule. Ici, on lit le dictionnaire
REELLEMENT construit par le run.

Cout mesure le 02/09/2026 : 0,36 s sur le Moulin Blanc -- lecture des 10,1 Mo
de `dsCad.txt` et verification des 15 346 armatures designees. A comparer aux
466 s du premier appel solveur.
"""

import io
import os
import re

#: Comment chaque forme de designation se lit dans le modele. La clef est
#: celle employee dans le dict `sens` d'une variable ; la valeur dit dans
#: quel fichier chercher et sous quelle forme les noms y sont ecrits.
#:
#: `axis` et `region_key` n'y figurent pas : ce ne sont pas des designations
#: d'elements du modele mais des reglages de la sensibilite elle-meme.
DESIGNATIONS = {
    "rebars":    ("dsCad.txt",  r"REBAR\('([^']+)'"),
    "solids":    ("dsCad.txt",  r"BLOCK\('([^']+)'"),
    "load_case": ("dsLoad.txt", r"LOAD_CASE\('([^']+)'"),
}


def _lire(chemin_ds, nom):
    chemin = os.path.join(chemin_ds, nom)
    if not os.path.isfile(chemin):
        return ""
    return io.open(chemin, encoding="utf-8", errors="replace").read()


def anomalies(param_config, params_names, chemin_ds):
    """La liste des incoherences entre l'etude et son modele, vide si sain.

    Rendre une LISTE et non lever : l'appelant voit ainsi TOUTES les
    anomalies d'un coup. Un refus au premier probleme oblige a relancer
    autant de fois qu'il y a de defauts -- et chaque relance d'un run coute
    le temps de son premier appel solveur.
    """
    mauvais = []

    # ------------------------------------------------------------------ #
    # 1. les reglages de sensibilite, sans toucher au modele              #
    # ------------------------------------------------------------------ #
    manquants = [p for p in params_names
                 if not param_config[p].get("sens", {}).get("region_key")]
    if manquants:
        mauvais.append("region_key manquant pour : %s" % ", ".join(manquants))

    clefs = [param_config[p].get("sens", {}).get("region_key")
             for p in params_names]
    doubles = sorted({k for k in clefs if k and clefs.count(k) > 1})
    if doubles:
        mauvais.append(
            "region_key employe deux fois : %s -- deux variables ecriraient "
            "leur sensibilite dans la meme region." % ", ".join(doubles))

    # ------------------------------------------------------------------ #
    # 2. chaque variable est un parametre du modele                       #
    # ------------------------------------------------------------------ #
    cad, load = _lire(chemin_ds, "dsCad.txt"), _lire(chemin_ds, "dsLoad.txt")
    if not cad and not load:
        mauvais.append("modele illisible ou vide : %s" % chemin_ds)
        return mauvais

    for p in params_names:
        motif = r"(?m)^" + re.escape(p) + r"\s*="
        n_cad, n_load = (len(re.findall(motif, cad)),
                         len(re.findall(motif, load)))
        if n_cad + n_load == 0:
            mauvais.append(
                "%r est declaree dans PARAM_CONFIG mais n'est pas un "
                "parametre du modele : `patch_params` refuserait, au premier "
                "appel solveur." % p)
        for n, fichier in ((n_cad, "dsCad.txt"), (n_load, "dsLoad.txt")):
            if n > 1:
                mauvais.append(
                    "%r est defini %d fois dans %s : `patch_params` ne "
                    "reecrit que la PREMIERE occurrence, et on ne sait pas "
                    "laquelle le solveur lit." % (p, n, fichier))

    # ------------------------------------------------------------------ #
    # 3. les elements designes existent                                   #
    # ------------------------------------------------------------------ #
    presents = {}
    for clef, (fichier, motif) in DESIGNATIONS.items():
        texte = cad if fichier == "dsCad.txt" else load
        presents[clef] = set(re.findall(motif, texte))

    for p in params_names:
        sens = param_config[p].get("sens", {})
        for clef, noms in sens.items():
            if clef not in DESIGNATIONS:
                continue
            demandes = [noms] if isinstance(noms, str) else list(noms)
            inconnus = [n for n in demandes if n not in presents[clef]]
            if inconnus:
                mauvais.append(
                    "%r designe %d %s absent(s) du modele (%s%s) : la region "
                    "de sensibilite serait VIDE, et son gradient nul -- "
                    "indiscernable d'une insensibilite physique."
                    % (p, len(inconnus), clef, ", ".join(inconnus[:3]),
                       ", ..." if len(inconnus) > 3 else ""))
    return mauvais


def verifier(param_config, params_names, chemin_ds, tracer=print):
    """Refuse de demarrer si l'etude et son modele ne s'accordent pas.

    Un `SystemExit` et non une exception : ce n'est pas un defaut de code
    mais une etude mal declaree, et le message doit etre lisible sans pile
    d'appels.
    """
    mauvais = anomalies(param_config, params_names, chemin_ds)
    if mauvais:
        raise SystemExit(
            "[coherence] l'etude ne s'accorde pas avec son modele :\n"
            + "\n".join("  - " + m for m in mauvais)
            + "\n  modele : %s" % chemin_ds)

    designes = sum(
        len(v) if not isinstance(v, str) else 1
        for p in params_names
        for k, v in param_config[p].get("sens", {}).items()
        if k in DESIGNATIONS)
    tracer("[coherence] %d variable(s), %d element(s) designe(s) : tous "
           "presents dans le modele." % (len(params_names), designes))
