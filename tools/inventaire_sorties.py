"""
Ce qu'un run a reellement produit, et ce que ca pese.

    python tools/inventaire_sorties.py --etude moulin_blanc
    python tools/inventaire_sorties.py --etude pure_flexion --journal C:\\tmp\\run.txt

Un run de fiabilite ecrit a DEUX endroits, et l'un des deux surprend :

  * le dossier de SORTIE de l'etude -- figures, journal, trace de la
    configuration. C'est celui qu'on regarde ;
  * le MODELE lui-meme, le `.ds` -- caches, dump de reprise, journal des
    points, et les fichiers de sortie du solveur, reecrits a chaque appel.
    C'est celui qu'on oublie, et c'est le plus gros.

L'outil les inventorie tous les deux, distingue ce qui est un RESULTAT de ce
qui est un fichier de TRAVAIL, et lit le journal s'il est fourni pour rendre
le compte des appels au solveur.

Ne demande ni Digital Structure ni OpenTURNS.
"""

import argparse
import io
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "_config"))

#: fichiers du `.ds` produits par la chaine, et ce qu'ils sont.
#: Un fichier absent de cette table est signale : le depot doit savoir ce que
#: son code ecrit.
CONNUS = [
    (r"^dsCad\.txt$",              "modele", "geometrie -- REECRITE a chaque evaluation"),
    (r"^dsLoad\.txt$",             "modele", "chargements -- reecrit a chaque evaluation"),
    (r"^dsNote\.txt$",             "modele", "note"),
    (r"\.stp$",                    "modele", "geometrie importee"),
    (r"^doe_cache\.json$",         "cache",  "plan d'experiences, relu si config_is_identical"),
    (r"^hf_grid.*cache.*\.json$",  "cache",  "grille haute fidelite"),
    (r"^hf_custom_cache\.json$",   "cache",  "grille haute fidelite sur mesure"),
    (r"^restart_state\.json$",     "reprise", "etat pour restart_enrich_only"),
    (r"^points_log\.jsonl$",       "trace",  "un enregistrement par point evalue"),
    (r"\.dscad$",                  "travail", "modele CAD binaire, reecrit a chaque appel"),
    (r"_kine\.dsmetares$",         "resultat", "alpha, statut solveur, sensibilites"),
    (r"_kine\.dsmed$",             "travail", "champs cinematiques -- volumineux"),
    (r"_kine\.dslog$",             "travail", "journal du solveur"),
    (r"_stat\.dsmed$",             "travail", "champs statiques"),
    (r"\.msh$",                    "travail", "maillages intermediaires"),
    (r"\.pos$",                    "travail", "sortie Gmsh"),
    (r"\.dsload$",                 "travail", "chargements compiles"),
    (r"^SOCP_history$",            "archive", "copie par point si save_history"),
    (r"^_doe_workers$",            "travail", "copies du modele pour le DOE parallele"),
    (r"^_hf_workers$",             "travail", "copies du modele pour la grille HF"),
    # Releves le 26/08/2026 en lancant cet outil : il les avait signales comme
    # non recenses, ce qui est exactement son role.
    (r"_mesh\.dsmetares$",         "resultat", "metadonnees du maillage"),
    (r"_mesh\.dslog$",             "travail", "journal du mailleur"),
    (r"_MeshQuality\.txt$",        "trace",  "qualite du maillage"),
    (r"\.dslogloc$",               "travail", "journal local du solveur"),
    (r"_cadSurf\.mesh$",           "travail", "surface CAD, format MeshGems"),
    (r"\.mesh$",                   "travail", "maillage au format MeshGems"),
    (r"\.sol$",                    "travail", "carte de tailles MeshGems"),
    (r"^coupe\.txt$",              "modele", "definition de coupe, fournie avec le modele"),
    (r"^solve_one\.json$",         "resultat", "sortie de tools/solve_one.py"),
    (r"^png EFF$",                 "resultat", "figures de l'enrichissement, un dossier par run"),
    (r"^configuration\.json$",     "trace",  "configuration effective du run"),
    (r"^globalplanche.*\.png$",    "resultat", "planche de synthese"),
    (r"\.png$",                    "resultat", "figure"),
    (r"^custom_hf_grid\.json$",    "modele", "grille HF sur mesure, fournie"),
]


def _classer(nom):
    for motif, categorie, quoi in CONNUS:
        if re.search(motif, nom):
            return categorie, quoi
    return "inconnu", ""


def _taille(chemin):
    if os.path.isfile(chemin):
        return os.path.getsize(chemin)
    total = 0
    for racine, _, fichiers in os.walk(chemin):
        for f in fichiers:
            try:
                total += os.path.getsize(os.path.join(racine, f))
            except OSError:
                pass
    return total


def _mo(octets):
    return octets / 1048576.0


def inventorier(dossier, titre):
    if not os.path.isdir(dossier):
        print("  %s : ABSENT (%s)" % (titre, dossier))
        return {}
    entrees = []
    for nom in sorted(os.listdir(dossier)):
        chemin = os.path.join(dossier, nom)
        categorie, quoi = _classer(nom)
        entrees.append((categorie, nom, _taille(chemin), quoi,
                        os.path.isdir(chemin)))

    print("")
    print("  %s" % titre)
    print("  %s" % dossier)
    print("  " + "-" * 74)
    par_categorie = {}
    for categorie, nom, octets, quoi, est_dossier in entrees:
        par_categorie.setdefault(categorie, 0)
        par_categorie[categorie] += octets
        marque = "/" if est_dossier else " "
        print("  %-9s %-38s %9.2f Mo  %s"
              % (categorie, (nom + marque)[:38], _mo(octets), quoi))
    print("  " + "-" * 74)
    for categorie in sorted(par_categorie, key=lambda c: -par_categorie[c]):
        print("  %-9s %58.2f Mo" % (categorie, _mo(par_categorie[categorie])))
    inconnus = [n for c, n, *_ in entrees if c == "inconnu"]
    if inconnus:
        print("")
        print("  NON RECENSES : %s" % inconnus)
        print("  Les ajouter a CONNUS : le depot doit savoir ce que son code ecrit.")
    return par_categorie


def lire_journal(chemin):
    """Compte les appels solveur et releve les grandeurs finales."""
    if not chemin or not os.path.isfile(chemin):
        return
    appels = 0
    modes, pf, beta = [], None, None
    duree_solv = 0.0
    for ligne in io.open(chemin, encoding="utf-8", errors="replace"):
        if "_kine.dsmetares" in ligne:
            appels += 1
        m = re.match(r"\s*mode\s+\d+\s*:\s*beta=.*", ligne)
        if m:
            modes.append(ligne.strip())
        if re.match(r"\s*Pf_IS\s+=", ligne):
            pf = ligne.strip()
        if re.match(r"\s*beta_IS\s+=", ligne):
            beta = ligne.strip()
        m = re.search(r"solveur ([\d.]+) s", ligne)
        if m:
            duree_solv += float(m.group(1))

    print("")
    print("  JOURNAL  %s" % chemin)
    print("  " + "-" * 74)
    print("  appels solveur reperes : %d" % appels)
    if modes:
        for m in modes[-4:]:
            print("  %s" % m)
    for x in (pf, beta):
        if x:
            print("  %s" % x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--etude", default="moulin_blanc",
                    help="nom du fichier d'etude, sans .toml")
    ap.add_argument("--journal", default=None,
                    help="journal du run, pour compter les appels solveur")
    args = ap.parse_args()

    from schema import charger                                  # noqa: PLC0415
    cfg = charger(os.path.join(REPO, "studies", args.etude + ".toml"))

    print("=" * 78)
    print("INVENTAIRE DES SORTIES -- etude %s" % args.etude)
    print("=" * 78)

    #: dossier de l'etude : le nom du `.toml` ne s'y ramene pas
    #: mecaniquement (`moulin_blanc` -> `Moulinblanc`).
    DOSSIERS = {"pure_flexion": "pure_flexion", "moulin_blanc": "Moulinblanc"}
    dossier_etude = os.path.join(REPO, DOSSIERS.get(args.etude, args.etude))

    total = {}
    for categories in (inventorier(cfg.chemin_ds, "MODELE (.ds) -- ecrit PENDANT le run"),
                       inventorier(os.path.join(dossier_etude, "output"),
                                   "SORTIES de l'etude")):
        for c, v in categories.items():
            total[c] = total.get(c, 0) + v

    lire_journal(args.journal)

    print("")
    print("=" * 78)
    print("  TOTAL %.1f Mo, dont %.1f Mo de fichiers de TRAVAIL reecrits a chaque appel"
          % (_mo(sum(total.values())), _mo(total.get("travail", 0))))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
