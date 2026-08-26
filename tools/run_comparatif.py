"""
Applique une configuration d'essai a un script AC, pour comparer deux versions.

    python tools/run_comparatif.py --patch <chemin/AC.py> --sortie <dossier>

Repond a la seule question que les tests ne posent pas : le script AC,
DANS SON ENSEMBLE, tourne-t-il encore ? Le harness verifie les modules
extraits un a un ; la baseline analytique n'importe aucun AC ; `--check`
n'execute que l'en-tete. Rien ne couvre les 3 000 lignes de `main`.

Les surcharges appliquees ne changent aucune physique. Elles rendent le run
court et reproductible des deux cotes :

    path_dir            le dossier de l'etude, au lieu du poste de l'auteur
    n_workers_DOE = 1   chemin sequentiel : pas de sous-processus
    save_history        desactive : ~8,8 Mo par point sinon
    print_HF            desactive : la grille de visualisation coute
                        49 appels SOCP, sans rien apporter a la comparaison
    n_max_EFF_points    plafonne, pour borner la duree

Le meme fichier de surcharges est applique aux deux versions : c'est ce qui
rend la comparaison valable.
"""

import argparse
import io
import os
import re
import sys

SURCHARGES = {
    "n_workers_DOE": "1",
    "save_history": "False",
    "print_HF": "False",
    "print_fullHF": "False",
    "print_3D": "False",
    "print_Pf": "False",
    "do_custom_hf": "False",
    "restart_enrich_only": "False",
    "config_is_identical": "False",     # on veut un vrai calcul, pas un cache
}


def patcher(chemin, dossier_etude, n_max_eff, dossier_sortie):
    src = io.open(chemin, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in src else "\n"
    texte = src.replace("\r\n", "\n")

    appliquees = []
    for nom, valeur in dict(SURCHARGES, n_max_EFF_points=str(n_max_eff)).items():
        motif = re.compile(r"(?m)^(\s+)%s(\s*)=\s*[^\n#]*" % re.escape(nom))
        neuf, n = motif.subn(lambda m: "%s%s%s= %s" % (m.group(1), nom, m.group(2), valeur),
                             texte, count=1)
        if n:
            texte, _ = neuf, appliquees.append("%s=%s" % (nom, valeur))
        else:
            print("  [patch] %s : introuvable, ignore" % nom, file=sys.stderr)

    # chemins : dossier de l'etude et sortie dediee
    texte = re.sub(r"(?m)^(\s+)path_dir\s*=\s*r?['\"][^'\"]*['\"]",
                   lambda m: "%spath_dir = r\"%s\"" % (m.group(1), dossier_etude), texte, count=1)
    texte = re.sub(r"(?m)^(\s+)out_dir_eff\s*=\s*.*",
                   lambda m: "%sout_dir_eff = r\"%s\"" % (m.group(1), dossier_sortie),
                   texte, count=1)
    os.makedirs(dossier_sortie, exist_ok=True)

    cible = os.path.join(os.path.dirname(chemin), "_run_comparatif.py")
    io.open(cible, "w", encoding="utf-8", newline="").write(
        texte.replace("\n", nl) if nl == "\r\n" else texte)
    print("  [patch] %s -> %s" % (os.path.basename(chemin), os.path.basename(cible)))
    print("  [patch] %s" % ", ".join(sorted(appliquees)))
    return cible


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--patch", required=True, help="script AC a preparer")
    ap.add_argument("--sortie", required=True, help="dossier des figures")
    ap.add_argument("--n-max-eff", type=int, default=8,
                    help="plafond de points d'enrichissement (defaut 8)")
    args = ap.parse_args()

    etude = os.path.dirname(os.path.abspath(args.patch))
    cible = patcher(os.path.abspath(args.patch), etude, args.n_max_eff,
                    os.path.abspath(args.sortie))
    print(cible)


if __name__ == "__main__":
    main()
