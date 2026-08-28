"""
Applique une configuration d'essai a un script AC, pour comparer deux versions.

    python tools/run_comparatif.py --patch <chemin/AC.py> --sortie <dossier>

Repond a la seule question que les tests ne posent pas : le script AC,
DANS SON ENSEMBLE, tourne-t-il encore ? Le harness verifie les modules
extraits un a un ; la baseline analytique n'importe aucun AC ; `--check`
n'execute que l'en-tete. Rien ne couvre les 3 000 lignes de `main`.

Les surcharges appliquees ne changent aucune physique. Elles rendent le run
court et reproductible des deux cotes :

    n_workers_DOE = 1   chemin sequentiel : pas de sous-processus
    save_history        desactive : ~8,8 Mo par point sinon
    print_HF            desactive : la grille de visualisation coute
                        49 appels SOCP, sans rien apporter a la comparaison
    n_max_EFF_points    plafonne, pour borner la duree

Le meme fichier de surcharges est applique aux deux versions : c'est ce qui
rend la comparaison valable.

OU LES SURCHARGES SONT INJECTEES, ET POURQUOI LA QUESTION COMPTE
----------------------------------------------------------------
Elles passent par `CFG.remplace(...)`, insere juste apres le chargement du
fichier d'etude et AVANT que le script n'imprime sa configuration.

La premiere version de cet outil reecrivait les lignes du bloc de liaison,
donc APRES l'impression : le journal du run annoncait `n_max_EFF_points=30`
alors que le run en appliquait 8. Un journal qui ment sur sa configuration
est pire qu'un journal muet -- il rend une comparaison A/B faussement
rassurante. Passer par `remplace()` a deux vertus de plus : la configuration
d'essai est VALIDEE comme une autre, et l'injection survit a la phase 5,
quand le bloc de liaison aura disparu.
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

#: instruction apres laquelle injecter les surcharges : le chargement du
#: fichier d'etude, donc avant l'impression du resume. Elle tient sur deux
#: lignes, d'ou le mode DOTALL et la quantification non gourmande.
ANCRE = re.compile(r"(?ms)^(\s*)CFG = _schema\.charger\(.*?\.toml\"\)\)")


def patcher(chemin, n_max_eff, dossier_sortie, cible=None):
    """Ecrit le script patche et renvoie son chemin.

    `cible` sert aux tests : sans elle, le script est ecrit A COTE de l'AC,
    la ou `launcher.py` le trouvera. Un test qui l'y ecrirait ecraserait --
    et, en nettoyant derriere lui, supprimerait -- le fichier d'un run en
    cours.
    """
    src = io.open(chemin, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in src else "\n"
    texte = src.replace("\r\n", "\n")

    # `dossier_sortie` passe par la MEME voie que les autres surcharges
    # depuis que le reglage agit (28/08/2026).
    surcharges = dict(SURCHARGES, n_max_EFF_points=str(n_max_eff),
                      dossier_sortie=repr(dossier_sortie))
    m = ANCRE.search(texte)
    if not m:
        raise SystemExit(
            "%s : point d'injection introuvable.\n"
            "Attendu un chargement `CFG = _schema.charger(... studies/<etude>.toml)`.\n"
            "Sans lui, la configuration d'essai ne serait pas appliquee -- et la\n"
            "comparaison porterait sur deux runs regles differemment."
            % os.path.basename(chemin))

    indent = m.group(1)
    injection = "\n%s# --- configuration d'essai imposee par tools/run_comparatif.py ---\n" % indent
    injection += "%sCFG = CFG.remplace(\n" % indent
    for nom, valeur in sorted(surcharges.items()):
        injection += "%s    %s=%s,\n" % (indent, nom, valeur)
    injection += "%s)" % indent
    texte = texte[:m.end()] + injection + texte[m.end():]
    appliquees = ["%s=%s" % (n, v) for n, v in sorted(surcharges.items())]

    # Sortie dediee, pour ne pas melanger les figures de deux versions.
    #
    # Cet outil REECRIVAIT la ligne `out_dir_eff = ...` du script, parce
    # que `dossier_sortie` ne servait a rien. Depuis le 28/08/2026 le
    # reglage agit : il suffit de le poser comme les autres, par
    # `CFG.remplace`. Une surcharge de moins qui passe par une expression
    # reguliere sur du code source.
    os.makedirs(dossier_sortie, exist_ok=True)

    cible = cible or os.path.join(os.path.dirname(chemin), "_run_comparatif.py")
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

    cible = patcher(os.path.abspath(args.patch), args.n_max_eff,
                    os.path.abspath(args.sortie))
    print(cible)


if __name__ == "__main__":
    main()
