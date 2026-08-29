r"""Aucune liaison `nom = CFG.nom` que plus rien ne lit.

CE QU'ON A TROUVE -- 29/08/2026
--------------------------------
Le bloc de liaison de chaque etude traduit le fichier d'etude en variables
locales, une ligne par reglage. Quand un morceau de code part dans un module,
son lecteur part avec lui -- et la ligne de liaison reste.

    pure_flexion   12 liaisons mortes sur 61
    Moulin Blanc   14 liaisons mortes sur 62

Cinq etaient devenues mortes le jour meme (`n_max_EFF_points`, `eps_taylor`,
`print_DOE`, `print_gepck_calls`, `THETA_MIN_KRG`) : leurs lecteurs avaient
suivi la boucle EFF, le plan initial et l'ajustement. Les autres l'etaient
depuis plus longtemps -- dont QUATRE des sept drapeaux de modele, sous un
commentaire qui expliquait soigneusement pourquoi ils sont derives.

CE QUE CELA COUTE
------------------
Rien a l'execution. Mais une liaison morte affirme au lecteur qu'un reglage
sert ICI, et c'est faux -- c'est le meme mensonge, en plus petit, que celui
de `dossier_sortie`, affiche par le resume de configuration alors qu'il ne
faisait rien.

POURQUOI CE TEST EST CONSCIENT DES PORTEES
-------------------------------------------
La premiere version de cette sonde collectait tous les `Name` en lecture du
fichier. Elle a donc declare `q = CFG.q` VIVANT, a cause de :

    **{f"dg_{q}": _wSOL[_k].get(f"dg_{q}") for q in params_names}

-- une variable de comprehension, qui a sa propre portee et n'a aucun rapport
avec le reglage `q`. Un faux NEGATIF, celui-la : la sonde disait « propre »
sur une liaison morte. C'est `symtable` qui tranche, parce qu'il connait les
portees ; `ast` seul ne les connait pas.
"""

import ast
import io
import os
import symtable

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)

ETUDES = ("pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py")


def _liaisons(arbre):
    """Les lignes `nom = CFG.<attr>` du flux principal."""
    main = [n for n in arbre.body if isinstance(n, ast.If)][0]
    liaisons = {}
    for n in main.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Attribute)
                and isinstance(n.value.value, ast.Name)
                and n.value.value.id == "CFG"):
            liaisons[n.targets[0].id] = n.lineno
    return liaisons


def _noms_vivants(src, nom_fichier):
    """Les noms du module qu'une portee lit VRAIMENT.

    Deux sources, et il faut les deux :

    * `symtable` dit quels noms sont references au niveau du module, et quels
      noms une portee imbriquee lit SANS les lier -- c'est-a-dire en allant
      les chercher au module. C'est lui qui ecarte les variables de
      comprehension, qui portent le meme nom mais vivent ailleurs ;
    * l'arbre dit ou, pour ecarter la ligne de liaison elle-meme : une
      affectation n'est pas une lecture, et un nom qui n'apparait que la est
      mort meme si `symtable` le voit.
    """
    st = symtable.symtable(src, nom_fichier, "exec")
    vus = set()

    def _descendre(bloc, racine):
        for s in bloc.get_symbols():
            if racine and s.is_referenced():
                vus.add(s.get_name())
            elif not racine and s.is_global() and s.is_referenced():
                vus.add(s.get_name())
        for enfant in bloc.get_children():
            _descendre(enfant, False)

    _descendre(st, True)
    return vus


def _mortes(src, nom_fichier):
    arbre = ast.parse(src, nom_fichier)
    liaisons = _liaisons(arbre)
    vivants = _noms_vivants(src, nom_fichier)
    ailleurs = set()
    for n in ast.walk(arbre):
        if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                and n.id in liaisons and n.lineno != liaisons[n.id]):
            ailleurs.add(n.id)
    return {nom: ligne for nom, ligne in liaisons.items()
            if nom not in (vivants & ailleurs)}


@pytest.mark.parametrize("script", ETUDES)
def test_aucune_liaison_de_configuration_n_est_morte(script):
    """Une ligne `nom = CFG.nom` que rien ne lit doit partir avec son lecteur."""
    chemin = os.path.join(_REPO, script)
    src = io.open(chemin, encoding="utf-8", errors="replace").read()
    mortes = _mortes(src, chemin)
    assert not mortes, (
        "%s : %d liaison(s) que plus rien ne lit.\n  %s\n"
        "Elles affirment au lecteur qu'un reglage sert ici. Si son lecteur a "
        "suivi du code extrait, la ligne doit partir avec lui."
        % (script, len(mortes),
           "\n  ".join("l.%d  %s" % (l, n) for n, l in sorted(
               mortes.items(), key=lambda kv: kv[1]))))


@pytest.mark.parametrize("script", ETUDES)
def test_la_sonde_voit_les_liaisons(script):
    """Une sonde qui ne trouve aucune liaison ne prouve rien.

    Si le bloc de liaison change de forme -- la phase 5 le fera disparaitre --
    ce test tombe et demande d'adapter la sonde, au lieu de la laisser
    approuver un fichier qu'elle ne sait plus lire.
    """
    chemin = os.path.join(_REPO, script)
    src = io.open(chemin, encoding="utf-8", errors="replace").read()
    assert len(_liaisons(ast.parse(src, chemin))) > 20


# --------------------------------------------------------------------------- #
# LES DENTS : la sonde doit MORDRE, et sur le bon cas                          #
# --------------------------------------------------------------------------- #
_MORTE = '''
if __name__ == '__main__':
    CFG = charger()
    vivant = CFG.vivant
    mort = CFG.mort
    print(vivant)
'''

_MASQUEE = '''
if __name__ == '__main__':
    CFG = charger()
    q = CFG.q

    def ailleurs(noms):
        return {f"dg_{q}": 0 for q in noms}
'''

_VIVANTE_PAR_UNE_FONCTION = '''
if __name__ == '__main__':
    CFG = charger()
    seuil = CFG.seuil

    def juger(x):
        return x < seuil
'''


def test_une_liaison_jamais_relue_est_vue():
    assert sorted(_mortes(_MORTE, "<essai>")) == ["mort"]


def test_une_variable_de_comprehension_ne_fait_pas_vivre_une_liaison():
    """LE faux negatif du 29/08/2026, fige en test.

    `for q in noms` lie `q` dans la portee de la comprehension. Une sonde qui
    ne compte que les `Name` du fichier voit une lecture et declare la liaison
    vivante -- elle est morte.
    """
    assert sorted(_mortes(_MASQUEE, "<essai>")) == ["q"], (
        "la sonde a repris son angle mort : une variable de comprehension "
        "homonyme fait passer une liaison morte pour vivante.")


def test_une_liaison_lue_par_une_fonction_imbriquee_est_vivante():
    """Le pendant : la sonde ne doit pas non plus condamner ce qui vit.

    C'est la forme NORMALE dans ces etudes -- toutes les fonctions du flux
    sont imbriquees et lisent les liaisons par fermeture.
    """
    assert _mortes(_VIVANTE_PAR_UNE_FONCTION, "<essai>") == {}
