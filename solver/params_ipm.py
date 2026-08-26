r"""Reglages du point interieur qui se decident depuis la configuration.

CE MODULE N'IMPORTE PAS DIGITAL STRUCTURE, et c'est tout son interet.

`digital_structure.py` ne s'importe pas sans licence, sans GPU et hors
Python 3.10 : rien de ce qui y vit ne peut etre teste sur un poste ordinaire ni
en CI. Le 26/08/2026, cette impossibilite a laisse passer un `AttributeError`
sur TOUTE evaluation (voir `tests/test_88_contrat_solveur_statique.py`).

Le choix du solveur lineaire est de la manipulation de dictionnaires : il n'a
aucun besoin du solveur pour etre verifie. Il vit donc ici, teste comme
n'importe quel code.
"""

#: Solveur lineaire du point interieur : nom -> valeur de `IPARM0[21]`.
#: Les valeurs viennent du commentaire des `InitSolver.py` :
#: « PT INT (1 = MKL PARDISO, 3 = MUMPS, 4 = CuDss) ».
#:
#: Pardiso (valeur 1, et le bloc `MyPardiso_params`) n'est PAS propose : il est
#: deprecie (Agnes, 26/08/2026). Les blocs de parametres Pardiso restent
#: transmis au solveur -- on ne touche pas aux `InitSolver.py` -- mais aucune
#: valeur proposee ici ne les selectionne.
#:
#: Doit rester aligne sur `_config/schema.py:SOLVEURS_LINEAIRES`, ce qu'un test
#: verifie.
SOLVEURS_LINEAIRES = {"mumps": 3, "cudss": 4}

#: L'indice d'`IPARM0` qui porte ce choix pour l'approche cinematique. Le
#: pendant statique est `IPARM0[11]` : laisse tel quel, l'analyse a la rupture
#: est cinematique et le solveur statique n'intervient pas dans la chaine.
IPARM0_SOLVEUR_LINEAIRE = 21


def imposer_solveur_lineaire(cinematic_params, nom, tracer=print):
    """Remplace la valeur d'`IPARM0[21]` dans `cinematic_params`, en place.

    Retourne la liste, pour pouvoir chainer.

    POURQUOI CE PARAMETRE EXISTE -- constat du 26/08/2026
    ------------------------------------------------------
    Le choix vivait uniquement dans l'`InitSolver.py` de l'etude, en clair
    mais sans que rien ne le remonte. Les deux etudes du depot avaient
    DIVERGE en silence :

        pure_flexion/InitSolver.py    IPARM0[21] = 3   MUMPS
        Moulinblanc/InitSolver.py     IPARM0[21] = 4   CuDss

    C'est aussi la que se separent les deux reproductibilites mesurees --
    2,9e-11 sur la flexion pure, 7,7e-06 sur le Moulin Blanc. Cela ne prouve
    rien : les deux modeles n'ont ni la meme taille ni le meme
    conditionnement. Mais tant que le backend n'etait pas un parametre
    visible, l'hypothese n'etait meme pas formulable.

    `nom = None` ne touche a rien : c'est le comportement d'avant l'existence
    du parametre, et il reste le defaut.

    NE CREE PAS l'entree si elle manque. Une `InitSolver.py` sans
    `IPARM0[21]` a une forme qu'on ne connait pas ; ajouter silencieusement un
    reglage a cote des autres serait pire que s'arreter.
    """
    if nom is None:
        return cinematic_params
    if nom not in SOLVEURS_LINEAIRES:
        raise ValueError(
            "solveur_lineaire=%r inconnu (attendu : %s). Pardiso est deprecie "
            "et n'est pas propose."
            % (nom, ", ".join(sorted(SOLVEURS_LINEAIRES))))
    voulu = SOLVEURS_LINEAIRES[nom]
    for entree in cinematic_params:
        if entree.get("table") == "IPARM0" \
                and entree.get("indices") == [IPARM0_SOLVEUR_LINEAIRE]:
            avant = entree["value"]
            entree["value"] = voulu
            if tracer is not None:
                if avant != voulu:
                    tracer("  [SOLVEUR LINEAIRE] %s : IPARM0[%d] %s -> %d"
                           % (nom, IPARM0_SOLVEUR_LINEAIRE, avant, voulu))
                else:
                    tracer("  [SOLVEUR LINEAIRE] %s : IPARM0[%d] = %d, deja la "
                           "valeur de l'InitSolver.py"
                           % (nom, IPARM0_SOLVEUR_LINEAIRE, voulu))
            return cinematic_params
    raise ValueError(
        "solveur_lineaire=%r demande, mais cinematic_params ne contient aucune "
        "entree IPARM0 indices=[%d] a remplacer. Verifier l'InitSolver.py de "
        "l'etude." % (nom, IPARM0_SOLVEUR_LINEAIRE))
