r"""Les variables aleatoires d'une etude : declarees en donnees, assemblees ici.

CE QUE CE MODULE REMPLACE
--------------------------
Chaque etude portait son catalogue en Python :

    PARAM_CONFIG_CAD = {
        'fc': {'sens': {"param": "COMPRESSIVE_STRENGTH", "solids": ["Block1"],
                        "region_key": "fc"},
               'loi': loi_fc, 'args': (48, 0.12)},
        'fy': {'sens': {"param": "YIELD_STRENGTH", "rebars": rebar_names,
                        "region_key": "fy"},
               'loi': loi_fy, 'args': (550, None)},
    }

Sept lignes, mais les seules du script qui nommaient encore `fc` et `fy` --
mesure du 02/09/2026 : dans les 1 097 lignes de l'etude de flexion, ces deux
noms n'apparaissent qu'ici et dans la surcouche analytique. Tout le reste de
la chaine ne connait que `PARAM_CONFIG`.

Le fichier d'etude les declare maintenant :

    [variables.fc]
    loi     = "fc"
    args    = [48, 0.12]
    param   = "COMPRESSIVE_STRENGTH"
    solides = ["Block1"]

    [variables.fy]
    loi   = "fy"
    args  = [550]
    param = "YIELD_STRENGTH"

et l'etude ne fournit que ce qui demande du code -- la SELECTION :

    PARAM_CONFIG = _variables.construire(
        CFG, elements={"fy": {"armatures": rebar_names}})

C'est l'option B', arbitree par Agnes le 02/09/2026. La frontiere : un NOM
est une donnee et va dans le TOML ; une SELECTION (« les armatures de nuance
fyd1 », 13 858 noms) est une propriete du modele et reste du code, dans
`_model/selection.py`.

CE QUI EST GAGNE, AU-DELA DE LA FORME
--------------------------------------
* `region_key` n'est plus declarable : il vaut le nom de la variable. Les
  deux etudes l'ecrivaient a la main -- identique au nom dans les quatre cas
  --, et un doublon aurait fait ecrire deux variables dans la meme region.
  Le deriver rend le doublon IMPOSSIBLE, la ou `coherence` devait le guetter.
* une loi mal orthographiee est refusee au chargement, avec la liste des
  choix, plutot qu'au premier tirage.
* une variable qui ne designe AUCUN element est refusee. Sa region de
  sensibilite serait vide, donc son gradient nul -- indiscernable d'une
  insensibilite physique. Meme classe de defaut que la selection vide de
  `_model/selection.py`.
"""

#: Les clefs de designation, du fichier d'etude vers celles que le solveur
#: attend dans `sens`. Le TOML est la surface utilisateur, donc en francais ;
#: les noms internes restent ceux de Digital Structure.
DESIGNATIONS = {
    "armatures": "rebars",
    "solides": "solids",
    "cas_de_charge": "load_case",
}

#: Reglages de la sensibilite qui ne designent pas des elements.
REGLAGES = {"axe": "axis"}


def porte_sur_un_chargement(catalogue):
    """Les variables du catalogue qui agissent sur un CAS DE CHARGE.

    Remplace le test `set(params_names) <= set(PARAM_CONFIG_CAD.keys())` que
    l'etude de flexion pure ecrivait : la distinction entre variables de
    geometrie et de chargement n'est plus portee par deux dictionnaires mais
    DERIVEE de la designation. `patch_params`, lui, n'en a jamais eu besoin --
    il cherche dans les deux fichiers du modele.
    """
    return [nom for nom, decl in catalogue.items()
            if "load_case" in decl["sens"]]


def construire(cfg, elements=None, tracer=None):
    """`PARAM_CONFIG` a partir de la declaration du TOML et des selections.

    `elements` : `{nom_de_variable: {"armatures": [...], ...}}`, ce que
    l'etude a selectionne sur le modele. Il l'emporte sur les noms litteraux
    du TOML, et s'y ajoute quand ils portent des clefs differentes.

    L'ORDRE est celui du fichier d'etude : `params_names` en derive, et avec
    lui l'ordre des colonnes du plan d'experiences. Le changer changerait des
    resultats sans le dire.
    """
    from lois import loi_nommee                             # noqa: PLC0415

    elements = dict(elements or {})
    inconnues = sorted(set(elements) - set(cfg.variables))
    if inconnues:
        raise ValueError(
            "elements designes pour %s, qui n'est pas declaree dans "
            "`[variables]` du fichier d'etude. Declarees : %s."
            % (", ".join(inconnues), ", ".join(cfg.variables) or "aucune"))
    if not cfg.variables:
        raise ValueError(
            "aucune variable aleatoire declaree : le fichier d'etude doit "
            "porter au moins une table `[variables.<nom>]`.")

    catalogue, lignes = {}, []
    for nom, decl in cfg.variables.items():
        sens = {"param": decl["param"], "region_key": nom}
        for clef_toml, clef_solveur in REGLAGES.items():
            if clef_toml in decl:
                sens[clef_solveur] = decl[clef_toml]

        designes = 0
        for clef_toml, clef_solveur in DESIGNATIONS.items():
            valeur = elements.get(nom, {}).get(clef_toml, decl.get(clef_toml))
            if valeur is None:
                continue
            sens[clef_solveur] = valeur
            designes += 1 if isinstance(valeur, str) else len(valeur)
        if not designes:
            raise ValueError(
                "la variable %r ne designe AUCUN element du modele. Ecrire "
                "des noms litteraux dans `[variables.%s]` (%s) ou les "
                "selectionner dans l'etude et les passer par `elements`.\n"
                "Une region de sensibilite vide ne leve rien : son gradient "
                "vaut zero, ce qui ne se distingue pas d'une insensibilite "
                "physique." % (nom, nom, ", ".join(sorted(DESIGNATIONS))))

        catalogue[nom] = {"sens": sens,
                          "loi": loi_nommee(decl["loi"]),
                          "args": tuple(decl["args"])}
        lignes.append("  %-10s loi_%-14s args=%-16s %s sur %d element(s)"
                      % (nom, decl["loi"], tuple(decl["args"]),
                         decl["param"], designes))

    if tracer is not None:
        tracer("[variables] %d variable(s) aleatoire(s) :\n%s"
               % (len(catalogue), "\n".join(lignes)))
    return catalogue
