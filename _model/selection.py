r"""Designer des elements du modele par ce qu'ils SONT, pas par leur nom.

POURQUOI CE MODULE
-------------------
Une variable aleatoire porte sur des elements du modele : des armatures, des
solides, un cas de charge. Les deux etudes les selectionnaient par une
expression reguliere ecrite sur place :

    rebar_names  = re.findall(r"REBAR\('([^']+)'", _cad_txt)
    group1_names = re.findall(r"REBAR\('([^']+)',[^\n]*GRADE=fyd1,", _cad_txt)

Trois choses avec cela.

D'abord, ces expressions sont illisibles pour qui n'ecrit pas de Python -- et
la prochaine etude sera ecrite par un ingenieur, pas par un codeur. Ici, la
meme selection se lit :

    armatures(cad)                    # toutes
    armatures(cad, grade="fyd1")      # celles de nuance fyd1

Ensuite, `re.findall` rend une LISTE VIDE quand il ne trouve rien. Une region
de sensibilite vide ne provoque aucune erreur : le gradient correspondant
vaut zero, et un zero de gradient ne se distingue pas d'une insensibilite
physique. Une faute de frappe dans `GRADE=fyd1` -- `fdy1`, `fyd_1` -- rendait
donc un resultat plausible et faux. Ici, une selection vide est REFUSEE, et
le message dit ce qui a ete cherche et quelles valeurs existent.

Enfin le decoupage : selectionner des elements est une propriete du MODELE,
pas de l'etude. C'est la seule part de la declaration des variables qui ait
besoin de code -- le reste (les lois, leurs parametres, le nom du parametre
solveur) est de la donnee, et vit dans le fichier `.toml` de l'etude.
Arbitrage d'Agnes du 02/09/2026, option B'.

CE QUE CE MODULE LIT
---------------------
Du TEXTE : `dsCad.txt` et `dsLoad.txt` sont les fichiers du modele, et ce
sont des scripts Python que Digital Structure execute. On n'a donc besoin ni
de Digital Structure ni d'une licence pour selectionner -- c'est ce qui rend
ces fonctions testables partout, comme `solver/analytique.py`.

Mesure du 02/09/2026 : dans les deux modeles versionnes, chaque appel
`REBAR(...)` tient sur UNE ligne et s'y termine -- 15 346 sur 15 346 pour le
Moulin Blanc, 24 sur 24 pour la flexion pure. L'analyse est donc faite ligne
a ligne, comme l'etait l'expression reguliere d'origine.
"""

import re

#: Les appels du modele que l'on sait selectionner, et dans quel fichier
#: chacun se trouve. La clef est le nom de la fonction du modele.
APPELS = {
    "REBAR": "dsCad.txt",
    "BLOCK": "dsCad.txt",
    "LOAD_CASE": "dsLoad.txt",
}

#: Noms de criteres acceptes en francais sans accent, vers l'attribut du
#: modele. Tout autre critere est majuscule tel quel : `grade` -> `GRADE`.
ALIAS = {
    "diametre": "DIAMETER",
    "distance": "DISTANCE",
    "nuance": "GRADE",
}


def _appels(texte, fonction):
    """Les `(nom, attributs)` de chaque appel `FONCTION('nom', ...)`.

    `attributs` est le dictionnaire des `CLEF=valeur` de la MEME ligne. Les
    valeurs sont rendues telles qu'ecrites dans le modele, sans conversion :
    `GRADE=fyd1` donne `'fyd1'`, `DIAMETER=5` donne `'5'`. Comparer des
    chaines evite d'avoir a deviner si `phi` est un nombre ou une variable du
    modele -- dans la flexion pure, `DIAMETER=phi` est une variable.
    """
    debut = re.compile(re.escape(fonction) + r"\(\s*'([^']+)'")
    attribut = re.compile(r"\b([A-Z][A-Z_]*)\s*=\s*([^,)\s]+)")
    out = []
    for ligne in texte.splitlines():
        m = debut.search(ligne)
        if m is None:
            continue
        out.append((m.group(1), dict(attribut.findall(ligne[m.end():]))))
    return out


def _selectionner(texte, fonction, criteres):
    attendus = {ALIAS.get(k.lower(), k.upper()): str(v)
                for k, v in criteres.items()}
    appels = _appels(texte, fonction)
    if not appels:
        raise ValueError(
            "aucun appel %s('...') dans le modele : soit ce n'est pas le bon "
            "fichier, soit les elements n'y sont pas NOMMES -- sans nom, ils "
            "ne peuvent pas etre designes a une region de sensibilite."
            % fonction)

    presents = set()
    for _, attrs in appels:
        presents |= set(attrs)
    inconnus = sorted(set(attendus) - presents)
    if inconnus:
        raise ValueError(
            "critere(s) %s inconnu(s) des appels %s du modele. Attributs "
            "presents : %s." % (", ".join(inconnus), fonction,
                                ", ".join(sorted(presents))))

    retenus = [nom for nom, attrs in appels
               if all(attrs.get(k) == v for k, v in attendus.items())]
    if not retenus:
        valeurs = {k: sorted({a.get(k) for _, a in appels if k in a})
                   for k in attendus}
        raise ValueError(
            "selection VIDE : aucun %s ne verifie %s.\n"
            "Valeurs presentes dans le modele : %s.\n"
            "Une region de sensibilite vide ne leve rien -- son gradient "
            "vaut zero, ce qui ne se distingue pas d'une insensibilite "
            "physique. D'ou ce refus."
            % (fonction, attendus, valeurs))
    return retenus


def armatures(cad, **criteres):
    """Les noms des armatures du modele, dans l'ordre du fichier.

        armatures(cad)                  toutes
        armatures(cad, grade="fyd1")    celles de nuance fyd1
        armatures(cad, diametre="5")    celles de diametre 5

    L'ORDRE EST CELUI DU MODELE, et il compte : c'est celui que produisait
    `re.findall`, et les regions de sensibilite le suivent. Le changer
    changerait des resultats sans le dire.

    Leve si la selection est vide -- voir l'en-tete du module.
    """
    return _selectionner(cad, "REBAR", criteres)


def solides(cad, **criteres):
    """Les noms des solides (`BLOCK`) du modele, dans l'ordre du fichier."""
    return _selectionner(cad, "BLOCK", criteres)


def cas_de_charge(load, **criteres):
    """Les noms des cas de charge (`LOAD_CASE`), dans l'ordre du fichier."""
    return _selectionner(load, "LOAD_CASE", criteres)


def nuances(cad):
    """Les nuances d'acier presentes, et combien d'armatures chacune.

    Sert a ECRIRE une etude : on regarde ce que le modele contient avant de
    declarer ses variables. Mesure du 02/09/2026 -- Moulin Blanc :
    `{'fyd1': 13858, 'fyd2': 1488}` ; flexion pure : `{'fyd': 24}`.
    """
    compte = {}
    for _, attrs in _appels(cad, "REBAR"):
        nuance = attrs.get("GRADE")
        if nuance is not None:
            compte[nuance] = compte.get(nuance, 0) + 1
    return compte
