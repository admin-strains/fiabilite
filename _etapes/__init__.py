r"""Les actions d'une etude de fiabilite, lancables SEPAREMENT.

POURQUOI CE DECOUPAGE
----------------------
Jusqu'au 26/08/2026, une etude etait un bloc : `AC3_<etude>.py`, 3 000 lignes,
tout ou rien. Refaire une figure demandait de rejouer le plan d'experiences et
l'enrichissement -- 47 heures sur le Moulin Blanc -- parce que rien ne
separait ce qui coute des appels solveur de ce qui n'en coute aucun.

Pire, la separation n'existait meme pas dans les noms : SEPT des onze
fonctions `print_*` pouvaient declencher un appel au solveur.

    print_planche_EFF -> _hf_from_custom_points -> run_HF

Une fonction dont le nom dit « imprime » lancait 225 appels, soit 29 heures.
C'est ce qui a fait annoncer a tort, le matin meme, que la grille arrivait en
dernier et qu'on pouvait l'interrompre sans rien perdre.

LES CINQ ACTIONS
-----------------
Chacune declare ce qu'elle lit, ce qu'elle ecrit, et CE QU'ELLE COUTE :

    action        lit                     ecrit           appels solveur
    ----------------------------------------------------------------------
    plan          config                  points          n0
    enrichir      config + points         points          <= n_max_EFF_points
    grille        config + geometrie      points          n_grid_hf^2
    analyser      config + points         resultats       0
    figurer       config + points + res.  images          0  -- GARANTI

Le zero de `figurer` n'est pas une intention : un test de fermeture
transitive interdit a toute fonction de trace d'atteindre le solveur. Ce qui
rend la surprise du 26/08 structurellement impossible.

CE QUI CIRCULE
---------------
Un seul artefact cher : le journal des points (`points.py`). Le metamodele
n'est PAS persiste -- il est reconstruit depuis les points, et c'est ce qui
permet de refaire une analyse ou une figure sans rappeler le solveur.

    ne persister que ce qui coute un appel solveur ; recalculer le reste.

ETAT D'AVANCEMENT (phase 0)
----------------------------
    points.py     l'artefact                        FAIT, 18 tests
    plan.py       ...                               a venir
    enrichir.py   ...                               a venir
    grille.py     ...                               a venir
    analyser.py   ...                               a venir
    figurer.py    ...                               a venir

Les caches actuels restent en place tant qu'une action n'est pas migree, avec
temoin : le code de production sert d'oracle a sa propre refonte
(`tools/extraction_temoin.py`).
"""

#: Les actions, dans l'ordre du flux, avec leur cout en appels solveur.
#: Sert au journal de run et aux tests d'exhaustivite.
ACTIONS = (
    ("plan",      "plan d'experiences initial",        "n0"),
    ("enrichir",  "enrichissement EFF",                "<= n_max_EFF_points"),
    ("grille",    "grille haute fidelite (figures)",   "n_grid_hf^2"),
    ("analyser",  "FORM multimodal + tirage d'importance", "0"),
    ("figurer",   "figures",                           "0"),
)

#: Les actions qui n'ont PAS le droit d'appeler le solveur. Verifie par un
#: test de fermeture transitive sur le graphe d'appel.
SANS_SOLVEUR = ("analyser", "figurer")
