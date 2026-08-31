r"""Ecrire un fichier sans detruire celui qu'on remplace.

LE DEFAUT -- MESURE LE 29/08/2026
----------------------------------
Neuf ecritures de ce depot suivaient le motif :

    json.dump(objet, open(fichier, "w"), indent=1)

`open(fichier, "w")` TRONQUE le fichier avant que `json.dump` ne serialise
quoi que ce soit. Une seule valeur non serialisable, et le fichier precedent
est remplace par un fragment :

    avant  : {"xt": [[1.0, 2.0]], "n_total": 1}
    apres  : {"xt": [[1.0, 2.0]], "coupe":

C'est pire qu'une suppression : le fichier existe encore, avec la bonne
taille apparente et la bonne date. La relecture suivante leve un
`JSONDecodeError`, que les caches attrapent et traduisent en « recalcul ».

CE QUE CELA POUVAIT COUTER
---------------------------
Ce motif portait :

    restart_state.json         jusqu'a 90 heures d'enrichissement
    hf_grid_cache*.json        225 points, soit 29 heures sur le Moulin Blanc
    doe_cache.json             le plan initial, et son filet de reprise
    hf_custom_cache.json       la grille de points libres

Le commentaire de `reprise.enregistrer` disait deja l'intention :
« une erreur de serialisation ne doit pas emporter le calcul qui vient de se
terminer ». Elle emportait le PRECEDENT, c'est-a-dire exactement ce que le
dump protegeait.

LE REMEDE, ET POURQUOI EN DEUX TEMPS
-------------------------------------
1. serialiser d'ABORD, en memoire : une valeur refusee leve alors sans que le
   fichier ait ete ouvert ;
2. ecrire dans un fichier temporaire, puis `os.replace` : un processus tue
   pendant l'ecriture laisse l'ancien fichier intact. C'est le scenario meme
   pour lequel tous ces filets existent.

`os.replace` est atomique sur Windows comme sur POSIX, tant que la source et
la destination sont sur le meme volume -- ce qui est le cas ici, le temporaire
etant ecrit a cote de sa cible.
"""

import json
import os
import time

#: Nombre de tentatives de renommage, et pause entre deux. Mesure du
#: 29/08/2026 sur `pure_flexion_grille` : UN `os.replace` sur 49 a echoue en
#: `PermissionError [WinError 5]`. Sous Windows, un renommage peut se voir
#: refuser l'acces quand un autre processus -- indexeur, antivirus -- tient
#: brievement la cible. Le filet du cache s'ecrit apres CHAQUE point : 225
#: fois sur le Moulin Blanc, donc un echec isole y est certain.
TENTATIVES = 5
PAUSE = 0.05


def ecrire_json(objet, fichier, indent=1):
    """Ecrit `objet` en JSON dans `fichier`, sans risque pour l'ancien.

    L'ordre des arguments est celui de `json.dump` a dessein : le
    remplacement se lit d'un coup d'oeil sur les neuf sites, et une
    inversion silencieuse est impossible -- un chemin n'est pas serialisable
    en JSON et un dictionnaire n'est pas un chemin.

    Leve ce que leve `json.dumps` -- les appelants savent deja quoi en faire,
    et ils le font desormais SANS avoir perdu le fichier precedent.
    """
    # Serialiser D'ABORD : une valeur refusee leve ici, sans que le fichier
    # ait ete ouvert. C'est ce qui ferme le defaut mesure.
    texte = json.dumps(objet, indent=indent)
    temporaire = fichier + ".tmp"
    try:
        with open(temporaire, "w") as fh:
            fh.write(texte)
        _remplacer(temporaire, fichier, texte)
    except Exception:
        # Ne pas laisser un `.tmp` derriere soi : au prochain passage il
        # ferait croire a une ecriture en cours.
        try:
            if os.path.exists(temporaire):
                os.remove(temporaire)
        except OSError:
            pass
        raise


def _remplacer(temporaire, fichier, texte):
    """Le renommage, avec ce qu'il faut de patience -- puis un repli.

    Le repli ecrit en place le texte DEJA serialise. Il ne peut donc plus
    detruire le fichier sur une valeur refusee -- le defaut du 29/08/2026 --
    et ne laisse exposee que la fenetre ou le processus serait tue pendant
    l'ecriture elle-meme, qui est ce qu'on avait avant.

    Mieux vaut un filet ecrit sans renommage qu'un filet pas ecrit du tout :
    ce fichier est ce qui evite de repayer des heures de solveur.
    """
    derniere = None
    for i in range(TENTATIVES):
        try:
            os.replace(temporaire, fichier)
            return
        except PermissionError as e:
            derniere = e
            time.sleep(PAUSE * (i + 1))
    with open(fichier, "w") as fh:
        fh.write(texte)
    try:
        os.remove(temporaire)
    except OSError:
        pass
    print("[ECRITURE] renommage refuse %d fois (%s) -- ecrit en place. "
          "Le contenu est bon ; seule l'atomicite est perdue."
          % (TENTATIVES, derniere), flush=True)
