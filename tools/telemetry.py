"""
Instrumentation de la chaine de fiabilite -- journal de run comparable.

Raison d'etre
-------------
Le chantier de restructuration consiste a deplacer beaucoup de code sans
changer un seul resultat. Pour tenir cette promesse il faut pouvoir, apres
chaque modification, repondre precisement a : **qu'est-ce qui a bouge, et de
combien ?** -- pas seulement « le beta final est-il le meme ».

Ce module enregistre un **journal de run** (JSONL, une ligne par evenement)
qui trace la chaine etape par etape : plan d'experiences, appels au solveur,
coefficients du metamodele, points d'enrichissement, iterations FORM,
resultat du tirage d'importance. Deux journaux se comparent ensuite avec
`baseline_compare.py`, qui dit ou et de combien ils divergent.

Trois principes
---------------
1. **Ne pas modifier le code teste.** Les fonctions de `_lib` sont
   instrumentees par enveloppement au moment de l'execution
   (`instrument_lib`), pas par edition. L'instrumentation survit donc a un
   deplacement de code, et le simple fait qu'elle continue a accrocher est
   une verification du contrat d'API.

2. **Empreinte ET statistiques.** Chaque tableau est enregistre avec son
   hachage (identique au bit pres, oui ou non) *et* ses statistiques
   (de combien a-t-il bouge). Un hachage seul ne distingue pas une refonte
   qui casse tout d'un dernier bit d'arrondi.

3. **Une baseline porte son propre bruit.** `baseline_run.py --repeat N`
   mesure la dispersion d'un run a l'autre. Sans ce plancher de bruit, on ne
   sait pas distinguer une derive due a une modification d'une fluctuation
   normale.

Determinisme (mesure le 25/08/2026)
-----------------------------------
* OpenTURNS part d'une graine par defaut **fixe** : deux processus donnent
  la meme suite. Ce determinisme est donc *implicite* et sensible a l'ordre
  des tirages -- un refactoring qui deplace un appel decale toute la suite.
  `pin_seeds()` le rend explicite ; `SetSeed(0)` reproduit exactement l'etat
  par defaut, la mise en place est donc neutre.
* `numpy.random` global n'est PAS reproductible, et il est utilise a un seul
  endroit : `branche3.uq_Kriging_helper_create_randIdx`, uniquement quand
  K != 1. La configuration actuelle est en LOO (K=1), donc cet alea est
  dormant -- mais il se reveillera si quelqu'un passe en K-fold.
* `_parallel_is` seme chaque bloc par son indice : reproductible.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: au-dela de cette taille, on ne stocke que hachage + statistiques
MAX_VALEURS_STOCKEES = 256


# --------------------------------------------------------------------------- #
# Empreintes                                                                   #
# --------------------------------------------------------------------------- #
def fingerprint(value):
    """Empreinte comparable d'une valeur : hachage exact + statistiques."""
    if value is None:
        return {"kind": "none"}

    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return {"kind": "scalar", "value": float(value)}

    if isinstance(value, str):
        return {"kind": "str", "value": value}

    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted(map(str, value.keys()))}

    arr = np.asarray(value)
    if arr.dtype == object:
        return {"kind": "object", "repr": repr(value)[:200]}

    a = arr.astype(float, copy=False).ravel()
    fp = {
        "kind": "array",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "hash": hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16],
        "n": int(a.size),
    }
    fini = a[np.isfinite(a)]
    if fini.size:
        fp["stats"] = {
            "min": float(fini.min()), "max": float(fini.max()),
            "mean": float(fini.mean()), "l2": float(np.linalg.norm(fini)),
        }
    if not np.all(np.isfinite(a)):
        fp["non_finis"] = int(np.sum(~np.isfinite(a)))
    if a.size <= MAX_VALEURS_STOCKEES:
        fp["values"] = [None if not np.isfinite(v) else float(v) for v in a]
    return fp


# --------------------------------------------------------------------------- #
# Journal                                                                      #
# --------------------------------------------------------------------------- #
class Journal:
    """Journal JSONL d'un run. Une ligne = un evenement."""

    def __init__(self, nom, outdir=None, config=None, note=None):
        self.nom = nom
        self.seq = 0
        self.t0 = time.perf_counter()
        self.outdir = outdir or os.path.join(REPO, "baselines", nom)
        os.makedirs(self.outdir, exist_ok=True)
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.outdir, "run_%s.jsonl" % horodatage)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._compteurs = {}
        self._emit("header", "run", {
            "nom": nom,
            "horodatage": horodatage,
            "note": note,
            "config": config or {},
            "environnement": environnement(),
        })

    # -- ecriture ----------------------------------------------------------
    def _emit(self, kind, stage, payload, name=None):
        self.seq += 1
        ligne = {"seq": self.seq, "kind": kind, "stage": stage,
                 "t": round(time.perf_counter() - self.t0, 6)}
        if name is not None:
            ligne["name"] = name
        ligne.update(payload)
        self._fh.write(json.dumps(ligne, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def probe(self, stage, **valeurs):
        """Enregistre des grandeurs nommees a une etape de la chaine."""
        for nom, v in sorted(valeurs.items()):
            occ = self._compteurs.setdefault((stage, nom), 0)
            self._compteurs[(stage, nom)] = occ + 1
            self._emit("probe", stage, {"occ": occ, "fp": fingerprint(v)}, name=nom)

    def event(self, stage, message, **extra):
        self._emit("event", stage, {"message": message, "extra": extra})

    def close(self, **resume):
        self._emit("footer", "run", {"duree": round(time.perf_counter() - self.t0, 3),
                                     "resume": {k: fingerprint(v) for k, v in sorted(resume.items())},
                                     "n_evenements": self.seq})
        self._fh.close()
        return self.path


# --------------------------------------------------------------------------- #
# Environnement et determinisme                                                #
# --------------------------------------------------------------------------- #
def environnement():
    """Tout ce qui peut expliquer un ecart entre deux journaux."""
    paquets = {}
    for m in ("numpy", "scipy", "openturns", "smt", "sklearn", "matplotlib"):
        try:
            paquets[m] = getattr(__import__(m), "__version__", "?")
        except Exception:
            paquets[m] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                capture_output=True, text=True, timeout=20).stdout.strip()
        sale = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                   capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:
        commit, sale = None, None
    return {"python": sys.version.split()[0], "plateforme": platform.platform(),
            "paquets": paquets, "git_commit": commit, "git_modifie": sale}


def pin_seeds(journal=None, ot_seed=0, np_seed=20260825):
    """
    Rend le determinisme EXPLICITE.

    `SetSeed(0)` reproduit exactement l'etat par defaut d'OpenTURNS (verifie),
    la mise en place ne change donc aucun resultat -- elle protege seulement
    d'un changement de defaut dans une version future.
    """
    poses = {}
    try:
        import openturns as ot
        ot.RandomGenerator.SetSeed(ot_seed)
        poses["openturns"] = ot_seed
    except Exception:
        poses["openturns"] = None
    np.random.seed(np_seed)
    poses["numpy_global"] = np_seed
    if journal is not None:
        journal.event("seeds", "graines posees", **poses)
    return poses


# --------------------------------------------------------------------------- #
# Instrumentation de _lib par enveloppement                                    #
# --------------------------------------------------------------------------- #
#: fonctions suivies -> (module, nom, arguments traces, mode)
#
#  mode "detaille"  : un evenement par appel. Reserve aux fonctions peu
#                     appelees et a fort contenu (les ajustements).
#  mode "agrege"    : un seul evenement en fin de run, portant le nombre
#                     d'appels et un condensat roulant de toutes les entrees
#                     et sorties.
#
#  Pourquoi l'agregation : les predictions sont appelees des milliers de fois
#  (chaque iteration FORM, chaque point de grille). Les tracer une a une
#  produit un journal de plusieurs Mo, et surtout un journal FRAGILE : si une
#  modification change le nombre d'iterations FORM, l'alignement par indice
#  d'occurrence saute et le comparateur ne sait plus quoi comparer a quoi.
#  Le condensat roulant, lui, repond exactement a la bonne question : la SUITE
#  des appels est-elle la meme ?
CIBLES_LIB = [
    ("branche1", "fit_pck", ("X", "Y"), "detaille"),
    ("branche1", "fit_gepck", ("X", "Y_aug"), "detaille"),
    ("branche1", "predict_pck", ("X_test",), "agrege"),
    ("branche1", "predict_gepck", ("X_test",), "agrege"),
    ("branche1", "predict_gradient_gepck", ("X_test",), "agrege"),
]


def _resume_modele(fm):
    """Ce qu'on veut voir bouger dans un metamodele ajuste."""
    if not isinstance(fm, dict):
        return None
    out = {}
    try:
        out["LOO"] = float(fm["Error"][0]["LOO"])
        out["theta"] = np.asarray(fm["Kriging"][0]["theta"], dtype=float).ravel()
        out["beta_pce"] = np.asarray(fm["Kriging"][0]["beta"], dtype=float).ravel()
        out["sigmaSQ"] = float(np.atleast_1d(fm["Kriging"][0]["sigmaSQ"]).ravel()[0])
        npoly = int(np.atleast_1d(fm["NumberOfPoly"])[0])
        out["n_poly"] = npoly
        idx = fm["idxranking"][0][:npoly]
        out["poly_indices"] = np.asarray(fm["AllIndices"][0][np.array(idx), :], dtype=float)
    except Exception:
        pass
    return out


def instrument_lib(journal, cibles=CIBLES_LIB):
    """
    Enveloppe les fonctions publiques de `_lib` pour journaliser chaque appel.

    N'edite aucun fichier : le remplacement se fait dans le module charge.
    Renvoie une fonction qui restaure l'etat d'origine.
    """
    restaurer = []
    agregats = {}
    for mod_nom, fn_nom, args_traces, mode in cibles:
        try:
            mod = __import__(mod_nom)
        except ImportError:
            journal.event("instrumentation", "module absent", module=mod_nom)
            continue
        original = getattr(mod, fn_nom, None)
        if original is None:
            journal.event("instrumentation", "fonction absente",
                          module=mod_nom, fonction=fn_nom)
            continue

        if mode == "agrege":
            agregats[fn_nom] = {"n": 0, "digest": hashlib.md5(), "duree": 0.0}

        def fabrique(original=original, fn_nom=fn_nom, args_traces=args_traces, mode=mode):
            import inspect
            params = list(inspect.signature(original).parameters)

            def enveloppe(*a, **kw):
                lies = dict(zip(params, a))
                lies.update(kw)
                entrees = {"in_" + n: lies.get(n) for n in args_traces if n in lies}
                t0 = time.perf_counter()
                res = original(*a, **kw)
                dt = time.perf_counter() - t0

                sorties = {}
                resume = _resume_modele(res)
                if resume:
                    sorties.update({"out_" + k: v for k, v in resume.items()})
                elif isinstance(res, tuple):
                    for i, r in enumerate(res):
                        sorties["out_%d" % i] = r
                else:
                    sorties["out"] = res

                if mode == "agrege":
                    ag = agregats[fn_nom]
                    ag["n"] += 1
                    ag["duree"] += dt
                    for nom, v in sorted({**entrees, **sorties}.items()):
                        fp = fingerprint(v)
                        ag["digest"].update(
                            (nom + "|" + str(fp.get("hash") or fp.get("value"))).encode())
                    return res

                journal.probe("lib:" + fn_nom, **entrees, **sorties)
                journal.event("lib:" + fn_nom, "appel", duree=round(dt, 6))
                return res

            enveloppe.__name__ = original.__name__
            enveloppe.__doc__ = original.__doc__
            enveloppe.__wrapped__ = original
            return enveloppe

        setattr(mod, fn_nom, fabrique())
        restaurer.append((mod, fn_nom, original))
        journal.event("instrumentation", "accroche", module=mod_nom, fonction=fn_nom)

    def restore():
        for fn_nom, ag in sorted(agregats.items()):
            journal.probe("lib:" + fn_nom,
                          n_appels=ag["n"],
                          condensat_suite=ag["digest"].hexdigest()[:16])
            journal.event("lib:" + fn_nom, "agrege",
                          n_appels=ag["n"], duree_totale=round(ag["duree"], 4))
        for mod, fn_nom, original in restaurer:
            setattr(mod, fn_nom, original)

    return restore
