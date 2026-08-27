r"""Le journal des points evalues : le SEUL artefact cher d'une etude.

LE PRINCIPE DE DECOUPAGE
-------------------------
    Ne persister que ce qui coute un appel au solveur.
    Tout le reste se recalcule.

Cette ligne n'avait jamais ete tracee, et c'est la cause du desordre : quatre
caches se recouvrent aujourd'hui -- `doe_cache.json`, `restart_state.json`,
`points_log.jsonl`, `hf_grid_cache.json` -- avec des regles de validite
differentes, des sous-ensembles qui se chevauchent, et un seul d'entre eux
reellement relu.

Or le flux de l'etude dit exactement quoi garder :

    g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)

Le metamodele est RECONSTRUIT depuis les points, a chaque fois. FORM et le
tirage d'importance n'en dependent que par lui. Le seul bien irremplacable,
c'est donc l'ensemble des points evalues :

    action              appels solveur
    plan                       n0        <- a persister
    enrichissement       <= n_max        <- a persister
    grille HF          n_grid_hf^2       <- a persister
    analyse FORM+IS             0        recalculable en ~23 s
    figures                     0        recalculable

CE QUE CE FORMAT REND POSSIBLE
-------------------------------
* **Decorreler les actions.** Refaire une figure ne demande plus de rejouer
  47 heures d'enrichissement : les points sont la.
* **Survivre a une panne.** Le journal est en AJOUT SEUL : un point ecrit est
  acquis, et une interruption ne peut pas corrompre ce qui precede. C'est ce
  qui manquait au plan d'experiences, dont le cache incremental etait ecrit
  puis jete a la relecture.
* **Refuser un melange.** Chaque enregistrement porte la signature sous
  laquelle il a ete calcule. Relire des points issus d'un autre solveur
  lineaire, d'un autre maillage ou d'un autre domaine devient impossible en
  silence -- c'etait le defaut commun aux huit points de reprise du depot.

CE FICHIER N'IMPORTE NI OPENTURNS NI DIGITAL STRUCTURE : il est verifiable
partout, y compris en integration continue.
"""

import json
import os

import numpy as np

#: D'ou vient un point. Sert a distinguer ce qui a nourri le metamodele de ce
#: qui n'a servi qu'a une figure -- distinction qu'aucun cache actuel ne fait.
ORIGINES = ("plan", "enrichissement", "grille", "fosm", "manuel")


class Point:
    """Une evaluation de l'etat limite, et ce qu'on en sait.

    `grad_u` peut etre None : Digital Structure rend parfois
    `Sensitivity = {None, None}` sur un point NUMERICAL_ERROR. Un gradient
    fabrique a zero affirmerait que l'etat limite est plat en ce point, et le
    metamodele l'ajusterait -- on garde donc l'absence telle quelle, et c'est
    l'appelant qui decide (cf. `exclure_points_sans_gradient`).
    """

    __slots__ = ("u", "x", "g", "grad_u", "origine", "sain", "diagnostic")

    def __init__(self, u, x, g, grad_u=None, origine="plan", sain=True,
                 diagnostic=None):
        if origine not in ORIGINES:
            raise ValueError("origine=%r inconnue (attendu : %s)"
                             % (origine, ", ".join(ORIGINES)))
        self.u = [float(v) for v in u]
        self.x = [float(v) for v in x] if x is not None else None
        self.g = float(g)
        self.grad_u = (None if grad_u is None or any(v is None for v in grad_u)
                       else [float(v) for v in grad_u])
        self.origine = origine
        self.sain = bool(sain)
        self.diagnostic = dict(diagnostic or {})

    @property
    def gradient_complet(self) -> bool:
        return self.grad_u is not None

    def en_dict(self) -> dict:
        return {"u": self.u, "x": self.x, "g": self.g, "grad_u": self.grad_u,
                "origine": self.origine, "sain": self.sain,
                "diagnostic": self.diagnostic}

    @classmethod
    def depuis_dict(cls, d) -> "Point":
        return cls(u=d["u"], x=d.get("x"), g=d["g"], grad_u=d.get("grad_u"),
                   origine=d.get("origine", "plan"), sain=d.get("sain", True),
                   diagnostic=d.get("diagnostic"))

    def __repr__(self):
        return "Point(u=%s, g=%+.6f, %s%s)" % (
            [round(v, 3) for v in self.u], self.g, self.origine,
            "" if self.gradient_complet else ", SANS GRADIENT")


class JournalPoints:
    """Un fichier `.jsonl` en AJOUT SEUL, plus la signature qui le qualifie.

    Format : une ligne d'en-tete portant la signature, puis un point par
    ligne. Le choix du JSONL n'est pas cosmetique -- il donne l'ajout atomique
    ligne a ligne, donc la survie a une interruption, ce qu'un `json.dump`
    complet ne donne pas : il reecrit tout, et une coupure au milieu laisse un
    fichier illisible.
    """

    def __init__(self, chemin, signature=None):
        self.chemin = chemin
        self.signature = signature
        self._points = []

    # ----------------------------------------------------------------- #
    def __len__(self):
        return len(self._points)

    def __iter__(self):
        return iter(self._points)

    def __getitem__(self, i):
        return self._points[i]

    # ----------------------------------------------------------------- #
    def ajouter(self, point: Point) -> Point:
        """Ajoute un point EN MEMOIRE ET SUR LE DISQUE, immediatement.

        L'ecriture est faite point par point, et non a la fin : un appel au
        solveur coute de la dizaine de secondes a plusieurs minutes, le perdre
        parce que le run s'arrete deux points plus loin n'est pas acceptable.
        """
        self._points.append(point)
        os.makedirs(os.path.dirname(os.path.abspath(self.chemin)), exist_ok=True)
        neuf = not os.path.exists(self.chemin) or os.path.getsize(self.chemin) == 0
        with open(self.chemin, "a", encoding="utf-8") as fh:
            if neuf:
                fh.write(json.dumps({"_signature": self.signature},
                                    ensure_ascii=False) + "\n")
            fh.write(json.dumps(point.en_dict(), ensure_ascii=False) + "\n")
        return point

    # ----------------------------------------------------------------- #
    @classmethod
    def relire(cls, chemin, signature=None, tracer=print) -> "JournalPoints":
        """Relit un journal, ou en rend un VIDE si rien n'est reutilisable.

        Ne leve jamais : un journal absent, tronque ou etranger doit conduire
        a recalculer, pas a interrompre. Mais il le DIT -- un cache ecarte en
        silence est aussi dangereux qu'un cache accepte a tort.
        """
        j = cls(chemin, signature)
        if not os.path.exists(chemin):
            return j
        try:
            with open(chemin, "r", encoding="utf-8") as fh:
                lignes = fh.read().splitlines()
        except Exception as exc:                       # noqa: BLE001
            if tracer:
                tracer("[POINTS] lecture impossible (%s) -> on repart de zero"
                       % exc)
            return j
        if not lignes:
            return j

        try:
            entete = json.loads(lignes[0])
        except Exception:                              # noqa: BLE001
            entete = {}
        sig_fichier = entete.get("_signature") if isinstance(entete, dict) else None

        if signature is not None and sig_fichier != signature:
            ecarts = ["%s : disque=%r  courant=%r"
                      % (k, (sig_fichier or {}).get(k), v)
                      for k, v in signature.items()
                      if (sig_fichier or {}).get(k) != v]
            if tracer:
                tracer("[POINTS] %s -> on repart de zero. %s"
                       % ("signature absente" if sig_fichier is None
                          else "signature differente",
                          "; ".join(ecarts) or "(cles absentes)"))
            return j

        # Une ligne tronquee, c'est une interruption pendant l'ecriture : on
        # garde tout ce qui precede. C'est l'interet de l'ajout seul.
        tronquees = 0
        for ligne in lignes[1:]:
            if not ligne.strip():
                continue
            try:
                j._points.append(Point.depuis_dict(json.loads(ligne)))
            except Exception:                          # noqa: BLE001
                tronquees += 1
        if tronquees and tracer:
            tracer("[POINTS] %d ligne(s) illisible(s) ignoree(s) -- "
                   "interruption pendant une ecriture" % tronquees)
        if tracer and j._points:
            tracer("[POINTS] %d point(s) repris de %s -> autant d'appels "
                   "solveur evites" % (len(j._points), chemin))
        return j

    # ----------------------------------------------------------------- #
    def selon(self, origine=None, avec_gradient=None, sain=None):
        """Les points qui repondent aux criteres donnes."""
        out = list(self._points)
        if origine is not None:
            attendues = (origine,) if isinstance(origine, str) else tuple(origine)
            out = [p for p in out if p.origine in attendues]
        if avec_gradient is not None:
            out = [p for p in out if p.gradient_complet is avec_gradient]
        if sain is not None:
            out = [p for p in out if p.sain is sain]
        return out

    def tableaux(self, points=None):
        """(u, g, grad_u) au format attendu par le metamodele.

        REFUSE les points sans gradient plutot que de fabriquer des zeros :
        `np.asarray([[None]], dtype=float)` rend `[[nan]]` SANS RIEN DIRE, et
        les NaN se propageraient dans l'ajustement.
        """
        pts = self._points if points is None else list(points)
        sans = [i for i, p in enumerate(pts) if not p.gradient_complet]
        if sans:
            raise ValueError(
                "points sans gradient aux indices %s : ils ne peuvent pas "
                "entrer dans un metamodele a gradients. Les filtrer avec "
                "`selon(avec_gradient=True)`, ou decider explicitement de leur "
                "sort (cf. `exclure_points_sans_gradient`)." % sans)
        if not pts:
            vide = np.zeros((0, 0))
            return vide, vide, vide
        u = np.array([p.u for p in pts], dtype=float)
        g = np.array([[p.g] for p in pts], dtype=float)
        grad = np.array([p.grad_u for p in pts], dtype=float)
        return u, g, grad

    def resume(self) -> str:
        par_origine = {}
        for p in self._points:
            par_origine[p.origine] = par_origine.get(p.origine, 0) + 1
        sans_grad = len(self.selon(avec_gradient=False))
        malades = len(self.selon(sain=False))
        bouts = ["%d point(s)" % len(self._points)]
        if par_origine:
            bouts.append(", ".join("%s=%d" % kv for kv in sorted(par_origine.items())))
        if sans_grad:
            bouts.append("%d sans gradient" % sans_grad)
        if malades:
            bouts.append("%d non converge(s)" % malades)
        return " | ".join(bouts)
