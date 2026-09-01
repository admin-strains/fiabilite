"""
Hygiene du depot : les fautes qu'aucun test fonctionnel n'attrape.

Ecrit apres en avoir commis une. Un script d'edition de la phase 3 a produit
des fins de ligne CR-CR-LF sur les 6 692 lignes des deux scripts AC. Python
les tolere -- le retour chariot surnumeraire n'est que de l'espace -- donc
**toute la suite est restee verte**. Seul le diff, devenu illisible
(6 795 lignes modifiees au lieu de 28), a revele la corruption.

Une modification qui rend le diff inexploitable detruit la revue de code,
c'est-a-dire la seule garantie qui reste quand les tests ne voient rien.
"""

import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IGNORES = {".git", "__pycache__", ".pytest_cache", "historique", "baselines",
           ".vs", "output"}


def _fichiers(extensions):
    for racine, dossiers, fichiers in os.walk(REPO):
        dossiers[:] = [d for d in dossiers if d not in IGNORES]
        for f in fichiers:
            if os.path.splitext(f)[1] in extensions:
                yield os.path.join(racine, f)


@pytest.mark.parametrize("ext", [{".py"}, {".md", ".txt", ".ini", ".bat", ".json"}])
def test_pas_de_retour_chariot_double(ext):
    """CR-CR-LF : invisible a l'execution, devastateur pour le diff."""
    fautifs = []
    for chemin in _fichiers(ext):
        with open(chemin, "rb") as fh:
            if b"\r\r" in fh.read():
                fautifs.append(os.path.relpath(chemin, REPO))
    assert not fautifs, "retours chariot doubles dans : %s" % fautifs


def test_les_sources_python_sont_syntaxiquement_valides():
    """Garde-fou des scripts d'edition automatique : une edition par
    substitution de texte peut produire du code qui ne compile plus, dans un
    fichier qu'aucun test n'importe -- donc sans que rien ne tombe."""
    fautifs = []
    for chemin in _fichiers({".py"}):
        with open(chemin, "rb") as fh:
            source = fh.read()
        try:
            compile(source, chemin, "exec")
        except SyntaxError as exc:
            fautifs.append("%s:%s : %s"
                           % (os.path.relpath(chemin, REPO), exc.lineno, exc.msg))
    assert not fautifs, "\n".join(fautifs)


#: Les gros fichiers qu'on assume, et pourquoi. Une exemption se DECLARE :
#: elargir le filtre sans le dire reviendrait a desarmer le garde-fou pour
#: tout le monde.
GROS_ASSUMES = {
    # Les deux modeles de test versionnes le 26/08/2026 sur demande d'Agnes.
    # `dsCad.txt` du Moulin Blanc fait 9,65 Mo : 188 094 lignes, dont 15 348
    # appels REBAR et 141 954 POINT. C'est une DONNEE de modele, pas un
    # journal echappe -- et sans elle le depot decrit une etude que personne
    # ne peut rejouer. Compresse par git, l'ensemble pese 1,6 Mo.
    # Voir `modeles/README.md`.
    "modeles": "donnees des deux modeles de test (cf. modeles/README.md)",
}


def test_pas_de_fichier_texte_volumineux():
    """Le depot trainait un journal de 3,8 Mo ne d'une redirection ratee
    (`> C:\\tmp\\form_out.txt` ayant perdu ses antislashes). Empecher le
    prochain -- sans interdire les donnees de modele, qui sont declarees
    dans `GROS_ASSUMES`."""
    gros = []
    for racine, dossiers, fichiers in os.walk(REPO):
        dossiers[:] = [d for d in dossiers
                       if d not in IGNORES and d not in GROS_ASSUMES]
        for f in fichiers:
            p = os.path.join(racine, f)
            if os.path.splitext(f)[1] in {".py", ".md", ".json", ".ini", ".bat", ".txt"} \
                    and os.path.getsize(p) > 2_000_000:
                gros.append("%s (%.1f Mo)"
                            % (os.path.relpath(p, REPO), os.path.getsize(p) / 1048576))
    assert not gros, "fichiers texte de plus de 2 Mo : %s" % gros


def test_les_exemptions_de_taille_existent_encore():
    """Une exemption qui ne protege plus rien doit disparaitre, sinon elle
    devient un trou ouvert par inadvertance."""
    manquants = [d for d in GROS_ASSUMES if not os.path.isdir(os.path.join(REPO, d))]
    assert not manquants, (
        "GROS_ASSUMES cite des dossiers qui n'existent plus : %s.\n"
        "Retirer l'exemption plutot que la laisser desarmer le garde-fou."
        % manquants)


#: paquets qui appartiennent a la couche des ETUDES, jamais au noyau.
#: `_lib/` ne doit dependre que de numpy et scipy : c'est la promesse du
#: README, celle qui rend le harness executable partout et la CI possible.
COUCHE_ETUDES = ("openturns", "sklearn", "smt", "matplotlib", "autograd", "STRAINS")


def test_le_noyau_ne_depend_que_de_numpy_et_scipy():
    """Un import de trop dans `_lib/` casse la portabilite en silence.

    Jusqu'a la phase 8, un tel import ne se voyait meme pas : il
    interrompait la COLLECTE de pytest, et la suite entiere ne tournait plus
    du tout au lieu de sauter un fichier. C'est ce qui est arrive a
    `test_84_extraction_graphiques.py`, qui importait matplotlib sans garde.

    Le meme controle tourne dans la CI, sur un runner ou seul
    `requirements/core.txt` est installe -- ce test-ci le rend visible AVANT
    de pousser.
    """
    import re
    fautifs = []
    for chemin in _fichiers({".py"}):
        if os.sep + "_lib" + os.sep not in chemin:
            continue
        with open(chemin, encoding="utf-8", errors="replace") as fh:
            texte = fh.read()
        # on retire les commentaires : une docstring qui CITE openturns est
        # legitime, un import ne l'est pas
        code = "\n".join(l.split("#")[0] for l in texte.splitlines())
        for nom in COUCHE_ETUDES:
            if re.search(r"^\s*(import|from)\s+%s\b" % re.escape(nom), code, re.M):
                fautifs.append("%s importe %s"
                               % (os.path.relpath(chemin, REPO), nom))
    assert not fautifs, (
        "la couche noyau doit rester installable sans licence :\n  "
        + "\n  ".join(fautifs))


def test_chaque_test_qui_exige_la_couche_etudes_le_declare():
    """Un fichier de test qui importe matplotlib, OpenTURNS ou scikit-learn
    au niveau module DOIT le faire par `pytest.importorskip`, sinon il
    interrompt la collecte de toute la suite sur un poste minimal.

    CE TEMOIN NE SUFFIT PAS, ET IL FAUT LE SAVOIR
    ----------------------------------------------
    Il ne voit que l'import DIRECT. Le 31/08/2026, trois fichiers cassaient
    la collecte sans qu'il bronche : ils n'importaient pas OpenTURNS, ils
    importaient `plan`, `evaluation`, `grille` et `figurer` -- des modules
    du depot qui, eux, l'importent. La sonde regardait une forme de
    surface, pas le mecanisme.

    C'est `test_la_suite_se_collecte_sans_la_couche_etudes` qui tient
    reellement la propriete. Celui-ci reste parce qu'il est instantane et
    que son message designe le coupable en clair."""
    import re
    fautifs = []
    for chemin in _fichiers({".py"}):
        if os.sep + "tests" + os.sep not in chemin or os.sep + "unit" + os.sep in chemin:
            continue
        with open(chemin, encoding="utf-8", errors="replace") as fh:
            lignes = fh.read().splitlines()
        code = "\n".join(l.split("#")[0] for l in lignes)
        for nom in ("matplotlib", "openturns", "sklearn", "smt"):
            motif = r"^\s*(import|from)\s+%s\b" % re.escape(nom)
            if re.search(motif, code, re.M) and "importorskip" not in code:
                fautifs.append("%s importe %s sans importorskip"
                               % (os.path.relpath(chemin, REPO), nom))
    assert not fautifs, "\n  ".join([""] + fautifs)


# --------------------------------------------------------------------- #
# la propriete elle-meme, pas un indice de la propriete
# --------------------------------------------------------------------- #
#: Ce que `requirements/core.txt` n'installe PAS. Le job `noyau` de
#: l'integration continue tourne exactement avec cette absence, sur quatre
#: combinaisons OS x version de Python.
#: NOM DISTINCT DE `COUCHE_ETUDES` (l. 104), et ce n'est pas un detail : les
#: deux listes ne servent pas la meme chose. Celle-ci enumere ce que
#: `requirements/core.txt` n'installe PAS, pour fabriquer l'absence ; l'autre
#: enumere ce que `_lib/` n'a pas le droit d'importer. Les confondre ferait
#: silencieusement changer de perimetre au controle du noyau.
ABSENT_SANS_LA_COUCHE_ETUDES = ("openturns", "smt", "matplotlib", "sklearn",
                                "autograd", "ndsplines", "psutil", "STRAINS")

_SOUS_PROCESSUS = '''
import sys

class _Absent:
    """Rend introuvables les modules de la couche etudes.

    `ModuleNotFoundError` et NON `ImportError` : c'est ce que leve Python
    quand un module est reellement absent, et un `except ModuleNotFoundError`
    quelque part dans la chaine ne verrait pas passer le second. Teste le
    31/08/2026 -- la version `ImportError` declarait 22 fichiers fautifs la
    ou un environnement reellement minimal n'en compte aucun.
    """
    def find_module(self, nom, chemin=None):
        return self.find_spec(nom, chemin)

    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in %r:
            raise ModuleNotFoundError("No module named %%r" %% nom, name=nom)
        return None

sys.meta_path.insert(0, _Absent())

import pytest
sys.exit(pytest.main(["--collect-only", "-q", "--no-header",
                      "-p", "no:cacheprovider", "-p", "no:randomly"]))
'''


def test_la_suite_se_collecte_sans_la_couche_etudes():
    r"""Sans OpenTURNS ni matplotlib, la suite doit SAUTER, pas s'interrompre.

    POURQUOI C'EST UNE PROPRIETE, ET PAS UN DETAIL
    -----------------------------------------------
    Un fichier qui importe OpenTURNS hors d'un `importorskip` ne fait pas
    echouer ses propres tests : il fait echouer la COLLECTE, et pytest
    interrompt alors la suite ENTIERE. Trois fichiers verts en local
    suffisent a rendre rouge un runner qui n'a que la couche noyau -- 808
    tests qui ne tournent plus, pour une ligne d'import.

    C'est ce qui est arrive : `test_106`, `test_108` et `test_110`
    importaient des modules du depot qui importent OpenTURNS. Corrige le
    31/08/2026.

    ON N'EMULE PAS L'ABSENCE, ON LA FABRIQUE. Un second environnement n'est
    pas necessaire : un `meta_path` qui refuse ces noms dans un
    sous-processus donne exactement le meme verdict que le venv minimal
    verifie le meme jour -- 833 collectes, zero erreur.
    """
    import subprocess
    import sys

    p = subprocess.run([sys.executable, "-c", _SOUS_PROCESSUS % (ABSENT_SANS_LA_COUCHE_ETUDES,)],
                       capture_output=True, text=True, cwd=REPO)
    if p.returncode != 0:
        fautifs = [l for l in (p.stdout or "").splitlines()
                   if l.startswith("ERROR ")]
        raise AssertionError(
            "la collecte s'interrompt sans la couche etudes (code %d).\n"
            "Sur le runner d'integration continue, c'est TOUTE la suite qui "
            "ne tourne pas.\n  %s\n\n"
            "Corriger en placant `pytest.importorskip(\"openturns\")` AVANT "
            "les imports du depot dans ces fichiers -- le module importe "
            "peut etre du depot et tirer OpenTURNS sans le nommer.\n%s"
            % (p.returncode, "\n  ".join(fautifs) or "(aucune ligne ERROR)",
               (p.stdout or "")[-1500:]))


def test_le_BLAS_tourne_sur_un_thread():
    r"""Le bridage de `conftest.py` mord-il VRAIMENT ?

    POURQUOI CE TEMOIN EXISTE
    --------------------------
    La premiere version bridait par `threadpoolctl.threadpool_limits(1)`
    depuis `pytest_configure`. Elle ne bridait que les bibliotheques DEJA
    CHARGEES : l'OpenBLAS que scipy charge ensuite restait a 7 threads. La
    suite est restee VERTE, les goldens ont continue de passer, et le
    bridage ne servait a rien. Un bridage qu'on croit actif est pire que pas
    de bridage : on regenere des goldens en croyant les avoir rendus
    portables.

    Ce test regarde ce que les bibliotheques DECLARENT, apres coup. numpy et
    scipy embarquent chacun le leur -- deux OpenBLAS distincts sur ce poste.
    """
    threadpoolctl = pytest.importorskip("threadpoolctl")
    import numpy   # noqa: F401  -- charge l'OpenBLAS de numpy
    import scipy.linalg   # noqa: F401  -- et celui de scipy, qui est un AUTRE

    bavardes = [(i.get("internal_api"), i.get("filepath"), i.get("num_threads"))
                for i in threadpoolctl.threadpool_info()
                if i.get("num_threads") != 1]
    assert not bavardes, (
        "%d bibliotheque(s) d'algebre lineaire tournent sur plusieurs "
        "threads pendant les tests :\n  %s\n\n"
        "Les goldens figent alors le nombre de coeurs de la machine, pas le "
        "code : a 7 threads contre 1, theta passe de [0.010000, 6.548512] a "
        "[0.341149, 6.244649] sur `flexion/PCK`.\n"
        "Le bridage est en tete de `tests/conftest.py` et doit rester AVANT "
        "`import numpy` -- OpenBLAS lit ces variables au chargement."
        % (len(bavardes),
           "\n  ".join("%s (%s) : %s threads" % (a, os.path.basename(b or "?"), c)
                       for a, b, c in bavardes)))
