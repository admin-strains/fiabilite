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
    interrompt la collecte de toute la suite sur un poste minimal."""
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
