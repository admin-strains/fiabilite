r"""Le lanceur doit dire COMMENT un run finit.

CE QUI A MOTIVE CE FICHIER -- 26/08/2026
-----------------------------------------
Un run de fumee est mort apres deux heures : Digital Structure a termine le
processus en pleine iteration IPM. Le montage de l'epoque etait

    python launcher.py ... 2>&1 | filtre >> journal

Dans un pipeline `cmd`, le code de retour est celui du DERNIER maillon -- le
filtre -- jamais celui de python. Le filtre a vu son entree se fermer, a rendu
0, et le `.bat` a ecrit « FIN ». **Un run mort etait indiscernable d'un run
reussi**, et il a fallu lire le journal ligne a ligne pour s'en apercevoir.

`tools/executer_filtre.py` LANCE la commande au lieu de la subir : il tient le
code de retour, l'ecrit en clair dans le journal, et le rend au systeme.

Ces tests lancent de vrais sous-processus Python. Aucune dependance lourde.
"""

import io
import os
import subprocess
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, os.path.join(_REPO, "tools"))

import executer_filtre                               # noqa: E402


def _lancer(code_python, journal):
    """Execute un bout de Python sous le lanceur, rend (code, contenu)."""
    ret = executer_filtre.executer(
        [sys.executable, "-u", "-c", code_python], journal)
    with io.open(journal, encoding="utf-8", errors="replace") as fh:
        return ret, fh.read()


def _corps(txt):
    """La sortie FILTREE seule, sans l'en-tete ni le bilan.

    L'en-tete recopie la commande -- volontairement, un journal qui ne dit pas
    ce qui a ete lance ne se compare a rien. Mais quand la commande est un
    `-c` qui contient litteralement « DS_PROF| », chercher ce texte dans TOUT
    le fichier le trouve dans l'en-tete et non dans la sortie. Le premier
    jet de ces tests s'y est fait prendre.
    """
    lignes = txt.splitlines()
    # Chaque run pose TROIS separateurs : ouverture, fermeture d'en-tete, et
    # pied. Le corps du DERNIER run commence donc juste apres l'avant-dernier.
    seps = [i for i, l in enumerate(lignes) if l.startswith("=" * 8)]
    depart = seps[-2] + 1 if len(seps) >= 2 else 0
    for i, l in enumerate(lignes[depart:], depart):
        if l.startswith("[filtre]"):
            return "\n".join(lignes[depart:i])
    return "\n".join(lignes[depart:])


# --------------------------------------------------------------------- #
# le filtrage
# --------------------------------------------------------------------- #
def test_ecarte_le_prefixe_et_garde_le_reste(tmp_path):
    j = str(tmp_path / "run.log")
    code, txt = _lancer(
        "print('garde A'); print('DS_PROF|bruit'); print('garde B')", j)
    assert code == 0
    assert "garde A" in txt and "garde B" in txt
    assert "DS_PROF|bruit" not in _corps(txt), "la ligne de bruit a survecu"
    assert "1 lignes gardees" not in txt      # il y en a 2
    assert "2 lignes gardees, 1 lignes DS_PROF ecartees" in txt


def test_le_compte_des_lignes_est_juste(tmp_path):
    j = str(tmp_path / "run.log")
    _, txt = _lancer(
        "\n".join(["print('DS_PROF|x')"] * 7 + ["print('vrai')"] * 3), j)
    assert "3 lignes gardees, 7 lignes DS_PROF ecartees" in txt


def test_la_sortie_d_erreur_est_conservee(tmp_path):
    """stderr est fusionne dans stdout : une trace d'exception ne doit pas se
    perdre -- c'est elle qui explique un echec."""
    j = str(tmp_path / "run.log")
    code, txt = _lancer(
        "import sys; sys.stderr.write('desastre\\n'); sys.exit(1)", j)
    assert code == 1
    assert "desastre" in txt


# --------------------------------------------------------------------- #
# le verdict -- la raison d'etre de l'outil
# --------------------------------------------------------------------- #
def test_un_succes_est_annonce_comme_tel(tmp_path):
    j = str(tmp_path / "run.log")
    code, txt = _lancer("print('ok')", j)
    assert code == 0
    assert "==== FIN : succes (code 0) ====" in txt


def test_un_echec_est_annonce_ET_propage(tmp_path):
    """LE test de non-regression du 26/08 : le journal doit dire ECHEC, et le
    code de retour doit sortir du lanceur."""
    j = str(tmp_path / "run.log")
    code, txt = _lancer("import sys; print('avant'); sys.exit(3)", j)
    assert code == 3, "le code de l'enfant doit etre rendu, pas celui du filtre"
    assert "ECHEC" in txt and "code 3" in txt
    assert "succes" not in txt


@pytest.mark.parametrize("code,attendu", [
    (0, "succes"),
    (1, "erreur generique"),
    (3221225477, "violation d'acces"),
    (3221225725, "debordement de pile"),
    (-9, "signal 9"),
    (42, "code 42"),
])
def test_les_codes_sont_traduits(code, attendu):
    """Un code brut comme 3221225477 ne dit rien ; 0xC0000005 dit tout."""
    assert attendu in executer_filtre._diagnostic(code)


def test_le_journal_porte_la_commande_et_les_dates(tmp_path):
    """Un journal qui ne dit pas ce qui a ete lance ne se compare a rien."""
    j = str(tmp_path / "run.log")
    _, txt = _lancer("print('x')", j)
    assert "DEBUT" in txt and "COMMANDE" in txt and "duree" in txt


def test_les_runs_successifs_s_ajoutent(tmp_path):
    """Le journal est ouvert en ajout : relancer ne doit pas effacer la trace
    du run precedent, qui est souvent celle qui explique l'echec."""
    j = str(tmp_path / "run.log")
    _lancer("print('premier')", j)
    _, txt = _lancer("print('second')", j)
    assert "premier" in txt and "second" in txt
    assert txt.count("==== FIN") == 2


# --------------------------------------------------------------------- #
# l'interface en ligne de commande
# --------------------------------------------------------------------- #
def test_la_ligne_de_commande_propage_aussi_le_code(tmp_path):
    j = str(tmp_path / "run.log")
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO, "tools", "executer_filtre.py"),
         "--journal", j, "--", sys.executable, "-c", "import sys; sys.exit(7)"],
        capture_output=True)
    assert proc.returncode == 7


def test_sans_separateur_l_outil_refuse(tmp_path):
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO, "tools", "executer_filtre.py"),
         "--journal", str(tmp_path / "x.log"), "echo", "coucou"],
        capture_output=True)
    assert proc.returncode == 2
    assert b"--" in proc.stderr


def test_le_prefixe_est_reglable(tmp_path):
    j = str(tmp_path / "run.log")
    ret = executer_filtre.executer(
        [sys.executable, "-u", "-c", "print('AUTRE|x'); print('garde')"],
        j, prefixe=b"AUTRE|")
    assert ret == 0
    with io.open(j, encoding="utf-8") as fh:
        txt = fh.read()
    assert "garde" in txt and "AUTRE|x" not in _corps(txt)
