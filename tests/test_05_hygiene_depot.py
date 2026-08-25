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


def test_pas_de_fichier_texte_volumineux():
    """Le depot trainait un journal de 3,8 Mo ne d'une redirection ratee
    (`> C:\\tmp\\form_out.txt` ayant perdu ses antislashes). Empecher le
    prochain."""
    gros = []
    for racine, dossiers, fichiers in os.walk(REPO):
        dossiers[:] = [d for d in dossiers if d not in IGNORES]
        for f in fichiers:
            p = os.path.join(racine, f)
            if os.path.splitext(f)[1] in {".py", ".md", ".json", ".ini", ".bat", ".txt"} \
                    and os.path.getsize(p) > 2_000_000:
                gros.append("%s (%.1f Mo)"
                            % (os.path.relpath(p, REPO), os.path.getsize(p) / 1048576))
    assert not gros, "fichiers texte de plus de 2 Mo : %s" % gros
