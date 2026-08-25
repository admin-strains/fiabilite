"""
Contraintes d'environnement -- ce qui doit rester vrai pour qu'une etude tourne.

Ces tests SAUTENT proprement sur un poste sans Digital Structure : le harness
garde sa promesse de tourner avec numpy + scipy seuls. Sur un poste DS, ils
verifient les deux contraintes qui rendent l'installation delicate, et qui
n'etaient jusqu'ici documentees nulle part ailleurs que dans un commentaire
approximatif de launcher3.py.

Contrainte 1 -- Python 3.10 exactement (ABI des .pyd de DS).
Contrainte 2 -- OpenTURNS importe AVANT l'ajout des repertoires DLL de DS,
                a cause d'une collision libblas / liblapack / zlib1.

Le test de la contrainte 2 echoue dans LES DEUX SENS : si le contournement
cesse d'etre necessaire, il le dit, pour qu'on supprime le contournement au
lieu de le trainer indefiniment.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT)
import launcher  # noqa: E402


def _ds_root():
    try:
        return launcher.find_ds_root()
    except SystemExit:
        return None


DS_ROOT = _ds_root()

pytestmark = pytest.mark.skipif(
    DS_ROOT is None, reason="Digital Structure absent de ce poste")


def _ot_present():
    try:
        import openturns  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Contrainte 1 : version de Python                                            #
# --------------------------------------------------------------------------- #
def test_les_pyd_de_ds_exigent_python_310():
    """Les modules compiles sont lies a python310.dll : le verifier sur le binaire,
    pas sur une convention."""
    import glob
    import re

    pyds = glob.glob(os.path.join(DS_ROOT, r"STRAINS\rupt\core\*.pyd"))
    assert pyds, "aucun .pyd trouve dans %s" % DS_ROOT

    tags = set()
    for f in pyds:
        with open(f, "rb") as fh:
            tags |= {m.decode().lower() for m in re.findall(rb"python3[0-9]{1,2}\.dll", fh.read())}
    assert tags == {"python310.dll"}, (
        "les .pyd de DS ne visent plus uniquement Python 3.10 mais %s -- "
        "mettre a jour launcher.check_python et requirements/studies.txt" % sorted(tags))


def test_le_launcher_refuse_une_mauvaise_version():
    with pytest.raises(SystemExit) as exc:
        vi = sys.version_info
        sys.version_info = (3, 13, 0, "final", 0)  # type: ignore[assignment]
        try:
            launcher.check_python()
        finally:
            sys.version_info = vi  # type: ignore[assignment]
    assert "python310" in str(exc.value)


# --------------------------------------------------------------------------- #
# Contrainte 2 : ordre d'import OpenTURNS / DS                                #
# --------------------------------------------------------------------------- #
_ORDER_SCRIPT = r'''
import os, sys
DLL = %(dll)r
if %(ot_first)r:
    import openturns
    for d in DLL:
        if os.path.isdir(d): os.add_dll_directory(d)
    sys.path.insert(0, %(ds)r)
    from STRAINS.rupt.core import CetSOLV
else:
    for d in DLL:
        if os.path.isdir(d): os.add_dll_directory(d)
    sys.path.insert(0, %(ds)r)
    from STRAINS.rupt.core import CetSOLV
    import openturns
print("IMPORTS_OK", openturns.__version__)
'''


def _run_order(ot_first, tmp_path):
    dll = [os.path.join(DS_ROOT, s) for s in launcher.DLL_SUBDIRS]
    src = _ORDER_SCRIPT % {"dll": dll, "ds": DS_ROOT, "ot_first": ot_first}
    script = tmp_path / ("ordre_%s.py" % ("ot" if ot_first else "ds"))
    script.write_text(src, encoding="utf-8")
    return subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, errors="replace", timeout=300)


@pytest.mark.skipif(not _ot_present(), reason="openturns absent")
@pytest.mark.slow
def test_openturns_avant_ds_fonctionne(tmp_path):
    p = _run_order(True, tmp_path)
    assert "IMPORTS_OK" in p.stdout, (
        "l ordre pratique par launcher.py ne fonctionne plus :\n"
        + p.stdout[-1500:] + "\n" + p.stderr[-1500:])


@pytest.mark.skipif(not _ot_present(), reason="openturns absent")
@pytest.mark.slow
def test_ds_avant_openturns_casse_toujours(tmp_path):
    """Si ce test echoue parce que l ordre inverse MARCHE, c est une bonne
    nouvelle : le contournement de launcher.py est devenu inutile et doit
    etre SUPPRIME, avec ce test."""
    p = _run_order(False, tmp_path)
    if "IMPORTS_OK" in p.stdout:
        pytest.fail(
            "importer DS avant OpenTURNS fonctionne desormais. La collision "
            "libblas/liblapack a disparu (mise a jour d OpenTURNS ou de DS). "
            "Supprimer le contournement d ordre dans launcher.py, ainsi que "
            "ce test et celui qui le precede.")
    assert "DLL load failed" in (p.stderr + p.stdout), (
        "l ordre inverse echoue, mais pas sur l erreur attendue :\n"
        + p.stderr[-1500:])


def test_les_dll_en_collision_sont_bien_differentes():
    """Documente le mecanisme exact de la collision, et previent s il change."""
    ot = pytest.importorskip("openturns")
    import glob
    import hashlib

    ot_dir = os.path.dirname(ot.__file__)
    ds_bin = os.path.join(DS_ROOT, r"STRAINS\rupt\core\bin")

    collisions = {}
    for name in ("libblas.dll", "liblapack.dll", "zlib1.dll"):
        a = glob.glob(os.path.join(ot_dir, "**", name), recursive=True)
        b = os.path.join(ds_bin, name)
        if a and os.path.isfile(b):
            ha = hashlib.md5(open(a[0], "rb").read()).hexdigest()
            hb = hashlib.md5(open(b, "rb").read()).hexdigest()
            collisions[name] = (ha != hb, os.path.getsize(a[0]), os.path.getsize(b))

    assert collisions, (
        "aucune des DLL connues pour entrer en collision n a ete trouvee des "
        "deux cotes -- le diagnostic de launcher.py est peut-etre perime")
    for name, (differe, taille_ot, taille_ds) in collisions.items():
        assert differe, (
            "%s est desormais IDENTIQUE des deux cotes (%d octets) : la cause "
            "de la contrainte d ordre a disparu, revoir launcher.py"
            % (name, taille_ot))


# --------------------------------------------------------------------------- #
# Le lanceur portable                                                          #
# --------------------------------------------------------------------------- #
def test_le_launcher_trouve_ds_sans_chemin_en_dur(monkeypatch):
    monkeypatch.delenv("DS_ROOT", raising=False)
    assert os.path.isdir(os.path.join(launcher.find_ds_root(), "STRAINS"))


def test_ds_root_invalide_donne_un_message_utile(monkeypatch, tmp_path):
    monkeypatch.setenv("DS_ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        launcher.find_ds_root()
    assert "STRAINS" in str(exc.value) and "front" in str(exc.value)
