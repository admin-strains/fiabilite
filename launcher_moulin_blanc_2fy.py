import sys; print("LAUNCHER MOULIN BLANC 2-FY START", flush=True); sys.stdout.flush()
import os, sys

# 2026-06-18 : LOG DETAILLE FORCE dans le dossier .ds du projet, horodate (jamais ecrase).
# On redirige au NIVEAU PROCESS (subprocess stdout) : c'est le SEUL moyen fiable sous Windows
# de capturer AUSSI la sortie C++ de STRAINS (os.dup2 cote Python ne suit pas le handle C++).
# Source unique du modelname : lu directement dans l'AC -> impossible de se tromper de dossier.
if os.environ.get("_FIAB_LOG_REDIRECTED") != "1":
    import re, subprocess, datetime
    _ac = r'C:\workspace\fiabilite\AC_moulin_blanc_2fy.py'
    _txt = open(_ac, encoding='utf-8').read()
    _m = re.search(r'(?m)^\s*modelname\s*=\s*"([^"]+)"', _txt)
    if _m is None:
        raise RuntimeError("modelname introuvable dans l'AC -> impossible de placer le log")
    _modelname = _m.group(1)
    _ds = r'C:\workspace\storage\admin\Moulin_Blanc' + os.sep + _modelname + '.ds'
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log = os.path.join(_ds, "log_2fy_" + _ts + ".log")
    print("LOG DETAILLE FORCE -> " + _log, flush=True)
    _env = dict(os.environ, _FIAB_LOG_REDIRECTED="1")
    with open(_log, "w") as _f:
        _r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                            stdout=_f, stderr=subprocess.STDOUT, env=_env)
    sys.exit(_r.returncode)

# ===================== (process enfant : tout va dans le log .ds) =====================
# Import openturns BEFORE adding STRAINS DLL dirs pour eviter le conflit MKL
import openturns as ot

# Setup DLL search paths BEFORE any STRAINS import
dll_dirs = [
    r'C:\workspace\front\STRAINS\rupt\core\bin',
    r'C:\workspace\front\STRAINS\rupt\core',
    r'C:\workspace\front\STRAINS\common\Dll',
    r'C:\workspace\front\STRAINS\rupt\core\bin\meshgems',
    r'C:\workspace\front\STRAINS\rupt\core\bin\mosek',
]
for d in dll_dirs:
    if os.path.isdir(d):
        os.add_dll_directory(d)

# Add STRAINS + fiabilite repo to Python path (HORS garde : les process enfants de l'IS
# parallele en ont besoin pour importer _parallel_is / branche1)
sys.path.insert(0, r'C:\workspace\front')
sys.path.insert(0, r'C:\workspace\fiabilite')
sys.path.insert(0, r'C:\workspace\fiabilite\_lib')   # pour _parallel_is (workers IS parallele)

# IMPORTANT (Windows / multiprocessing 'spawn') : l'exec de l'AC DOIT etre garde par __main__,
# sinon chaque process enfant (cree par ProcessPoolExecutor pour l'IS parallele _IS_PARALLEL=1)
# re-execute tout l'AC. Les enfants importent ce launcher sous '__mp_main__' -> la garde les bloque.
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    # Run the target script (fiabilite 2-fy Moulin Blanc)
    exec(open(r'C:\workspace\fiabilite\AC_moulin_blanc_2fy.py').read(), {'__name__': '__main__'})
