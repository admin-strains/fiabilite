import sys; print("LAUNCHER CANTILEVER-S START", flush=True); sys.stdout.flush()
import os, sys

# Log detaille force dans le .ds du projet (horodate). Modelname lu dans l'AC -> bon dossier.
if os.environ.get("_FIAB_LOG_REDIRECTED") != "1":
    import re, subprocess, datetime
    _ac = r'C:\workspace\fiabilite\AC_cantilever_s.py'
    _txt = open(_ac, encoding='utf-8').read()
    _m = re.search(r'(?m)^\s*modelname\s*=\s*"([^"]+)"', _txt)
    if _m is None:
        raise RuntimeError("modelname introuvable dans l'AC -> impossible de placer le log")
    _modelname = _m.group(1)
    _ds = r'C:\workspace\storage\admin\Moulin_Blanc' + os.sep + _modelname + '.ds'
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log = os.path.join(_ds, "log_cantilever_s_" + _ts + ".log")
    print("LOG DETAILLE FORCE -> " + _log, flush=True)
    _env = dict(os.environ, _FIAB_LOG_REDIRECTED="1")
    with open(_log, "w") as _f:
        _r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                            stdout=_f, stderr=subprocess.STDOUT, env=_env)
    sys.exit(_r.returncode)

# ===================== (process enfant : tout va dans le log .ds) =====================
import openturns as ot   # avant les DLL STRAINS (conflit MKL)

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

sys.path.insert(0, r'C:\workspace\front')
sys.path.insert(0, r'C:\workspace\fiabilite')
sys.path.insert(0, r'C:\workspace\fiabilite\_lib')   # _parallel_is (workers IS parallele)

# Garde __main__ (Windows/multiprocessing 'spawn') : les enfants IS importent ce launcher
# sous '__mp_main__' -> la garde empeche la re-execution de l'AC.
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    exec(open(r'C:\workspace\fiabilite\AC_cantilever_s.py').read(), {'__name__': '__main__'})
