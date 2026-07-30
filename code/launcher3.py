import sys; print("LAUNCHER3 START", flush=True); sys.stdout.flush()
import os, sys

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

# Add STRAINS to Python path
sys.path.insert(0, r'C:\workspace\front')
sys.path.insert(0, r'C:\_workingDir\dir_fiabilite\lib')

# Run the target script — guard __main__ + freeze_support pour que les
# subprocesses DOE paralleles (Windows spawn) ne re-executent pas tout.
import multiprocessing as mp
if __name__ == "__main__":
    mp.freeze_support()
    exec(open(r'C:\_workingDir\dir_fiabilite\code\AC3_fiabilite.py').read(), {'__name__': '__main__'})
