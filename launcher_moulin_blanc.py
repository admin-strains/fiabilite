import sys; print("LAUNCHER MOULIN BLANC START", flush=True); sys.stdout.flush()
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

# Add STRAINS + fiabilite repo to Python path
sys.path.insert(0, r'C:\workspace\front')
sys.path.insert(0, r'C:\workspace\fiabilite')

# Run the target script (calcul fiabilite Moulin Blanc 35K)
exec(open(r'C:\workspace\fiabilite\AC_moulin_blanc.py').read(), {'__name__': '__main__'})
