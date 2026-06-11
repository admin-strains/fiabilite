import os, sys
sys.path.insert(0, r'C:\workspace\front')
sys.path.insert(0, r'C:\workspace\front_mohamad')
for d in [r'C:\workspace\front\STRAINS\rupt\core\bin',
          r'C:\workspace\front\STRAINS\rupt\core',
          r'C:\workspace\front\STRAINS\common\Dll',
          r'C:\workspace\front\STRAINS\rupt\core\bin\meshgems',
          r'C:\workspace\front\STRAINS\rupt\core\bin\mosek',
          r'C:\workspace\front\.pixi\envs\default\Library\bin']:
    try: os.add_dll_directory(d)
    except: pass

from STRAINS.rupt.core import CetVISU
print("=== Calling CetVISU.RenderResult with GROUPS=['InputData'] ===", flush=True)
try:
    CetVISU.RenderResult(
        "Yield_analysis0",
        0,
        PATH=r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds",
        GROUPS=[b"InputData"]
    )
    print("=== RenderResult SUCCESS ===", flush=True)
except Exception as e:
    print(f"=== RenderResult FAILED: {e} ===", flush=True)
    import traceback
    traceback.print_exc()

print("=== Files in .ds after call ===", flush=True)
import glob
for f in sorted(glob.glob(r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\*InputData*")):
    print(f"  {f}", flush=True)
