"""Test direct CetVISU pour generer _rebar_inputdata.dsviewres."""
import os, sys, glob
PATH_DS = r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds"

# Clear previous output
for f in glob.glob(os.path.join(PATH_DS, "*rebar_inputdata*.dsviewres")):
    print(f"Removing old: {f}")
    os.remove(f)

sys.path.insert(0, r'C:\workspace\front')
sys.path.insert(0, r'C:\workspace\front_mohamad')
for d in [r'C:\workspace\front\STRAINS\rupt\core\bin',
          r'C:\workspace\front\STRAINS\rupt\core',
          r'C:\workspace\front\STRAINS\common\Dll']:
    try: os.add_dll_directory(d)
    except: pass

from STRAINS.rupt.core import CetVISU
print("=== Calling CetVISU.RenderResult to generate _rebar_inputdata ===", flush=True)
try:
    CetVISU.RenderResult(
        "Yield_analysis0",
        0,
        PATH=PATH_DS,
        WRITE_OPTIONS={"input_data_rebars": True, "stress_rebars": True}
    )
    print("=== SUCCESS ===", flush=True)
except Exception as e:
    print(f"=== FAILED: {e} ===", flush=True)
    import traceback
    traceback.print_exc()

print("=== Files after call ===", flush=True)
for f in sorted(glob.glob(os.path.join(PATH_DS, "*rebar*.dsviewres"))):
    print(f"  {f} ({os.path.getsize(f)}b)", flush=True)
