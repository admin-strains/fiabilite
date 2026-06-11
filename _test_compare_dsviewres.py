"""Generate _Strain.dsviewres + _InputData.dsviewres and compare binary shapes."""
import os, sys, glob, struct

PATH_DS = r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds"
for f in glob.glob(os.path.join(PATH_DS, "Yield_analysis0_0_*.dsviewres")):
    if "rebar" not in f and "mesh" not in f:
        os.remove(f)
        print(f"removed {os.path.basename(f)}")

sys.path.insert(0, r'C:\workspace\front')
for d in [r'C:\workspace\front\STRAINS\rupt\core\bin',
          r'C:\workspace\front\STRAINS\rupt\core',
          r'C:\workspace\front\STRAINS\common\Dll']:
    try: os.add_dll_directory(d)
    except: pass

from STRAINS.rupt.core import CetVISU

# Generate both InputData and Strain
print("=== Gen InputData ===", flush=True)
CetVISU.RenderResult("Yield_analysis0", 0, PATH=PATH_DS, GROUPS=[b"InputData"])
print("=== Gen Strain ===", flush=True)
CetVISU.RenderResult("Yield_analysis0", 0, PATH=PATH_DS, GROUPS=[b"Strain"])

print("\n=== Files ===", flush=True)
for f in sorted(glob.glob(os.path.join(PATH_DS, "Yield_analysis0_0_*.dsviewres"))):
    print(f"{os.path.basename(f)} {os.path.getsize(f)}b")
