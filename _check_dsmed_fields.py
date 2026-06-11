import sys, os
# Strip 01_3RDPARTY from PATH (hdf5 conflict workaround)
os.environ["PATH"] = ";".join([p for p in os.environ.get("PATH","").split(";") if "01_3RDPARTY" not in p])
import h5py
target = sys.argv[1] if len(sys.argv) > 1 else r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\Yield_analysis0_0_kine.dsmed"
print(f"Checking : {target}")
print(f"Exists : {os.path.exists(target)}")
if not os.path.exists(target):
    sys.exit(0)
with h5py.File(target, "r") as f:
    print(f"Top-level keys : {list(f.keys())}")
    if "CHA" in f:
        fields = list(f["CHA"].keys())
        print(f"\nFields in CHA ({len(fields)}):")
        for fld in sorted(fields):
            print(f"  - {fld}")
    elif "FAS" in f:
        print(f"Has FAS but no CHA. Keys deeper: {[k for k in f.keys()]}")
