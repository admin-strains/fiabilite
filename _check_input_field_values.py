import sys, os
os.environ["PATH"] = ";".join([p for p in os.environ.get("PATH","").split(";") if "01_3RDPARTY" not in p])
import h5py
import numpy as np
target = sys.argv[1] if len(sys.argv) > 1 else r"C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\Yield_analysis0_0_kine.dsmed"

with h5py.File(target, "r") as f:
    for fname in ["input____FC", "input____FT", "input____FY"]:
        if fname in f["CHA"]:
            print(f"--- {fname} ---")
            grp = f["CHA"][fname]
            # Walk the HDF5 tree to find the data array
            def find_arr(g, depth=0):
                for k in g.keys():
                    obj = g[k]
                    if isinstance(obj, h5py.Dataset):
                        arr = np.array(obj)
                        if arr.dtype.kind == 'f' and arr.size > 0:
                            print(f"  Dataset {obj.name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
                            return True
                    elif isinstance(obj, h5py.Group) and depth < 5:
                        if find_arr(obj, depth+1): return True
                return False
            find_arr(grp)
        else:
            print(f"--- {fname} : ABSENT ---")
