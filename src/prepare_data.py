import numpy as np, struct, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "data"; OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "base": (OUT/"sift_base.fvecs",        516000000, 128),
    "query":(OUT/"sift_query.fvecs",         5160000, 128),
    "gt":   (OUT/"sift_groundtruth.ivecs",   4040000, 100),
}

def validate():
    ok = True
    for p, expect_bytes, expect_dim in FILES.values():
        if not p.exists():
            print(f"[ERROR] Missing file: {p}"); ok = False; continue
        size = p.stat().st_size
        if size != expect_bytes:
            print(f"[ERROR] {p} has {size} bytes; expected {expect_bytes}"); ok = False; continue
        with open(p, "rb") as f:
            d = struct.unpack("i", f.read(4))[0]
        if d != expect_dim:
            print(f"[ERROR] {p} header dim={d}; expected {expect_dim}"); ok = False
    return ok

def read_fvecs(path: pathlib.Path) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    a = a.reshape(-1, d + 1)
    return a[:, 1:].view(np.float32)

def read_ivecs(path: pathlib.Path) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    a = a.reshape(-1, d + 1)
    return a[:, 1:]

if __name__ == "__main__":
    if not validate():
        print("Fix files in data/ and rerun."); sys.exit(1)
    xb = read_fvecs(OUT/"sift_base.fvecs")
    xq = read_fvecs(OUT/"sift_query.fvecs")
    gt = read_ivecs(OUT/"sift_groundtruth.ivecs")
    np.save(OUT/"xb.npy", xb); np.save(OUT/"xq.npy", xq); np.save(OUT/"gt.npy", gt)
    print("Saved data/*.npy")
