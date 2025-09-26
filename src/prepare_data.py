import urllib.request, numpy as np, struct, pathlib, tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)

HF = "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main"
DIRECT_FILES = {
    "sift_base.fvecs":        f"{HF}/sift_base.fvecs?download=true",
    "sift_query.fvecs":       f"{HF}/sift_query.fvecs?download=true",
    "sift_groundtruth.ivecs": f"{HF}/sift_groundtruth.ivecs?download=true",
}

TEXMEX_TAR = "http://corpus-texmex.irisa.fr/sift.tar.gz"

def _download(url: str, dst: pathlib.Path):
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dst)

def fetch():
    try:
        for fname, url in DIRECT_FILES.items():
            dst = OUT / fname
            if not dst.exists():
                print(f"Downloading {fname} from HF...")
                _download(url, dst)
        return
    except Exception as e:
        print(f"[WARN] Direct download failed ({e}). Falling back to TEXMEX tarball...")
    tar_path = OUT / "sift.tar.gz"
    if not tar_path.exists():
        print("Downloading sift.tar.gz from TEXMEX...")
        _download(TEXMEX_TAR, tar_path)
    print("Extracting needed files from sift.tar.gz...")
    with tarfile.open(tar_path, "r:gz") as tf:
        members = {m.name: m for m in tf.getmembers()}
        wanted = {
            "sift/sift_base.fvecs":        OUT / "sift_base.fvecs",
            "sift/sift_query.fvecs":       OUT / "sift_query.fvecs",
            "sift/sift_groundtruth.ivecs": OUT / "sift_groundtruth.ivecs",
        }
        for inner, dst in wanted.items():
            if not dst.exists():
                tf.extract(members[inner], path=OUT)
                (OUT / inner).rename(dst)

def read_fvecs(path: pathlib.Path) -> np.ndarray:
    with open(path, "rb") as f:
        buf = f.read()
    dim = struct.unpack_from("i", buf, 0)[0]
    arr = np.frombuffer(buf, dtype=np.float32, offset=4).reshape(-1, dim + 1)[:, 1:]
    return arr.astype(np.float32)

def read_ivecs(path: pathlib.Path) -> np.ndarray:
    with open(path, "rb") as f:
        buf = f.read()
    dim = struct.unpack_from("i", buf, 0)[0]
    arr = np.frombuffer(buf, dtype=np.int32).reshape(-1, dim + 1)[:, 1:]
    return arr.astype(np.int32)

if __name__ == "__main__":
    fetch()
    xb = read_fvecs(OUT / "sift_base.fvecs")
    xq = read_fvecs(OUT / "sift_query.fvecs")
    gt = read_ivecs(OUT / "sift_groundtruth.ivecs")
    np.save(OUT / "xb.npy", xb)
    np.save(OUT / "xq.npy", xq)
    np.save(OUT / "gt.npy", gt)
    print("Saved data/*.npy")
