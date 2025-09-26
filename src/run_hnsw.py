import numpy as np, hnswlib, pickle, time, os, json, pathlib
from eval_utils import recall_at_k, measure_latency, index_size_bytes, hw_info

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

xb = np.load(DATA/"xb.npy")
xq = np.load(DATA/"xq.npy")
gt = np.load(DATA/"gt.npy")

dim = xb.shape[1]
n = xb.shape[0]
k = 10

def build_and_run(M=16, efC=200, efS_list=(64,128,256), save_path="results/hnsw.idx"):
    idx = hnswlib.Index(space='l2', dim=dim)
    idx.init_index(max_elements=n, M=M, ef_construction=efC)  # build quality vs RAM/time
    t0 = time.time()
    idx.add_items(xb, np.arange(n))
    build_s = time.time() - t0

    # persist to measure size
    pickle.dump(idx, open(save_path, "wb"))
    size_b = index_size_bytes(save_path)

    rows = []
    for efS in efS_list:
        idx.set_ef(efS)  # search effort: larger → higher recall, higher latency
        # search_fn closure returning (D,I)
        def search_fn(q, kk):
            I, D = idx.knn_query(q, k=kk)
            return D, I
        lat = measure_latency(lambda qq,k: search_fn(qq,k), xq, k)
        D, I = search_fn(xq, k)
        rec = recall_at_k(I, gt, k)

        rows.append({
            "dataset": "SIFT1M",
            "method": "HNSW",
            "knobs": f"M={M},efC={efC},efS={efS}",
            "recall@10": round(rec, 4),
            **lat,
            "build_time_s": round(build_s, 2),
            "index_bytes": size_b,
            "hardware": json.dumps(hw_info()),
            "seed": 42
        })
        print(rows[-1])
    return rows

if __name__ == "__main__":
    rows = build_and_run()
    import pandas as pd
    out = RESULTS/"runs.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Appended to {out}")
