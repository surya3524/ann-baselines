# src/eval_utils.py
import numpy as np, time, os, psutil, platform, json

def recall_at_k(I_pred, I_true, k=10):
    # I_true shape: (nq, >=k); I_pred: (nq, k)
    correct = 0
    gt_k = I_true[:, :k]
    for i in range(I_pred.shape[0]):
        correct += len(set(I_pred[i]).intersection(gt_k[i]))
    return correct / (I_pred.shape[0] * k)

def measure_latency(search_fn, q, k=10):
    # per-query timing to get p95/p99 robustly
    times = []
    for i in range(q.shape[0]):
        t0 = time.perf_counter()
        search_fn(q[i:i+1], k)    # expects (1, d) → (D, I)
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)*1000),
        "p95_ms": float(np.percentile(arr, 95)*1000),
        "p99_ms": float(np.percentile(arr, 99)*1000)
    }

def index_size_bytes(pathlike):
    try:
        return os.path.getsize(pathlike)
    except:
        return None

def hw_info():
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "system": platform.platform()
    }
