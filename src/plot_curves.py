import pandas as pd, matplotlib.pyplot as plt, pathlib, json

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
df = pd.read_csv(RES/"runs.csv")

# Fig A: Recall vs median latency (ms)
for method, g in df.groupby("method"):
    plt.plot(g["median_ms"], g["recall@10"], marker='o', label=method)
plt.xlabel("Median latency (ms)"); plt.ylabel("Recall@10"); plt.legend()
plt.title("Recall vs Latency (SIFT1M)")
plt.tight_layout(); RES.joinpath("figs").mkdir(exist_ok=True)
plt.savefig(RES/"figs/recall_vs_latency.png"); plt.clf()

# Fig B: Memory vs Recall (index bytes per vector)
df["bytes_per_vec"] = df["index_bytes"] / 1_000_000 / 1_000_000  # MB → small scale
for method, g in df.groupby("method"):
    plt.plot(g["bytes_per_vec"], g["recall@10"], marker='o', label=method)
plt.xlabel("Index size (GB equivalent per entire index)"); plt.ylabel("Recall@10"); plt.legend()
plt.title("Memory vs Recall (SIFT1M)")
plt.tight_layout(); plt.savefig(RES/"figs/memory_vs_recall.png"); plt.clf()

# Fig C: p95 vs p99 latency scatter
for method, g in df.groupby("method"):
    plt.scatter(g["p95_ms"], g["p99_ms"], label=method)
plt.xlabel("p95 (ms)"); plt.ylabel("p99 (ms)"); plt.legend()
plt.title("Tail Latency (SIFT1M)")
plt.tight_layout(); plt.savefig(RES/"figs/tail_latency_cdf_proxy.png"); plt.clf()

print("Saved figures to results/figs/")
