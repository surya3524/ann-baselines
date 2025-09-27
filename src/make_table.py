import pandas as pd, pathlib, json
RES = pathlib.Path(__file__).resolve().parents[1] / "results"
df = pd.read_csv(RES/"runs.csv")

def row_hw(s):
    try:
        d = json.loads(s)
        return f"{d.get('machine','')}, RAM={d.get('ram_gb','?')}GB"
    except:
        return ""

df["Hardware/Seed"] = df["hardware"].apply(row_hw) + ", seed=" + df["seed"].astype(str)

cols = ["dataset","method","knobs","recall@10","median_ms","p95_ms","p99_ms","build_time_s","index_bytes","Hardware/Seed"]
pretty = df[cols].copy()
pretty["index_bytes"] = (pretty["index_bytes"]/ (1024**2)).round(1).astype(str) + " MB"
pretty.rename(columns={"median_ms":"median (ms)","p95_ms":"p95 (ms)","p99_ms":"p99 (ms)","build_time_s":"build (s)"}, inplace=True)

md = pretty.to_markdown(index=False)
(RES/"table.md").write_text(md)
print("Wrote results/table.md")
print(md)
