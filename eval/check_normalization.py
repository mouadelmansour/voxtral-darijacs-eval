import sys, re, pyarrow.parquet as pq
sys.path.insert(0, sys.argv[2])
from normalization import normalize_arabic_asr, trace_normalization
root = sys.argv[1]
dq = pq.read_table(f"{root}/data/DarijaCS-Probe/train-00000-of-00001.parquet", columns=["id", "type", "Text"]).to_pandas()
dp = pq.read_table(f"{root}/data/DarijaCS-Prism/train-00000-of-00001.parquet", columns=["Filename", "Text"]).to_pandas()
for i in [30, 35, 61]:
    print(f"\n### Probe {i}")
    for stage, txt in trace_normalization(dq.loc[dq.id == i, "Text"].iloc[0]):
        print(f"  {stage:32s} {txt!r}")
print("\n### Exemples Prism")
for t in dp["Text"].head(3):
    print(repr(t[:150]), "\n  ->", repr(normalize_arabic_asr(t)[:150]))
br = re.compile(r"\[[^\]]*\]")
for name, d in [("Prism", dp), ("Probe", dq)]:
    hits = d[d["Text"].str.contains(br)]
    print(f"\n{name}: {len(hits)} références avec [annotations] :",
          sorted({m for t in hits["Text"] for m in br.findall(t)}))
    print(f"{name}: références vides après normalisation :",
          sum(normalize_arabic_asr(t) == "" for t in d["Text"]))
