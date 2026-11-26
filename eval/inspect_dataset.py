import sys, os, io, glob
from collections import Counter
import pyarrow.parquet as pq, soundfile as sf

root = sys.argv[1]
files = sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
               if os.path.isfile(p) and "/.cache/" not in p)
print("== Fichiers ==")
for p in files:
    print(f"{os.path.getsize(p)/1e6:10.2f} MB  {os.path.relpath(p, root)}")

for p in [f for f in files if f.endswith(".parquet")]:
    print(f"\n================ {os.path.relpath(p, root)} ================")
    t = pq.read_table(p)
    print("Lignes :", t.num_rows)
    print("Schéma :\n", t.schema)
    df = t.to_pandas()
    is_audio = lambda c: isinstance(df[c].iloc[0], dict) and ("bytes" in df[c].iloc[0] or "path" in df[c].iloc[0])
    audio_cols = [c for c in df.columns if is_audio(c)]
    for c in [c for c in df.columns if c not in audio_cols]:
        n = df[c].astype(str).nunique()
        if n <= 30:
            print(f"\n[{c}] {n} valeurs distinctes : {df[c].astype(str).value_counts().to_dict()}")
        else:
            print(f"\n[{c}] {n} valeurs distinctes. Exemples :")
            for v in df[c].head(3): print("   ", repr(v)[:300])
    for c in audio_cols:
        fmt, durs, errs = Counter(), [], 0
        for a in df[c]:
            try:
                src = io.BytesIO(a["bytes"]) if a.get("bytes") else os.path.join(os.path.dirname(p), a["path"])
                i = sf.info(src)
                fmt[(i.samplerate, i.channels, i.format, i.subtype)] += 1
                durs.append(i.duration)
            except Exception as e:
                errs += 1; last = e
        print(f"\n[AUDIO {c}] (sr, canaux, format, subtype) : {dict(fmt)}")
        if durs:
            s = sorted(durs)
            print(f"  n={len(s)}  total={sum(s)/3600:.2f} h  min={s[0]:.2f}s  "
                  f"médiane={s[len(s)//2]:.2f}s  max={s[-1]:.2f}s")
        if errs: print(f"  {errs} audios illisibles, dernière erreur : {last!r}")
