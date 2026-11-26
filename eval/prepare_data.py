import sys, os, subprocess, tempfile
import pyarrow.parquet as pq, soundfile as sf, pandas as pd
root, out = sys.argv[1], sys.argv[2]
for subset, idcol in [("Prism", "Filename"), ("Probe", "id")]:
    df = pq.read_table(f"{root}/data/DarijaCS-{subset}/train-00000-of-00001.parquet").to_pandas()
    d = os.path.join(out, subset); os.makedirs(d, exist_ok=True)
    rows = []
    for _, r in df.iterrows():
        sid = f"{subset.lower()}_{r[idcol]}"
        ext = os.path.splitext(r["audio"].get("path") or "")[1] or ".bin"
        dst = os.path.join(d, sid + ".wav")
        with tempfile.NamedTemporaryFile(suffix=ext) as f:
            f.write(r["audio"]["bytes"]); f.flush()
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", f.name,
                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", dst], check=True)
        info = sf.info(dst)
        meta = {f"meta_{k.replace(' ', '_')}": (None if pd.isna(v) else v)
                for k, v in r.items() if k not in ("audio", "Text")}
        rows.append({"sample_id": sid, "subset": subset, "audio_path": os.path.abspath(dst),
                     "duration_s": round(info.duration, 3), "sr": info.samplerate,
                     "reference": r["Text"], **meta})
    pd.DataFrame(rows).to_csv(os.path.join(out, f"manifest_{subset.lower()}.csv"), index=False)
    print(subset, len(rows), "fichiers,", round(sum(x["duration_s"] for x in rows) / 3600, 3), "h,",
          "sr =", {x["sr"] for x in rows})
