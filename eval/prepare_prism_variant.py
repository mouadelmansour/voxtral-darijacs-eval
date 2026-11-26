"""Variante de prétraitement de Prism : canal gauche seul (au lieu de la moyenne des canaux)
et rééchantillonnage soxr (au lieu du rééchantillonneur par défaut de ffmpeg).
Usage : python eval/prepare_prism_variant.py data-test/DarijaCS-asr-eval data-test/prepared_left_soxr"""
import os, subprocess, sys, tempfile
import pandas as pd, pyarrow.parquet as pq, soundfile as sf

root, out = sys.argv[1], sys.argv[2]
df = pq.read_table(f"{root}/data/DarijaCS-Prism/train-00000-of-00001.parquet").to_pandas()
d = os.path.join(out, "Prism"); os.makedirs(d, exist_ok=True)
rows = []
for _, r in df.iterrows():
    sid = f"prism_{r['Filename']}"
    dst = os.path.abspath(os.path.join(d, sid + ".wav"))
    with tempfile.NamedTemporaryFile(suffix=".m4a") as f:
        f.write(r["audio"]["bytes"]); f.flush()
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", f.name,
                        "-af", "pan=mono|c0=c0,aresample=resampler=soxr", "-ar", "16000", "-c:a", "pcm_f32le", dst], check=True)
    info = sf.info(dst)
    rows.append({"sample_id": sid, "subset": "Prism", "audio_path": dst, "duration_s": round(info.duration, 3),
                 "sr": info.samplerate, "reference": r["Text"]})
pd.DataFrame(rows).to_csv(os.path.join(out, "manifest_prism.csv"), index=False)
print("Prism (canal gauche + soxr) :", len(rows), "fichiers")
