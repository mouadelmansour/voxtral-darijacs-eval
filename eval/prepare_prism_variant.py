"""Variante de prétraitement de Prism : canal gauche seul (au lieu de la moyenne des canaux),
puis rééchantillonnage 16 kHz avec soxr en Python (au lieu du rééchantillonneur par défaut de ffmpeg).
Usage : python eval/prepare_prism_variant.py data-test/DarijaCS-asr-eval data-test/prepared_left_soxr"""
import os, subprocess, sys, tempfile
import numpy as np, pandas as pd, pyarrow.parquet as pq, soundfile as sf, soxr

root, out = sys.argv[1], sys.argv[2]
df = pq.read_table(f"{root}/data/DarijaCS-Prism/train-00000-of-00001.parquet").to_pandas()
d = os.path.join(out, "Prism"); os.makedirs(d, exist_ok=True)
rows, rates = [], set()
for _, r in df.iterrows():
    sid = f"prism_{r['Filename']}"
    dst = os.path.abspath(os.path.join(d, sid + ".wav"))
    with tempfile.NamedTemporaryFile(suffix=".m4a") as f, tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        f.write(r["audio"]["bytes"]); f.flush()
        # décodage AAC -> canal gauche, fréquence d'origine conservée
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", f.name,
                        "-af", "pan=mono|c0=c0", "-c:a", "pcm_f32le", tmp.name], check=True)
        x, sr = sf.read(tmp.name, dtype="float32")
    rates.add(sr)
    y = soxr.resample(x, sr, 16000, quality="VHQ").astype(np.float32)
    sf.write(dst, y, 16000, subtype="FLOAT")
    rows.append({"sample_id": sid, "subset": "Prism", "audio_path": dst, "duration_s": round(len(y) / 16000, 3),
                 "sr": 16000, "reference": r["Text"]})
pd.DataFrame(rows).to_csv(os.path.join(out, "manifest_prism.csv"), index=False)
print("Prism (canal gauche + soxr VHQ) :", len(rows), "fichiers | fréquences d'origine :", sorted(rates))
