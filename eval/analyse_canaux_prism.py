"""Analyse des canaux audio des 357 enregistrements Prism d'origine (AAC 48 kHz) :
corrélation gauche/droite, niveaux, opposition de phase éventuelle, croisés avec le CER.
Usage : python eval/analyse_canaux_prism.py [run_dir]   (défaut : eval/runs/run_20261112_0440)"""
import subprocess, sys, tempfile, numpy as np, pandas as pd, soundfile as sf, pyarrow.parquet as pq
run = sys.argv[1] if len(sys.argv) > 1 else "eval/runs/run_20261112_0440"
df = pq.read_table("data-test/DarijaCS-asr-eval/data/DarijaCS-Prism/train-00000-of-00001.parquet").to_pandas()
cer = pd.read_csv(f"{run}/predictions_scored.csv", dtype={"sample_id": str},
                  keep_default_na=False).set_index("sample_id")["cer"].astype(float)
db = lambda v: 20*np.log10(np.sqrt(np.mean(v**2)) + 1e-12)
rows = []
for _, r in df.iterrows():
    with tempfile.NamedTemporaryFile(suffix=".m4a") as f, tempfile.NamedTemporaryFile(suffix=".wav") as w:
        f.write(r["audio"]["bytes"]); f.flush()
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", f.name, "-c:a", "pcm_f32le", w.name], check=True)
        x, sr = sf.read(w.name, always_2d=True)
    sid = f"prism_{r['Filename']}"
    row = {"id": sid, "canaux": x.shape[1], "noise": r["Noise"], "cer": round(100*cer[sid], 1)}
    if x.shape[1] == 2:
        L, R = x[:, 0], x[:, 1]
        row.update(gauche=db(L), droite=db(R), moyenne=db((L+R)/2), diff=db((L-R)/2), corr_gd=np.corrcoef(L, R)[0, 1])
    rows.append(row)
t = pd.DataFrame(rows).round(2)
st = t[t["canaux"] == 2].copy()
st["suspect"] = (st["corr_gd"] < 0.5) | (st["diff"] > st["moyenne"] - 3) | ((st["gauche"] - st["droite"]).abs() > 6)
t.to_csv("eval/analyse_canaux_prism.csv", index=False)
print("Mono :", int((t["canaux"] == 1).sum()), "| Stéréo :", len(st))
print("Corrélation G/D des fichiers stéréo : min", st["corr_gd"].min(), "| médiane", st["corr_gd"].median())
print("Fichiers stéréo suspects :", int(st["suspect"].sum()))
cols = ["id", "corr_gd", "gauche", "droite", "moyenne", "diff", "noise", "cer"]
print(st[st["suspect"]].sort_values("corr_gd")[cols].to_string(index=False))
print("\nCER moyen : mono", round(t[t["canaux"] == 1]["cer"].mean(), 2), "% | stéréo", round(st["cer"].mean(), 2), "%")
print("\nprism_133 :"); print(t[t["id"] == "prism_133"].to_string(index=False))
