"""Télécharge FLEURS arabe (ar_eg, split test) depuis google/fleurs et prépare un manifeste au même format
que DarijaCS (WAV 16 kHz mono float32). Référence = raw_transcription.
Usage (sur login1, qui a Internet) : python eval/prepare_fleurs.py data-test/fleurs_ar_eg"""
import csv, os, subprocess, sys, tarfile
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
files = HfApi().list_repo_files("google/fleurs", repo_type="dataset")
tsv = [f for f in files if "ar_eg" in f and f.endswith("test.tsv")]
tar = [f for f in files if "ar_eg" in f and "test" in f and f.endswith(".tar.gz")]
print("Fichiers trouvés :", tsv, tar)
assert len(tsv) == 1 and len(tar) == 1, "structure inattendue du dépôt google/fleurs"
tsv_p = hf_hub_download("google/fleurs", tsv[0], repo_type="dataset")
tar_p = hf_hub_download("google/fleurs", tar[0], repo_type="dataset")

raw_dir = os.path.join(out, "raw")
os.makedirs(raw_dir, exist_ok=True)
with tarfile.open(tar_p) as t:
    t.extractall(raw_dir, filter="data")
wavs = {}
for root, _, fs in os.walk(raw_dir):
    for f in fs:
        wavs[f] = os.path.join(root, f)

wav_dir = os.path.join(out, "wav")
os.makedirs(wav_dir, exist_ok=True)
rows = []
with open(tsv_p, encoding="utf-8") as f:
    for i, r in enumerate(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
        fid, fname, raw_tr = r[0], r[1], r[2]
        if fname not in wavs:
            print("audio manquant :", fname); continue
        sid = f"fleurs_{i:04d}"
        dst = os.path.abspath(os.path.join(wav_dir, sid + ".wav"))
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", wavs[fname], "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_f32le", dst], check=True)
        info = sf.info(dst)
        rows.append({"sample_id": sid, "subset": "FLEURS", "audio_path": dst, "duration_s": round(info.duration, 3),
                     "sr": info.samplerate, "reference": raw_tr, "meta_fleurs_id": fid, "meta_file": fname})
with open(os.path.join(out, "manifest_fleurs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"FLEURS ar_eg test : {len(rows)} fichiers, {sum(r['duration_s'] for r in rows)/3600:.3f} h")
