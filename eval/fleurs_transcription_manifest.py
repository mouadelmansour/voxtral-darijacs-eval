"""Variante du manifeste FLEURS avec la colonne `transcription` (au lieu de `raw_transcription`)
comme référence, pour vérifier que le choix de colonne ne change pas le CER.
Usage : python eval/fleurs_transcription_manifest.py   (puis compute_metrics.py sur ce manifeste)"""
import csv, pandas as pd
from huggingface_hub import hf_hub_download
tsv = hf_hub_download("google/fleurs", "data/ar_eg/test.tsv", repo_type="dataset")
tr = {}
with open(tsv, encoding="utf-8") as f:
    for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
        tr[r[1]] = r[3]                     # colonne 1 = nom du fichier audio, colonne 3 = transcription
m = pd.read_csv("data-test/fleurs_ar_eg/manifest_fleurs.csv", dtype={"sample_id": str})
raw = m["reference"].copy()
m["reference"] = m["meta_file"].map(tr)
assert m["reference"].notna().all(), "fichier audio sans transcription"
m.to_csv("data-test/fleurs_ar_eg/manifest_fleurs_transcription.csv", index=False)
print("Références différentes entre les deux colonnes :", int((raw != m["reference"]).sum()), "sur", len(m))
print("Exemple raw_transcription :", raw.iloc[0][:120])
print("Exemple transcription     :", m["reference"].iloc[0][:120])
