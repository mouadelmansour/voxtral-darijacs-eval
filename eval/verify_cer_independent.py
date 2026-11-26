"""Recalcul indépendant du CER/WER (distance de Levenshtein codée ici, sans jiwer).
Usage : python verify_cer_independent.py predictions_scored.csv [dossier_du_modele_pour_renormaliser]"""
import csv, sys, os

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
norm = None
if len(sys.argv) > 2:
    sys.path.insert(0, os.path.abspath(sys.argv[2]))
    from normalization import normalize_arabic_asr as norm
agg = {}
mism = 0
for r in rows:
    ref, hyp = r["reference_norm"], r["hypothesis_norm"]
    if norm is not None:
        ref2, hyp2 = norm(r["reference"]), norm(r["hypothesis_raw"])
        mism += (ref2 != ref) + (hyp2 != hyp)
        ref, hyp = ref2, hyp2
    a = agg.setdefault(r["subset"], [0, 0, 0, 0, 0])
    a[0] += lev(ref, hyp); a[1] += len(ref)
    a[2] += lev(ref.split(), hyp.split()); a[3] += len(ref.split()); a[4] += 1
for s, (ce, cn, we, wn, n) in sorted(agg.items()):
    print(f"{s:6s} n={n}  CER={100*ce/cn:.2f} %  ({ce} erreurs / {cn} caractères)  WER={100*we/wn:.2f} %  ({we} / {wn} mots)")
if norm is not None:
    print("Différences entre la normalisation refaite et celle du fichier :", mism)
