"""Compare deux exécutions échantillon par échantillon (fichiers predictions_scored.csv).
Usage : python eval/compare_runs.py RUN_A/predictions_scored.csv RUN_B/predictions_scored.csv "nom A" "nom B" """
import sys
import pandas as pd

a = pd.read_csv(sys.argv[1], dtype={"sample_id": str}, keep_default_na=False)
b = pd.read_csv(sys.argv[2], dtype={"sample_id": str}, keep_default_na=False)
na, nb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("A", "B")
m = a.merge(b, on="sample_id", suffixes=("_a", "_b"))
print(f"=== {na}  vs  {nb} : {len(m)} échantillons en commun")
for s, g in m.groupby("subset_a"):
    same = (g.hypothesis_norm_a.str.strip() == g.hypothesis_norm_b.str.strip()).sum()
    ca = 100 * (g.c_S_a + g.c_D_a + g.c_I_a).sum() / g.c_N_a.sum()
    cb = 100 * (g.c_S_b + g.c_D_b + g.c_I_b).sum() / g.c_N_b.sum()
    print(f"{s:7s} transcriptions identiques : {same}/{len(g)} ({100*same/len(g):.1f} %)   "
          f"CER {na} = {ca:.2f} %   CER {nb} = {cb:.2f} %   écart = {cb-ca:+.2f} pt")
m["delta"] = 100 * (m.cer_b - m.cer_a)
print("\nPlus grands écarts par échantillon (CER B - CER A, en points) :")
for r in m.reindex(m.delta.abs().sort_values(ascending=False).index).head(8).itertuples():
    print(f"  {r.sample_id:11s} {r.delta:+7.1f}   A: {r.hypothesis_norm_a[:70]!r}\n  {'':11s} {'':7s}   B: {r.hypothesis_norm_b[:70]!r}")
