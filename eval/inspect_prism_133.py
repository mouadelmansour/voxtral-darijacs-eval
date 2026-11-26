"""Examen de prism_133 (aucune transcription produite) : sorties de chaque exécution et niveau sonore,
comparé à ses voisins prism_132 et prism_134.

Usage : python eval/inspect_prism_133.py [run_principal] [run_transformers]
Sans argument : les exécutions du cluster (run_20261112_0440 et verif_20261112_2307)."""
import os, sys
import numpy as np, pandas as pd, soundfile as sf

if len(sys.argv) > 1:
    main = sys.argv[1]
    tf = sys.argv[2] if len(sys.argv) > 2 else os.path.join(main, "transformers")
    runs = [("vLLM (principal)", f"{main}/predictions_scored.csv"), ("Transformers", f"{tf}/predictions_scored.csv"),
            ("vLLM répétition", f"{main}/verification/A_repetition/predictions_scored.csv"),
            ("vLLM gauche+soxr", f"{main}/verification/C_prism_gauche_soxr/predictions_scored.csv")]
else:
    runs = [("vLLM (principal)",   "eval/runs/run_20261112_0440/predictions_scored.csv"),
            ("Transformers",       "eval/runs/run_20261112_0440/transformers/predictions_scored.csv"),
            ("vLLM répétition",    "eval/runs/verif_20261112_2307/A_repetition/predictions_scored.csv"),
            ("vLLM gauche+soxr",   "eval/runs/verif_20261112_2307/C_prism_gauche_soxr/predictions_scored.csv")]
r = None
for name, p in runs:
    if not os.path.exists(p):
        print(f"{name:18s} (absent : {p})"); continue
    r = pd.read_csv(p, dtype={"sample_id": str}, keep_default_na=False).set_index("sample_id").loc["prism_133"]
    print(f"{name:18s} CER = {100*float(r.cer):5.1f} %   transcription = {r.hypothesis_raw[:150]!r}")
if r is not None:
    print("\nRÉFÉRENCE :", r.reference)

def niveau(path):
    x, sr = sf.read(path)
    rms = lambda v: 20*np.log10(np.sqrt(np.mean(v**2)) + 1e-12)
    return len(x)/sr, rms(x), np.abs(x).max(), [round(float(rms(x[i:i+sr])), 1) for i in range(0, len(x), sr)]
for sid in ["prism_133", "prism_132", "prism_134"]:
    d, g, pk, per_s = niveau(f"data-test/prepared/Prism/{sid}.wav")
    print(f"\n{sid} : {d:.1f} s | niveau moyen {g:.1f} dBFS | pic {pk:.2f}\n  niveau par seconde (dBFS) : {per_s}")
