"""Construit la page web de l'analyse (un seul fichier HTML autonome) à partir des résultats.

Usage :
  python analysis/build_site.py --results results --analysis analysis/out --out docs/site/index.html

Entrées : results/vllm_realtime/{metrics.json,predictions_scored.csv}, results/transformers/metrics.json,
sorties de analysis/error_analysis.py. Seules les valeurs du model card (lues sur ses figures
benchmark_scores.png et avg_evals_by_delay.png) sont écrites dans ce script.
"""
import argparse, json, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--analysis", default="analysis/out")
ap.add_argument("--template", default=os.path.join(os.path.dirname(__file__), "site", "template.html"))
ap.add_argument("--out", default="docs/site/index.html")
a = ap.parse_args()

M = json.load(open(os.path.join(a.results, "vllm_realtime", "metrics.json"), encoding="utf-8"))
MT = json.load(open(os.path.join(a.results, "transformers", "metrics.json"), encoding="utf-8"))
df = pd.read_csv(os.path.join(a.results, "vllm_realtime", "predictions_scored.csv"), dtype={"sample_id": str}, keep_default_na=False)


def pick(m):
    c, w = m["cer"], m["wer"]
    return {"n": m["n_echantillons"], "h": m["duree_h"], "cer": c["taux_corpus_pct"], "lo": c["ic95_bootstrap_pct"][0],
            "hi": c["ic95_bootstrap_pct"][1], "cerMacro": c["moyenne_par_echantillon_pct"], "wer": w["taux_corpus_pct"],
            "werLo": w["ic95_bootstrap_pct"][0], "werHi": w["ic95_bootstrap_pct"][1],
            "S": c["S"], "D": c["D"], "I": c["I"], "N": c["N"], "sdi": c["part_S_D_I_pct"]}


main = {k: pick(v) for k, v in M["resultats_principaux"].items()}
vent = {"Prism": {k: {g: pick(x) for g, x in v.items()} for k, v in M["ventilations"]["Prism"].items()},
        "Probe": {g: pick(x) for g, x in M["ventilations"]["Probe"]["type (multi-étiquettes)"].items()}}
sens = {k: pick(v) for k, v in M["sensibilite"].items()}
tfm = {g: MT["resultats_principaux"][g] for g in ("Prism", "Probe")}
tfs = MT["sensibilite"]
TF = {g: {"cer": tfm[g]["cer"]["taux_corpus_pct"], "lo": tfm[g]["cer"]["ic95_bootstrap_pct"][0],
          "hi": tfm[g]["cer"]["ic95_bootstrap_pct"][1], "wer": tfm[g]["wer"]["taux_corpus_pct"]} for g in tfm}

var = pd.read_csv(os.path.join(a.analysis, "cer_variantes.csv"))
var = var[~((var.subset == "Probe") & var.variante.str.contains("prism_"))]
variants = [{"subset": r.subset, "label": r.variante, "cer": r["CER_%"]} for _, r in var.iterrows()]
variants += [
    {"subset": "Prism", "label": "Transformers (contrôle)", "cer": TF["Prism"]["cer"]},
    {"subset": "Probe", "label": "Officielle, annotations [..] retirées",
     "cer": M["sensibilite"]["Probe_annotations_crochets_retirees_avant_normalisation"]["cer"]["taux_corpus_pct"]},
    {"subset": "Probe", "label": "Transformers (contrôle)", "cer": TF["Probe"]["cer"]},
    {"subset": "Probe", "label": "Transformers + annotations [..] retirées",
     "cer": tfs["Probe_annotations_crochets_retirees_avant_normalisation"]["cer"]["taux_corpus_pct"]},
]

cats = pd.read_csv(os.path.join(a.analysis, "categories_erreurs_mots.csv"))
def grp(k):
    if k.startswith("quasi"): return "Variante orthographique / clitique"
    if k.startswith("changement"): return "Changement d'écriture (latin ↔ arabe)"
    if k.startswith("substitution"): return "Substitution franche"
    if k.startswith("suppression"): return "Suppression"
    return "Insertion"
cats["g"] = cats.categorie.map(grp)
catd = {s: {g: int(v) for g, v in d.groupby("g").n_mots.sum().items()} for s, d in cats.groupby("subset")}
script = pd.read_csv(os.path.join(a.analysis, "erreurs_par_ecriture.csv")).to_dict(orient="records")

ex_ids = [("prism_280", "Code-switching : mots français écrits en arabe ; nom propre (موكادور → مقادور)"),
          ("probe_23", "Code-switching technique transcrit en écriture arabe"),
          ("probe_66", "Tech CS : écriture arabe et ordre des mots modifié"),
          ("prism_84", "Audio bruité : substitutions franches, termes scientifiques déformés"),
          ("probe_54", "Clip court (2,2 s) bruité : segmentation erronée de mots anglais"),
          ("probe_113", "Annotation non prononcée comptée en suppression ; emprunts en écriture arabe"),
          ("prism_3", "Nombre et expression idiomatique (« لا ستة حمص ») mal reconnus"),
          ("prism_247", "Variantes orthographiques et clitiques seulement"),
          ("prism_246", "Référence erronée dans le dataset (copie de prism_247)"),
          ("prism_133", "Aucune transcription (vLLM et Transformers) : parole presque inaudible sous le bruit"),
          ("probe_109", "Toponymes marocains reconnus sans erreur")]
d = df.set_index("sample_id")
examples = [{"id": i, "type": t, "ref": d.loc[i, "reference_norm"], "hyp": d.loc[i, "hypothesis_norm"],
             "cer": round(100 * float(d.loc[i, "cer"]), 1)} for i, t in ex_ids if i in d.index]

rows = []
for r in df.itertuples():
    tag = r.meta_type if r.subset == "Probe" else f"{r.meta_Domain} · bruit {r.meta_Noise} · {r.meta_Pace or 'vitesse ?'}"
    rows.append([r.sample_id, r.subset, round(100 * float(r.cer), 1), round(100 * float(r.wer), 1),
                 round(float(r.duration_s), 1), tag or "sans étiquette", r.reference_norm, r.hypothesis_norm,
                 round(float(r.inference_time_s), 2) if str(r.inference_time_s) else None])

DATA = {
    "main": main, "vent": vent, "sens": sens, "variants": variants, "cats": catd, "script": script,
    "examples": examples, "rows": rows, "time": M["temps_inference"],
    "tf": {**TF, "sens": {"brackets": tfs["Probe_annotations_crochets_retirees_avant_normalisation"]["cer"]["taux_corpus_pct"],
                          "no30": tfs["Probe_sans_probe_30"]["cer"]["taux_corpus_pct"]}},
    # Valeurs du model card, lues sur benchmark_scores.png et avg_evals_by_delay.png
    "card": {"Realtime": {"Moyenne 7 benchmarks": 8.82, "ISMA (= Prism)": 6.68, "Darija in the Wild (= Probe)": 9.69, "MGB-2": 5.31,
                          "SADA": 13.75, "Casablanca": 20.34, "FLEURS": 2.66, "MASC clean": 3.33},
             "Transcribe": {"Moyenne 7 benchmarks": 7.91, "ISMA (= Prism)": 4.96, "Darija in the Wild (= Probe)": 8.97, "MGB-2": 4.83,
                            "SADA": 12.25, "Casablanca": 18.60, "FLEURS": 2.27, "MASC clean": 3.50},
             "delay": [[80, 10.64], [160, 9.76], [320, 9.17], [480, 8.82], [720, 8.80]]},
}
tpl = open(a.template, encoding="utf-8").read()
assert tpl.count("/*DATA*/") == 1
data = json.dumps(DATA, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
open(a.out, "w", encoding="utf-8").write(tpl.replace("/*DATA*/", data))
print("Page écrite :", a.out, f"({len(rows)} transcriptions)")
