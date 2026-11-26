"""Construit la page web de l'analyse (un seul fichier HTML autonome) à partir des résultats.

Usage :
  python analysis/build_site.py --results results --analysis analysis/out --out docs/site/index.html

Entrées : results/vllm_realtime/{metrics.json,predictions_scored.csv}, results/transformers/metrics.json,
sorties de analysis/error_analysis.py. Seules les valeurs du model card (lues sur ses figures
benchmark_scores.png) sont écrites dans ce script.
"""
import argparse, json, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--analysis", default="analysis/out")
ap.add_argument("--template", default=os.path.join(os.path.dirname(__file__), "site", "template.html"))
ap.add_argument("--out", default="docs/site/index.html")
ap.add_argument("--repro", default="results/run_repro_20261125",
                help="exécution complète contenant verification/ et analysis/ (tests de robustesse)")
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
    # Valeurs du model card, lues sur benchmark_scores.png
    "card": {"Realtime": {"ISMA (= Prism)": 6.68, "Darija in the Wild (= Probe)": 9.69, "MGB-2": 5.31,
                          "SADA": 13.75, "Casablanca": 20.34, "FLEURS": 2.66, "MASC clean": 3.33},
             "Transcribe": {"ISMA (= Prism)": 4.96, "Darija in the Wild (= Probe)": 8.97, "MGB-2": 4.83,
                            "SADA": 12.25, "Casablanca": 18.60, "FLEURS": 2.27, "MASC clean": 3.50}},
}


def verification(repro):
    """Tests de robustesse : lus dans verification/*/metrics.json, verification/SYNTHESE.txt,
    analysis/comparaison_original.txt et analysis/analyse_canaux_prism.csv."""
    import re
    V = os.path.join(repro, "verification")
    if not os.path.isdir(V):
        return None
    def res(d, sub):
        m = json.load(open(os.path.join(V, d, "metrics.json"), encoding="utf-8"))["resultats_principaux"][sub]
        return {"cer": m["cer"]["taux_corpus_pct"], "lo": m["cer"]["ic95_bootstrap_pct"][0], "hi": m["cer"]["ic95_bootstrap_pct"][1],
                "wer": m["wer"]["taux_corpus_pct"], "n": m["n_echantillons"], "h": m["duree_h"]}
    def identical(path):
        out, cur = {}, None
        for line in open(path, encoding="utf-8"):
            h = re.match(r"=== (.+?)\s+vs\s+(.+?) : ", line)
            if h:
                cur = h.group(2).strip(); out[cur] = {}
            m = re.match(r"(\w+)\s+transcriptions identiques : (\d+)/(\d+)", line)
            if m and cur:
                out[cur][m.group(1)] = [int(m.group(2)), int(m.group(3))]
        return out
    syn = identical(os.path.join(V, "SYNTHESE.txt"))
    indep = []
    for line in open(os.path.join(V, "SYNTHESE.txt"), encoding="utf-8"):
        m = re.match(r"(Prism|Probe)\s+n=(\d+)\s+CER=([\d.]+) %", line)
        if m:
            indep.append({"subset": m.group(1), "n": int(m.group(2)), "cer": float(m.group(3))})
    rep = identical(os.path.join(repro, "analysis", "comparaison_original.txt"))
    ch = pd.read_csv(os.path.join(repro, "analysis", "analyse_canaux_prism.csv"))
    st = ch[ch["canaux"] == 2]
    suspect = (st["corr_gd"] < 0.5) | (st["diff"] > st["moyenne"] - 3) | ((st["gauche"] - st["droite"]).abs() > 6)
    return {"fleurs": res("D_fleurs", "FLEURS"), "fleursTr": res("D_fleurs_transcription", "FLEURS"),
            "C": res("C_prism_gauche_soxr", "Prism"),
            "A": syn.get("A répétition"), "B": syn.get("B pace 1"), "E": syn.get("E format Mistral"),
            "TF": syn.get("Transformers"), "repro": next(iter(rep.values()), None), "independent": indep,
            "channels": {"stereo": int(len(st)), "mono": int((ch["canaux"] == 1).sum()), "suspects": int(suspect.sum()),
                         "corrMin": round(float(st["corr_gd"].min()), 2), "corrMed": round(float(st["corr_gd"].median()), 2),
                         "cerMono": round(float(ch[ch["canaux"] == 1]["cer"].mean()), 2), "cerStereo": round(float(st["cer"].mean()), 2)}}


DATA["verif"] = verification(a.repro)
tpl = open(a.template, encoding="utf-8").read()
assert tpl.count("/*DATA*/") == 1
data = json.dumps(DATA, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
open(a.out, "w", encoding="utf-8").write(tpl.replace("/*DATA*/", data))
print("Page écrite :", a.out, f"({len(rows)} transcriptions)")
