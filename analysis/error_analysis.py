"""Analyses complémentaires sur predictions_scored.csv (sortie de eval/compute_metrics.py) :

  cer_variantes.csv              CER selon plusieurs méthodes de calcul (section « D'où vient l'écart »)
  erreurs_par_ecriture.csv       taux d'erreur des mots de référence en écriture arabe / latine / chiffres
  categories_erreurs_mots.csv    nature des erreurs de mots (variante, substitution franche, changement d'écriture...)
  integrite.txt                  sorties vides, position des suppressions (troncature en fin de flux ?),
                                 cas où la prédiction correspond mieux à une autre référence

Usage : python analysis/error_analysis.py results/vllm_realtime/predictions_scored.csv analysis/out
Les classements de la partie « catégories » sont heuristiques (seuil de Levenshtein, détection d'écriture
par caractères) : les proportions sont des ordres de grandeur.
"""
import collections, os, re, sys, unicodedata
import jiwer
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
df = pd.read_csv(src, dtype={"sample_id": str}, keep_default_na=False)
CARD = {"Prism": 6.68, "Probe": 9.69}          # valeurs du model card (benchmark_scores.png)

# ---------------------------------------------------------------- 1. variantes de calcul du CER
def corpus(g, rc, hc, f=lambda x: x):
    R = [f(x) for x in g[rc]]; H = [f(x) for x in g[hc]]
    keep = [i for i, x in enumerate(R) if x.strip()]
    return round(100 * jiwer.process_characters([R[i] for i in keep], [H[i] for i in keep]).cer, 2)

def macro(g, rc, hc, f=lambda x: x):
    v = [jiwer.process_characters(f(a), f(b)).cer for a, b in zip(g[rc], g[hc]) if f(a).strip()]
    return round(100 * float(np.mean(v)), 2)

nosp = lambda x: re.sub(r"\s+", "", x)
def light(x):  # normalisation minimale, NON officielle : minuscules, ponctuation retirée
    x = unicodedata.normalize("NFKC", x).lower(); x = re.sub(r"[^\w\s]", " ", x); return " ".join(x.split())

rows = []
for s, g in df.groupby("subset"):
    g2 = g[g.sample_id != "prism_133"]
    rows += [
        (s, "Principal : officielle, corpus, espaces inclus", corpus(g, "reference_norm", "hypothesis_norm")),
        (s, "Officielle, corpus, sans espaces", corpus(g, "reference_norm", "hypothesis_norm", nosp)),
        (s, "Officielle, moyenne par échantillon", macro(g, "reference_norm", "hypothesis_norm")),
        (s, "Officielle, moyenne par échantillon, sans espaces", macro(g, "reference_norm", "hypothesis_norm", nosp)),
        (s, "Officielle, corpus, sans prism_133 (sortie vide)", corpus(g2, "reference_norm", "hypothesis_norm")),
        (s, "Officielle, corpus, sans prism_246 ni prism_133", corpus(g[~g.sample_id.isin(["prism_246", "prism_133"])], "reference_norm", "hypothesis_norm")),
        (s, "Normalisation minimale (non officielle)", corpus(g, "reference", "hypothesis_raw", light)),
        (s, "Aucune normalisation (textes bruts)", corpus(g, "reference", "hypothesis_raw", lambda x: " ".join(x.split()))),
    ]
var = pd.DataFrame(rows, columns=["subset", "variante", "CER_%"])
var["model_card_%"] = var.subset.map(CARD)
var["ecart_pp"] = (var["CER_%"] - var["model_card_%"]).round(2)
var.to_csv(os.path.join(out, "cer_variantes.csv"), index=False)

# ---------------------------------------------------------------- 2. erreurs par écriture
LAT = re.compile(r"[A-Za-zÀ-ÿ]"); ARA = re.compile(r"[؀-ۿ]"); DIG = re.compile(r"\d")
def script(w):
    if LAT.search(w) and not ARA.search(w): return "latin"
    if DIG.fullmatch(w) or (DIG.search(w) and not ARA.search(w) and not LAT.search(w)): return "chiffre"
    if ARA.search(w): return "arabe"
    return "autre"

stats = collections.defaultdict(collections.Counter)
for r in df.itertuples():
    ref, hyp = r.reference_norm.split(), r.hypothesis_norm.split()
    for c in jiwer.process_words(r.reference_norm, r.hypothesis_norm).alignments[0]:
        if c.type == "insert":
            continue
        R = ref[c.ref_start_idx:c.ref_end_idx]
        H = hyp[c.hyp_start_idx:c.hyp_end_idx] if c.type != "delete" else []
        for i, w in enumerate(R):
            k = (r.subset, script(w)); stats[k]["total"] += 1
            if c.type == "equal": stats[k]["correct"] += 1
            elif c.type == "delete": stats[k]["supprime"] += 1
            else:
                hs = script(H[i]); sc = script(w)
                if sc == "latin" and hs == "arabe": stats[k]["vers_arabe"] += 1
                elif sc == "arabe" and hs == "latin": stats[k]["vers_latin"] += 1
                else: stats[k]["meme"] += 1
ecr = pd.DataFrame([dict(subset=s, ecriture_ref=sc, mots=c["total"],
                         correct_pct=round(100 * c["correct"] / c["total"], 1),
                         erreur_pct=round(100 * (c["total"] - c["correct"]) / c["total"], 1),
                         subst_meme_ecriture=c["meme"], subst_vers_autre_ecriture=c["vers_arabe"] + c["vers_latin"],
                         supprime=c["supprime"]) for (s, sc), c in sorted(stats.items())])
ecr.to_csv(os.path.join(out, "erreurs_par_ecriture.csv"), index=False)

# ---------------------------------------------------------------- 3. catégories d'erreurs de mots
def sc2(w): return "latin" if LAT.search(w) and not ARA.search(w) else ("arabe" if ARA.search(w) else "autre")
cat = collections.defaultdict(collections.Counter)
for r in df.itertuples():
    ref, hyp = r.reference_norm.split(), r.hypothesis_norm.split()
    for c in jiwer.process_words(r.reference_norm, r.hypothesis_norm).alignments[0]:
        if c.type == "equal": continue
        if c.type == "insert":
            cat[r.subset]["insertion"] += c.hyp_end_idx - c.hyp_start_idx; continue
        R = ref[c.ref_start_idx:c.ref_end_idx]
        if c.type == "delete":
            for w in R: cat[r.subset]["suppression_latin" if sc2(w) == "latin" else "suppression_arabe/autre"] += 1
            continue
        for w, h in zip(R, hyp[c.hyp_start_idx:c.hyp_end_idx]):
            a, b = sc2(w), sc2(h)
            if a != b and "autre" not in (a, b): k = f"changement_ecriture ({a}->{b})"
            elif Levenshtein.normalized_distance(w, h) <= 0.34: k = f"quasi-identique ({a}, variante/clitique)"
            else: k = f"substitution_franche ({a})"
            cat[r.subset][k] += 1
pd.DataFrame([(s, k, v, round(100 * v / sum(c.values()), 1)) for s, c in cat.items() for k, v in c.most_common()],
             columns=["subset", "categorie", "n_mots", "part_pct"]).to_csv(os.path.join(out, "categories_erreurs_mots.csv"), index=False)

# ---------------------------------------------------------------- 4. contrôles d'intégrité
lines = []
empty = df[df.hypothesis_norm.str.strip() == ""]
lines.append(f"Sorties vides : {len(empty)} -> {', '.join(empty.sample_id)}")
pos, tails = collections.defaultdict(list), []
for r in df.itertuples():
    al = jiwer.process_words(r.reference_norm, r.hypothesis_norm).alignments[0]
    n = max(len(r.reference_norm.split()), 1)
    for c in al:
        if c.type == "delete": pos[r.subset] += [i / n for i in range(c.ref_start_idx, c.ref_end_idx)]
    t = al[-1]; tails.append(t.ref_end_idx - t.ref_start_idx if t.type == "delete" else 0)
for s, p in pos.items():
    p = np.array(p)
    lines.append(f"{s} : {len(p)} mots supprimés, dont {100*(p >= .9).mean():.1f} % dans les 10 % finaux de la référence")
lines.append(f"Suppressions en fin d'enregistrement : {int((np.array(tails) > 0).sum())} enregistrements, {int(sum(tails))} mots")
for s, g in df.groupby("subset"):
    g = g.reset_index(drop=True)
    for _, r in g[g.cer > (0.25 if s == "Prism" else 0.3)].iterrows():
        if not r.hypothesis_norm.strip(): continue
        best = min((jiwer.cer(x, r.hypothesis_norm), g.sample_id[j]) for j, x in enumerate(g.reference_norm) if x.strip())
        if best[1] == r.sample_id: flag = ""
        elif best[0] < 0.3: flag = "  <-- correspond à une autre référence (référence décalée ?)"
        else: flag = "  <-- ne correspond à aucune référence du dataset (référence erronée ?)"
        lines.append(f"{r.sample_id}: CER propre {r.cer:.3f} | meilleure référence {best[1]} ({best[0]:.3f}){flag}")
open(os.path.join(out, "integrite.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines[:4]))
print("Écrit dans", out)
