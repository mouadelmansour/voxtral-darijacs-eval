"""Graphiques du rapport (PNG). Toutes les valeurs mesurées viennent des metrics.json (vLLM et Transformers)
et des CSV produits par analysis/error_analysis.py ; les valeurs du model card sont lues sur benchmark_scores.png.

Usage :
  python analysis/make_figures.py --results results --analysis analysis/out --out docs/figures          # écran
  python analysis/make_figures.py --results results --analysis analysis/out --out docs/figures --print  # PDF A4
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CARD = "#8f8d86"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results", help="dossier contenant vllm_realtime/ et transformers/")
ap.add_argument("--analysis", default="analysis/out", help="sorties de analysis/error_analysis.py")
ap.add_argument("--out", default="docs/figures")
ap.add_argument("--print", dest="printable", action="store_true", help="mise en page portrait, 220 dpi (PDF A4)")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
DPI = 220 if A.printable else 160
M = json.load(open(os.path.join(A.results, "vllm_realtime", "metrics.json"), encoding="utf-8"))
MT = json.load(open(os.path.join(A.results, "transformers", "metrics.json"), encoding="utf-8"))
CARD_VAL = {"Prism": 6.68, "Probe": 9.69}
TF = {g: (MT["resultats_principaux"][g]["cer"]["taux_corpus_pct"], *MT["resultats_principaux"][g]["cer"]["ic95_bootstrap_pct"])
      for g in ("Prism", "Probe")}


def clean(ax, grid_axis="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fr(x):
    return f"{x:.2f}".replace(".", ",")


# --- Fig 1 : model card vs mesures -------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.6))
groups = ["Prism", "Probe"]
labels = ["DarijaCS-Prism\n(« ISMA » dans le model card)", "DarijaCS-Probe\n(« Darija in the Wild » dans le model card)"]
series = [("Model card (annoncé)", CARD, [(CARD_VAL[g], None, None) for g in groups]),
          ("Mesuré : vLLM Realtime 480 ms (protocole principal)", BLUE,
           [(M["resultats_principaux"][g]["cer"]["taux_corpus_pct"], *M["resultats_principaux"][g]["cer"]["ic95_bootstrap_pct"]) for g in groups]),
          ("Mesuré : Transformers (contrôle croisé)", ORANGE, [TF[g] for g in groups])]
w = 0.26
for k, (name, col, vals) in enumerate(series):
    for i, (v, lo, hi) in enumerate(vals):
        x = i + (k - 1) * (w + 0.02)
        ax.bar(x, v, w, color=col, label=name if i == 0 else None, zorder=2)
        if lo is not None:
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=4, linewidth=1, zorder=3)
        ax.text(x, (hi if hi else v) + 0.35, fr(v) + " %", ha="center", va="bottom", color=INK, fontsize=9)
ax.set_xticks(range(2), labels)
ax.set_ylabel("CER (%) - plus bas = meilleur")
ax.set_ylim(0, 18.5)
clean(ax)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("CER annoncé vs mesuré (normalisation officielle, délai 480 ms)\nbarres d'erreur : IC 95 % bootstrap",
             loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(A.out, "fig1_modelcard_vs_mesure.png"), dpi=DPI)

# --- Fig 2 : ventilations Prism ----------------------------------------------------------
V = M["ventilations"]["Prism"]
panels = [("Bruit", "Noise", ["clean", "little", "medium", "noise"]),
          ("Vitesse d'élocution", "Pace", ["slow", "medium", "fast", "non_annote"]),
          ("Code-switching", "Code_switching", ["0", "1"]),
          ("Domaine", "Domain", None)]
if A.printable:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
else:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.8, 1.6]})
axes = axes.flat if A.printable else axes
axes = list(axes)
for ax, (title, key, order) in zip(axes, panels):
    d = V[key]
    order = order or sorted(d, key=lambda k: d[k]["cer"]["taux_corpus_pct"])
    for j, k in enumerate(order):
        c = d[k]["cer"]; v = c["taux_corpus_pct"]; lo, hi = c["ic95_bootstrap_pct"]
        ax.barh(j, v, 0.6, color=BLUE, zorder=2)
        ax.errorbar(v, j, xerr=[[v - lo], [hi - v]], color=INK, capsize=3, linewidth=1, zorder=3)
        ax.text(hi + 0.4, j, f"{fr(v)} %", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(range(len(order)), [f"{k.replace('non_annote', 'non annoté')} (n={d[k]['n_echantillons']})" for k in order])
    ax.axvline(CARD_VAL["Prism"], color=CARD, linestyle="--", linewidth=1.2, zorder=1)
    ax.set_xlim(0, 25)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    clean(ax, "x")
axes[0].text(CARD_VAL["Prism"] + 0.3, -0.75, "6,68 % annoncé\n(ensemble)", color=INK2, fontsize=8, va="top")
fig.suptitle("DarijaCS-Prism : CER mesuré (vLLM, 480 ms) par condition - IC 95 %", x=0.01, ha="left", color=INK, fontsize=11.5)
fig.tight_layout()
fig.savefig(os.path.join(A.out, "fig2_prism_ventilations.png"), dpi=DPI)

# --- Fig 3 : Probe par type ----------------------------------------------------------------
d = M["ventilations"]["Probe"]["type (multi-étiquettes)"]
names = {"non_annote": "sans étiquette (parole de base)", "None": "« None » (+ autre étiquette)"}
order = sorted(d, key=lambda k: d[k]["cer"]["taux_corpus_pct"])
fig, ax = plt.subplots(figsize=(8.5, 4.4))
for j, k in enumerate(order):
    c = d[k]["cer"]; v = c["taux_corpus_pct"]; lo, hi = c["ic95_bootstrap_pct"]
    ax.barh(j, v, 0.6, color=BLUE, zorder=2)
    ax.errorbar(v, j, xerr=[[v - lo], [hi - v]], color=INK, capsize=3, linewidth=1, zorder=3)
    ax.text(hi + 0.4, j, f"{fr(v)} %", va="center", color=INK, fontsize=8.5)
ax.set_yticks(range(len(order)), [f"{names.get(k, k)} (n={d[k]['n_echantillons']})" for k in order])
ax.axvline(CARD_VAL["Probe"], color=CARD, linestyle="--", linewidth=1.2)
ax.text(CARD_VAL["Probe"] + 0.3, -0.9, "9,69 % annoncé (ensemble)", color=INK2, fontsize=8, va="center")
ax.set_ylim(-1.3, len(order) - 0.5)
ax.set_xlim(0, 34)
ax.set_xlabel("CER (%)")
clean(ax, "x")
ax.set_title("DarijaCS-Probe : CER mesuré (vLLM, 480 ms) par type de difficulté\n(un échantillon peut porter plusieurs étiquettes) - IC 95 %",
             loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(A.out, "fig3_probe_types.png"), dpi=DPI)

# --- Fig 4 : catégories d'erreurs de mots -----------------------------------------------
C = pd.read_csv(os.path.join(A.analysis, "categories_erreurs_mots.csv"))
def grp(k):
    if k.startswith("quasi"): return "Variante orthographique / clitique"
    if k.startswith("changement"): return "Changement d'écriture (latin ↔ arabe)"
    if k.startswith("substitution"): return "Substitution franche"
    if k.startswith("suppression"): return "Suppression"
    return "Insertion"
C["groupe"] = C.categorie.map(grp)
G = C.groupby(["subset", "groupe"]).n_mots.sum().unstack(fill_value=0)
cats = ["Variante orthographique / clitique", "Substitution franche", "Changement d'écriture (latin ↔ arabe)", "Suppression", "Insertion"]
cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
fig, ax = plt.subplots(figsize=(10, 3.2))
for i, s in enumerate(["Prism", "Probe"]):
    tot = G.loc[s].sum(); left = 0
    for cat, col in zip(cats, cols):
        p = 100 * G.loc[s, cat] / tot
        ax.barh(i, p, 0.55, left=left, color=col, edgecolor=SURF, linewidth=2, label=cat if i == 0 else None)
        if p >= 7:
            ax.text(left + p / 2, i, f"{p:.0f} %", ha="center", va="center", color=INK, fontsize=9)
        left += p
ax.set_yticks([0, 1], [f"Prism ({int(G.loc['Prism'].sum())} erreurs)", f"Probe ({int(G.loc['Probe'].sum())} erreurs)"])
ax.set_xlim(0, 100); ax.invert_yaxis(); ax.set_xlabel("part des erreurs de mots (%)")
clean(ax, "x")
ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.32), fontsize=9)
ax.set_title("Nature des erreurs de mots (vLLM, après normalisation officielle)", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(A.out, "fig4_categories_erreurs.png"), dpi=DPI)

# --- Fig 5 : robustesse de l'écart -------------------------------------------------------
var = pd.read_csv(os.path.join(A.analysis, "cer_variantes.csv"))
var = var[~((var.subset == "Probe") & var.variante.str.contains("prism_"))]
var["variante"] = var.variante.str.replace("Officielle, corpus, sans prism_246 ni prism_133", "Officielle, sans prism_246 ni prism_133")
SB = "Probe_annotations_crochets_retirees_avant_normalisation"
extra = pd.DataFrame([
    ("Probe", "Officielle, annotations [..] retirées", M["sensibilite"][SB]["cer"]["taux_corpus_pct"]),
    ("Prism", "Transformers (contrôle)", TF["Prism"][0]),
    ("Probe", "Transformers (contrôle)", TF["Probe"][0]),
    ("Probe", "Transformers + annotations [..] retirées", MT["sensibilite"][SB]["cer"]["taux_corpus_pct"]),
], columns=["subset", "variante", "CER_%"])
var = pd.concat([var[["subset", "variante", "CER_%"]], extra], ignore_index=True)
fig, axes = plt.subplots(2, 1, figsize=(9.5, 9.8), sharex=True) if A.printable else plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
for ax, s in zip(axes, ["Prism", "Probe"]):
    v = var[var.subset == s].sort_values("CER_%", ascending=False).reset_index(drop=True)
    for j, r in v.iterrows():
        main = r.variante.startswith("Principal")
        ax.plot([CARD_VAL[s], r["CER_%"]], [j, j], color=GRID, linewidth=2, zorder=1)
        ax.scatter(r["CER_%"], j, s=70 if main else 45, color=BLUE, zorder=3, edgecolor=SURF, linewidth=2)
        ax.text(r["CER_%"] + 0.3, j, fr(r["CER_%"]), va="center", fontsize=8.5, color=INK)
    ax.axvline(CARD_VAL[s], color=CARD, linestyle="--", linewidth=1.4)
    ax.text(CARD_VAL[s] - 0.2, -0.9, f"annoncé {fr(CARD_VAL[s])} %", ha="right", va="center", color=INK2, fontsize=8.5)
    ax.set_ylim(-1.3, len(v) - 0.5)
    ax.set_yticks(range(len(v)), v.variante, fontsize=8.5)
    ax.set_xlim(4, 18)
    ax.set_title(f"DarijaCS-{s}", loc="left", color=INK, fontsize=10.5)
    clean(ax, "x")
    ax.set_xlabel("CER (%)")
fig.suptitle("Aucune variante de calcul testée ne rejoint la valeur annoncée", x=0.01, ha="left", color=INK, fontsize=11.5)
fig.tight_layout()
fig.savefig(os.path.join(A.out, "fig5_robustesse_ecart.png"), dpi=DPI)
print("ok")
