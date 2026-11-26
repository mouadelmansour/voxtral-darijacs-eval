"""Calcul du CER / WER sur DarijaCS (Prism et Probe, toujours séparés).

Résultat principal : normalisation officielle `normalize_arabic_asr` appliquée à la
référence ET à l'hypothèse, références non modifiées, tous les échantillons.
Les analyses de sensibilité sont calculées à part et étiquetées comme telles.

Usage :
  python compute_metrics.py \
      --manifests data-test/prepared/manifest_prism.csv data-test/prepared/manifest_probe.csv \
      --predictions eval/runs/RUN/predictions.csv \
      --normalization-dir Voxtral-Mini-4B-Realtime-Arabic \
      --out eval/runs/RUN
"""
import argparse, hashlib, json, os, re, sys
import numpy as np
import pandas as pd
import jiwer

BRACKETS = re.compile(r"\[[^\]]*\]")
THIRD_PARTY_COL = "meta_darija-stt-solutions-2607_tr"


def edit_counts(ref, hyp, level):
    """Retourne S, D, I, N pour une paire (ref, hyp) déjà normalisée."""
    if ref == "":
        n_ins = len(hyp) if level == "char" else len(hyp.split())
        return 0, 0, n_ins, 0
    o = (jiwer.process_characters if level == "char" else jiwer.process_words)(ref, hyp)
    return o.substitutions, o.deletions, o.insertions, o.substitutions + o.deletions + o.hits


def score(df, ref_col, hyp_col):
    df = df.copy()
    for lvl, p in (("char", "c"), ("word", "w")):
        r = [edit_counts(a, b, lvl) for a, b in zip(df[ref_col], df[hyp_col])]
        df[[f"{p}_S", f"{p}_D", f"{p}_I", f"{p}_N"]] = pd.DataFrame(r, index=df.index)
        n = df[f"{p}_N"].replace(0, np.nan)
        df["cer" if p == "c" else "wer"] = (df[f"{p}_S"] + df[f"{p}_D"] + df[f"{p}_I"]) / n
    return df


def bootstrap_ci(err, n, iters=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), size=(iters, len(err)))
    rates = err[idx].sum(1) / n[idx].sum(1)
    return [round(float(np.percentile(rates, 2.5)) * 100, 2), round(float(np.percentile(rates, 97.5)) * 100, 2)]


def summarize(df):
    """Métriques agrégées au niveau corpus (somme des éditions / somme des longueurs de référence)."""
    d = df[df["c_N"] > 0]
    out = {"n_echantillons": int(len(df)), "n_echantillons_scores": int(len(d)),
           "duree_h": round(float(df["duration_s"].sum()) / 3600, 3)}
    for p, name in (("c", "cer"), ("w", "wer")):
        S, D, I, N = (int(d[f"{p}_{k}"].sum()) for k in "SDIN")
        err = (d[f"{p}_S"] + d[f"{p}_D"] + d[f"{p}_I"]).to_numpy()
        out[name] = {
            "taux_corpus_pct": round(100 * (S + D + I) / N, 2) if N else None,
            "ic95_bootstrap_pct": bootstrap_ci(err, d[f"{p}_N"].to_numpy()) if N else None,
            "moyenne_par_echantillon_pct": round(100 * float(d[name].mean()), 2) if len(d) else None,
            "S": S, "D": D, "I": I, "N": N,
            "part_S_D_I_pct": [round(100 * x / max(S + D + I, 1), 1) for x in (S, D, I)],
        }
    empty = df[df["c_N"] == 0]
    if len(empty):
        out["references_vides_insertions_car"] = {r.sample_id: int(r.c_I) for r in empty.itertuples()}
    return out


def by_group(df, col, multilabel=False):
    s = df[col].map(lambda v: "non_annote" if pd.isna(v) else
                    (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()))
    if multilabel:
        tags = s.str.split(",").apply(lambda xs: sorted({x.strip() for x in xs if x.strip()}))
        groups = sorted({t for ts in tags for t in ts})
        return {g: summarize(df[tags.apply(lambda ts: g in ts)]) for g in groups}
    return {g: summarize(df[s == g]) for g in sorted(s.unique())}


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--normalization-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.normalization_dir))
    from normalization import normalize_arabic_asr as norm

    man = pd.concat([pd.read_csv(m, dtype={"sample_id": str}) for m in a.manifests], ignore_index=True)
    pred = pd.read_csv(a.predictions, dtype={"sample_id": str}, keep_default_na=False)
    missing = sorted(set(man.sample_id) - set(pred.sample_id))
    extra = sorted(set(pred.sample_id) - set(man.sample_id))
    if missing or extra:
        sys.exit(f"ERREUR : {len(missing)} prédictions manquantes {missing[:10]}, "
                 f"{len(extra)} prédictions inconnues {extra[:10]}. Aucun échantillon n'est ignoré en silence.")
    df = man.merge(pred, on="sample_id", how="left", validate="one_to_one")
    df["hypothesis_raw"] = df["hypothesis_raw"].fillna("").astype(str)
    df["reference_norm"] = df["reference"].map(norm)
    df["hypothesis_norm"] = df["hypothesis_raw"].map(norm)
    df = score(df, "reference_norm", "hypothesis_norm")

    metrics = {"protocole_metriques": {
        "normalisation": "normalize_arabic_asr (normalization.py officiel) sur référence et hypothèse",
        "cer": "corpus : somme(S+D+I) / somme(N caractères de référence), espaces inclus (jiwer.process_characters)",
        "wer": "corpus : somme(S+D+I) / somme(N mots de référence) (jiwer.process_words)",
        "ic95": "bootstrap percentile sur les échantillons, 1000 tirages, graine 0",
        "jiwer": getattr(jiwer, "__version__", "voir pip freeze"),
    }, "resultats_principaux": {}, "ventilations": {}, "sensibilite": {}}

    prism, probe = df[df.subset == "Prism"], df[df.subset == "Probe"]
    for name, sub in (("Prism", prism), ("Probe", probe)):
        if len(sub):
            metrics["resultats_principaux"][name] = summarize(sub)

    if len(prism):
        metrics["ventilations"]["Prism"] = {c.replace("meta_", ""): by_group(prism, c)
                                            for c in ("meta_Domain", "meta_Code_switching", "meta_Pace", "meta_Noise")
                                            if c in prism}
    if len(probe) and "meta_type" in probe:
        metrics["ventilations"]["Probe"] = {"type (multi-étiquettes)": by_group(probe, "meta_type", multilabel=True)}

    # --- Analyses de sensibilité (jamais présentées comme résultat principal) ---
    sens = metrics["sensibilite"]
    if len(probe):
        p2 = probe.copy()
        p2["reference_norm"] = p2["reference"].map(lambda t: norm(BRACKETS.sub(" ", t)))
        sens["Probe_annotations_crochets_retirees_avant_normalisation"] = summarize(score(p2, "reference_norm", "hypothesis_norm"))
        sens["Probe_sans_probe_30"] = summarize(probe[probe.sample_id != "probe_30"])
        if THIRD_PARTY_COL in probe:
            p3 = probe.copy()
            p3["third_norm"] = p3[THIRD_PARTY_COL].fillna("").astype(str).map(norm)
            sens["Probe_systeme_tiers_colonne_dataset_darija-stt-solutions-2607"] = summarize(score(p3, "reference_norm", "third_norm"))
    if len(prism):
        h = prism["audio_path"].map(md5)
        dup = prism[(h.duplicated()) & (prism["reference"].duplicated())]
        sens["Prism_doublons_exacts_retires"] = {"retires": dup.sample_id.tolist(),
                                                 **summarize(prism.drop(dup.index))}

    if "inference_time_s" in df:
        t = pd.to_numeric(df["inference_time_s"], errors="coerce")
        metrics["temps_inference"] = {
            "moyenne_s_par_echantillon": round(float(t.mean()), 3),
            "total_s": round(float(t.sum()), 1),
            "rtf_global": round(float(t.sum() / df["duration_s"].sum()), 4),
        }

    os.makedirs(a.out, exist_ok=True)
    keep = ["sample_id", "subset", "duration_s", "reference", "hypothesis_raw", "reference_norm", "hypothesis_norm",
            "cer", "wer", "c_S", "c_D", "c_I", "c_N", "w_S", "w_D", "w_I", "w_N"]
    keep += [c for c in df.columns if c.startswith("meta_") or c == "inference_time_s"]
    df[keep].to_csv(os.path.join(a.out, "predictions_scored.csv"), index=False)
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    for name, m in metrics["resultats_principaux"].items():
        print(f"{name:6s} n={m['n_echantillons']:4d}  CER={m['cer']['taux_corpus_pct']}% IC95={m['cer']['ic95_bootstrap_pct']}  "
              f"WER={m['wer']['taux_corpus_pct']}%")
    for k, m in sens.items():
        print(f"[sensibilité] {k}: CER={m['cer']['taux_corpus_pct']}%  WER={m['wer']['taux_corpus_pct']}%")


if __name__ == "__main__":
    main()
