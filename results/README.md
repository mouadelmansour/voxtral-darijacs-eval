# Résultats

## `run_20261112_0440/` : évaluation principale du 12 novembre 2026 (fichiers originaux)

Exécution vLLM Realtime (délai 480 ms, température 0) du job PBS 13530, nœud gpu06 (NVIDIA L40S).

| Fichier | Contenu |
|---|---|
| `metrics.json` | CER / WER de Prism et Probe, IC 95 %, ventilations, analyses de sensibilité, temps d'inférence |
| `predictions_scored.csv` | Les 466 enregistrements : référence, prédiction brute (`hypothesis_raw`), versions normalisées, CER, WER, nombres de substitutions, suppressions et insertions, temps d'inférence |
| `environment.txt` | Nœud, GPU, empreintes SHA256 des fichiers du modèle, configuration du délai, version de ffmpeg, `pip freeze` |

Vérification : `python eval/verify_cer_independent.py results/run_20261112_0440/predictions_scored.csv`
redonne Prism 9,03 % (5133 erreurs / 56 870 caractères) et Probe 13,42 % (1396 / 10 406).

## `verif_20261112_2307/SYNTHESE.txt` : tests de robustesse du 12-13 novembre 2026

Synthèse produite par `eval/run_verify.pbs` : comparaisons enregistrement par enregistrement (répétition,
envoi en temps réel, variante audio, format Mistral, vLLM contre Transformers), recalcul indépendant du CER
et options effectives du serveur vLLM. Résultat FLEURS de cette exécution : CER 2,81 % [IC 95 % 2,14 – 3,76], n = 428.

## `run_repro_20261125/` : nouvelle exécution complète depuis le dépôt (25 novembre 2026)

Exécution de `scripts/run_pipeline.pbs` (job PBS 13677) sur un clone de ce dépôt, après un nouveau téléchargement
du modèle et du dataset aux révisions fixées et une nouvelle préparation audio (`scripts/run_pipeline.sh download prepare`).

- Évaluation vLLM : **466/466 transcriptions identiques** à `run_20261112_0440` ; Prism 9,03 %, Probe 13,42 %
  (`analysis/comparaison_original.txt`).
- Contrôle Transformers (`transformers/`), tests de robustesse A à E et FLEURS (`verification/`) : FLEURS 2,81 % [2,14 – 3,76].
- Analyses (`analysis/`) : variantes de calcul du CER, erreurs par écriture, catégories d'erreurs, analyse des canaux stéréo,
  examen de prism_133, contrôles d'intégrité.
