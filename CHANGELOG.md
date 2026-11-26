# Chronologie

Le code a été développé et exécuté sur le cluster avant la création de ce dépôt Git.
Les commits reprennent, dans l'ordre, les versions successives réellement utilisées
(y compris les versions corrigées ensuite). Leurs dates sont celles de la création du dépôt ;
les dates réelles des exécutions sont ci-dessous.

## 2026-11-12 : préparation et évaluation principale

- Téléchargement du modèle (révision `ceb7bc33`) et du dataset (révision `e2f4d96b`).
  Vérification que les fichiers du modèle sont identiques (SHA256) à ceux de `mistralai/Voxtral-Mini-4B-Realtime-Arabic`.
- Inspection du dataset (`inspect_dataset.py`) : l'audio de Prism (AAC 48 kHz) n'est pas lisible par libsndfile.
  Diagnostic (`diag_audio.py`), puis conversion avec ffmpeg (`prepare_data.py`).
- Vérification de la normalisation (`check_normalization.py`) : les annotations de silence de Probe
  (`[silence]`, `[صمت 4 ثواني]`) sont conservées comme des mots.
- Vérification du délai (`check_delay.py`) : 480 ms dans `tekken.json`, 6 tokens dans `config.json`.
- `run_eval.pbs`, première version : refusée par le cluster (ressources hors des limites de la queue `gpu_1d`).
  Ressources alignées sur la queue (20 CPU, 180 Go, 24 h).
- Exécution de 04:19 : échec, GPU déjà occupé par un autre processus (CUDA out of memory).
- Exécution de 04:29 : échec à la compilation JIT de FlashInfer.
  Ajout de `VLLM_USE_FLASHINFER_SAMPLER=0` et d'un contrôle de la mémoire GPU libre avant le lancement de vLLM.
- **Exécution de 04:40 (job 13530, nœud gpu06) : évaluation principale réussie**, 466/466 transcriptions
  (`eval/runs/run_20261112_0440`). Prism 9,03 %, Probe 13,42 %.
- Contrôle Transformers dans le même job : échec, paquet `accelerate` absent. Après installation,
  job dédié `run_transformers.pbs` (job 13531) : Prism 8,98 %, Probe 12,17 %.

## 2026-11-12 23:07 : tests de robustesse (`run_verify.pbs`)

- `compute_metrics.py` : calcul des résultats pour tout sous-ensemble (et non plus seulement Prism et Probe),
  pour évaluer FLEURS.
- `prepare_prism_variant.py`, première version : échec, le ffmpeg de conda-forge n'inclut pas soxr.
  Remplacé par un rééchantillonnage soxr en Python.
- Résultats (`eval/runs/verif_20261112_2307`) : répétition identique (466/466), temps réel identique (109/109),
  format Mistral identique (466/466), Prism canal gauche + soxr 9,22 %, FLEURS ar_eg 2,81 %.

## 2026-11-13 : contrôles complémentaires

- FLEURS avec la colonne `transcription` (`fleurs_transcription_manifest.py`) : 2,81 %, identique.
- `prism_133` (aucune transcription) : même résultat avec vLLM et Transformers ; fichier mono de niveau normal,
  parole presque inaudible sous le bruit à l'écoute (`inspect_prism_133.py`).
- Canaux stéréo de Prism (`analyse_canaux_prism.py`) : 0 fichier suspect sur 187, corrélation G/D médiane 0,98.

## 2026-11-24 : publication

- Création du dépôt, image Podman (`Containerfile`) et pipeline hors cluster (`scripts/run_pipeline.sh`).

## 2026-11-25 : sauvegarde des résultats

- Le dossier de travail du cluster a été supprimé. Les résultats originaux de l'évaluation principale
  (`metrics.json`, `predictions_scored.csv`, `environment.txt`) et la synthèse des tests de robustesse
  sont conservés dans `results/`.
- Ajout de `scripts/run_pipeline.pbs` pour régénérer toutes les sorties à partir du dépôt seul.

## 2026-11-25 : reproduction complète depuis le dépôt

- Nouvelle exécution de tout le pipeline à partir d'un clone du dépôt (job 13677) : 466/466 transcriptions identiques
  à l'exécution du 12 novembre, mêmes CER (Prism 9,03 %, Probe 13,42 %), FLEURS 2,81 %.
  Les sorties perdues (Transformers, tests de robustesse, analyses) sont régénérées dans `results/run_repro_20261125/`.
