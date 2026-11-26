# Évaluation indépendante de Voxtral-Mini-4B-Realtime-Arabic sur DarijaCS

Ce dépôt contient tout le code utilisé pour évaluer le modèle
[`Morocco-MITNRA-Labs/Voxtral-Mini-4B-Realtime-Arabic`](https://huggingface.co/Morocco-MITNRA-Labs/Voxtral-Mini-4B-Realtime-Arabic)
sur le dataset public [`Morocco-MITNRA-Labs/DarijaCS-asr-eval`](https://huggingface.co/datasets/Morocco-MITNRA-Labs/DarijaCS-asr-eval)
(sous-ensembles Prism et Probe), avec le protocole décrit dans le model card :
délai de transcription de 480 ms, température 0, audio à 16 kHz, inférence en streaming
et normalisation officielle `normalize_arabic_asr`.

Il fournit :

- le code d'inférence (vLLM Realtime en streaming, et Transformers en contrôle croisé) ;
- le calcul des métriques, les analyses d'erreurs et les tests de robustesse ;
- les jobs PBS exécutés sur le cluster, tels quels ;
- une image Podman pour reproduire l'évaluation sur une autre machine.

## Résultats

| Benchmark | Model card | Mesuré (vLLM Realtime) | IC 95 % | Mesuré (Transformers) |
|---|---|---|---|---|
| FLEURS ar_eg, test (contrôle) | 2,66 % | **2,81 %** | 2,14 – 3,76 | – |
| DarijaCS-Prism (« ISMA » du model card) | 6,68 % | **9,03 %** | 8,11 – 10,11 | 8,98 % |
| DarijaCS-Probe (« Darija in the Wild » du model card) | 9,69 % | **13,42 %** | 11,16 – 16,27 | 12,17 % |

CER calculé sur l'ensemble du corpus, espaces compris, après `normalize_arabic_asr` appliquée à la référence
et à la prédiction. Les 466 enregistrements de DarijaCS sont évalués, sans exclusion.

La correspondance ISMA = Prism et Darija in the Wild = Probe est **déduite** de la figure
`benchmark_composition.png` du model card : mêmes totaux (357 et 109) et mêmes effectifs dans les 9 domaines de Prism.
Elle n'a pas été confirmée par les auteurs.

### Tests de robustesse

| Test | Résultat |
|---|---|
| Deuxième exécution identique | 466/466 transcriptions identiques |
| Envoi de l'audio en temps réel (`--pace 1`) | 109/109 transcriptions identiques (Probe) |
| Chargement forcé au format Mistral | 466/466 transcriptions identiques |
| Prism : canal gauche + rééchantillonnage soxr | 9,22 % (contre 9,03 %) |
| Canaux stéréo de Prism (187 fichiers) | 0 fichier suspect, corrélation G/D médiane 0,98 |
| CER recalculé avec une implémentation indépendante | chiffres identiques |
| FLEURS avec la colonne `transcription` au lieu de `raw_transcription` | 2,81 % (identique) |

Le même pipeline reproduit le score annoncé sur FLEURS (+0,15 point) mais pas sur DarijaCS (+2,35 et +3,73 points).

## Contenu du dépôt

```
eval/                          code exécuté sur le cluster (chemins relatifs à la racine du projet)
  inspect_dataset.py           structure du dataset, colonnes, formats et durées audio
  diag_audio.py                diagnostic du format audio de Prism (AAC 48 kHz)
  check_normalization.py       effet de normalize_arabic_asr sur les références
  check_delay.py               configuration audio et délai (tekken.json, config.json)
  prepare_data.py              audio 16 kHz mono float32 + manifestes CSV
  infer_vllm_realtime.py       client WebSocket /v1/realtime de vLLM (protocole principal)
  infer_transformers.py        contrôle croisé Transformers (do_sample=False)
  compute_metrics.py           CER / WER, ventilations, IC bootstrap, analyses de sensibilité
  prepare_fleurs.py            FLEURS ar_eg (test) au même format
  prepare_prism_variant.py     Prism en canal gauche + soxr
  fleurs_transcription_manifest.py   FLEURS avec la colonne `transcription`
  compare_runs.py              comparaison de deux exécutions, enregistrement par enregistrement
  verify_cer_independent.py    recalcul du CER sans jiwer (Levenshtein codé à la main)
  inspect_prism_133.py         examen de l'enregistrement sans transcription
  analyse_canaux_prism.py      analyse des canaux stéréo de Prism
  check_gpu.pbs, run_eval.pbs, run_transformers.pbs, run_verify.pbs   jobs PBS du cluster
analysis/                      analyse d'erreurs, figures et page web du rapport
scripts/
  download_assets.sh           modèle et dataset aux révisions exactes
  run_pipeline.sh              pipeline complet hors cluster (utilisé par l'image)
Containerfile, container/environment.yml, requirements.lock   construction de l'image Podman
results/                       résultats de l'évaluation sur le cluster (prédictions, métriques)
CHANGELOG.md                   chronologie des exécutions et des corrections
```

## Versions exactes

| Élément | Version |
|---|---|
| Modèle | `Morocco-MITNRA-Labs/Voxtral-Mini-4B-Realtime-Arabic` @ `ceb7bc331024bc8a75fa61366b43bc9545b258e5` (poids identiques, SHA256, à `mistralai/Voxtral-Mini-4B-Realtime-Arabic` @ `7c60cf2a`) |
| Dataset | `Morocco-MITNRA-Labs/DarijaCS-asr-eval` @ `e2f4d96bbf1009101534d48e7c05bd28f19c0506` |
| Contrôle | `google/fleurs`, configuration `ar_eg`, split `test` (428 enregistrements) |
| Délai | `transcription_delay_ms = 480` (`tekken.json`), `default_num_delay_tokens = 6` (`config.json`) |
| Logiciels | Python 3.11, vLLM 0.33.0, PyTorch 2.14.0, Transformers 5.20.2, mistral_common 1.13.1, jiwer 4.0.0, ffmpeg 7.1 (liste complète : `requirements.lock`) |
| Matériel | 1 × NVIDIA L40S 46 Go, BF16 |

## Paramètres d'inférence

- **vLLM** : `vllm serve <modèle> --enforce-eager --override-generation-config '{"temperature": 0.0}'`,
  variable `VLLM_USE_FLASHINFER_SAMPLER=0` (la compilation JIT de FlashInfer échouait sur le cluster ;
  sans effet en décodage glouton).
- **Streaming** : `session.update` → `input_audio_buffer.commit` → blocs `input_audio_buffer.append`
  (PCM 16 bits, 4096 octets = 128 ms) → `input_audio_buffer.commit` avec `final: true` → `transcription.delta`
  → `transcription.done`. Un enregistrement à la fois.
- **Transformers** : `VoxtralRealtimeForConditionalGeneration.generate(do_sample=False)`, BF16.
- **Audio** : Prism (AAC 48 kHz, en partie stéréo) converti avec ffmpeg en mono (moyenne des canaux), 16 kHz, float32.
  Probe est déjà en WAV 16 kHz mono.

## Reproduire avec Podman

Prérequis : un GPU NVIDIA avec au moins 30 Go libres, le pilote NVIDIA et
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) configuré en CDI
(`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`). Le modèle et les données étant publics,
aucun jeton Hugging Face n'est nécessaire.

```bash
podman build -t voxtral-darijacs-eval:1.0 -f Containerfile .
mkdir -p work
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
  --userns=keep-id:uid=57439,gid=57439 \
  -v "$PWD/work:/workspace/work" \
  voxtral-darijacs-eval:1.0 all
```

L'étape `all` enchaîne : téléchargement (modèle, dataset, FLEURS), préparation audio, évaluation vLLM,
contrôle Transformers, tests de robustesse et analyses. Les résultats sont écrits dans `work/runs/run_<date>/`.

Pour lancer les étapes séparément, fixez le dossier de résultats avec `RUN` :

```bash
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
  --userns=keep-id:uid=57439,gid=57439 -v "$PWD/work:/workspace/work" \
  -e RUN=eval/runs/run_mistral voxtral-darijacs-eval:1.0 download prepare evaluate
```

Étapes disponibles : `download`, `prepare`, `evaluate`, `transformers`, `verify`, `analysis`, `all`.
L'option `--userns=keep-id:uid=57439,gid=57439` fait correspondre votre utilisateur à celui de l'image
(`mambauser`), pour qu'il puisse écrire dans `work/`.

## Reproduire sur un cluster PBS

Les jobs de `eval/*.pbs` sont ceux exécutés sur le cluster, sans modification. Ils supposent un environnement conda
`voxtral_eval` (Python 3.11, `pip install -r requirements.lock`, `conda install -c conda-forge ffmpeg=7.1`)
et s'exécutent depuis la racine du projet :

```bash
bash scripts/download_assets.sh .                # crée ./Voxtral-Mini-4B-Realtime-Arabic et ./data-test/DarijaCS-asr-eval
python eval/prepare_data.py data-test/DarijaCS-asr-eval data-test/prepared
qsub eval/run_eval.pbs                           # vLLM Realtime + métriques + contrôle Transformers
python eval/prepare_fleurs.py data-test/fleurs_ar_eg
python eval/prepare_prism_variant.py data-test/DarijaCS-asr-eval data-test/prepared_left_soxr
qsub eval/run_verify.pbs                         # tests A à E
```

`run_transformers.pbs` et `run_verify.pbs` contiennent les noms des dossiers de l'exécution du cluster
(`eval/runs/run_20261112_0440`). Adaptez-les à votre propre exécution.

## Limites connues

- La correspondance ISMA = Prism et Darija in the Wild = Probe est déduite de la composition, pas confirmée.
- La version des données (audio d'origine ou AAC publié, version des références) et le checkpoint utilisés
  pour les chiffres du model card ne sont pas connus.
- Le délai effectif n'apparaît pas dans les logs de vLLM : il est établi à partir de la configuration du modèle.
- Sur Probe, vLLM et Transformers diffèrent de 1,25 point (77 % de transcriptions identiques), sans explication à ce stade.
- Un seul benchmark de contrôle (FLEURS) a été évalué.

## Données et licences

- **DarijaCS-Prism et DarijaCS-Probe** : Ministry of Artificial Intelligence, Digital Transformation and Administration Reform,
  Kingdom of Morocco (2026), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Les fichiers de `results/`
  contiennent les références du dataset, sans modification, et leurs versions normalisées.
- **Modèle** : Apache 2.0.
- **FLEURS** : Google, CC BY 4.0.

Ni le modèle ni les données ne sont inclus dans ce dépôt : ils sont téléchargés aux révisions indiquées.
