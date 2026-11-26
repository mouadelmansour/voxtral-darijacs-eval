#!/usr/bin/env bash
# Pipeline complet hors cluster (image Podman ou machine Linux avec un GPU NVIDIA).
#
# Reprend les commandes et options des jobs exécutés sur le cluster
# (eval/run_eval.pbs, eval/run_transformers.pbs, eval/run_verify.pbs), sans les directives PBS
# ni le chargement de modules. Les scripts de eval/ sont utilisés tels quels.
#
# Usage : bash scripts/run_pipeline.sh ÉTAPE [ÉTAPE ...]
#   download      modèle + dataset aux révisions fixées
#   prepare       audio 16 kHz mono (DarijaCS), variante Prism canal gauche + soxr, FLEURS ar_eg
#   evaluate      vLLM Realtime (délai 480 ms, température 0) + métriques       -> $RUN
#   transformers  contrôle croisé Transformers + métriques                       -> $RUN/transformers
#   verify        tests de robustesse A à E + comparaisons                       -> $RUN/verification
#   analysis      analyse d'erreurs, analyse des canaux audio, figures           -> $RUN/analysis
#   all           toutes les étapes dans l'ordre
#
# Variables : WORK (données, modèle, résultats ; défaut ./work), RUN (défaut work/runs/run_AAAAMMJJ_HHMM)
set -eo pipefail
cd "$(dirname "$0")/.."

WORK="${WORK:-$PWD/work}"
mkdir -p "$WORK"/{runs,data-test,hf_cache} logs
export HF_HOME="${HF_HOME:-$WORK/hf_cache}" VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1

# Les scripts de eval/ utilisent des chemins relatifs à la racine du projet, comme sur le cluster.
link () { [ -e "$2" ] || [ -L "$2" ] || ln -s "$1" "$2"; }
link "$WORK/data-test" data-test
link "$WORK/Voxtral-Mini-4B-Realtime-Arabic" Voxtral-Mini-4B-Realtime-Arabic
link "$WORK/runs" eval/runs

MODEL_DIR="$PWD/Voxtral-Mini-4B-Realtime-Arabic"
NAME="Morocco-MITNRA-Labs/Voxtral-Mini-4B-Realtime-Arabic"
P=data-test/prepared
MANIFESTS="$P/manifest_prism.csv $P/manifest_probe.csv"
RUN="${RUN:-eval/runs/run_$(date +%Y%m%d_%H%M)}"
PORT=$((8100 + RANDOM % 800))
SERVER_PID=""

gpu_guard () {
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "Mémoire GPU libre : ${free} MiB"
  [ "${free:-0}" -ge 30000 ] || { echo "!! GPU déjà occupé : arrêt"; nvidia-smi; exit 2; }
}
start_server () {   # $1 = fichier log, puis options supplémentaires de vllm serve
  local log=$1; shift
  vllm serve "$MODEL_DIR" --served-model-name "$NAME" --port "$PORT" --enforce-eager \
    --override-generation-config '{"temperature": 0.0}' "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 90); do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { echo "Serveur vLLM prêt ($log)"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "Le serveur vLLM s'est arrêté"; tail -60 "$log"; exit 1; }
    sleep 10
  done
  echo "Serveur vLLM non prêt après 15 min"; tail -60 "$log"; exit 1
}
stop_server () { [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }; SERVER_PID=""; }
trap stop_server EXIT
infer () { python -u eval/infer_vllm_realtime.py --port "$PORT" --model "$NAME" --concurrency 1 "$@" || true; }
metrics () { python -u eval/compute_metrics.py --manifests $2 --predictions "$1/predictions.csv" \
               --normalization-dir "$MODEL_DIR" --out "$1" || echo "!! métriques non calculées pour $1"; }

step_download () { bash scripts/download_assets.sh "$WORK"; }

step_prepare () {
  python eval/prepare_data.py data-test/DarijaCS-asr-eval data-test/prepared
  python eval/prepare_prism_variant.py data-test/DarijaCS-asr-eval data-test/prepared_left_soxr
  python eval/prepare_fleurs.py data-test/fleurs_ar_eg
  python eval/fleurs_transcription_manifest.py
}

step_evaluate () {
  mkdir -p "$RUN"; gpu_guard
  { date -Is; hostname; nvidia-smi; sha256sum "$MODEL_DIR"/*.json "$MODEL_DIR"/normalization.py
    python eval/check_delay.py "$MODEL_DIR"; ffmpeg -version | head -1; pip freeze; } > "$RUN/environment.txt" 2>&1
  start_server "$RUN/vllm_server.log"
  infer --manifests $MANIFESTS --out "$RUN/predictions.csv" --pace 0
  stop_server
  metrics "$RUN" "$MANIFESTS"
}

step_transformers () {
  mkdir -p "$RUN/transformers"; gpu_guard
  python -u eval/infer_transformers.py --model-dir "$MODEL_DIR" --manifests $MANIFESTS --out "$RUN/transformers/predictions.csv"
  metrics "$RUN/transformers" "$MANIFESTS"
}

step_verify () {
  local V="$RUN/verification"
  mkdir -p "$V"/{A_repetition,B_probe_pace1,C_prism_gauche_soxr,D_fleurs,D_fleurs_transcription,E_format_mistral}
  gpu_guard
  start_server "$V/vllm_server_defaut.log"
  infer --manifests $MANIFESTS --out "$V/A_repetition/predictions.csv"
  infer --manifests $P/manifest_probe.csv --pace 1 --out "$V/B_probe_pace1/predictions.csv"
  infer --manifests data-test/prepared_left_soxr/manifest_prism.csv --out "$V/C_prism_gauche_soxr/predictions.csv"
  infer --manifests data-test/fleurs_ar_eg/manifest_fleurs.csv --out "$V/D_fleurs/predictions.csv"
  stop_server
  start_server "$V/vllm_server_mistral.log" --tokenizer-mode mistral --config-format mistral --load-format mistral
  infer --manifests $MANIFESTS --out "$V/E_format_mistral/predictions.csv"
  stop_server
  metrics "$V/A_repetition" "$MANIFESTS"
  metrics "$V/B_probe_pace1" "$P/manifest_probe.csv"
  metrics "$V/C_prism_gauche_soxr" "data-test/prepared_left_soxr/manifest_prism.csv"
  metrics "$V/D_fleurs" "data-test/fleurs_ar_eg/manifest_fleurs.csv"
  cp "$V/D_fleurs/predictions.csv" "$V/D_fleurs_transcription/predictions.csv"
  metrics "$V/D_fleurs_transcription" "data-test/fleurs_ar_eg/manifest_fleurs_transcription.csv"
  metrics "$V/E_format_mistral" "$MANIFESTS"
  {
    python eval/compare_runs.py "$RUN/predictions_scored.csv" "$V/A_repetition/predictions_scored.csv" "run principal" "A répétition"
    python eval/compare_runs.py "$RUN/predictions_scored.csv" "$V/B_probe_pace1/predictions_scored.csv" "run principal" "B pace 1"
    python eval/compare_runs.py "$RUN/predictions_scored.csv" "$V/C_prism_gauche_soxr/predictions_scored.csv" "run principal" "C gauche+soxr"
    python eval/compare_runs.py "$RUN/predictions_scored.csv" "$V/E_format_mistral/predictions_scored.csv" "run principal" "E format Mistral"
    [ -f "$RUN/transformers/predictions_scored.csv" ] && \
      python eval/compare_runs.py "$RUN/predictions_scored.csv" "$RUN/transformers/predictions_scored.csv" "vLLM" "Transformers"
    for f in "$RUN/predictions_scored.csv" "$RUN/transformers/predictions_scored.csv"; do
      [ -f "$f" ] && python eval/verify_cer_independent.py "$f" "$MODEL_DIR"
    done
  } > "$V/SYNTHESE.txt" 2>&1 || true
  cat "$V/SYNTHESE.txt"
}

step_analysis () {
  local A="$RUN/analysis"; mkdir -p "$A"
  python analysis/error_analysis.py "$RUN/predictions_scored.csv" "$A"
  python eval/analyse_canaux_prism.py "$RUN" && mv -f eval/analyse_canaux_prism.csv "$A/" || true
  python eval/inspect_prism_133.py "$RUN" "$RUN/transformers" > "$A/prism_133.txt" 2>&1 || true
  if [ -f "$RUN/transformers/metrics.json" ]; then
    mkdir -p "$A/results/vllm_realtime" "$A/results/transformers"
    cp "$RUN/metrics.json" "$A/results/vllm_realtime/"; cp "$RUN/transformers/metrics.json" "$A/results/transformers/"
    python analysis/make_figures.py --results "$A/results" --analysis "$A" --out "$A/figures"
  fi
}

[ $# -eq 0 ] && set -- all
for s in "$@"; do
  case "$s" in
    all) for t in download prepare evaluate transformers verify analysis; do echo "=== $t"; "step_$t"; done ;;
    download|prepare|evaluate|transformers|verify|analysis) echo "=== $s"; "step_$s" ;;
    *) echo "Étape inconnue : $s"; exit 1 ;;
  esac
done
echo "Terminé. Résultats : $RUN"
