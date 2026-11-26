#!/usr/bin/env bash
# Télécharge le modèle et le dataset aux révisions exactes utilisées pour l'évaluation.
# Usage : bash scripts/download_assets.sh [dossier_de_travail]   (défaut : $WORK, sinon ./work)
set -euo pipefail

WORK="${1:-${WORK:-$PWD/work}}"
MODEL_REPO="Morocco-MITNRA-Labs/Voxtral-Mini-4B-Realtime-Arabic"
MODEL_REV="ceb7bc331024bc8a75fa61366b43bc9545b258e5"     # poids identiques (SHA256) à mistralai/...@7c60cf2a
DATA_REPO="Morocco-MITNRA-Labs/DarijaCS-asr-eval"
DATA_REV="e2f4d96bbf1009101534d48e7c05bd28f19c0506"

mkdir -p "$WORK/data-test"
hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$WORK/Voxtral-Mini-4B-Realtime-Arabic"
hf download "$DATA_REPO" --repo-type dataset --revision "$DATA_REV" --local-dir "$WORK/data-test/DarijaCS-asr-eval"

# Contrôle d'intégrité des fichiers du modèle utilisés par l'évaluation
( cd "$WORK/Voxtral-Mini-4B-Realtime-Arabic" && sha256sum tekken.json params.json config.json normalization.py )
echo "Attendu : tekken.json 2a88a9b8…  params.json 2ace010e…  config.json 560dcfc0…  normalization.py 501255ef…"
