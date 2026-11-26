# Image d'évaluation de Voxtral-Mini-4B-Realtime-Arabic sur DarijaCS.
#
# Construction :  podman build -t voxtral-darijacs-eval:1.0 -f Containerfile .
# Exécution    :  voir README.md (GPU NVIDIA requis, via CDI : --device nvidia.com/gpu=all)
#
# Le modèle et les données ne sont pas inclus dans l'image : ils sont téléchargés au premier lancement
# dans le volume monté sur /workspace/work, aux révisions exactes utilisées pour l'évaluation.

FROM docker.io/mambaorg/micromamba:2.0.5

LABEL org.opencontainers.image.title="voxtral-darijacs-eval" \
      org.opencontainers.image.description="Évaluation indépendante de Voxtral-Mini-4B-Realtime-Arabic sur DarijaCS (Prism, Probe) et FLEURS ar_eg"

# 1. Python 3.11 + ffmpeg 7.1 (conda-forge), comme sur le cluster
COPY --chown=$MAMBA_USER:$MAMBA_USER container/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

# Les commandes RUN suivantes s'exécutent dans l'environnement activé
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# 2. Paquets Python aux versions exactes du cluster (vLLM 0.33.0, torch 2.14.0, transformers 5.20.2…)
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

# 3. Code
USER root
RUN mkdir -p /workspace && chown $MAMBA_USER:$MAMBA_USER /workspace
USER $MAMBA_USER
WORKDIR /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER eval/ eval/
COPY --chown=$MAMBA_USER:$MAMBA_USER analysis/ analysis/
COPY --chown=$MAMBA_USER:$MAMBA_USER scripts/ scripts/

# 4. Réglages identiques aux jobs du cluster
ENV WORK=/workspace/work \
    HF_HOME=/workspace/work/hf_cache \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    PYTHONUNBUFFERED=1

# Le point d'entrée de micromamba active l'environnement, puis lance le pipeline.
# Argument : download | prepare | evaluate | transformers | verify | analysis | all (défaut)
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "bash", "scripts/run_pipeline.sh"]
CMD ["all"]
