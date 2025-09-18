# ---------- Build args ----------
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# ---------- Stage: base (final efetivo) ----------
FROM ${BASE_IMAGE} AS base

# Args de build (mantidos do template)
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Ambientes úteis
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Dependências de SO
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
 && ln -sf /usr/bin/python3.12 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip \
 && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Instala uv e cria venv isolada
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
 && ln -s /root/.local/bin/uv /usr/local/bin/uv \
 && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
 && uv venv /opt/venv

# Usa a venv em tudo
ENV PATH="/opt/venv/bin:${PATH}"

# comfy-cli e deps
RUN uv pip install comfy-cli pip setuptools wheel

# Instala ComfyUI (via comfy-cli)
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# (Opcional) Upgrade de PyTorch se precisar de outra CUDA
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# Diretório de trabalho do ComfyUI
WORKDIR /comfyui

# ---------- Extra model paths (oficial) ----------
# Este arquivo já aponta para /runpod-volume/models/*
ADD src/extra_model_paths.yaml ./

# Volta para raiz
WORKDIR /

# ---------- Runtime do worker ----------
RUN uv pip install runpod requests websocket-client

# Scripts do worker
ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# Manager helper scripts
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

ENV PIP_NO_INPUT=1

COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# ---------- Custom Nodes ----------
# Coloque todos os nodes em vendor/<node> no seu repo antes do build
# Ex.: vendor/ComfyUI-WD14-Tagger/, vendor/was-node-suite-comfyui/, etc.
COPY vendor/ /comfyui/custom_nodes/

# Instala requirements de TODOS os nodes, exceto WD14; WD14 por último
RUN set -eux; \
    WD14_DIR="/comfyui/custom_nodes/ComfyUI-WD14-Tagger"; \
    # instala primeiro os outros
    if [ -d /comfyui/custom_nodes ]; then \
      find /comfyui/custom_nodes -maxdepth 2 -type f -name requirements.txt \
        ! -path "${WD14_DIR}/*" -print -exec uv pip install --no-cache-dir -r {} \; ; \
    fi; \
    # agora o WD14 por ÚLTIMO (para vencer conflitos)
    if [ -f "${WD14_DIR}/requirements.txt" ]; then \
      uv pip install --upgrade --no-cache-dir -r "${WD14_DIR}/requirements.txt"; \
    fi; \
    python -m pip check || true

# ---------- Modelos via Network Volume ----------
# NADA é baixado na build; os modelos virão do volume montado em /runpod-volume
# O extra_model_paths.yaml já aponta para /runpod-volume/models/*
# (Se quiser, crie as pastas padrão no primeiro boot do container)
RUN mkdir -p /runpod-volume/models

# Comando padrão
CMD ["/start.sh"]

# ---------- Stage final (mantido por compatibilidade) ----------
FROM base AS final
# Sem cópias adicionais: a imagem final é o que construímos acima.
