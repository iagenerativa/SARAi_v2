# skills/Dockerfile - Skills-as-Services v2.12 Phoenix
# Hardware: CPU-only, <50MB RAM per client
# Security: cap_drop=ALL, read_only, tmpfs, no-new-privileges

# ══════════════════════════════════════════════════════════════════════════
# Stage 1: Builder (reutiliza runtime existente como base)
# ══════════════════════════════════════════════════════════════════════════
FROM ghcr.io/iagenerativa/sarai_v2:runtime-v2.11 AS base

WORKDIR /build

# Copiar requirements del skill
COPY skills/requirements.txt .

# Instalar deps de gRPC (ligeras)
RUN pip install --no-cache-dir -r requirements.txt

# ══════════════════════════════════════════════════════════════════════════
# Stage 2: Runtime (imagen mínima)
# ══════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

# Instalar solo libgomp (para llama-cpp-python)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar deps instaladas
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copiar stubs gRPC generados (CRÍTICO: deben existir antes del build)
COPY skills/__init__.py /app/skills/
COPY skills/skills_pb2.py /app/skills/
COPY skills/skills_pb2_grpc.py /app/skills/
COPY skills/runtime.py /app/skills/

# Copiar código del skill específico
ARG SKILL
COPY skills/${SKILL}/ /app/skills/${SKILL}/

# Copiar core (para embeddings, etc. si el skill lo necesita)
COPY core/ /app/core/

# Crear directorios necesarios (tmpfs se montará aquí)
RUN mkdir -p /tmp /app/models /app/state && \
    chmod 1777 /tmp

# Usuario no-root (seguridad)
RUN useradd -m -u 1000 skilluser && \
    chown -R skilluser:skilluser /app

USER skilluser

# Exponer puerto gRPC
EXPOSE 50051

# Healthcheck (gRPC reflection)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); channel.channel_ready()" || exit 1

# Entrypoint con USR1 para hot-reload
CMD ["python", "-m", "skills.runtime", "--skill", "${SKILL}"]
