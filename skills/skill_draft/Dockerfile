# Dockerfile skill_draft - Draft LLM Service
# Qwen3-VL-4B-Instruct IQ4_NL containerizado

# ========================================
# Stage 1: Builder
# ========================================
FROM python:3.11-slim as builder

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copiar requirements desde skill_draft
COPY skills/skill_draft/requirements.txt .

# Instalar Python packages
RUN pip install --user --no-cache-dir -r requirements.txt

# ========================================
# Stage 2: Runtime
# ========================================
FROM python:3.11-slim

# Instalar solo runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar packages instalados del builder
COPY --from=builder /root/.local /usr/local

# Crear directorio de trabajo
WORKDIR /app

# Copiar código del servicio y protobuf stubs
COPY skills/skill_draft/server.py .
COPY skills/skills_pb2.py .
COPY skills/skills_pb2_grpc.py .
COPY skills/__init__.py .

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV GRPC_PORT=50051
ENV N_THREADS=2
ENV HF_CACHE_DIR=/models

# Crear directorios
RUN mkdir -p /models

# Healthcheck para Kubernetes/Docker Compose
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); channel.close()" || exit 1

# Puerto expuesto
EXPOSE 50051

# Usuario no-root
RUN useradd -m -u 1000 sarai && chown -R sarai:sarai /app /models
USER sarai

# Comando de inicio
CMD ["python", "server.py"]
