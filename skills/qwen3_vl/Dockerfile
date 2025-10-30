# Dockerfile qwen3_vl - Vision-Language Model Service
# Qwen2-VL-7B-Instruct containerizado

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

# Copiar requirements desde qwen3_vl
COPY skills/qwen3_vl/requirements.txt .

# Instalar Python packages
RUN pip install --user --no-cache-dir -r requirements.txt

# ========================================
# Stage 2: Runtime
# ========================================
FROM python:3.11-slim

# Instalar solo runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar packages instalados del builder
COPY --from=builder /root/.local /usr/local

# Crear directorio de trabajo
WORKDIR /app

# Copiar código del servicio y protobuf stubs
COPY skills/qwen3_vl/server.py .
COPY skills/skills_pb2.py .
COPY skills/skills_pb2_grpc.py .
COPY skills/__init__.py .

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV GRPC_PORT=50051
ENV HF_HOME=/models/hf_cache

# Crear directorios
RUN mkdir -p /models /app/state

# Healthcheck para Kubernetes/Docker Compose
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); channel.close()" || exit 1

# Puerto expuesto
EXPOSE 50051

# Usuario no-root
RUN useradd -m -u 1000 sarai && chown -R sarai:sarai /app /models
USER sarai

# Comando de inicio
CMD ["python", "server.py"]
