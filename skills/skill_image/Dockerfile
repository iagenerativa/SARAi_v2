# skill_image - Dockerfile Multi-Stage v2.16
# Procesamiento de imágenes containerizado (0MB host RAM)

# ========================================
# Stage 1: Builder
# ========================================
FROM python:3.11-slim as builder

# Build dependencies (mínimas para compilar packages Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copiar requirements desde skill_image
COPY skills/skill_image/requirements.txt .

# Instalar dependencias Python (--user para copiar después)
RUN pip install --user --no-cache-dir -r requirements.txt

# ========================================
# Stage 2: Runtime
# ========================================
FROM python:3.11-slim

# Runtime dependencies (mínimas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar packages desde builder
COPY --from=builder /root/.local /usr/local

# Crear usuario no-root
RUN useradd -m -u 1000 sarai

# Workdir
WORKDIR /app

# Copiar código del servicio y protobuf stubs
COPY skills/skill_image/server.py .
COPY skills/skills_pb2.py .
COPY skills/skills_pb2_grpc.py .
COPY skills/__init__.py .

# Crear directorio para cache de imágenes
RUN mkdir -p /app/cache && chown -R sarai:sarai /app

# Cambiar a usuario no-root
USER sarai

# Exponer puerto gRPC
EXPOSE 50052

# Health check (cada 30s)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import grpc; from skills import skills_pb2, skills_pb2_grpc; \
        channel = grpc.insecure_channel('localhost:50052'); \
        stub = skills_pb2_grpc.SkillsServiceStub(channel); \
        response = stub.Health(skills_pb2.HealthReq()); \
        exit(0 if response.status == 'HEALTHY' else 1)"

# Comando de inicio
CMD ["python3", "server.py"]
