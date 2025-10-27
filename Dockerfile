# Dockerfile v2.4 - SARAi Bundle de Producción
# Multi-stage build para imagen final ~1.9GB
# Incluye HEALTHCHECK para orquestadores (Docker, K8s, Swarm)

# ======== STAGE 1: Builder ========
FROM python:3.11-slim as builder

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copiar archivos del proyecto
COPY requirements.txt setup.py ./
COPY agents/ ./agents/
COPY core/ ./core/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY tests/ ./tests/

# Crear directorios necesarios
RUN mkdir -p models/gguf models/trm_base models/trm_mini logs state

# Descargar modelos GGUF ANTES de instalar deps Python
# Esto mejora el caching de capas si solo cambias código
RUN python3 scripts/download_gguf_models.py || echo "⚠️ Download script no disponible, saltando..."

# Instalar dependencias en --user (se copiará a runtime stage)
RUN pip install --user --no-cache-dir -r requirements.txt
RUN pip install --user --no-cache-dir llama-cpp-python


# ======== STAGE 2: Runtime ========
FROM python:3.11-slim

# Instalar solo runtime dependencies (sin build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar instalación de Python desde builder
COPY --from=builder /root/.local /usr/local

# Copiar modelos GGUF descargados
COPY --from=builder /build/models /app/models

# Copiar código fuente
COPY --from=builder /build/agents /app/agents
COPY --from=builder /build/core /app/core
COPY --from=builder /build/scripts /app/scripts
COPY --from=builder /build/config /app/config

# Crear directorios de runtime
RUN mkdir -p /app/logs /app/state

WORKDIR /app

# Variables de entorno
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Exponer puerto del health dashboard
EXPOSE 8080

# 🚀 HEALTHCHECK: Verifica que el servicio responde
# - interval: cada 30s comprueba salud
# - timeout: espera máximo 5s por respuesta
# - start-period: da 60s de gracia al iniciar (carga de modelos)
# - retries: 3 fallos consecutivos → contenedor unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Comando por defecto (puede sobrescribirse)
# Si health_dashboard.py no existe, usar main.py
CMD if [ -f "sarai/health_dashboard.py" ]; then \
        python -m sarai.health_dashboard; \
    else \
        python main.py; \
    fi
