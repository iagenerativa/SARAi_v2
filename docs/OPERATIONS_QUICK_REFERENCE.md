# SARAi v2.14 - Guía Rápida de Operaciones

> Comandos esenciales para operación diaria, troubleshooting y validación del sistema.

**Última actualización**: 2025-01-01  
**Versión del sistema**: v2.14 (Unified Model Wrapper + VisCoder2)

---

## 📋 Tabla de Contenidos

- [Setup Inicial](#setup-inicial)
- [Validación del Sistema](#validación-del-sistema)
- [Operación Diaria](#operación-diaria)
- [Troubleshooting](#troubleshooting)
- [Auditoría y Seguridad](#auditoría-y-seguridad)
- [Benchmarking](#benchmarking)
- [Docker](#docker)
- [Logs](#logs)
- [Emergency Procedures](#emergency-procedures)

---

## Setup Inicial

### Instalación Completa (20-30 min)

```bash
# Setup completo: venv + deps + llama.cpp + modelos GGUF
make install

# Verificar estrategia llama.cpp detectada
make show-llama-strategy

# Validar instalación
make validate
```

### Variables de Entorno Críticas

```bash
# .env mínimo
OLLAMA_BASE_URL=http://localhost:11434
SOLAR_MODEL_NAME=hf.co/upstage/SOLAR-10.7B-Instruct-v1.0-GGUF:Q4_K_M
VISCODER2_MODEL_NAME=hf.co/mradermacher/VisCoder2-7B-GGUF:Q4_K_M
HOME_ASSISTANT_URL=http://localhost:8123

# Opcional: HMAC para auditoría
HMAC_SECRET_KEY=your-secret-key-here
```

---

## Validación del Sistema

### Validación Rápida (30 seg)

```bash
# Todas las secciones
make validate

# Una sección específica
make validate-section SECTION=config
make validate-section SECTION=health
make validate-section SECTION=tests
```

### Validación Completa (5-10 min)

```bash
# Ejecutar checklist completo de auditoría
bash scripts/run_audit_checklist.sh

# Ver informe de auditoría
cat logs/audit_report_$(date +%Y-%m-%d).md
```

### Verificación de KPIs

```bash
# RAM P99 (objetivo: ≤12 GB)
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().used/(1024**3):.1f} GB')"

# Tests (objetivo: 111/111 passing)
pytest tests/ -v --tb=short | grep -E "passed|failed"

# Health endpoint
curl -s http://localhost:8080/health | jq '.status'

# Metrics Prometheus
curl -s http://localhost:8080/metrics | grep sarai_
```

---

## Operación Diaria

### Levantar el Sistema

```bash
# Dashboard de salud (http://localhost:8080/health)
make health

# O con uvicorn directamente
uvicorn sarai.health_dashboard:app --host 0.0.0.0 --port 8080
```

### Ejecutar SARAi

```bash
# Modo interactivo
python main.py

# Con logging verbose
python main.py --verbose

# Procesar un archivo de texto
python main.py --input input.txt --output response.txt
```

### Skills Phoenix

```bash
# Probar detección de skills
python -c "
from core.mcp import detect_and_apply_skill
skill = detect_and_apply_skill('Cómo crear una API REST en Python', 'solar')
print(f'Skill detectado: {skill[\"name\"] if skill else \"ninguno\"}')
"

# Ver configuración de todos los skills
python -c "from core.skill_configs import SKILLS; import json; print(json.dumps(list(SKILLS.keys()), indent=2))"
```

### Layers Architecture

```bash
# Ver estado de Layer 2 (Tone Memory)
cat state/layer2_tone_memory.jsonl | tail -n 10

# Contar entradas (max 256)
wc -l state/layer2_tone_memory.jsonl

# Test de Layer 3 (Tone Bridge)
python -c "
from core.layer3_fluidity.tone_bridge import get_tone_bridge
bridge = get_tone_bridge()
profile = bridge.update('happy', 0.8, 0.7)
print(f'Estilo inferido: {profile.style}')
print(f'Filler hint: {profile.filler_hint}')
"
```

---

## Troubleshooting

### Modelos no Cargan

```bash
# Verificar configuración de modelos
python -c "
from core.unified_model_wrapper import ModelRegistry
registry = ModelRegistry()
registry.load_config()
print(f'Modelos configurados: {len(registry._config)}')
for name in registry._config.keys():
    print(f'  - {name}')
"

# Verificar Ollama accesible
curl -s ${OLLAMA_BASE_URL:-http://localhost:11434}/api/version

# Test de conexión a Ollama
python scripts/test_ollama_connection.py
```

### Tests Fallan

```bash
# Test específico con output detallado
pytest tests/test_unified_wrapper.py -v -s --tb=long

# Solo tests de integración
pytest tests/test_unified_wrapper_integration.py -v

# Con coverage
pytest tests/ --cov=core --cov=agents --cov-report=term-missing
```

### RAM Excedida

```bash
# Ver uso actual de RAM por proceso
ps aux | grep python | awk '{print $2, $4, $11}' | sort -k2 -rn | head -n 10

# Limpiar cache de modelos
python -c "
from core.model_pool import get_model_pool
pool = get_model_pool()
pool.clear_all()
print('Cache de modelos limpiado')
"

# Verificar límites en config
grep -A 5 "memory:" config/sarai.yaml
```

### Health Dashboard No Responde

```bash
# Ver logs de uvicorn
tail -f logs/uvicorn.log

# Verificar puerto 8080 libre
lsof -i :8080

# Iniciar en puerto alternativo
uvicorn sarai.health_dashboard:app --host 0.0.0.0 --port 8081
```

---

## Auditoría y Seguridad

### Verificar Integridad de Logs

```bash
# Logs de voz (HMAC)
python scripts/verify_voice_audit.py

# Logs web (SHA-256)
python -m core.web_audit --verify $(date +%Y-%m-%d)

# Verificar sidecars
ls -lh logs/*.sha256 logs/*.hmac
```

### Supply Chain Verification

```bash
# Verificar firma de release
cosign verify \
  --certificate-identity-regexp="https://github.com/user/sarai/.*" \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  ghcr.io/user/sarai:v2.14

# Verificar SBOM
cosign verify-attestation --type spdxjson ghcr.io/user/sarai:v2.14 | jq .

# Verificar build environment
cosign verify-attestation --type custom ghcr.io/user/sarai:v2.14 | \
  jq '.payload | @base64d | fromjson | .predicate'
```

### Verificar Sin IPs Hardcodeadas

```bash
# En código Python
grep -r "192\.168\|10\.\|172\." --include="*.py" core/ agents/ scripts/ tests/

# En configs
grep -r "192\.168\|10\.\|172\." config/ .env*

# Debe retornar vacío (exit code 1)
```

---

## Benchmarking

### Benchmark Completo

```bash
# Ejecutar benchmark de v2.14
make benchmark VERSION=v2.14

# Comparar con versión anterior
make benchmark-compare OLD=v2.13 NEW=v2.14

# Ver histórico
make benchmark-history
```

### Benchmark Rápido (Debug)

```bash
# Solo 2 iteraciones
make benchmark-quick VERSION=v2.14

# O con script directo
python scripts/benchmark_wrapper_overhead.py --iterations 2
```

### Métricas Específicas

```bash
# Latencia Ollama
python -c "
import time
import requests
url = 'http://localhost:11434/api/generate'
payload = {'model': 'solar-pro', 'prompt': 'Test', 'stream': False}
start = time.time()
r = requests.post(url, json=payload)
print(f'Latencia: {(time.time()-start)*1000:.1f}ms')
"

# Overhead wrapper
python scripts/benchmark_wrapper_overhead.py --backend ollama
```

---

## Docker

### Hardening Validation

```bash
# Ejecutar checklist de hardening
make validate-hardening

# O manualmente
docker inspect sarai-omni-engine | jq '.[0].HostConfig.SecurityOpt'
docker inspect sarai-omni-engine | jq '.[0].HostConfig.CapDrop'
docker inspect sarai-omni-engine | jq '.[0].HostConfig.ReadonlyRootfs'
```

### Build y Deploy

```bash
# Build local
docker-compose build

# Build multi-arch
docker buildx build --platform linux/amd64,linux/arm64 -t sarai:v2.14 .

# Deploy con hardening
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

---

## Logs

### Ubicaciones

```bash
logs/
├── feedback_log.jsonl               # Feedback implícito
├── voice_interactions_2025-01-01.jsonl
├── voice_interactions_2025-01-01.jsonl.hmac
├── web_queries_2025-01-01.jsonl
├── web_queries_2025-01-01.jsonl.sha256
└── audit_report_2025-01-01.md
```

### Consultas Útiles

```bash
# Últimas 10 interacciones
tail -n 10 logs/feedback_log.jsonl | jq .

# Filtrar por skill usado
jq 'select(.skill_used == "programming")' logs/feedback_log.jsonl

# Contar skills por tipo
jq -r '.skill_used // "none"' logs/feedback_log.jsonl | sort | uniq -c

# Ver búsquedas web de hoy
jq . logs/web_queries_$(date +%Y-%m-%d).jsonl

# Verificar integridad de logs
python scripts/verify_all_logs.py
```

---

## Emergency Procedures

### Safe Mode Activado

```bash
# Verificar estado
python -c "from core.audit import is_safe_mode; print(f'Safe Mode: {is_safe_mode()}')"

# Desactivar (SOLO si se verificó integridad)
rm state/SAFE_MODE

# Re-verificar logs antes de desactivar
python scripts/verify_all_logs.py
```

### Sistema No Responde

```bash
# 1. Verificar RAM
free -h

# 2. Matar procesos Python
pkill -f "python.*sarai"

# 3. Limpiar cache
rm -rf state/*.pkl __pycache__

# 4. Restart limpio
make clean
python main.py
```

### Recuperación de Desastre

```bash
# Backup de estado actual
tar -czf backup_$(date +%Y%m%d).tar.gz state/ logs/ config/

# Limpieza total
make distclean

# Reinstalación
make install

# Restaurar estado
tar -xzf backup_YYYYMMDD.tar.gz
```

---

## Referencia Rápida de Comandos

| Comando | Descripción | Tiempo |
|---------|-------------|--------|
| `make install` | Setup completo | 20-30 min |
| `make validate` | Validación rápida | 30 seg |
| `make health` | Dashboard de salud | - |
| `make benchmark VERSION=vX.X` | Benchmark completo | 5-10 min |
| `make clean` | Limpia artefactos | <1 seg |
| `make distclean` | Limpieza total | 5 seg |
| `pytest tests/ -v` | Todos los tests | 2-3 min |
| `python scripts/quick_validate.py` | Validación subsistemas | 30 seg |
| `bash scripts/run_audit_checklist.sh` | Auditoría completa | 10-15 min |

---

## Contacto y Referencias

- **Checklist de Auditoría**: `docs/AUDIT_CHECKLIST.md`
- **Guía Master**: `.github/copilot-instructions.md`
- **Estado Actual**: `STATUS_ACTUAL.md`
- **Arquitectura**: `ARCHITECTURE.md`
- **Changelog**: `CHANGELOG.md`

---

**Mantra v2.14**: 
_"Validación antes que velocidad. Auditoría antes que confianza.  
El sistema que se auto-verifica es el sistema que no falla."_
