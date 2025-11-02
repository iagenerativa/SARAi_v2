# SARAi v2.14 - Checklist de Auditoría y Verificación

**Propósito**: Validación rápida del estado del sistema para auditoría, operación y troubleshooting.

**Audiencia**: Operadores, auditores de seguridad, DevSecOps, equipos de seguimiento.

**Última actualización**: 2025-11-01

---

## ✅ Checklist Rápida (5 minutos)

### 1. Configuración Base

```bash
# Verificar variables de entorno críticas
echo "OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-NO CONFIGURADO}"
echo "SOLAR_MODEL_NAME: ${SOLAR_MODEL_NAME:-NO CONFIGURADO}"
echo "HOME_ASSISTANT_URL: ${HOME_ASSISTANT_URL:-NO CONFIGURADO}"

# Validar que no hay IPs hardcodeadas en configuración
grep -r "192\.168\|10\.[0-9]" config/ || echo "✅ Sin IPs hardcodeadas"

# Verificar modelos configurados
python -c "from core.unified_model_wrapper import ModelRegistry; r = ModelRegistry(); r.load_config(); print(f'✅ {len(r._config)} modelos configurados')"
```

**Criterio de éxito**: 
- Variables de entorno definidas en `.env`
- 0 IPs hardcodeadas en `config/`
- ≥8 modelos en registry

---

### 2. Health Endpoints

```bash
# Verificar /health (HTML para humanos)
curl -s http://localhost:8080/health | grep -q "HEALTHY" && echo "✅ /health OK" || echo "❌ /health FAIL"

# Verificar /health (JSON para monitoreo)
curl -s -H "Accept: application/json" http://localhost:8080/health | jq '.status' | grep -q "HEALTHY" && echo "✅ /health JSON OK"

# Verificar /metrics (Prometheus)
curl -s http://localhost:8080/metrics | grep -q "sarai_" && echo "✅ /metrics OK" || echo "❌ /metrics FAIL"
```

**Criterio de éxito**:
- `/health` retorna status "HEALTHY"
- `/metrics` expone métricas con prefijo `sarai_`
- Content negotiation funciona (HTML vs JSON)

---

### 3. Tests Unitarios e Integración

```bash
# Tests del Unified Wrapper (13 tests)
pytest tests/test_unified_wrapper.py -v --tb=short

# Tests de integración (si Ollama disponible)
pytest tests/test_unified_wrapper_integration.py -v -m "not requires_ollama" || echo "⚠️ Ollama no disponible"

# Tests de Skills Phoenix (84 tests)
pytest tests/test_graph_skills_integration.py -v

# Tests de Layers Architecture (14 tests)
pytest tests/test_layer_integration.py -v
```

**Criterio de éxito**:
- Unified Wrapper: 13/13 passing
- Skills Phoenix: 84/84 passing
- Layers: 14/14 passing
- Total: ≥111 tests en verde

---

### 4. Auditoría de Logs (HMAC + SHA-256)

```bash
# Verificar estructura de logs
ls -lh logs/ | grep -E "\.jsonl$|\.sha256$|\.hmac$" && echo "✅ Logs estructurados"

# Verificar integridad de logs web (ejemplo)
DATE=$(date +%Y-%m-%d)
python -m core.web_audit --verify $DATE && echo "✅ Logs web íntegros" || echo "❌ Corrupción detectada"

# Verificar logs de voz (si existen)
if [ -f logs/voice_interactions_$DATE.jsonl ]; then
    python scripts/verify_voice_audit.py $DATE && echo "✅ Logs voz íntegros"
fi

# Contar líneas auditadas hoy
LINES=$(wc -l logs/*_$DATE.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
echo "📊 Interacciones auditadas hoy: ${LINES:-0}"
```

**Criterio de éxito**:
- Logs JSONL con sidecars `.sha256` o `.hmac`
- Verificación de integridad pasa (0 corrupciones)
- Timestamps recientes (<24h)

---

### 5. Supply Chain (Cosign + SBOM)

```bash
# Verificar firma de la última release (requiere cosign instalado)
LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v2.14.0")

# Instalar cosign si no está disponible
if ! command -v cosign &> /dev/null; then
    echo "⚠️ cosign no instalado. Instalar: https://github.com/sigstore/cosign"
else
    # Verificar firma de imagen Docker
    cosign verify \
        --certificate-identity-regexp="https://github.com/iagenerativa/SARAi_v2/.*" \
        --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
        ghcr.io/iagenerativa/sarai_v2:$LATEST_TAG && echo "✅ Imagen firmada y verificada"
fi

# Verificar SBOM (si existe en release)
if [ -f sbom.spdx.json ]; then
    jq '.packages | length' sbom.spdx.json && echo "✅ SBOM presente"
fi
```

**Criterio de éxito**:
- Imagen Docker firmada con Cosign
- SBOM disponible y válido
- Certificado OIDC del CI/CD oficial

---

### 6. Hardening de Contenedores (Docker)

```bash
# Verificar no-new-privileges
docker inspect sarai-omni-engine 2>/dev/null | jq '.[0].HostConfig.SecurityOpt' | grep -q "no-new-privileges:true" && echo "✅ no-new-privileges OK"

# Verificar cap_drop ALL
docker inspect sarai-omni-engine 2>/dev/null | jq '.[0].HostConfig.CapDrop' | grep -q "ALL" && echo "✅ cap_drop ALL OK"

# Verificar read-only filesystem
docker inspect sarai-omni-engine 2>/dev/null | jq '.[0].HostConfig.ReadonlyRootfs' | grep -q "true" && echo "✅ read-only OK"

# Test de escalada (debe fallar)
docker exec sarai-omni-engine sudo ls 2>&1 | grep -q "effective uid is not 0" && echo "✅ Escalada bloqueada" || echo "⚠️ sudo posible"
```

**Criterio de éxito**:
- `no-new-privileges: true`
- `cap_drop: ALL`
- `read_only: true`
- sudo/escalada bloqueados

---

### 7. Benchmark y Overhead

```bash
# Ejecutar benchmark completo del wrapper
python scripts/benchmark_wrapper_overhead.py

# Verificar overhead ≤5%
python scripts/benchmark_wrapper_overhead.py 2>&1 | grep -q "Overhead.*<5%" && echo "✅ Overhead dentro de objetivo"

# Verificar RAM en uso (requiere sistema en ejecución)
python -c "import psutil; ram_gb = psutil.virtual_memory().used / (1024**3); print(f'RAM actual: {ram_gb:.1f} GB'); exit(0 if ram_gb <= 12.0 else 1)" && echo "✅ RAM bajo límite"
```

**Criterio de éxito**:
- Wrapper overhead ≤5%
- RAM P99 ≤12 GB
- Latencia P50 ≤20s

---

### 8. Skills Phoenix (Detección)

```bash
# Verificar detección de skills
python -c "
from core.mcp import detect_and_apply_skill

queries = [
    ('Cómo crear una función en Python', 'programming'),
    ('Analizar error de base de datos', 'diagnosis'),
    ('Estrategia de inversión ROI', 'financial'),
    ('Escribe una historia corta', 'creative'),
]

for query, expected in queries:
    skill = detect_and_apply_skill(query, 'solar')
    detected = skill['name'] if skill else None
    status = '✅' if detected == expected else '❌'
    print(f'{status} {query[:30]}... → {detected}')
"
```

**Criterio de éxito**:
- 7 skills detectables (programming, diagnosis, financial, creative, reasoning, cto, sre)
- Long-tail patterns funcionan (0 falsos positivos)
- Temperature y max_tokens correctos por skill

---

### 9. Layers Architecture (Estado)

```bash
# Verificar Layer 2 (tone memory) persistencia
if [ -f state/layer2_tone_memory.jsonl ]; then
    ENTRIES=$(wc -l < state/layer2_tone_memory.jsonl)
    echo "📊 Tone memory: $ENTRIES entradas (max 256)"
    [ $ENTRIES -le 256 ] && echo "✅ Buffer dentro de límite"
fi

# Verificar Layer 3 (tone bridge) estilos
python -c "
from core.layer3_fluidity.tone_bridge import get_tone_bridge
bridge = get_tone_bridge()
# Test con emoción positiva alta energía
profile = bridge.update('happy', 0.8, 0.7)
print(f'✅ Estilo inferido: {profile.style}')
assert profile.style == 'energetic_positive'
"
```

**Criterio de éxito**:
- Layer 1: Detección de emoción funcional
- Layer 2: Buffer JSONL ≤256 entradas
- Layer 3: 9 estilos inferibles

---

## 📋 Checklist Extendida (Auditoría Completa - 30 minutos)

### 10. Configuración de Modelos (models.yaml)

```bash
# Validar sintaxis YAML
python -c "import yaml; yaml.safe_load(open('config/models.yaml'))" && echo "✅ YAML válido"

# Verificar backends configurados
python -c "
import yaml
config = yaml.safe_load(open('config/models.yaml'))
backends = set(m.get('backend') for m in config.values() if isinstance(m, dict))
print(f'Backends: {sorted(backends)}')
assert len(backends) >= 5, 'Menos de 5 backends configurados'
print('✅ Backends suficientes')
"

# Verificar que GGUF tiene model_path
python -c "
import yaml
config = yaml.safe_load(open('config/models.yaml'))
gguf_models = [name for name, cfg in config.items() if cfg.get('backend') == 'gguf']
for model in gguf_models:
    assert 'model_path' in config[model], f'{model} sin model_path'
print(f'✅ {len(gguf_models)} modelos GGUF con path')
"
```

---

### 11. Memoria y Recursos

```bash
# Verificar que ModelPool respeta límites
python -c "
from core.model_pool import get_model_pool
pool = get_model_pool()

# Verificar max_concurrent_llms
import yaml
config = yaml.safe_load(open('config/sarai.yaml'))
max_llms = config.get('runtime', {}).get('max_concurrent_llms', 2)
assert max_llms <= 2, 'max_concurrent_llms debe ser ≤2'
print(f'✅ max_concurrent_llms: {max_llms}')

# Verificar TTL configurado
ttl = config.get('memory', {}).get('model_ttl_seconds', 45)
assert ttl >= 30, 'TTL muy bajo'
print(f'✅ model_ttl_seconds: {ttl}s')
"
```

---

### 12. Seguridad de Skills (Firejail + chattr)

```bash
# Verificar que skills críticos usan firejail
if command -v firejail &> /dev/null; then
    echo "✅ firejail disponible"
    
    # Verificar configuración de skills
    grep -r "use_firejail.*true" config/sarai.yaml && echo "✅ Skills con sandbox"
fi

# Verificar logs append-only (requiere permisos)
if command -v lsattr &> /dev/null; then
    lsattr logs/*.jsonl 2>/dev/null | grep -q "^----a" && echo "✅ Logs append-only (chattr +a)"
fi
```

---

### 13. Red y Conectividad

```bash
# Verificar que contenedores usan red interna
docker network inspect sarai_internal 2>/dev/null | jq '.[0].Internal' | grep -q "true" && echo "✅ Red interna aislada"

# Verificar que no hay acceso externo no autorizado
docker exec sarai-omni-engine ping -c 1 8.8.8.8 2>&1 | grep -q "Network is unreachable" && echo "✅ Sin acceso externo"

# Verificar whitelist de puertos expuestos
EXPOSED=$(docker ps --filter "name=sarai" --format "{{.Ports}}" | grep -oE "[0-9]+->")
echo "Puertos expuestos: $EXPOSED"
```

---

### 14. Degradación Elegante (Fallbacks)

```bash
# Test de fallback: Simular fallo de Ollama
python -c "
import os
os.environ['OLLAMA_BASE_URL'] = 'http://localhost:99999'  # Puerto inválido

from core.unified_model_wrapper import get_model

try:
    solar = get_model('solar_short')
    solar.invoke('test')
    print('❌ Debería haber fallado')
except Exception as e:
    print(f'✅ Fallback correcto: {type(e).__name__}')
"

# Verificar que sistema sigue respondiendo con LFM2 local
python -c "
from core.unified_model_wrapper import get_model
lfm2 = get_model('lfm2')
response = lfm2.invoke('Di hola')
assert len(response) > 0
print('✅ Fallback a LFM2 funciona')
"
```

---

### 15. Rendimiento E2E (Latencia Real)

```bash
# Medir latencia de query simple
time python -c "
from core.graph import create_sarai_graph
graph = create_sarai_graph()
result = graph.invoke({'input': 'Hola'})
print(result['response'][:50])
"

# Verificar que tarda <5s para query simple
# (en producción: P50 ~19.5s para queries complejas)
```

---

## 🎯 KPIs Mínimos para Aprobar Auditoría

| KPI | Mínimo Aceptable | Cómo Verificar |
|-----|------------------|----------------|
| **Tests passing** | ≥95% | `pytest tests/ -v` |
| **Health endpoint** | HTTP 200 | `curl http://localhost:8080/health` |
| **RAM P99** | ≤12 GB | Sección 7 (Benchmark) |
| **Logs íntegros** | 100% | Sección 4 (Auditoría) |
| **Imagen firmada** | Cosign OK | Sección 5 (Supply Chain) |
| **Hardening score** | ≥95/100 | Sección 6 (Docker) |
| **Skills detectables** | 7/7 | Sección 8 (Phoenix) |
| **Sin IPs hardcodeadas** | 0 | Sección 1 (Config) |

---

## 📝 Plantilla de Reporte de Auditoría

```markdown
# Reporte de Auditoría SARAi v2.14

**Fecha**: YYYY-MM-DD
**Auditor**: [Nombre]
**Versión**: v2.14.x
**Entorno**: [dev|staging|prod]

## Resumen Ejecutivo

- ✅/❌ Configuración base
- ✅/❌ Health endpoints
- ✅/❌ Tests (XXX/111 passing)
- ✅/❌ Auditoría de logs
- ✅/❌ Supply chain
- ✅/❌ Hardening Docker
- ✅/❌ Benchmark
- ✅/❌ Skills Phoenix

## Hallazgos

### Críticos
- [Listar si hay]

### Altos
- [Listar si hay]

### Medios
- [Listar si hay]

### Bajos
- [Listar si hay]

## Recomendaciones

1. [Acción 1]
2. [Acción 2]

## Estado Final

**APROBADO / APROBADO CON OBSERVACIONES / RECHAZADO**
```

---

## 🔗 Referencias

- Documento maestro: `.github/copilot-instructions.md`
- Estado actual: `STATUS_ACTUAL.md`
- Guía de wrapper: `docs/UNIFIED_WRAPPER_GUIDE.md`
- Roadmap: `ROADMAP_v2.16_OMNI_LOOP.md`

---

**Mantenimiento**: Actualizar este checklist con cada versión mayor (v2.15, v2.16, etc.)
