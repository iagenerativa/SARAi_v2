# SARAi v2.11 "Omni-Sentinel" - Roadmap de Implementación

## 📋 Resumen Ejecutivo

**Estado actual**: Blueprint arquitectónico sellado, código base parcialmente implementado.

**Objetivo**: Sistema de producción 100% funcional con voz empática, RAG web y skills domóticos.

**Fecha objetivo**: 2025-11-15 (3 semanas)

**KPI de éxito**: Todos los tests pasan + Docker funcional + Validación KPIs en hardware real.

---

## 🎯 Hitos de Desarrollo (Milestones)

### Milestone 1: Core Funcional (Semana 1)
**Fecha límite**: 2025-11-03  
**Objetivo**: Sistema base operativo sin voz ni skills avanzados.

#### Tasks:
- [ ] **M1.1**: Implementar TRM-Classifier completo
  - Archivo: `core/trm_classifier.py`
  - Estado: Parcial (falta cabeza web_query)
  - Criterio: `pytest tests/test_trm_classifier.py` pasa 100%

- [ ] **M1.2**: Implementar MCP v2 con Fast-Cache
  - Archivo: `core/mcp.py`
  - Estado: Implementado (validar persistencia)
  - Criterio: `pytest tests/test_mcp.py` pasa 100%

- [ ] **M1.3**: ModelPool con GGUF dinámico
  - Archivo: `core/model_pool.py`
  - Estado: Implementado (validar prefetch)
  - Criterio: RAM P99 ≤ 12GB bajo carga

- [ ] **M1.4**: Graph básico (hard/soft routing)
  - Archivo: `core/graph.py`
  - Estado: Implementado (sin RAG ni voz)
  - Criterio: Respuesta end-to-end funcional

- [ ] **M1.5**: Descarga de modelos GGUF
  - Script: `scripts/download_gguf_models.py`
  - Estado: Implementado
  - Criterio: SOLAR + LFM2 descargados correctamente

**Validación M1**:
```bash
make install  # Sin errores
python main.py --test-mode  # Responde a query simple
```

---

### Milestone 2: RAG + Web (Semana 2)
**Fecha límite**: 2025-11-10  
**Objetivo**: Búsqueda web funcional con auditoría HMAC.

#### Tasks:
- [ ] **M2.1**: RAG Agent completo
  - Archivo: `agents/rag_agent.py`
  - Estado: Implementado (validar SearXNG)
  - Criterio: `pytest tests/test_rag_agent.py` pasa 100%

- [ ] **M2.2**: Web Cache con TTL dinámico
  - Archivo: `core/web_cache.py`
  - Estado: Implementado
  - Criterio: Cache hit rate 40-60%

- [ ] **M2.3**: Web Audit con HMAC
  - Archivo: `core/web_audit.py`
  - Estado: Implementado
  - Criterio: `make audit-log` verifica integridad

- [ ] **M2.4**: Docker SearXNG configurado
  - Archivo: `docker-compose.override.yml`
  - Estado: Implementado
  - Criterio: SearXNG responde en localhost:8888

- [ ] **M2.5**: Integración RAG en Graph
  - Archivo: `core/graph.py`
  - Estado: Pendiente
  - Criterio: Routing web_query > 0.7 funcional

**Validación M2**:
```bash
docker-compose up searxng -d
python -m agents.rag_agent --query "¿Clima en Tokio?"
# Debe retornar respuesta sintetizada + log HMAC
```

---

### Milestone 3: Voz Empática (Semana 3 - Parte 1)
**Fecha límite**: 2025-11-13  
**Objetivo**: Motor de voz Omni-3B operativo con latencia <250ms.

#### Tasks:
- [ ] **M3.1**: Audio Router con LID
  - Archivo: `agents/audio_router.py`
  - Estado: ✅ Implementado
  - Criterio: `pytest tests/test_audio_router.py` pasa 100%

- [ ] **M3.2**: Omni Pipeline (Qwen3-VL-4B-Instruct)
  - Archivo: `agents/omni_pipeline.py`
  - Estado: Implementado (validar ONNX)
  - Criterio: Latencia voz-a-voz <250ms en i7

- [ ] **M3.3**: NLLB Translation Server (opcional)
  - Archivo: `agents/nllb_server.py`
  - Estado: Pendiente
  - Criterio: Traducción es→fr funcional

- [ ] **M3.4**: HMAC Audit para Voz
  - Archivo: `core/web_audit.py` (extendido)
  - Estado: Documentado en copilot-instructions
  - Criterio: Logs de voz verificables

- [ ] **M3.5**: Integración Voz en Graph
  - Archivo: `core/graph.py`
  - Estado: Pendiente
  - Criterio: Input de audio procesado correctamente

**Validación M3**:
```bash
python -m agents.omni_pipeline --test audio_sample.wav
# Debe retornar audio + texto + latencia <250ms
```

---

### Milestone 4: Skills Domóticos (Semana 3 - Parte 2)
**Fecha límite**: 2025-11-15  
**Objetivo**: Home Assistant + Network Diag operativos.

#### Tasks:
- [ ] **M4.1**: Home Ops Skill
  - Archivo: `skills/home_ops.py`
  - Estado: ✅ Implementado
  - Criterio: `pytest tests/test_home_ops.py` pasa 100%

- [ ] **M4.2**: Network Diag Skill
  - Archivo: `skills/network_diag.py`
  - Estado: Pendiente
  - Criterio: Ping + traceroute sandboxed

- [ ] **M4.3**: Skills en MoE Router
  - Archivo: `core/graph.py`
  - Estado: Pendiente
  - Criterio: Routing skill > 0.7 ejecuta skill

- [ ] **M4.4**: Firejail Sandboxing
  - Config: Validar perfiles firejail
  - Estado: Documentado
  - Criterio: Skills no pueden escribir fuera de /tmp

- [ ] **M4.5**: Home Assistant Mock (testing)
  - Archivo: `tests/mock_ha.py`
  - Estado: Pendiente
  - Criterio: Tests no requieren HA real

**Validación M4**:
```bash
python -m skills.home_ops --action turn_on --entity light.living_room
# Debe ejecutar sin errores (con HA mock)
```

---

### Milestone 5: Docker + Hardening (Continuo)
**Fecha límite**: 2025-11-15  
**Objetivo**: Todos los servicios dockerizados con hardening aplicado.

#### Tasks:
- [x] **M5.1**: Docker Hardening (omni_pipeline)
  - Estado: ✅ Completado
  - Criterio: `make validate-hardening` pasa

- [x] **M5.2**: Docker Hardening (searxng)
  - Estado: ✅ Completado
  - Criterio: `make validate-hardening` pasa

- [ ] **M5.3**: Docker Hardening (sarai backend)
  - Archivo: `Dockerfile` principal
  - Estado: Pendiente
  - Criterio: security_opt + cap_drop aplicados

- [ ] **M5.4**: Dockerfile.omni optimizado
  - Archivo: `Dockerfile.omni`
  - Estado: Implementado (validar build)
  - Criterio: Imagen <1.5GB

- [ ] **M5.5**: Health Dashboard REST
  - Archivo: `sarai/health_dashboard.py`
  - Estado: Implementado (validar métricas)
  - Criterio: `/health` retorna JSON correcto

**Validación M5**:
```bash
docker-compose build --no-cache
docker-compose up -d
curl http://localhost:8080/health
# Debe retornar {"status": "HEALTHY", ...}
```

---

## 🧪 Testing Strategy

### Tests Unitarios
**Ubicación**: `tests/`  
**Framework**: pytest

- `test_trm_classifier.py`: TRM-Classifier (hard/soft/web_query)
- `test_mcp.py`: MCP α/β weights + cache
- `test_audio_router.py`: Audio routing + fallback
- `test_rag_agent.py`: RAG pipeline + sentinel
- `test_home_ops.py`: Home Assistant skill
- `test_web_cache.py`: Web cache TTL + hit rate

**Comando**:
```bash
pytest tests/ -v -s --tb=short
```

**Criterio de éxito**: 100% tests pasan.

---

### Tests de Integración
**Ubicación**: `tests/integration/`

- `test_full_pipeline.py`: Input → Graph → Output
- `test_rag_integration.py`: RAG + SearXNG real
- `test_voice_integration.py`: Audio → Omni → Response
- `test_skills_integration.py`: Skills + LangGraph

**Comando**:
```bash
pytest tests/integration/ -v -s
```

**Criterio de éxito**: 100% tests pasan con servicios Docker arriba.

---

### Tests de Rendimiento (Benchmarks)
**Ubicación**: `tests/sarai_bench.py`

- Latencia P50/P99 (Critical/Normal/RAG)
- RAM P99 bajo carga
- MCP cache hit rate
- Voz latencia end-to-end

**Comando**:
```bash
make bench
```

**Criterio de éxito**: Todos los KPIs dentro de rango objetivo.

---

### Golden Queries (Regresión)
**Ubicación**: `tests/golden_queries.jsonl`

Queries críticas que NUNCA deben fallar:
1. "Explica recursión en Python" (hard)
2. "Me siento frustrado con este código" (soft)
3. "¿Quién ganó el Oscar 2024?" (RAG)
4. [Audio en español] (voz)
5. "Enciende las luces de la sala" (skill)

**Comando**:
```bash
python tests/test_golden_queries.py
```

**Criterio de éxito**: 0% regresión.

---

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] Todos los tests pasan (unit + integration + bench)
- [ ] Golden queries 100% correctas
- [ ] Docker images buildeadas sin errores
- [ ] `.env` configurado correctamente
- [ ] Modelos GGUF descargados

### Deployment
- [ ] `docker-compose up -d` sin errores
- [ ] Health endpoint responde en 5s
- [ ] Logs de arranque sin warnings críticos
- [ ] SearXNG responde a búsquedas
- [ ] Home Assistant (si aplica) conectado

### Post-Deployment
- [ ] Latencia P50 ≤ 20s (normal queries)
- [ ] Latencia P99 ≤ 2s (critical queries)
- [ ] RAM P99 ≤ 12GB
- [ ] 0% regresión en golden queries
- [ ] Logs HMAC verificables

---

## 📊 KPIs a Validar en Hardware Real

| KPI | Objetivo | Comando de Validación |
|-----|----------|----------------------|
| **RAM P99** | ≤ 12 GB | `python scripts/monitor_ram.py --duration 300` |
| **Latencia P50 (Normal)** | ≤ 20 s | `make bench` |
| **Latencia P99 (Critical)** | ≤ 2 s | `make bench` |
| **Latencia P50 (RAG)** | ≤ 30 s | `python -m agents.rag_agent --benchmark` |
| **Latencia Voz (P50)** | ≤ 250 ms | `python -m agents.omni_pipeline --benchmark` |
| **Hard-Acc** | ≥ 0.85 | `pytest tests/test_trm_classifier.py` |
| **Empathy** | ≥ 0.75 | `pytest tests/test_mcp.py` |
| **Web Cache Hit Rate** | 40-60% | `python -m core.web_cache --stats` |
| **Regresión MCP** | 0% | `python tests/test_golden_queries.py` |
| **Disponibilidad** | 99.9% | `make validate-hardening` |

---

## 🔧 Troubleshooting Guide

### Problema: RAM excede 12GB
**Síntomas**: OOM killer termina procesos  
**Solución**:
1. Verificar `max_concurrent_llms: 2` en `config/sarai.yaml`
2. Comprobar que Omni-3B usa ONNX q4 (no PyTorch)
3. Aumentar `model_ttl_seconds` para evitar recargas frecuentes

### Problema: Latencia >30s
**Síntomas**: Respuestas lentas  
**Solución**:
1. Verificar CPU threads: `n_threads: 6` (dejar 2 libres)
2. Comprobar que usa GGUF (no transformers en CPU)
3. Validar prefetcher activo: logs de TRM-Mini

### Problema: Docker build falla
**Síntomas**: Error en stage builder  
**Solución**:
1. Limpiar caché: `docker builder prune -a`
2. Verificar GGUF paths en Dockerfile
3. Comprobar espacio en disco: `df -h`

### Problema: SearXNG no responde
**Síntomas**: RAG falla con timeout  
**Solución**:
1. Verificar contenedor arriba: `docker ps | grep searxng`
2. Logs del contenedor: `docker logs sarai-searxng`
3. Reiniciar servicio: `docker-compose restart searxng`

### Problema: Voz con latencia >500ms
**Síntomas**: Audio lento  
**Solución**:
1. Verificar ONNX Runtime instalado: `pip show onnxruntime`
2. Comprobar zram en Pi-4: `zramctl`
3. Validar GPU no está activa (solo CPU): `nvidia-smi` no debe existir

---

## 📅 Cronograma Detallado

### Semana 1 (2025-10-28 → 2025-11-03)
**Lunes-Martes**: M1.1, M1.2 (TRM + MCP)  
**Miércoles-Jueves**: M1.3, M1.4 (ModelPool + Graph)  
**Viernes**: M1.5 + Validación M1

### Semana 2 (2025-11-04 → 2025-11-10)
**Lunes-Martes**: M2.1, M2.2 (RAG + Cache)  
**Miércoles-Jueves**: M2.3, M2.4 (Audit + SearXNG)  
**Viernes**: M2.5 + Validación M2

### Semana 3 - Parte 1 (2025-11-11 → 2025-11-13)
**Lunes**: M3.2, M3.4 (Omni Pipeline + Audit)  
**Martes**: M3.3, M3.5 (NLLB + Graph Integration)  
**Miércoles**: Validación M3

### Semana 3 - Parte 2 (2025-11-14 → 2025-11-15)
**Jueves**: M4.2, M4.3, M4.4 (Network Diag + MoE + Firejail)  
**Viernes**: M4.5 + M5.3 + Validación M4 + M5  
**Viernes tarde**: Deployment completo + Validación KPIs final

---

## 🎯 Definición de "Done"

Un milestone se considera **COMPLETADO** cuando:
1. ✅ Todos los tests asociados pasan
2. ✅ Código reviewed (self-review con checklist)
3. ✅ Documentación actualizada (docstrings + README)
4. ✅ KPI específico validado en hardware
5. ✅ Commit con mensaje descriptivo

**Criterio global de "Done" para v2.11**:
- ✅ 100% tests pasan (unit + integration + golden)
- ✅ Docker build + up funcional
- ✅ Todos los KPIs dentro de objetivo
- ✅ Documentación completa (CHANGELOG + ARCHITECTURE + copilot-instructions)
- ✅ Health endpoint operativo
- ✅ 0% regresión en golden queries

---

## 🔄 Proceso de Desarrollo Iterativo

1. **Seleccionar task** del milestone actual
2. **Implementar** código (TDD: test primero si es posible)
3. **Validar** con test específico
4. **Commit** con mensaje descriptivo
5. **Actualizar** todo list (marcar como completado)
6. **Repetir** hasta completar milestone
7. **Validación de milestone** (ejecutar todos los tests del milestone)
8. **Pasar al siguiente milestone**

---

## 📝 Notas Finales

- **Prioridad**: M1 → M2 → M3 → M4 → M5
- **Bloqueos**: M2 requiere M1 completo, M3 requiere M2, etc.
- **Paralelización**: M5 (Docker) puede hacerse en paralelo con M1-M4
- **Contingencia**: Si un milestone se atrasa, posponer features opcionales (NLLB, Network Diag)
- **Comunicación**: Actualizar todo list diariamente

**Mantra de desarrollo**: _"Cada commit debe dejar el sistema en estado deployable. No romper nunca el main branch."_

---

**Versión del Planning**: 1.0  
**Fecha de creación**: 2025-10-27  
**Última actualización**: 2025-10-27  
**Responsable**: Noel  
**Revisado por**: SARAi  
**Licencia**: CC BY-NC-SA 4.0  

---

## 📄 Nota sobre Licencia y Contribuciones

SARAi v2.11 está licenciado bajo **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

**Para contribuidores**:
- Al enviar un Pull Request, aceptas que tu contribución se licencie bajo CC BY-NC-SA 4.0
- Mantén la atribución original al autor (Noel)
- No uses el proyecto para fines comerciales sin permiso
- Comparte tus modificaciones bajo la misma licencia

Ver archivo `LICENSE` para términos legales completos.
