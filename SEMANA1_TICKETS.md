# Semana 1: Consolidación Núcleo (MoE + Prioridades)

**Período**: 31 octubre - 6 noviembre 2025  
**Responsable**: Equipo SARAi  
**Estado**: 🟢 EN PROGRESO

---

## 📋 Tickets Granulares

### T1.1: ~~Implementar carga dinámica de skills MoE~~ **CANCELADO - Malentendido Conceptual**
**Prioridad**: ~~CRÍTICA~~ **OBSOLETO**  
**Estimación**: ~~8 horas~~ **0 horas (no hacer)**  
**Dependencias**: Ninguna  
**Owner**: Backend Team  
**Estado**: ❌ CANCELADO (31 Oct 2025)

**Razón de cancelación**:
**MALENTENDIDO**: Interpreté skills como 6 LLMs separados (CodeLlama, Mistral, etc).  
**REALIDAD**: Skills son **CONFIGURACIONES** (prompts especializados + parámetros) para SOLAR/LFM2.

Ejemplo:
```yaml
skills:
  programming:
    model: "solar"  # ← Usa SOLAR existente
    temperature: 0.2  # Precisión para código
    system_prompt: "Eres un experto en programación..."
    domains: ["código", "python", "debugging"]
```

**Beneficio**: Especialización profunda (prompts expertos) SIN cargar modelos adicionales (0 GB RAM extra).

**Archivos a REVERTIR**:
- ~~`core/model_pool.py` (+150 LOC)~~ → Eliminar `get_skill()`, mantener solo `get()`
- ~~`config/sarai.yaml` (+90 LOC)~~ → Skills quedan como referencia, NO se cargan como LLMs
- ~~`tests/test_model_pool_skills.py` (+320 LOC)~~ → Borrar archivo

---

### T1.2: ~~Activar `route_to_skills` en MCP runtime~~ **CANCELADO - Reemplazado por Skill Selection**
**Prioridad**: ~~CRÍTICA~~ **OBSOLETO**  
**Estimación**: ~~6 horas~~ **0 horas (no hacer)**  
**Dependencias**: T1.1 completado  
**Owner**: Backend Team  
**Estado**: ❌ CANCELADO (31 Oct 2025)

**Razón de cancelación**:
El método correcto NO es `execute_skills_moe()` (que cargaba LLMs), sino `select_skill_and_generate()` que:
1. Recibe scores TRM (ej: {"programming": 0.85})
2. Selecciona skill de mayor score
3. Carga config especializada (prompt + temperature)
4. Genera con SOLAR/LFM2 usando ese contexto

**Archivos a REVERTIR**:
- ~~`core/mcp.py` (+110 LOC)~~ → Eliminar `execute_skills_moe()`
- ~~`tests/test_mcp_skills.py` (+300 LOC)~~ → Borrar archivo

---

### T1.1-FINAL: Skill Configs + TRM Heads Especializados (8h)
**Prioridad**: CRÍTICA  
**Estimación**: 8 horas  
**Dependencias**: Ninguna  
**Owner**: ML Team  
**Estado**: ⏳ PENDIENTE

**Objetivo**: Implementar skills como **CONFIGURACIONES** (prompts + parámetros) que mejoran SOLAR/LFM2, NO como LLMs separados.

**Arquitectura**:
```
Input → TRM-Router (6 heads) → Skill Config → SOLAR/LFM2 con prompt especializado
```

**Tareas**:
- [ ] Crear `core/skill_configs.py` con 6 configs especializados:
  - `programming`: SOLAR + temp=0.2 + prompt experto en código
  - `diagnosis`: SOLAR + temp=0.3 + prompt análisis de logs/RCA
  - `finance`: SOLAR + temp=0.4 + prompt analista financiero (CFA)
  - `creative`: LFM2 + temp=0.9 + prompt escritor creativo
  - `reasoning`: SOLAR + temp=0.5 + prompt lógica formal
  - `general`: SOLAR + temp=0.7 + prompt asistente genérico
- [ ] Modificar TRM-Router con 6 heads (programming, diagnosis, finance, creative, reasoning, general)
- [ ] Implementar `MCP.select_skill_and_generate()`:
  - Recibe scores TRM → selecciona best skill
  - Carga config (prompt + temperature + modelo)
  - Genera con SOLAR/LFM2 usando contexto especializado
- [ ] Dataset sintético: 10K queries clasificadas por skill
- [ ] Entrenar TRM-Router con accuracy >85%
- [ ] 12 tests (6 skills × 2 tests cada uno)

**Archivos afectados**:
- `core/skill_configs.py` (NUEVO, +180 LOC: diccionario SKILL_CONFIGS)
- `core/trm_classifier.py` (+30 LOC: 6 heads especializados)
- `core/mcp.py` (+60 LOC: `select_skill_and_generate()`)
- `core/graph.py` (+25 LOC: integración LangGraph)
- `scripts/generate_skill_dataset.py` (NUEVO, +200 LOC)
- `scripts/train_trm_skills.py` (NUEVO, +150 LOC)
- `tests/test_skill_configs.py` (NUEVO, +120 LOC)

**Criterios de aceptación**:
- ✅ Input "Escribe código Python para decorador" → TRM: `programming=0.85`
- ✅ MCP carga `SKILL_CONFIGS["programming"]` → temp=0.2, prompt experto
- ✅ SOLAR genera con contexto especializado (NO CodeLlama separado)
- ✅ Accuracy TRM >85% en test set (1K queries)
- ✅ Latencia <50ms para clasificación TRM
- ✅ 12/12 tests pasando
- ✅ **0 GB RAM adicional** (usa SOLAR/LFM2 existentes)

**Beneficio vs Skills MoE**:
- ❌ Skills MoE: 6 LLMs × 800 MB = +4.8 GB RAM
- ✅ Skill Configs: 6 prompts en YAML = +0 GB RAM
- 🎯 Especialización: Prompts expertos > LLMs genéricos de 7B

---

### T1.2-FINAL: MCP Skill Selection (6h)
**Prioridad**: CRÍTICA  
**Estimación**: 6 horas  
**Dependencias**: T1.1-FINAL completado  
**Owner**: Backend Team  
**Estado**: ⏳ PENDIENTE

**Objetivo**: Integrar skill selection en LangGraph con feedback loop.

**Tareas**:
- [ ] Modificar `core/graph.py` para usar `select_skill_and_generate()` en vez de routing hard/soft simple
- [ ] Nodo `classify_skill` retorna scores de 6 skills
- [ ] Nodo `select_config` carga configuración del skill ganador
- [ ] Nodo `generate_specialized` ejecuta SOLAR/LFM2 con prompt especializado
- [ ] Feedback logger registra skill usado + confianza
- [ ] 6 tests E2E (un flow por skill)

**Archivos afectados**:
- `core/graph.py` (+40 LOC: nodos especializados)
- `core/feedback.py` (+25 LOC: log skill metadata)
- `tests/test_graph_skills.py` (NUEVO, +150 LOC: E2E tests)

**Criterios de aceptación**:
- ✅ Query "¿Cómo optimizo SQL?" → programming skill → SOLAR temp=0.2
- ✅ Query "Cuenta un cuento de dragones" → creative skill → LFM2 temp=0.9
- ✅ Logs incluyen: `{"skill": "programming", "confidence": 0.85, "model": "solar"}`
- ✅ 6/6 tests E2E pasando
- ✅ Latencia total <30s (incluye clasificación + generación)

---

### T1.3-FINAL: MCP Continuous Learning (6h)  
**Owner**: Performance Team

**Tareas**:
- [ ] Crear wrapper `PriorityQueue` con 3 niveles (critical/normal/batch)
- [ ] Modificar `main.py` para clasificar requests entrantes
- [ ] Activar batching `llama.cpp` solo en cola "batch"
- [ ] Configurar `n_parallel` dinámico según carga
- [ ] Tests: latencia por nivel, no starvation de cola normal

**Archivos afectados**:
- `core/batch_prioritizer.py` (+120 LOC modificadas)
- `main.py` (+60 LOC)
- `core/model_pool.py` (+40 LOC para n_parallel dinámico)
- `tests/test_batch_prioritizer.py` (+150 LOC)

**Criterios de aceptación**:
- ✅ Requests marcados `critical` → P99 ≤ 2s
- ✅ Requests normales → P50 ≤ 20s
- ✅ Batching se activa solo con ≥3 requests en cola
- ✅ No starvation: cola normal procesa al menos 1 req cada 5s
- ✅ 8/8 tests pasando

---

### T1.4: Implementar métricas de latencia por nivel
**Prioridad**: MEDIA  
**Estimación**: 4 horas  
**Dependencias**: T1.3 completado  
**Owner**: Observability Team

**Tareas**:
- [ ] Añadir histogramas Prometheus por nivel de prioridad
- [ ] Endpoint `/metrics` expone `sarai_latency_seconds{priority="critical|normal|batch"}`
- [ ] Dashboard Grafana con 3 paneles (P50, P95, P99 por nivel)
- [ ] Alertas si P99 critical > 2s

**Archivos afectados**:
- `sarai/health_dashboard.py` (+50 LOC)
- `extras/grafana_god.json` (+80 LOC)
- `tests/test_metrics_priority.py` (nuevo, +60 LOC)

**Criterios de aceptación**:
- ✅ Métricas visibles en `/metrics`
- ✅ Grafana muestra 3 series de latencia
- ✅ Alerta dispara webhook si P99 critical > 2s
- ✅ 3/3 tests pasando

---

### T1.5: Benchmark end-to-end Semana 1
**Prioridad**: MEDIA  
**Estimación**: 6 horas  
**Dependencias**: T1.1, T1.2, T1.3, T1.4 completados  
**Owner**: QA Team

**Tareas**:
- [ ] Ejecutar 100 queries mixtas (50% normal, 30% batch, 20% critical)
- [ ] Validar KPIs documentados (RAM ≤12GB, latencias por nivel)
- [ ] Generar reporte de rendimiento con gráficos
- [ ] Identificar cuellos de botella y documentar en issues

**Archivos afectados**:
- `tests/benchmark_semana1.py` (nuevo, +200 LOC)
- `docs/BENCHMARK_SEMANA1_REPORT.md` (nuevo)

**Criterios de aceptación**:
- ✅ RAM P99 ≤ 12GB durante todo el test
- ✅ Latencia P99 critical ≤ 2s
- ✅ Latencia P50 normal ≤ 20s
- ✅ Skills se cargan/descargan correctamente (sin leaks)
- ✅ Informe generado con recomendaciones

---

## 📊 Métricas de seguimiento Semana 1

| Métrica | Target | Estado |
|---------|--------|--------|
| Tests implementados | 32/32 | 0/32 ⏳ |
| LOC añadidas | ~870 | 0 ⏳ |
| RAM P99 | ≤ 12GB | - ⏳ |
| Latencia P99 Critical | ≤ 2s | - ⏳ |
| Latencia P50 Normal | ≤ 20s | - ⏳ |
| Skills funcionales | 6 | 0 ⏳ |

---

## 🎯 Definition of Done (Semana 1)

- ✅ Todos los tickets T1.1–T1.5 marcados como completados
- ✅ 32/32 tests pasando en CI
- ✅ Benchmark ejecutado con resultados dentro de targets
- ✅ Documentación actualizada (CHANGELOG, STATUS)
- ✅ Code review aprobado por al menos 1 revisor
- ✅ Merge a rama `master` sin conflictos

---

**Inicio**: 31 octubre 2025  
**Revisión mid-week**: 3 noviembre 2025  
**Cierre**: 6 noviembre 2025  
**Handoff a Semana 2**: 7 noviembre 2025 (AM)
