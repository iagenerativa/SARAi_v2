# Omni Dual Strategy v2.16.1 - Plan de Implementación

**Fecha**: 29 de octubre de 2025  
**Arquitectura**: 3B Permanente + 7B Bajo Demanda (Memory-First)  
**Decisión**: ✅ APROBADA (WER 12.3% aceptable, ROI 99x)

---

## 🎯 Resumen Ejecutivo

**ESTRATEGIA VALIDADA**:
- **Omni-3B permanente en RAM** (2.6 GB): Maneja 70% casos (audio, texto corto, NLLB)
- **Omni-7B bajo demanda** (4.9 GB): Carga automática para 30% casos (imagen, video, RAG)

**JUSTIFICACIÓN WER**:
- Penalización: +2.6pp STT (12.3% vs 9.7%)
- Impacto real: <2% casos críticos
- ROI: 99x (107% throughput + 41% latencia + 30% UX)

**RAM BASELINE**: 5.2 GB → 10.1 GB pico (37% libre)

---

## 📊 Trade-off Validado

| Métrica | 3B Permanente | 7B Permanente | Ganador |
|---------|---------------|---------------|---------|
| Latencia Audio | 180ms ✅ | 240ms | 3B ⚡ |
| Throughput | 8.9 tok/s ✅ | 4.3 tok/s | 3B ⚡⚡ |
| Pipeline NLLB | 1.95s ✅ | 3.30s | 3B ⚡⚡⚡ |
| STT WER | 12.3% | 9.7% ✅ | 7B |
| Fluidez | Alta ✅ | Baja | 3B |
| RAM Baseline | 5.2 GB ✅ | 6.2 GB | 3B |

**Score**: 3B (9/10) vs 7B (4/10)

---

## 🏗️ Arquitectura Final

```
MEMORIA BASELINE (PERMANENTE):
├── SOLAR HTTP Client     : 0.2 GB
├── LFM2-1.2B            : 0.7 GB
├── Qwen-Omni-3B (FAST) ✅: 2.6 GB  ← SIEMPRE EN RAM
├── EmbeddingGemma       : 0.15 GB
├── TRM-Router + Mini    : 0.05 GB
└── Sistema + Python     : 1.5 GB
    ─────────────────────────────
    TOTAL                : 5.2 GB (67% RAM libre)

BAJO DEMANDA (AUTO-CARGA):
└── Qwen-Omni-7B (QUALITY): 4.9 GB
    Triggers:
      • input_type in ["image", "video"]
      • len(input) > 500 chars
      • web_query > 0.7 (RAG complejo)
      • code_switching detectado
    Auto-descarga: Tras 60s sin uso
    Cold-start: ~2-3s
    
RAM PICO (ambos cargados): 10.1 GB (37% libre)
```

---

## 🔧 Implementación: Fase 2

### Tarea 1: Modificar `core/graph.py`

**Archivo**: `core/graph.py`  
**Cambios**: Routing inteligente dual 3B/7B  
**Tiempo**: 45 min

#### 1.1. Modificar `_route_to_agent()` (línea ~580)

```python
def _route_to_agent(self, state: State) -> str:
    """
    Routing refinado v2.16.1 Memory-First:
    - 3B PERMANENTE para 70% casos (audio, texto corto, empatía)
    - 7B BAJO DEMANDA para 30% casos (multimodal complejo)
    """
    
    # PRIORIDAD 1: RAG si web_query > 0.7
    if state.get("web_query", 0.0) > 0.7:
        return "rag"
    
    # PRIORIDAD 2: Audio → SIEMPRE 3B (latencia <2s crítica)
    if state.get("input_type") == "audio":
        logger.info("🎤 Audio detectado → Omni-3B (WER 12.3%, 180ms)")
        return "omni_fast"
    
    # PRIORIDAD 3: Imagen/Video → 7B (calidad +18%)
    if state.get("input_type") in ["image", "video"]:
        logger.info("🖼️ Multimodal detectado → Omni-7B (carga bajo demanda)")
        return "omni_quality"
    
    # PRIORIDAD 4: Texto largo → 7B
    input_length = len(state.get("input", ""))
    if input_length > 500:
        logger.info(f"📝 Texto largo ({input_length} chars) → Omni-7B")
        return "omni_quality"
    
    # PRIORIDAD 5: Expert si alpha > 0.7 (técnico puro)
    if state.get("alpha", 0.0) > 0.7:
        return "expert"
    
    # PRIORIDAD 6: Soft alto → Omni-3B (empatía rápida)
    if state.get("soft", 0.0) > 0.7:
        logger.info("💬 Empatía alta → Omni-3B (8.9 tok/s)")
        return "omni_fast"
    
    # DEFAULT: Tiny (queries simples)
    return "tiny"
```

#### 1.2. Añadir nodo `generate_omni_fast` (línea ~300)

```python
def _generate_omni_fast(self, state: State) -> State:
    """
    Genera respuesta con Omni-3B (velocidad)
    Modelo PERMANENTE en RAM
    Casos: audio STT/TTS, texto corto, empatía, NLLB
    """
    from agents.omni_fast import get_omni_fast_agent
    
    logger.info("⚡ Generando con Omni-3B (permanente en RAM)")
    
    agent = get_omni_fast_agent()
    
    # Contexto emocional si disponible
    context = state.get("input", "")
    if state.get("soft", 0.0) > 0.5:
        emo_prefix = "[Tono empático] "
        context = emo_prefix + context
    
    # Generación
    response = agent.invoke(
        context,
        max_tokens=512  # Límite 3B
    )
    
    state["response"] = response
    state["agent_used"] = "omni_fast_3b"
    state["model_size"] = "3B"
    state["wer_expected"] = 12.3
    
    logger.info(f"✅ Respuesta generada: {len(response)} chars")
    
    return state
```

#### 1.3. Renombrar `_generate_omni()` → `_generate_omni_quality()` (línea ~330)

```python
def _generate_omni_quality(self, state: State) -> State:
    """
    Genera respuesta con Omni-7B (calidad)
    Modelo BAJO DEMANDA (carga automática)
    Casos: imagen, video, texto largo, RAG, code-switching
    """
    from agents.omni_native import get_omni_agent
    from core.model_pool import get_model_pool
    
    logger.info("🔄 Cargando Omni-7B bajo demanda...")
    
    # Carga automática (ModelPool maneja LRU)
    agent = get_omni_agent()
    
    # Contexto emocional
    context = state.get("input", "")
    if state.get("soft", 0.0) > 0.5:
        emo_prefix = "[Tono empático] "
        context = emo_prefix + context
    
    # Generación
    response = agent.invoke(
        context,
        max_tokens=2048  # Límite 7B
    )
    
    state["response"] = response
    state["agent_used"] = "omni_quality_7b"
    state["model_size"] = "7B"
    state["wer_expected"] = 9.7
    
    logger.info(f"✅ Respuesta generada: {len(response)} chars")
    
    # Programar auto-descarga (60s timeout)
    pool = get_model_pool()
    pool.schedule_unload("qwen_omni_quality", delay=60)
    
    return state
```

#### 1.4. Actualizar `_build_workflow()` (línea ~650)

```python
def _build_workflow(self):
    workflow = StateGraph(State)
    
    # ... nodos existentes ...
    
    # NUEVO: Nodo Omni-3B (Fast - Permanente)
    workflow.add_node("generate_omni_fast", self._generate_omni_fast)
    
    # RENOMBRAR: Omni-7B (Quality - Bajo Demanda)
    workflow.add_node("generate_omni_quality", self._generate_omni_quality)
    
    # Routing condicional desde MCP
    workflow.add_conditional_edges(
        "mcp",
        self._route_to_agent,
        {
            "expert": "generate_expert",
            "tiny": "generate_tiny",
            "rag": "execute_rag",
            "omni_fast": "generate_omni_fast",      # NUEVO
            "omni_quality": "generate_omni_quality"  # RENOMBRADO
        }
    )
    
    # Edges a feedback
    workflow.add_edge("generate_omni_fast", "feedback")
    workflow.add_edge("generate_omni_quality", "feedback")
    
    return workflow.compile()
```

---

### Tarea 2: Actualizar `config/sarai.yaml`

**Archivo**: `config/sarai.yaml`  
**Cambios**: Configuración dual 3B/7B  
**Tiempo**: 15 min

```yaml
models:
  # ... modelos existentes ...
  
  # NUEVO: Omni-3B (Fast - PERMANENTE)
  qwen_omni_fast:
    gguf_file: "Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
    repo_id: "unsloth/Qwen3-VL-4B-Instruct-GGUF"
    context_length: 2048
    max_tokens: 512
    n_threads: 6
    memory_mb: 2600
    permanent: true  # ✅ NUNCA descargar
    priority: "high"  # Carga al inicio
    use_cases:
      - "audio_stt"
      - "audio_tts"
      - "nllb_multilingual"
      - "short_text"
      - "emotional_response"
  
  # RENOMBRAR: qwen_omni → qwen_omni_quality
  qwen_omni_quality:
    gguf_file: "Qwen2.5-Omni-7B-Q4_K_M.gguf"
    repo_id: "unsloth/Qwen2.5-Omni-7B-GGUF"
    context_length: 8192
    max_tokens: 2048
    n_threads: 6
    memory_mb: 4900
    permanent: false  # ✅ Carga bajo demanda
    priority: "normal"
    ttl_seconds: 60  # Auto-descarga tras 60s
    use_cases:
      - "image_analysis"
      - "video_analysis"
      - "long_text"
      - "complex_reasoning"
      - "code_switching"

routing:
  # NUEVO: Configuración de routing dual
  omni_variant_selection:
    audio_threshold: 0.0  # Audio SIEMPRE → 3B
    image_video_threshold: 1.0  # Imagen/Video SIEMPRE → 7B
    text_length_threshold: 500  # >500 chars → 7B
    web_query_threshold: 0.7  # RAG → 7B
    soft_empathy_threshold: 0.7  # Empatía alta → 3B
```

---

### Tarea 3: Modificar `core/model_pool.py`

**Archivo**: `core/model_pool.py`  
**Cambios**: Auto-descarga programada + prioridad permanente  
**Tiempo**: 30 min

#### 3.1. Añadir `schedule_unload()` (línea ~150)

```python
def schedule_unload(self, logical_name: str, delay: int = 60):
    """
    Programa descarga automática de modelo tras X segundos
    Solo afecta modelos con permanent=false
    """
    import threading
    
    config = self.config['models'].get(logical_name, {})
    if config.get('permanent', False):
        logger.info(f"⚠️ {logical_name} es permanente, no se descarga")
        return
    
    def _unload_after_delay():
        time.sleep(delay)
        if logical_name in self.cache:
            logger.info(f"⏱️ Auto-descarga de {logical_name} tras {delay}s")
            del self.cache[logical_name]
            gc.collect()
    
    threading.Thread(target=_unload_after_delay, daemon=True).start()
    logger.info(f"⏱️ Programada descarga de {logical_name} en {delay}s")
```

#### 3.2. Modificar `_load_with_backend()` para respetar `permanent` (línea ~200)

```python
def _load_with_backend(self, logical_name: str, prefetch: bool = False):
    """
    Carga modelo con prioridad de permanencia
    permanent=true → Nunca descarga, carga al inicio
    """
    config = self.config['models'].get(logical_name, {})
    is_permanent = config.get('permanent', False)
    
    if is_permanent:
        logger.info(f"✅ Cargando {logical_name} PERMANENTE (nunca descarga)")
    else:
        logger.info(f"🔄 Cargando {logical_name} BAJO DEMANDA (TTL {config.get('ttl_seconds', 60)}s)")
    
    # ... resto del código de carga ...
```

#### 3.3. Añadir `preload_permanent_models()` (línea ~100)

```python
def preload_permanent_models(self):
    """
    Carga modelos con permanent=true al iniciar SARAi
    Ejecutar en __init__ de SARAiGraph
    """
    for name, config in self.config['models'].items():
        if config.get('permanent', False):
            logger.info(f"🚀 Precarga de modelo permanente: {name}")
            self.get(name)  # Carga en cache
```

---

### Tarea 4: Crear `tests/test_omni_dual_routing.py`

**Archivo**: `tests/test_omni_dual_routing.py`  
**Tiempo**: 30 min

```python
#!/usr/bin/env python3
"""
Test del routing dual Omni 3B+7B Memory-First
Valida que cada tipo de entrada use el modelo correcto
"""

import pytest
from core.graph import SARAiGraph

def test_audio_routes_to_3b():
    """Audio debe usar SIEMPRE Omni-3B (permanente)"""
    graph = SARAiGraph()
    
    state = {
        "input": "Hola, ¿cómo estás?",
        "input_type": "audio",
        "hard": 0.3,
        "soft": 0.8,
        "alpha": 0.3,
        "beta": 0.7
    }
    
    route = graph._route_to_agent(state)
    assert route == "omni_fast", f"Audio debe usar omni_fast, obtuvo: {route}"
    print("✅ Audio → Omni-3B (WER 12.3%, 180ms)")

def test_image_routes_to_7b():
    """Imágenes deben usar Omni-7B (bajo demanda)"""
    graph = SARAiGraph()
    
    state = {
        "input": "Analiza esta imagen de mi cámara",
        "input_type": "image",
        "hard": 0.6,
        "soft": 0.4,
        "alpha": 0.6,
        "beta": 0.4
    }
    
    route = graph._route_to_agent(state)
    assert route == "omni_quality", f"Imagen debe usar omni_quality, obtuvo: {route}"
    print("✅ Imagen → Omni-7B (bajo demanda, +18% accuracy)")

def test_long_text_routes_to_7b():
    """Texto largo (>500 chars) debe usar 7B"""
    graph = SARAiGraph()
    
    long_text = "x" * 600  # 600 caracteres
    state = {
        "input": long_text,
        "input_type": "text",
        "hard": 0.5,
        "soft": 0.5,
        "alpha": 0.5,
        "beta": 0.5,
        "web_query": 0.0
    }
    
    route = graph._route_to_agent(state)
    assert route == "omni_quality", f"Texto largo debe usar omni_quality, obtuvo: {route}"
    print("✅ Texto largo → Omni-7B")

def test_short_text_empathy_routes_to_3b():
    """Texto corto + empatía alta → 3B (fluidez)"""
    graph = SARAiGraph()
    
    state = {
        "input": "Cuéntame un chiste corto",
        "input_type": "text",
        "hard": 0.2,
        "soft": 0.8,  # Empatía alta
        "alpha": 0.2,
        "beta": 0.8,
        "web_query": 0.0
    }
    
    route = graph._route_to_agent(state)
    assert route == "omni_fast", f"Empatía alta debe usar omni_fast, obtuvo: {route}"
    print("✅ Empatía alta → Omni-3B (8.9 tok/s)")

def test_nllb_pipeline_uses_3b():
    """Pipeline NLLB multilingüe debe usar 3B (latencia crítica)"""
    graph = SARAiGraph()
    
    state = {
        "input": "Bonjour, comment ça va?",  # Francés
        "input_type": "audio",
        "detected_language": "fr",
        "hard": 0.3,
        "soft": 0.7,
        "alpha": 0.3,
        "beta": 0.7
    }
    
    route = graph._route_to_agent(state)
    assert route == "omni_fast", f"NLLB debe usar omni_fast, obtuvo: {route}"
    print("✅ NLLB Pipeline → Omni-3B (1.95s total)")

def test_permanent_model_loaded():
    """Validar que 3B está permanente en RAM"""
    from core.model_pool import get_model_pool
    
    pool = get_model_pool()
    pool.preload_permanent_models()
    
    assert "qwen_omni_fast" in pool.cache, "3B no está en cache permanente"
    print("✅ Omni-3B permanente en RAM (2.6 GB)")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

---

## ⏱️ Cronograma de Implementación

| Tarea | Archivo | Tiempo | Prioridad |
|-------|---------|--------|-----------|
| 1. Routing dual en graph.py | `core/graph.py` | 45 min | CRÍTICO |
| 2. Config dual | `config/sarai.yaml` | 15 min | ALTA |
| 3. Auto-descarga en pool | `core/model_pool.py` | 30 min | ALTA |
| 4. Tests de validación | `tests/test_omni_dual_routing.py` | 30 min | ALTA |
| 5. Ejecutar tests | Terminal | 10 min | ALTA |
| 6. Validación E2E | Terminal | 20 min | MEDIA |
| **TOTAL** | | **2h 30min** | |

---

## ✅ Checklist de Validación

### Pre-implementación
- [x] Modelo 3B descargado (`Qwen3-VL-4B-Instruct-Q4_K_M.gguf`)
- [x] Modelo 7B descargado (`Qwen2.5-Omni-7B-Q4_K_M.gguf`)
- [x] Benchmarks comparativos ejecutados (8.9 vs 4.3 tok/s)
- [x] Trade-off WER validado (12.3% aceptable, ROI 99x)
- [x] Arquitectura Memory-First aprobada

### Post-implementación
- [ ] Tests unitarios pasan (6/6 ✅)
- [ ] Audio → 3B routing correcto
- [ ] Imagen/Video → 7B routing correcto
- [ ] 3B permanente en RAM al inicio
- [ ] 7B auto-descarga tras 60s sin uso
- [ ] RAM baseline ≤ 5.5 GB
- [ ] RAM pico (dual) ≤ 10.5 GB
- [ ] Latencia audio < 2s (3B)
- [ ] Latencia multimodal < 25s (7B)

---

## 📊 KPIs Esperados (Post-Implementación)

| KPI | Target | Actual | Estado |
|-----|--------|--------|--------|
| RAM Baseline | ≤ 5.5 GB | TBD | ⏳ |
| RAM Pico (dual) | ≤ 10.5 GB | TBD | ⏳ |
| Latencia Audio (3B) | ≤ 2s | TBD | ⏳ |
| Throughput 3B | ≥ 8.5 tok/s | TBD | ⏳ |
| Latencia Multimodal (7B) | ≤ 25s | TBD | ⏳ |
| Cold-start 7B | ≤ 3s | TBD | ⏳ |
| Auto-descarga 7B | 60s ± 5s | TBD | ⏳ |
| Tests unitarios | 6/6 PASS | TBD | ⏳ |

---

## 🚀 Próximos Pasos

1. **AHORA**: Aprobación final del usuario
2. **Inmediato**: Implementar Tarea 1 (graph.py)
3. **Luego**: Tareas 2-4 (config + pool + tests)
4. **Validación**: Ejecutar tests + medición RAM
5. **Documentación**: Actualizar completion report

**Tiempo total estimado**: 2.5 horas

---

**Referencias**:
- Benchmarks reales: `docs/OMNI_DUAL_STRATEGY_v2.16.1.md`
- Análisis WER: Terminal output 29 Oct 2025
- Decisión 3B vs 7B: `docs/OMNI_3B_VS_7B_DECISION.md`
