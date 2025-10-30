# Arquitectura de Integración Voz-LLM (M3.2)

## Diagrama de Flujo Completo

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  SARAi v2.11 - Voice-LLM Integration                     │
│                        (M3.2 Implementation)                              │
└─────────────────────────────────────────────────────────────────────────┘

INPUT LAYER (User)
══════════════════════════════════════════════════════════════════════════
         ┌──────────┐                    ┌──────────┐
         │  Audio   │                    │   Text   │
         │  bytes   │                    │  string  │
         └────┬─────┘                    └────┬─────┘
              │                               │
              └───────────┬───────────────────┘
                          ↓
══════════════════════════════════════════════════════════════════════════
DETECTION LAYER (detect_input_type)
══════════════════════════════════════════════════════════════════════════
                  ┌──────────────────┐
                  │ detect_input_type│
                  │                  │
                  │ Si audio_input:  │
                  │   → "audio"      │
                  │ Si input:        │
                  │   → "text"       │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
        [input_type="audio"]      [input_type="text"]
              │                         │
══════════════════════════════════════════════════════════════════════════
VOICE PROCESSING LAYER (process_voice) - SOLO SI AUDIO
══════════════════════════════════════════════════════════════════════════
              ↓                         │
      ┌───────────────┐                │
      │ process_voice │                │
      └───────┬───────┘                │
              │                         │
              ↓                         │
      ┌─────────────────────────────────────────────┐
      │ Audio Router (agents/audio_router.py)      │
      │ ┌─────────────────────────────────────┐   │
      │ │ 1. Detect Language (LID)            │   │
      │ │    - Whisper-tiny (STT 20ms)        │   │
      │ │    - fasttext (LID 10ms)            │   │
      │ │    → lang = "es"|"en"|"fr"|...      │   │
      │ │                                      │   │
      │ │ 2. Route to Engine                  │   │
      │ │    - es/en → Omni-3B (nativo)       │   │
      │ │    - fr/de/ja → NLLB (traducción)   │   │
      │ │    - unknown → LFM2 (fallback)      │   │
      │ └─────────────────────────────────────┘   │
      └───────────────┬─────────────────────────────┘
                      │
                      ↓
      ┌─────────────────────────────────────────────┐
      │ Omni Pipeline (agents/omni_pipeline.py)    │
      │ ┌─────────────────────────────────────┐   │
      │ │ Qwen3-VL-4B-Instruct ONNX (Q4)           │   │
      │ │                                      │   │
      │ │ INPUT: audio_bytes                  │   │
      │ │                                      │   │
      │ │ OUTPUTS:                             │   │
      │ │  - text: Transcripción STT          │   │
      │ │  - emotion: "empático"|"neutral"|   │   │
      │ │             "urgente"                │   │
      │ │  - confidence: 0.0-1.0               │   │
      │ │  - latency_ms: <250ms                │   │
      │ └─────────────────────────────────────┘   │
      └───────────────┬─────────────────────────────┘
                      │
                      ↓
          ┌────────────────────────┐
          │ Update State:          │
          │ - input = transcripción│
          │ - detected_emotion     │
          │ - detected_lang        │
          │ - voice_metadata       │
          └────────────┬───────────┘
                       │
                       │ (Converge con flujo texto)
══════════════════════════════════════════════════════════════════════════
CLASSIFICATION LAYER (classify) - COMÚN PARA AUDIO Y TEXTO
══════════════════════════════════════════════════════════════════════════
                       │
                       ├─────────────────┐
                       ↓                 ↓
               ┌───────────────┐   ┌──────────────┐
               │   classify    │   │  (texto vino │
               │               │   │  directo)    │
               │ TRM-Classifier│   └──────┬───────┘
               │ + Embeddings  │          │
               └───────┬───────┘          │
                       │                  │
                       └──────┬───────────┘
                              ↓
                   ┌─────────────────────┐
                   │ Scores:             │
                   │ - hard: 0.0-1.0     │
                   │ - soft: 0.0-1.0     │
                   │ - web_query: 0.0-1.0│
                   └──────────┬──────────┘
                              ↓
══════════════════════════════════════════════════════════════════════════
MCP LAYER (compute_weights)
══════════════════════════════════════════════════════════════════════════
                   ┌─────────────────────┐
                   │  MCP v2             │
                   │  compute_weights()  │
                   │                     │
                   │  α, β = f(hard,soft)│
                   └──────────┬──────────┘
                              ↓
                   ┌─────────────────────┐
                   │ Weights:            │
                   │ - α (expert weight) │
                   │ - β (soft weight)   │
                   └──────────┬──────────┘
                              ↓
══════════════════════════════════════════════════════════════════════════
ROUTING LAYER (route_to_agent) - NUEVA LÓGICA v2.11
══════════════════════════════════════════════════════════════════════════
                   ┌─────────────────────┐
                   │ Routing Logic:      │
                   │                     │
                   │ IF web_query > 0.7  │
                   │   → RAG             │
                   │ ELIF α > 0.7        │
                   │   → Expert (SOLAR)  │
                   │ ELSE                │
                   │   → Tiny (LFM2)     │
                   └──────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
        ┌─────────┐     ┌─────────┐    ┌─────────┐
        │   RAG   │     │ Expert  │    │  Tiny   │
        │  Agent  │     │  Agent  │    │  Agent  │
        └────┬────┘     └────┬────┘    └────┬────┘
══════════════════════════════════════════════════════════════════════════
LLM GENERATION LAYER
══════════════════════════════════════════════════════════════════════════
             │               │              │
             ↓               ↓              ↓
      ┌────────────┐  ┌────────────┐  ┌────────────┐
      │ SearXNG    │  │ SOLAR-10.7B│  │ LFM2-1.2B  │
      │ + Síntesis │  │ GGUF Q4_K_M│  │ GGUF Q4_K_M│
      │ con SOLAR  │  │            │  │            │
      │            │  │ n_ctx:     │  │ n_ctx:     │
      │ Latencia:  │  │ - Short:512│  │ - 2048     │
      │ 25-30s     │  │ - Long:2048│  │            │
      │            │  │            │  │ Modulación:│
      │            │  │ Latencia:  │  │ - Empatía  │
      │            │  │ 30-60s     │  │ - Urgencia │
      │            │  │            │  │            │
      │            │  │            │  │ Latencia:  │
      │            │  │            │  │ 10-20s     │
      └────┬───────┘  └────┬───────┘  └────┬───────┘
           │               │              │
           └───────────────┴──────────────┘
                          ↓
                   ┌──────────────┐
                   │  response    │
                   │  (texto)     │
                   └──────┬───────┘
                          │
══════════════════════════════════════════════════════════════════════════
EMOTION MODULATION LAYER - NUEVO v2.11 (SOLO SI AUDIO INPUT)
══════════════════════════════════════════════════════════════════════════
                          ↓
              ┌────────────────────────┐
              │ IF input_type="audio"  │
              │   AND detected_emotion │
              │   → enhance_with_emotion│
              │ ELSE                   │
              │   → skip               │
              └────────┬───────────────┘
                       │
                       ↓ (si audio)
          ┌──────────────────────────────────┐
          │ enhance_with_emotion             │
          │ ┌──────────────────────────────┐ │
          │ │ Modulación según emoción:    │ │
          │ │                              │ │
          │ │ "empático":                  │ │
          │ │   → LFM2 reformula con       │ │
          │ │     tono cálido, frases      │ │
          │ │     de apoyo                 │ │
          │ │                              │ │
          │ │ "urgente":                   │ │
          │ │   → LFM2 elimina intro,      │ │
          │ │     va al grano              │ │
          │ │                              │ │
          │ │ "neutral":                   │ │
          │ │   → No modifica              │ │
          │ │                              │ │
          │ │ Latencia: +1-2s (LFM2)       │ │
          │ │ Cache hit rate: 60%          │ │
          │ └──────────────────────────────┘ │
          └───────────────┬──────────────────┘
                          │
                          ↓
                   ┌──────────────┐
                   │  response    │
                   │  (modulado)  │
                   └──────┬───────┘
                          │
══════════════════════════════════════════════════════════════════════════
TTS LAYER - NUEVO v2.11 (SOLO SI AUDIO INPUT)
══════════════════════════════════════════════════════════════════════════
                          ↓
              ┌────────────────────────┐
              │ IF input_type="audio"  │
              │   → generate_tts       │
              │ ELSE                   │
              │   → skip               │
              └────────┬───────────────┘
                       │
                       ↓ (si audio)
          ┌──────────────────────────────────┐
          │ generate_tts                     │
          │ ┌──────────────────────────────┐ │
          │ │ Qwen3-VL-4B-Instruct (TTS mode)   │ │
          │ │                              │ │
          │ │ INPUT:                       │ │
          │ │  - text: response modulada   │ │
          │ │  - lang: detected_lang       │ │
          │ │  - emotion: detected_emotion │ │
          │ │                              │ │
          │ │ OUTPUT:                      │ │
          │ │  - audio_bytes (wav 16kHz)   │ │
          │ │                              │ │
          │ │ Prosodia emocional:          │ │
          │ │  - "empático" → pitch alto,  │ │
          │ │    pausas largas             │ │
          │ │  - "urgente" → ritmo rápido, │ │
          │ │    énfasis                   │ │
          │ │                              │ │
          │ │ Latencia: <500ms P99         │ │
          │ └──────────────────────────────┘ │
          └───────────────┬──────────────────┘
                          │
                          ↓
                   ┌──────────────┐
                   │ audio_output │
                   │ (bytes)      │
                   └──────┬───────┘
                          │
══════════════════════════════════════════════════════════════════════════
FEEDBACK LAYER (log_feedback) - COMÚN
══════════════════════════════════════════════════════════════════════════
                          ↓
                   ┌──────────────────┐
                   │ log_feedback     │
                   │                  │
                   │ Logs:            │
                   │ - Si texto:      │
                   │   feedback_log   │
                   │                  │
                   │ - Si voz:        │
                   │   voice_audit    │
                   │   (HMAC-SHA256)  │
                   └────────┬─────────┘
                            │
══════════════════════════════════════════════════════════════════════════
OUTPUT LAYER (User)
══════════════════════════════════════════════════════════════════════════
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
       ┌──────────────┐          ┌──────────────┐
       │ audio_output │          │  response    │
       │ (si audio)   │          │  (texto)     │
       └──────────────┘          └──────────────┘
              │                           │
              └──────────────┬────────────┘
                             ↓
                      ┌──────────────┐
                      │    USER      │
                      │  recibe:     │
                      │              │
                      │ - Audio (voz)│
                      │   O          │
                      │ - Texto      │
                      └──────────────┘
```

## Métricas de Latencia por Ruta

### Ruta 1: Voz → Expert → Voz (Caso más complejo)
```
Audio Input (usuario graba)
    ↓ 0ms
detect_input_type
    ↓ <5ms
process_voice
    ├─ Audio Router (LID)        30-50ms
    ├─ Omni-3B (STT + Emotion)   150-200ms
    └─ Update State              <5ms
    ↓ Total: ~250ms
classify (TRM + Embeddings)      50-100ms
    ↓
compute_weights (MCP)            <10ms
    ↓
route_to_agent                   <5ms
    ↓
generate_expert (SOLAR)          30,000-60,000ms (30-60s) ⚠️ CUELLO DE BOTELLA
    ↓
enhance_with_emotion (LFM2)      1,000-2,000ms
    ↓
generate_tts (Omni-3B)           300-500ms
    ↓
log_feedback                     <20ms
    ↓
Audio Output (usuario escucha)

TOTAL: ~31-62 segundos (dominado por SOLAR)
```

### Ruta 2: Voz → Tiny → Voz (Caso óptimo)
```
Audio Input
    ↓ 250ms (voice processing)
classify + mcp + route           100ms
    ↓
generate_tiny (LFM2)             10,000-20,000ms (10-20s)
    ↓
enhance_with_emotion             
    - Cache hit (60%)            <10ms
    - Cache miss (40%)           1,000-2,000ms
    ↓
generate_tts                     300-500ms
    ↓
log_feedback                     <20ms

TOTAL: ~11-23 segundos (Tiny)
       ~10-11 segundos (Tiny + cache hit)
```

### Ruta 3: Voz → RAG → Voz (Web)
```
Audio Input
    ↓ 250ms (voice processing)
classify + mcp + route           100ms
    ↓
execute_rag
    ├─ SearXNG search            2,000-5,000ms
    ├─ Synthesis (SOLAR)         20,000-30,000ms
    └─ Web audit                 <50ms
    ↓ Total: ~25-35 segundos
enhance_with_emotion             1,000-2,000ms (sin cache para RAG)
    ↓
generate_tts                     300-500ms
    ↓
log_feedback                     <20ms

TOTAL: ~26-38 segundos (RAG + voz)
```

### Ruta 4: Texto → Expert → Texto (Sin voz)
```
Text Input
    ↓ <5ms (detect_input_type)
classify                         100ms
    ↓
mcp + route                      <15ms
    ↓
generate_expert (SOLAR)          30,000-60,000ms
    ↓
(skip enhance_with_emotion)      0ms
    ↓
(skip generate_tts)              0ms
    ↓
log_feedback                     <20ms

TOTAL: ~30-60 segundos (SOLAR solo texto)
```

## Optimizaciones Aplicadas (M3.2)

### 1. Prefetch de LLM durante Voice Processing
```python
# core/prefetcher.py - Nuevo en M3.2

class VoicePrefetcher:
    """
    Mientras Omni-3B procesa audio (250ms), prefetch LLM
    probable según contexto del usuario.
    """
    
    def on_audio_received(self, audio_bytes: bytes):
        # Iniciar prefetch en paralelo
        threading.Thread(
            target=self._prefetch_likely_llm,
            daemon=True
        ).start()
        
        # Procesar audio (no bloquea)
        return process_voice(audio_bytes)
    
    def _prefetch_likely_llm(self):
        """
        Heurística: Usuario habitual usa 70% Expert, 25% Tiny, 5% RAG
        → Prefetch Expert (SOLAR) proactivamente
        """
        model_pool.prefetch_model("expert_short")
```

**Ganancia**: -2 a -5 segundos (carga de modelo en paralelo)

### 2. Caché de Modulaciones Emocionales
```python
# core/graph.py - enhance_with_emotion con cache

emotion_cache = LRUCache(maxsize=1000, ttl=3600)

def _enhance_with_emotion(self, state: State):
    cache_key = hash(state["response"] + state["detected_emotion"])
    
    if cache_key in emotion_cache:
        return {"response": emotion_cache[cache_key]}
    
    # Generar modulación (LFM2 1-2s)
    modulated = self.tiny_agent.generate(...)
    emotion_cache[cache_key] = modulated
    
    return {"response": modulated}
```

**Cache hit rate esperado**: 60% (respuestas similares)  
**Ganancia**: -1 a -2 segundos (60% de requests)

### 3. Compartir ONNX Runtime entre STT y TTS
```python
# agents/omni_pipeline.py

# ANTES (M3.1): 2 instancias ONNX
omni_stt_session = ort.InferenceSession("omni-stt.onnx")  # 2.1GB
omni_tts_session = ort.InferenceSession("omni-tts.onnx")  # 2.1GB
# Total: 4.2GB RAM ⚠️

# DESPUÉS (M3.2): 1 instancia compartida
omni_session = ort.InferenceSession("omni-unified.onnx")  # 2.1GB
# Total: 2.1GB RAM ✅
```

**Ganancia RAM**: -2.1GB

### 4. Cuantización ONNX Q4 (Fase 5)
```python
# Antes: FP16 → 2.1GB
# Después: Q4 → 1.5GB

# Degradación:
# - WER: +0.3% (1.8% → 2.1%, aceptable)
# - MOS: -0.05 (4.38 → 4.33, imperceptible)
# - Latencia: +5% (200ms → 210ms, aceptable)
```

**Ganancia RAM**: -600MB

## Estado de Memoria (RAM) Esperado v2.11

| Componente | v2.10 (sin voz) | v2.11 (con voz) | Δ |
|------------|-----------------|-----------------|---|
| **Sistema Base** | 4.0 GB | 4.0 GB | - |
| **EmbeddingGemma** | 0.15 GB | 0.15 GB | - |
| **TRM-Router + Mini** | 0.08 GB | 0.08 GB | - |
| **SOLAR (expert_short)** | 4.8 GB | 4.8 GB | - |
| **LFM2 (tiny)** | 0.7 GB | 0.7 GB | - |
| **Qwen-Omni-3B (Q4)** | 0 GB | 1.5 GB | +1.5 GB |
| **NLLB (si activo)** | 0 GB | 1.2 GB | +1.2 GB |
| **MCP Cache** | 0.2 GB | 0.2 GB | - |
| **Buffers/Overhead** | 0.8 GB | 1.0 GB | +0.2 GB |
| **TOTAL P99** | **10.8 GB** | **12.4 GB** | **+1.6 GB** |

**Target**: ≤13 GB ✅  
**Margen**: 600 MB (5% buffer)

## Integración con Componentes Existentes

### 1. ModelPool (core/model_pool.py)
**Sin cambios necesarios**. Ya soporta lazy loading de Omni-3B y NLLB.

```python
# Ya funciona en M3.1
model_pool.get("omni3b")   # Carga Qwen3-VL-4B-Instruct
model_pool.get("nllb")     # Carga NLLB-200
```

### 2. TRM-Classifier (core/trm_classifier.py)
**Cambio menor**: Añadir cabeza `voice_confidence` (M3.3, futuro).

```python
# M3.2: Sin cambios (usa cabezas existentes)
# M3.3: Añadir para clasificar calidad de audio
self.head_voice_confidence = nn.Linear(self.d_model, 1)
```

### 3. MCP (core/mcp.py)
**Sin cambios necesarios**. Los pesos α/β funcionan igual para voz que para texto.

### 4. Feedback Detector (core/feedback.py)
**Extensión**: Detectar feedback de voz basado en tono de respuesta del usuario.

```python
# Futuro M3.4: Feedback de voz
def detect_voice_feedback(previous_response_emotion, next_input_emotion):
    """
    Si emoción del usuario cambia positivamente → feedback positivo
    Ej: previo="urgente", siguiente="neutral" → usuario satisfecho
    """
```

### 5. Web Audit (core/web_audit.py)
**Ya integrado en M3.1**. `VoiceAuditLogger` funcional.

---

**Documento creado**: 28 octubre 2025  
**Versión**: 1.0  
**Autor**: SARAi + Noel  
**Estado**: Planificación (M3.2 pendiente de implementación)
