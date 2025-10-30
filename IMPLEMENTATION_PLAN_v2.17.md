# SARAi v2.17 - Plan de Implementación Full-Duplex

## 📅 Estado Actual: 30 de octubre de 2025

---

## ✅ **COMPLETADO** - Capa 1: I/O Full-Duplex

### Componentes Implementados:

#### **Canal IN** ✅

1. **`vosk_streaming.py`** ✅
   - Vosk STT en modo streaming (chunks de 100ms)
   - Clase `VoskStreamingSession` con buffer inteligente
   - Detección de fin de frase por silencio (500ms)
   - Test standalone funcional

2. **`bert_embedder.py`** ✅ (reutilizado, archivo existente)
   - BERT-es embeddings (768-dim)
   - Similitud semántica
   - Batch encoding

3. **`lora_router.py`** ✅
   - Router con LoRA (Low-Rank Adaptation)
   - Clasificación: TRM / LLM / Traducir
   - ~2M params entrenables (LoRA adapters)
   - Sistema de entrenamiento incluido
   - Save/Load funcional

4. **`input_thread.py`** ✅
   - Orquestador del Canal IN
   - 3 threads:
     - Audio Capture (arecord + VAD)
     - STT Processing (Vosk streaming)
     - Routing (BERT + LoRA)
   - Colas thread-safe
   - Estadísticas en tiempo real

#### **Canal OUT** ✅

1. **`output_thread.py`** ✅
   - Thread 1: Response Generation (TRM/LLM/NLLB)
   - Thread 2: TTS Streaming (Piper)
   - Thread 3: Audio Playback (aplay)
   - TRM Cache con respuestas comunes (< 50ms)
   - Integración LFM2 para generación completa
   - Espera inteligente (no interrumpe usuario)

2. **`orchestrator.py`** ✅
   - Coordina Canal IN + Canal OUT
   - Cola compartida de decisiones
   - Flag compartido (usuario hablando)
   - Estadísticas unificadas
   - Modo interactivo completo

### Test de Capa 1:
```bash
# Test componentes individuales
python3 -m core.layer1_io.vosk_streaming    # ✅ Streaming STT
python3 -m core.layer1_io.lora_router       # ✅ Router con datos sintéticos
python3 -m core.layer1_io.input_thread      # ✅ Canal IN
python3 -m core.layer1_io.output_thread     # ✅ Canal OUT
python3 -m core.layer1_io.orchestrator      # ✅ Orquestador

# Test integración completa
python3 tests/test_layer1_fullduplex.py     # ✅ Full-Duplex E2E
```

### Archivos Creados (Capa 1):
```
core/layer1_io/
├── __init__.py                 ✅ (actualizado)
├── vosk_streaming.py           ✅ (323 líneas)
├── lora_router.py              ✅ (384 líneas)
├── input_thread.py             ✅ (267 líneas)
├── output_thread.py            ✅ (386 líneas)
└── orchestrator.py             ✅ (241 líneas)

tests/
└── test_layer1_fullduplex.py   ✅ (118 líneas)

Total: ~1,719 líneas de código productivo
```

---

## ⏳ **PENDIENTE** - Capa 2: Coherencia Persistente (RAG)

### Componentes a Crear:

1. **`embedding_gemma.py`** ⏳
   - Wrapper para EmbeddingGemma 2B
   - Embeddings de 2048-dim
   - Lazy loading

2. **`qdrant_store.py`** ⏳
   - Cliente Qdrant local
   - Indexación de turnos conversacionales
   - Retrieval K=5 turnos relevantes

3. **`conversation_rag.py`** ⏳
   - RAG conversacional completo
   - Manejo de interrupciones
   - Reconstrucción de contexto

### Test Capa 2:
```python
# Test indexación
rag = ConversationRAG()
rag.index_turn(
    user="¿Qué es la IA?",
    assistant="La IA es..."
)

# Test recuperación
context = rag.retrieve_context("sobre lo que dijiste de IA", k=3)
# → Retorna turno anterior con embeddings similares
```

---

## ⏳ **PENDIENTE** - Capa 3: Gestión de Fluidez (Fillers)

### Componentes a Crear:

1. **`filler_manager.py`** ⏳
   - Detecta latencia > 800ms
   - Reproduce coletillas pregrabadas
   - Estrategia adaptativa (no spam de fillers)

2. **`latency_detector.py`** ⏳
   - Monitorea tiempo desde última palabra
   - Predice latencia LLM
   - Trigger de fillers

3. **`generate_fillers.py`** ⏳
   - Script para generar coletillas con Piper TTS
   - "Mmm...", "Déjame pensar...", "Interesante..."
   - Guardar en `assets/fillers/*.wav`

4. **`assets/fillers/fillers.yaml`** ⏳
   - Metadata de cada filler
   - Duración, contexto de uso, gap mínimo

### Fillers a Generar:
```yaml
fillers:
  - id: hmm_01
    file: hmm_01.wav
    text: "Mmm..."
    duration_ms: 400
    usage: "Latencia 800-1200ms"
  
  - id: thinking_01
    file: thinking_01.wav
    text: "Déjame pensar..."
    duration_ms: 800
    usage: "Latencia 1200-2000ms"
  
  - id: interesting_01
    file: interesting_01.wav
    text: "Interesante, necesito considerar..."
    duration_ms: 1200
    usage: "Latencia > 2000ms"
```

---

## ⏳ **PENDIENTE** - Capa 4: Orquestación Dinámica (LoRA Scheduler)

### Componentes a Crear:

1. **`lora_scheduler.py`** ⏳
   - LoRA para ajuste dinámico de recursos
   - Prioriza threads según contexto:
     - Usuario hablando → STT priority
     - Usuario callado + LLM → LLM priority
     - Respuesta lista → TTS priority
   - Ajusta `n_threads` de LFM2 dinámicamente

2. **`resource_monitor.py`** ⏳
   - Monitorea CPU, RAM, tamaño de colas
   - Detecta cuellos de botella
   - Feed al LoRA Scheduler

### Políticas de Priorización:
```python
# Ejemplo de ajuste dinámico
context = {
    "input_queue_size": 3,      # Usuario hablando mucho
    "llm_queue_size": 0,        # LLM idle
    "cpu_usage": 0.85,          # CPU alta
    "user_speaking": True
}

# LoRA Scheduler decide:
adjustments = {
    "boost_stt": True,          # Prioridad a STT
    "llm_n_threads": 2,         # Reduce LLM temporalmente
    "tts_pause": True           # Pausa TTS (usuario hablando)
}
```

---

## ⏳ **PENDIENTE** - Integración Final

### Orquestador Maestro:

**`core/orchestrator_fullduplex.py`** ⏳
- Coordina las 4 capas
- Gestiona ciclo de vida de threads
- Métricas globales
- Health monitoring

### Test E2E:

**`tests/test_fullduplex.py`** ⏳
```python
# Test conversación completa
orchestrator = FullDuplexOrchestrator()
orchestrator.start()

# Usuario: "Hola, ¿cómo estás?"
# → Canal IN → Router: TRM
# → Canal OUT → TRM cache: "¡Hola! Muy bien, gracias."
# → Latencia total: ~1.2s

# Usuario: "Explícame la relatividad"
# → Canal IN → Router: LLM
# → Capa 3 → Filler: "Mmm, déjame pensar..."
# → Canal OUT → LLM genera → TTS → Audio
# → Latencia percibida: ~1.5s (filler oculta 2.3s reales)
```

---

## 📊 KPIs Objetivo v2.17

| Métrica | Objetivo | Componente |
|---------|----------|------------|
| **Latencia STT** | < 150ms/chunk | Vosk Streaming ✅ |
| **Latencia Router** | < 30ms | LoRA Router ✅ |
| **Latencia TRM** | < 50ms | TRM Classifier ⏳ |
| **Latencia LLM** | < 2.5s | LFM2 ⏳ |
| **Latencia percibida** | < 1.5s | Fillers ⏳ |
| **TRM Hit Rate** | > 40% | LoRA Router entrenado ⏳ |
| **RAG Coherencia** | > 90% | Capa 2 ⏳ |
| **CPU P99** | < 85% | LoRA Scheduler ⏳ |

---

## 🚀 Roadmap de Implementación

### **✅ Semana 1: Capa 1 Completa** (COMPLETADO - 30 Oct 2025)
- [x] Canal IN (vosk_streaming, lora_router, input_thread)
- [x] Canal OUT (output_thread)
- [x] Orquestador Capa 1
- [x] Test E2E Capa 1
- [x] Documentación completa

**Logros:**
- ~1,719 líneas de código productivo
- 6 componentes nuevos totalmente funcionales
- Sistema full-duplex operativo
- TRM cache implementado (respuestas < 50ms)
- Integración LFM2 completa

### **⏳ Semana 2: Capa 2 - RAG** (Estimado: 2 días)
- [ ] EmbeddingGemma wrapper
- [ ] Qdrant setup
- [ ] Conversation RAG
- [ ] Test interrupciones

### **Semana 3: Capa 3 - Fillers** (2 días)
- [ ] Generar coletillas con Piper
- [ ] Filler Manager
- [ ] Latency Detector
- [ ] Test fillers en conversación

### **Semana 4: Capa 4 - Scheduler** (2 días)
- [ ] LoRA Scheduler
- [ ] Resource Monitor
- [ ] Test priorización dinámica

### **Semana 5: Integración Final** (3 días)
- [ ] Orquestador maestro
- [ ] Test E2E completo
- [ ] Benchmarking
- [ ] Optimización final

---

## 📁 Estructura de Archivos Actual

```
core/
├── layer1_io/
│   ├── __init__.py                 ✅
│   ├── vosk_streaming.py           ✅ (nuevo, 323 líneas)
│   ├── lora_router.py              ✅ (nuevo, 384 líneas)
│   ├── input_thread.py             ✅ (nuevo, 267 líneas)
│   ├── output_thread.py            ✅ (nuevo, 386 líneas)
│   └── orchestrator.py             ✅ (nuevo, 241 líneas)
│
├── layer2_memory/
│   ├── __init__.py                 ✅
│   ├── embedding_gemma.py          ⏳
│   ├── qdrant_store.py             ⏳
│   └── conversation_rag.py         ⏳
│
├── layer3_fluidity/
│   ├── __init__.py                 ✅
│   ├── filler_manager.py           ⏳
│   ├── latency_detector.py         ⏳
│   └── generate_fillers.py         ⏳
│
├── layer4_orchestration/
│   ├── __init__.py                 ✅
│   ├── lora_scheduler.py           ⏳
│   └── resource_monitor.py         ⏳
│
└── orchestrator_fullduplex.py      ⏳ (orquestador maestro 4 capas)

assets/
└── fillers/                        ✅ (dir creado)
    ├── hmm_01.wav                  ⏳
    ├── thinking_01.wav             ⏳
    ├── interesting_01.wav          ⏳
    └── fillers.yaml                ⏳

tests/
├── test_layer1_fullduplex.py       ✅ (nuevo, 118 líneas)
└── test_fullduplex_complete.py     ⏳ (test 4 capas)

docs/
├── ARCHITECTURE_FULLDUPLEX_v2.17.md ✅
└── IMPLEMENTATION_PLAN_v2.17.md     ✅ (este archivo)

state/
└── trm_cache.json                  ✅ (generado automáticamente)
```

---

## 🔧 Dependencias Adicionales Necesarias

```bash
# Ya instaladas:
# - vosk (v0.3.45) ✅
# - transformers ✅
# - torch ✅
# - numpy ✅

# Por instalar:
pip install qdrant-client          # Capa 2: BD vectorial
pip install sentence-transformers  # EmbeddingGemma (opcional)
pip install peft                   # LoRA training (Capa 4)
pip install psutil                 # Resource Monitor (Capa 4)
pip install sounddevice            # Test streaming (opcional)
```

---

## 📝 Notas de Implementación

### Canal IN (Completado):
- ✅ VAD simple por energía (threshold=0.02)
- ✅ Vosk streaming con detección de fin por silencio (500ms)
- ✅ BERT embeddings (768-dim)
- ✅ LoRA Router con 3 clases (TRM/LLM/Traducir)
- ✅ Estadísticas en tiempo real

### Canal OUT (Pendiente):
- Coordinar con VAD de usuario (no interrumpir mientras habla)
- TTS streaming con Piper (chunks de 200ms)
- Buffer de respuesta proactivo (LLM genera ANTES de silencio)

### Capa 2 (Pendiente):
- Qdrant local (no cloud, privacidad)
- Indexar cada turno con timestamp
- Retrieval por similitud semántica (K=5)
- Recovery automático en interrupciones

### Capa 3 (Pendiente):
- Fillers con personalidad (voz de SARAi, no robóticos)
- Gap mínimo 3s entre fillers (naturalidad)
- Predicción de latencia (no esperar timeout)

### Capa 4 (Pendiente):
- LoRA entrenado con logs de conversaciones reales
- Ajuste de prioridades cada 100ms
- Evitar thrashing (cambios muy frecuentes)

---

## ✅ Próximo Paso Inmediato

**Crear `output_thread.py` y completar Capa 1**

```python
# output_thread.py debe:
1. Esperar decisiones de input_thread
2. Si TRM → Lookup en cache → TTS inmediato
3. Si LLM → LFM2 genera (streaming) → Buffer → Espera silencio → TTS
4. Si Traducir → NLLB → LLM → TTS
5. Coordinarse con Filler Manager (Capa 3)
```

---

**Estado: Capa 1 (Canal IN) ✅ Completa y testeada**
**Siguiente: Capa 1 (Canal OUT) - Estimado 2-3 horas**
