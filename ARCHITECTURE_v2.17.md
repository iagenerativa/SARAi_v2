# 🏗️ SARAi v2.17 - Arquitectura Profesional de 4 Capas

**Fecha**: 30 de octubre de 2025  
**Versión**: 2.17.0  
**Estado**: En Implementación

---

## 📋 Visión General

Sistema de voz conversacional profesional con gestión de memoria, fluidez natural y orquestación dinámica de recursos.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA 1: I/O Asíncrono                        │
│  Hilo IN (Audio → Texto) | Hilo OUT (Texto → Audio)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓↑
┌─────────────────────────────────────────────────────────────────┐
│              CAPA 2: Memoria Conversacional (RAG)               │
│  EmbeddingGemma 2B + Qdrant → Contexto multi-turno             │
└─────────────────────────────────────────────────────────────────┘
                              ↓↑
┌─────────────────────────────────────────────────────────────────┐
│           CAPA 3: Gestión de Fluidez Natural                    │
│  Sherpa Streaming + Fillers → Engagement durante latencias     │
└─────────────────────────────────────────────────────────────────┘
                              ↓↑
┌─────────────────────────────────────────────────────────────────┐
│        CAPA 4: Orquestación Dinámica (LoRA Scheduler)          │
│  LoRA Adapter → Optimización de recursos en tiempo real        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 CAPA 1: I/O Asíncrono

### **Propósito**
Procesamiento paralelo de entrada (audio → texto) y salida (texto → audio) con colas thread-safe.

### **Componentes**

#### **Hilo IN (Input Thread)**
```
Audio Raw → Sherpa VAD → Vosk STT → BERT-es → DistilBART-es → LoRA Router
                                                                      ↓
                                                    [Traducir] → NLLB-200
                                                    [Normal] → Proyector
                                                    [TRM] → Cache rápida
                                                    [LLM] → LFM2-1.2B
```

**Componentes clave**:

| Componente | Función | Latencia | Tamaño |
|------------|---------|----------|--------|
| **Sherpa-ONNX VAD** | Detecta segmentos de voz | <50ms | ~10MB |
| **Vosk STT** | Transcripción español | 100-200ms | 38MB |
| **BERT-es** | Embeddings semánticos | 50ms | 110MB |
| **DistilBART-es** | Comprensión contextual | 80ms | 290MB |
| **LoRA Router** | Clasifica tipo de query | 10ms | 5MB |
| **NLLB-200** | Traducción multiidioma | 200ms | 600MB |

#### **Hilo OUT (Output Thread)**
```
Cola de Respuestas → Talker (?) → Kitter TTS (Piper) → Audio Out
```

**Componentes clave**:

| Componente | Función | Latencia | Tamaño |
|------------|---------|----------|--------|
| **Talker** | Generador de tokens audio | TBD | TBD |
| **Kitter TTS** | Síntesis de voz (Piper) | 395ms | 60MB |

### **Colas Thread-Safe**

```python
from queue import Queue

audio_raw_queue = Queue(maxsize=10)       # Audio bruto del micrófono
audio_chunks_queue = Queue(maxsize=50)    # Chunks tras VAD
text_queue = Queue(maxsize=20)            # Texto transcrito
routing_decision_queue = Queue(maxsize=20)  # Decisión del LoRA Router
response_queue = Queue(maxsize=10)        # Respuestas del LLM
audio_output_queue = Queue(maxsize=10)    # Audio final para reproducir
```

### **LoRA Router - Clases de Salida**

```python
class RouterDecision:
    TRANSLATE = "translate"  # Texto no español → NLLB → Español
    NORMAL = "normal"        # Español estándar → Proyector directo
    TRM = "trm"              # Respuesta común → Cache TRM
    LLM = "llm"              # Razonamiento profundo → LFM2-1.2B
```

**Criterios de decisión**:
- **TRANSLATE**: Idioma detectado != español (usando langdetect)
- **TRM**: Query en cache de respuestas frecuentes (similitud >0.95)
- **LLM**: Query compleja, requiere razonamiento
- **NORMAL**: Por defecto

---

## 🧠 CAPA 2: Memoria Conversacional (RAG)

### **Propósito**
Mantener contexto multi-turno y coherencia conversacional mediante recuperación de turnos relevantes.

### **Componentes**

| Componente | Función | Tamaño |
|------------|---------|--------|
| **EmbeddingGemma 2B** | Embeddings de alta calidad | 2GB |
| **Qdrant** (o ChromaDB) | Base de datos vectorial | Variable |

### **Flujo de Datos**

```python
# Indexación (cada turno)
user_input = "¿Cómo preparo café?"
embedding = embedding_gemma.encode(user_input)  # (2048,)
qdrant.upsert(
    collection="conversation_history",
    points=[{
        "id": turn_id,
        "vector": embedding.tolist(),
        "payload": {
            "text": user_input,
            "role": "user",
            "timestamp": datetime.now().isoformat()
        }
    }]
)

# Retrieval (antes de generar respuesta)
query_embedding = embedding_gemma.encode(current_query)
relevant_turns = qdrant.search(
    collection="conversation_history",
    query_vector=query_embedding,
    limit=5  # Top-5 turnos relevantes
)

# Contexto extendido para LLM
context = "\n".join([turn.payload["text"] for turn in relevant_turns])
prompt = f"{context}\n\nUsuario: {current_query}\nAsistente:"
```

### **Gestión de Memoria**

```python
# Limpieza automática (mantener últimas 100 conversaciones)
MAX_CONVERSATIONS = 100
MAX_TURNS_PER_CONVERSATION = 20

# Política de eviction: FIFO (First In, First Out)
if total_turns > MAX_CONVERSATIONS * MAX_TURNS_PER_CONVERSATION:
    oldest_conversation_id = get_oldest_conversation()
    qdrant.delete(filter={"conversation_id": oldest_conversation_id})
```

---

## 🎙️ CAPA 3: Gestión de Fluidez Natural

### **Propósito**
Mantener engagement del usuario durante latencias de procesamiento mediante fillers (coletillas) y streaming TTS.

### **Componentes**

| Componente | Función |
|------------|---------|
| **Latency Detector** | Monitorea tiempo de respuesta |
| **Filler Manager** | Reproduce coletillas pregrabadas |
| **Sherpa-ONNX Streaming** | TTS streaming (low latency) |

### **Fillers Pregrabados**

```yaml
# assets/fillers/fillers.yaml
fillers:
  - id: hmm_01
    file: hmm_01.wav
    text: "Mmm..."
    duration_ms: 800
    use_case: "Inicio de pensamiento"
  
  - id: thinking_01
    file: thinking_01.wav
    text: "Déjame pensar..."
    duration_ms: 1200
    use_case: "Query compleja detectada"
  
  - id: interesting_01
    file: interesting_01.wav
    text: "Interesante..."
    duration_ms: 1000
    use_case: "Reconocimiento de input complejo"
  
  - id: wait_moment_01
    file: wait_moment_01.wav
    text: "Un momento..."
    duration_ms: 900
    use_case: "Búsqueda en memoria"
```

### **Política de Activación**

```python
class LatencyDetector:
    THRESHOLDS = {
        "immediate": 200,      # <200ms: Sin filler
        "acceptable": 800,     # 200-800ms: Filler corto ("Mmm")
        "noticeable": 2000,    # 800-2000ms: Filler medio ("Déjame pensar")
        "critical": 5000       # >2000ms: Filler + streaming partial
    }
    
    def select_filler(self, estimated_latency_ms: int, query_type: str):
        if estimated_latency_ms < self.THRESHOLDS["acceptable"]:
            return None  # Sin filler
        
        elif estimated_latency_ms < self.THRESHOLDS["noticeable"]:
            return "hmm_01"
        
        elif query_type == "complex":
            return "thinking_01"
        
        else:
            return "wait_moment_01"
```

### **Sherpa-ONNX Streaming TTS**

**Ventaja sobre Piper**: Genera audio chunk por chunk (latencia perceived <100ms)

```python
# Streaming TTS (chunked output)
for text_chunk in response_stream:
    audio_chunk = sherpa_tts.synthesize_chunk(text_chunk)
    audio_output_queue.put(audio_chunk)  # Reproducción inmediata
```

---

## ⚙️ CAPA 4: Orquestación Dinámica (LoRA Scheduler)

### **Propósito**
Optimizar uso de CPU/RAM en tiempo real ajustando recursos según tipo de query.

### **Componentes**

| Componente | Función |
|------------|---------|
| **LoRA Adapter** | Ajuste fino de LFM2 para routing |
| **Resource Monitor** | Monitoreo de CPU/RAM |
| **Priority Queue** | Priorización de queries |

### **LoRA Adapter sobre LFM2**

```python
# LoRA config
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # Rank bajo para eficiencia
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Solo attention
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Aplicar a LFM2
lfm2_lora = get_peft_model(lfm2_base, lora_config)
# Tamaño: +5MB sobre LFM2 (vs 1.2GB base)
```

**Entrenamiento del LoRA**:
- Dataset: Logs históricos de `routing_decision_queue`
- Objetivo: Predecir clase óptima (TRM/LLM/TRANSLATE)
- Métricas: Accuracy routing + Latencia reducida

### **Dynamic Resource Allocation**

```python
class ResourceScheduler:
    def adjust_threads(self, query_type: str, cpu_usage: float):
        """
        Ajusta n_threads de LFM2 según carga
        """
        if query_type == "TRM":
            return 2  # TRM es lightweight
        
        elif cpu_usage > 80:
            return 4  # CPU saturada → reducir threads
        
        elif query_type == "LLM":
            return 8  # Query compleja → max threads
        
        else:
            return 6  # Balance normal
    
    def adjust_batch_size(self, queue_length: int):
        """
        Batching dinámico si hay múltiples queries
        """
        if queue_length >= 3:
            return 4  # Procesar 4 queries en batch
        else:
            return 1  # Sin batching
```

### **Priority Queue**

```python
import heapq

class PriorityQueue:
    priorities = {
        "TRM": 1,       # Máxima prioridad (cache rápida)
        "NORMAL": 2,
        "LLM": 3,
        "TRANSLATE": 4  # Menor prioridad (más lento)
    }
    
    def __init__(self):
        self.heap = []
    
    def put(self, query_type: str, data: dict):
        priority = self.priorities.get(query_type, 5)
        heapq.heappush(self.heap, (priority, data))
    
    def get(self):
        return heapq.heappop(self.heap)[1]
```

---

## 📊 Métricas de Rendimiento Objetivo

| Métrica | Objetivo v2.17 | Actual v2.16 |
|---------|----------------|--------------|
| **Latencia P50 (TRM)** | ≤500ms | N/A |
| **Latencia P50 (LLM)** | ≤2s | 7.2s |
| **Latencia P50 (Traducir)** | ≤3s | N/A |
| **RAM P99** | ≤14GB | 10.8GB |
| **CPU Utilization** | 70-85% | 60% |
| **Cache Hit Rate (TRM)** | ≥40% | N/A |
| **Filler Activation** | 30-50% | 0% |
| **Context Recall (RAG)** | ≥0.85 | N/A |

---

## 🚀 Roadmap de Implementación

### **Fase 1: Capa 1 - I/O Asíncrono** (ACTUAL)
- [x] Sherpa-ONNX VAD
- [x] Vosk STT wrapper
- [x] BERT-es embedder
- [ ] DistilBART-es processor
- [ ] LoRA Router (crear y entrenar)
- [ ] Kitter TTS wrapper
- [ ] Input/Output threads

### **Fase 2: Capa 2 - Memoria RAG**
- [ ] EmbeddingGemma 2B wrapper
- [ ] Qdrant setup local
- [ ] Indexación de turnos
- [ ] Retrieval K=5

### **Fase 3: Capa 3 - Fluidez Natural**
- [ ] Grabar fillers en español
- [ ] Sherpa-ONNX streaming TTS
- [ ] Latency detector
- [ ] Filler injection automática

### **Fase 4: Capa 4 - LoRA Scheduler**
- [ ] LoRA adapter sobre LFM2
- [ ] Resource monitor
- [ ] Priority queue
- [ ] Dynamic adjustment

### **Fase 5: Integración Final**
- [ ] Orquestador maestro
- [ ] Test E2E
- [ ] Benchmarking
- [ ] Documentación

---

## 🔧 Configuración del Sistema

```yaml
# config/sarai_v2.17.yaml
system:
  version: "2.17.0"
  max_ram_gb: 14
  max_cpu_percent: 85

layer1_io:
  vad:
    provider: "sherpa"  # "sherpa" | "silero"
    window_ms: 30
    min_speech_ms: 250
    min_silence_ms: 500
  
  stt:
    provider: "vosk"
    model: "vosk-model-small-es-0.42"
    sample_rate: 16000
  
  router:
    model: "lora_router_v1"
    threshold_trm: 0.95
    threshold_translate: 0.7

layer2_memory:
  embedding_model: "google/embedding-gemma-2b"
  vector_db: "qdrant"  # "qdrant" | "chromadb"
  collection: "conversation_history"
  top_k: 5
  max_conversations: 100

layer3_fluidity:
  filler_threshold_ms: 800
  streaming_tts: true
  tts_provider: "sherpa"  # "sherpa" | "piper"

layer4_orchestration:
  lora_enabled: true
  dynamic_threads: true
  priority_queue: true
  resource_monitor_interval_ms: 100
```

---

**Autor**: SARAi Development Team  
**Licencia**: Ver LICENSE_GUIDE.md
