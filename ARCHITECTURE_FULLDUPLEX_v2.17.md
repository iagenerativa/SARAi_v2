# SARAi v2.17 - Arquitectura Full-Duplex Streaming

## 🎯 Objetivo

Sistema de conversación por voz **full-duplex** con:
- ✅ 2 canales independientes (entrada siempre escuchando + salida proactiva)
- ✅ Streaming real (LLM procesa MIENTRAS el usuario habla)
- ✅ Coherencia persistente (BD vectorial para contexto)
- ✅ Latencia oculta con fillers pregrabados
- ✅ TRM para respuestas cached (sin LLM)
- ✅ LoRA para priorización dinámica de recursos

---

## 🏗️ Arquitectura de 4 Capas

### **CAPA 1: I/O Full-Duplex**

#### **Canal IN (Siempre Escuchando)**
```python
# Thread 1: Audio Capture (continuo)
Micrófono (16kHz) 
    → Sherpa-ONNX VAD (cada 30ms)
    → Vosk STT streaming (cada 100ms)
    → Text Buffer (acumula hasta silencio o 5s)
    → Queue: input_text_queue

# Thread 2: Intent Classification (continuo)
input_text_queue
    → BERT-es embeddings (300M)
    → LoRA Router (7M params) → Decisión:
        ├─ [TRM] → TRM-Classifier → Respuesta cached (< 50ms)
        ├─ [LLM] → LFM2-1.2B → Generación completa
        └─ [Traducir] → NLLB-200 → Traducción → LLM
    → Queue: llm_input_queue

# Thread 3: LLM Processing (proactivo)
llm_input_queue
    → LFM2 genera tokens (streaming)
    → COMIENZA SIN ESPERAR FIN DE INPUT
    → Queue: llm_output_queue (tokens parciales)
```

**KPI Canal IN**:
- VAD latency: < 30ms
- STT latency: < 150ms (cada chunk 100ms)
- Router latency: < 20ms
- TRM hit rate: > 40% (evita LLM)

#### **Canal OUT (Preparación Proactiva)**
```python
# Thread 4: Response Buffer (espera silencio)
llm_output_queue (tokens streaming)
    → Acumula respuesta parcial
    → Monitorea VAD del usuario
    → Si VAD detecta silencio > 300ms → Queue: tts_queue

# Thread 5: Filler Manager (latency hiding)
Latency Detector:
    IF (tiempo desde última palabra usuario) > 800ms
        AND respuesta LLM no lista
    THEN:
        → Reproduce filler pregrabado ("Mmm...", "Déjame pensar...")
        → assets/fillers/*.wav (22kHz, pre-generadas)

# Thread 6: TTS Streaming
tts_queue
    → Piper TTS (streaming mode)
    → Audio chunks (cada 200ms)
    → Playback inmediato (no espera síntesis completa)
```

**KPI Canal OUT**:
- Filler trigger: latencia > 800ms
- TTS chunk latency: < 200ms
- Audio playback delay: < 50ms

---

### **CAPA 2: Coherencia Persistente (RAG Conversacional)**

```python
# Vector Store: Qdrant local (o ChromaDB existente)
class ConversationRAG:
    def __init__(self):
        self.embedding_model = EmbeddingGemma2B()  # 2B params
        self.vector_store = Qdrant(path="state/qdrant_db")
        self.turn_history = []  # Buffer en RAM (últimos 10 turnos)
    
    def index_turn(self, user_text: str, assistant_text: str):
        """Indexa cada turno conversacional"""
        turn = {
            "user": user_text,
            "assistant": assistant_text,
            "timestamp": time.time(),
            "embedding": self.embedding_model.encode(user_text)
        }
        self.vector_store.add(turn)
        self.turn_history.append(turn)
    
    def retrieve_context(self, current_query: str, k=5):
        """Recupera contexto relevante si hay interrupción"""
        query_emb = self.embedding_model.encode(current_query)
        similar_turns = self.vector_store.search(query_emb, k=k)
        return similar_turns
    
    def handle_interruption(self, partial_input: str):
        """Si usuario interrumpe, recupera contexto de BD"""
        context = self.retrieve_context(partial_input, k=3)
        # Reconstruye estado conversacional
        return self._build_context_prompt(context)
```

**Flujo de Interrupción**:
```
Usuario habla: "Oye, sobre lo que dijiste antes de..."
    ↓
Sistema interrumpido (respuesta a medias)
    ↓
Capa 2 RAG:
    → Embedding del fragmento
    → Búsqueda en Qdrant
    → Encuentra: turno #3 (hace 2 min): "...modelo de IA..."
    → Reconstruye contexto
    ↓
LLM recibe: "Contexto recuperado: [turno #3]..."
```

**KPI Capa 2**:
- Indexación latency: < 100ms (async)
- Retrieval latency: < 200ms
- Context hits: > 80% en interrupciones

---

### **CAPA 3: Gestión de Fluidez (Filler System)**

```python
# assets/fillers/fillers.yaml
fillers:
  - id: hmm_01
    file: hmm_01.wav
    duration_ms: 400
    usage: "Inicio de procesamiento largo"
    
  - id: thinking_01
    file: thinking_01.wav
    text: "Déjame pensar..."
    duration_ms: 800
    usage: "Consulta compleja detectada"
    
  - id: interesting_01
    file: interesting_01.wav
    text: "Interesante..."
    duration_ms: 600
    usage: "Usuario dice algo inesperado"

class FillerManager:
    def __init__(self):
        self.fillers = self._load_fillers()
        self.last_filler_time = 0
        self.min_filler_gap = 3.0  # No más de 1 filler cada 3s
    
    def should_trigger_filler(self, latency_ms: float, query_complexity: str):
        """Decide si reproducir filler"""
        if time.time() - self.last_filler_time < self.min_filler_gap:
            return None  # Evita spam de fillers
        
        if latency_ms > 1500 and query_complexity == "high":
            return self.fillers["thinking_01"]
        elif latency_ms > 800:
            return self.fillers["hmm_01"]
        return None
    
    def play_filler(self, filler: dict):
        """Reproduce filler mientras LLM trabaja"""
        subprocess.run(["aplay", "-q", f"assets/fillers/{filler['file']}"])
        self.last_filler_time = time.time()
```

**Estrategia de Fillers**:
- **Latencia 800-1200ms**: "Mmm..." (corto)
- **Latencia 1200-2000ms**: "Déjame pensar..." (mediano)
- **Latencia > 2000ms**: "Interesante, necesito considerar..." (largo)
- **Gap mínimo**: 3 segundos entre fillers (evita artificio)

---

### **CAPA 4: Orquestación Dinámica (LoRA Scheduler)**

```python
class LoRAScheduler:
    """
    LoRA Adapter que ajusta prioridades de recursos en tiempo real
    Entrenado sobre LFM2-1.2B para optimización dinámica
    """
    
    def __init__(self):
        self.lora_adapter = self._load_lora()  # 7M params adicionales
        self.resource_monitor = ResourceMonitor()
    
    def adjust_priorities(self, context: dict):
        """
        Context: {
            "input_queue_size": 3,
            "llm_queue_size": 1,
            "cpu_usage": 0.85,
            "ram_available_gb": 4.2,
            "active_threads": 6
        }
        """
        # LoRA predice ajustes óptimos
        adjustments = self.lora_adapter.predict(context)
        
        # Aplicar ajustes dinámicamente
        if adjustments["boost_stt"]:
            # STT tiene prioridad (usuario hablando)
            self._set_thread_priority("vosk_stt", priority="high")
            self._set_llm_n_threads(2)  # Reduce LLM temporalmente
        
        elif adjustments["boost_llm"]:
            # LLM tiene prioridad (generando respuesta)
            self._set_thread_priority("lfm2", priority="high")
            self._set_llm_n_threads(4)  # Maximiza LLM
        
        elif adjustments["boost_tts"]:
            # TTS tiene prioridad (respuesta lista)
            self._set_thread_priority("piper_tts", priority="high")
    
    def detect_bottleneck(self):
        """Detecta cuello de botella en tiempo real"""
        if self.resource_monitor.cpu_usage > 0.9:
            return "cpu"
        elif self.resource_monitor.input_queue_size > 5:
            return "input_processing"
        elif self.resource_monitor.llm_latency > 3000:
            return "llm"
        return None
```

**Políticas de Priorización**:

| Situación | Acción LoRA |
|-----------|-------------|
| Usuario hablando | STT priority=high, LLM n_threads=2 |
| Usuario callado + LLM procesando | LLM priority=high, n_threads=4 |
| Respuesta lista | TTS priority=high, LLM pause |
| CPU > 90% | Reduce n_threads global, activa TRM cache |
| Input queue > 5 | Boost STT, reduce TTS chunking |

---

## 🔄 Flujo Completo de Conversación

### **Escenario 1: Conversación Fluida (Sin Latencia)**
```
T=0s:  Usuario: "Hola, ¿cómo estás?"
T=0.03s: VAD detecta habla
T=0.15s: Vosk STT: "Hola, ¿cómo"
T=0.25s: Vosk STT: "Hola, ¿cómo estás?"
T=0.28s: BERT embeddings
T=0.30s: LoRA Router → [TRM] (respuesta común)
T=0.35s: TRM hit: "¡Hola! Muy bien, gracias por preguntar."
T=0.40s: VAD detecta silencio usuario
T=0.45s: Piper TTS inicia (streaming)
T=0.65s: Audio "¡Hola!" reproducido
T=1.5s:  Audio completo reproducido

Total E2E: 1.5s (sin LLM, gracias a TRM)
```

### **Escenario 2: Consulta Compleja (Con Fillers)**
```
T=0s:    Usuario: "Explícame la teoría de la relatividad"
T=0.03s: VAD detecta habla
T=0.4s:  Vosk STT: "Explícame la teoría de la relatividad"
T=0.43s: BERT embeddings
T=0.45s: LoRA Router → [LLM] (consulta compleja)
T=0.50s: LFM2 comienza a procesar (NO espera fin usuario)
T=1.2s:  VAD detecta silencio usuario
T=1.3s:  ⚠️ Latencia detector: LLM aún procesando (solo 800ms transcurridos)
T=1.5s:  🎵 Filler: "Mmm, déjame pensar..." (reproduce mientras LLM trabaja)
T=2.8s:  LLM genera primeros tokens: "La teoría de la relatividad..."
T=3.0s:  Piper TTS streaming inicia
T=3.2s:  Audio "La teoría..." reproducido
T=7.5s:  Respuesta completa reproducida

Usuario percibe: 1.5s de "pensamiento natural" (filler) + respuesta fluida
Latencia real LLM: 2.3s (oculta por filler)
```

### **Escenario 3: Interrupción del Usuario**
```
T=0s:    Usuario: "Cuéntame sobre IA"
T=0.4s:  LLM generando: "La inteligencia artificial es..."
T=1.2s:  Usuario interrumpe: "Espera, sobre lo que dijiste antes..."
T=1.25s: ⚠️ Sistema detecta interrupción
T=1.27s: Capa 2 RAG activa:
         → Embedding: "sobre lo que dijiste antes"
         → Qdrant search: encuentra turno #5 (hace 30s)
         → Contexto recuperado: "...el aprendizaje profundo..."
T=1.5s:  LLM recibe: "Contexto: turno #5 [aprendizaje profundo]. Nueva consulta: sobre lo que dijiste antes..."
T=2.8s:  Respuesta coherente: "Claro, sobre el aprendizaje profundo que mencioné..."

Sistema mantiene coherencia a pesar de interrupción
```

---

## 📊 KPIs Objetivo v2.17

| Métrica | Objetivo | Cómo se Logra |
|---------|----------|---------------|
| **Latencia Percibida** | < 1.5s | Fillers + Streaming TTS |
| **Latencia Real E2E** | < 3.5s | Capa 1 optimizada |
| **TRM Hit Rate** | > 40% | Router bien entrenado |
| **Filler Naturalidad** | > 4.5/5 | Coletillas pregrabadas profesionales |
| **Coherencia Multi-turno** | > 90% | RAG con EmbeddingGemma |
| **CPU Usage P99** | < 85% | LoRA Scheduler dinámico |
| **Interrupciones Manejadas** | 100% | Capa 2 recovery |

---

## 🛠️ Dependencias Nuevas

```bash
# Sherpa-ONNX (VAD + TTS streaming)
pip install sherpa-onnx

# EmbeddingGemma 2B
pip install sentence-transformers

# Qdrant (BD vectorial)
pip install qdrant-client

# BERT-es
pip install transformers

# LoRA training
pip install peft
```

---

## 📁 Archivos Clave a Crear

```
core/layer1_io/
├── sherpa_vad.py          # VAD streaming (30ms chunks)
├── vosk_streaming.py      # STT streaming (100ms chunks)
├── bert_embedder.py       # BERT-es embeddings
├── lora_router.py         # Router con LoRA (TRM/LLM/Traducir)
├── input_thread.py        # Orquestador Canal IN
└── output_thread.py       # Orquestador Canal OUT

core/layer2_memory/
├── embedding_gemma.py     # EmbeddingGemma 2B wrapper
├── qdrant_store.py        # Qdrant client abstraction
└── conversation_rag.py    # RAG conversacional

core/layer3_fluidity/
├── filler_manager.py      # Gestión de coletillas
├── latency_detector.py    # Detecta delays > 800ms
└── generate_fillers.py    # Script para grabar fillers con Piper

core/layer4_orchestration/
├── lora_scheduler.py      # LoRA resource scheduler
└── resource_monitor.py    # CPU/RAM/Queue monitoring

core/orchestrator.py       # Orquestador maestro (4 capas)

tests/
└── test_fullduplex.py     # Test integración completa
```

---

## 🚀 Próximos Pasos

1. ✅ Crear estructura de directorios
2. ⏳ Implementar Capa 1 (I/O Full-Duplex)
3. ⏳ Implementar Capa 2 (RAG Conversacional)
4. ⏳ Implementar Capa 3 (Filler System)
5. ⏳ Implementar Capa 4 (LoRA Scheduler)
6. ⏳ Integración y testing E2E
7. ⏳ Benchmarking y optimización final

---

**Filosofía v2.17**: 
_"La conversación natural no espera. El sistema anticipa, oculta latencias y mantiene coherencia incluso en el caos de una interrupción."_
