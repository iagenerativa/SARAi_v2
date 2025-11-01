# 🏗️ Arquitectura de 3 Layers - SARAi v2.13

## 📊 Visión General

SARAi utiliza una arquitectura de **3 capas** para procesar información de forma modular:

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT (Audio/Texto)                  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
              ┌──────────────────────┐
              │   LAYER 1: I/O       │
              │   (Input/Output)     │
              │                      │
              │ • Captura audio      │
              │ • STT (Vosk)         │
              │ • Emotion detection  │
              │ • VAD (Sherpa)       │
              │ • TTS (MeloTTS)      │
              └──────────┬───────────┘
                         ↓
              ┌──────────────────────┐
              │   LAYER 2: Memory    │
              │   (Contexto/Tono)    │
              │                      │
              │ • Tone Memory Buffer │
              │ • Persistencia JSONL │
              │ • Historial emocional│
              └──────────┬───────────┘
                         ↓
              ┌──────────────────────┐
              │   LAYER 3: Fluidity  │
              │   (Transiciones)     │
              │                      │
              │ • Tone Bridge        │
              │ • Style inference    │
              │ • Filler hints       │
              │ • Smooth transitions │
              └──────────┬───────────┘
                         ↓
              ┌──────────────────────┐
              │    GRAPH (Core)      │
              │  TRM → MCP → LLM     │
              └──────────────────────┘
```

---

## 🎯 Layer 1: I/O (Input/Output)

### Propósito
**Captura y entrega de datos** (audio, texto, imagen).

### Componentes

#### 1. `true_fullduplex.py` - Multiprocessing I/O ⚡
**Función**: Sistema full-duplex real con 3 procesos independientes

**Características**:
- ✅ Sin GIL compartido (paralelismo REAL)
- ✅ Latencia interrupción <10ms
- ✅ Audio duplex nativo (PortAudio)

**Procesos**:
```python
Process 1: Input  → STT (Vosk) + Emotion Detection
Process 2: Output → LLM (SOLAR/LFM2) + TTS (MeloTTS)
Process 3: Main   → Orchestration + User I/O
```

#### 2. `audio_emotion_lite.py` - Detección Emocional 🎭
**Función**: Análisis ligero de tono/emoción desde audio

**Features extraídas**:
- Pitch (media, std, jitter)
- Energía (RMS)
- MFCC (13 coeficientes)
- Spectral features (centroid, rolloff, flux)
- Formantes (F1, F2)

**Modelo**: RandomForest (CPU-friendly)

**Emociones detectadas**:
- neutral
- happy
- sad
- angry
- fearful

**API**:
```python
from core.layer1_io.audio_emotion_lite import detect_emotion

# Desde archivo de audio
emotion_scores = detect_emotion("audio.wav")
# → {"neutral": 0.1, "happy": 0.7, "sad": 0.1, "angry": 0.05, "fearful": 0.05}

# Desde array numpy
emotion_scores = detect_emotion(audio_array, sr=16000)
```

**Output**:
```python
{
    "label": "happy",           # Emoción dominante
    "scores": {...},            # Scores por emoción
    "valence": 0.8,            # Positivo/negativo
    "arousal": 0.6,            # Energía/activación
    "confidence": 0.7          # Confianza del modelo
}
```

#### 3. `vosk_streaming.py` - STT Streaming 🎤
**Función**: Speech-to-Text en tiempo real con Vosk

**Características**:
- Modelo: vosk-model-small-es-0.42 (91 MB)
- Latency: ~50ms
- CPU-only, sin GPU requerida

#### 4. `sherpa_vad.py` - Voice Activity Detection 🔊
**Función**: Detecta cuando el usuario está hablando

**Uso**: Evita procesar silencio innecesario

#### 5. `lora_router.py` - Routing Dinámico de Modelos 🔀
**Función**: Enruta a diferentes checkpoints LoRA según contexto

**Ejemplo**:
```python
# Ruta a checkpoint optimizado para código
lora_checkpoint = lora_router.route("Escribe código Python")
# → "state/lora_programming.pt"
```

---

## 🧠 Layer 2: Memory (Contexto/Tono)

### Propósito
**Persistencia de contexto y tono emocional** para mantener coherencia en conversaciones largas.

### Componentes

#### 1. `tone_memory.py` - Memoria de Tono 📝
**Función**: Buffer persistente de eventos de tono

**Características**:
- Persistencia JSONL (no se pierde al reiniciar)
- Buffer en memoria (deque, max 256 entries)
- Thread-safe (con locks)

**API**:
```python
from core.layer2_memory.tone_memory import ToneMemoryBuffer

tone_memory = ToneMemoryBuffer()

# Agregar entrada
tone_memory.append({
    "label": "happy",
    "valence": 0.8,
    "arousal": 0.6,
    "timestamp": time.time()
})

# Obtener entradas recientes
recent_tones = tone_memory.recent(limit=10)
# → [{"label": "happy", "valence": 0.8, ...}, ...]

# Iterar sobre historial
for entry in tone_memory.iter_recent(limit=20):
    print(entry["label"], entry["valence"])
```

**Persistencia**:
```bash
# Archivo: state/layer2_tone_memory.jsonl
{"label": "happy", "valence": 0.8, "arousal": 0.6, "timestamp": 1730000000.0}
{"label": "neutral", "valence": 0.5, "arousal": 0.5, "timestamp": 1730000010.0}
{"label": "sad", "valence": 0.2, "arousal": 0.4, "timestamp": 1730000020.0}
```

**Uso en MCP**:
```python
# Ajustar α/β según historial de tono
historical_tone = tone_memory.recent(limit=5)
avg_valence = sum(t["valence"] for t in historical_tone) / len(historical_tone)

if avg_valence < 0.3:  # Usuario frustrado
    beta = min(beta + 0.15, 1.0)  # Aumentar empatía
    alpha = 1.0 - beta
```

---

## 🌊 Layer 3: Fluidity (Transiciones)

### Propósito
**Transiciones suaves de tono** y coordinación entre layers.

### Componentes

#### 1. `tone_bridge.py` - Puente de Tono 🌉
**Función**: Mapea valence/arousal a estilos de respuesta

**Smoothing**: Exponential moving average (α=0.25)

**API**:
```python
from core.layer3_fluidity.tone_bridge import ToneStyleBridge, ToneProfile

bridge = ToneStyleBridge(smoothing=0.25)

# Actualizar con nuevo tono
profile = bridge.update(
    label="happy",
    valence=0.8,
    arousal=0.6
)

print(profile.style)        # → "energetic_positive"
print(profile.filler_hint)  # → "match_energy_positive"
```

**Estilos inferidos**:

| Valence | Arousal | Style | Filler Hint |
|---------|---------|-------|-------------|
| ≥0.65 | ≥0.6 | energetic_positive | match_energy_positive |
| ≥0.65 | <0.6 | warm_positive | calm_positive_fillers |
| ≤0.35 | ≥0.6 | urgent_support | short_assurance_fillers |
| ≤0.35 | <0.6 | soft_support | soothing_fillers |
| mid | ≥0.7 | focused_alert | steadying_fillers |
| mid | ≤0.3 | low_energy | gentle_engagement |
| mid | mid | neutral_support | neutral_fillers |

**Uso en modulación**:
```python
# Obtener estilo actual
profile = bridge.snapshot()

if profile.style == "urgent_support":
    # Usuario estresado → respuesta breve y tranquilizadora
    prompt_style = "brief and reassuring"
elif profile.style == "energetic_positive":
    # Usuario energético → match energía
    prompt_style = "enthusiastic and upbeat"
else:
    # Neutral
    prompt_style = "balanced and clear"
```

#### 2. `sherpa_coordinator.py` - Coordinador VAD 🎛️
**Función**: Coordina Voice Activity Detection entre layers

**Uso**: Sincroniza detección de voz con procesamiento de audio

---

## 🔗 Integración con Graph

### Flujo Completo

```python
# 1. Layer1: Captura audio + detección emocional
audio_bytes = capture_audio()
emotion = detect_emotion(audio_bytes)

# 2. Layer2: Guardar en memoria de tono
tone_memory.append({
    "label": emotion["label"],
    "valence": emotion["valence"],
    "arousal": emotion["arousal"]
})

# 3. Layer3: Inferir estilo
profile = tone_bridge.update(
    label=emotion["label"],
    valence=emotion["valence"],
    arousal=emotion["arousal"]
)

# 4. Graph: Ajustar MCP según tono
mcp_weights = mcp.compute_weights(scores, context, tone_profile=profile)

# 5. Graph: Generar respuesta con estilo apropiado
response = llm.generate(query, style=profile.style)

# 6. Layer1: TTS con tono apropiado
audio_output = tts.synthesize(response, emotion=profile.last_label)
```

### Modificaciones en `core/graph.py`

#### Paso 1: Detectar emoción (Layer1)
```python
from core.layer1_io.audio_emotion_lite import detect_emotion

def classify_intent_node(state: State):
    # Si input es audio, detectar emoción
    if state.get("input_type") == "audio" and state.get("audio_path"):
        emotion = detect_emotion(state["audio_path"])
        state["emotion"] = emotion
    
    # Clasificación TRM como antes
    scores = trm_router.invoke(state["input"])
    return {"hard": scores["hard"], "soft": scores["soft"]}
```

#### Paso 2: Guardar en memoria (Layer2)
```python
from core.layer2_memory.tone_memory import ToneMemoryBuffer

tone_memory = ToneMemoryBuffer()

def log_emotion_node(state: State):
    if state.get("emotion"):
        tone_memory.append({
            "label": state["emotion"]["label"],
            "valence": state["emotion"]["valence"],
            "arousal": state["emotion"]["arousal"]
        })
    return state
```

#### Paso 3: Ajustar MCP (Layer2 → MCP)
```python
def compute_weights_node(state: State):
    # Obtener tono histórico
    recent_tones = tone_memory.recent(limit=5)
    
    # Calcular α/β con ajuste de tono
    alpha, beta = mcp.compute_weights(
        scores={"hard": state["hard"], "soft": state["soft"]},
        context=state["input"],
        tone_history=recent_tones
    )
    
    return {"alpha": alpha, "beta": beta}
```

#### Paso 4: Aplicar estilo (Layer3)
```python
from core.layer3_fluidity.tone_bridge import ToneStyleBridge

tone_bridge = ToneStyleBridge()

def modulate_soft_node(state: State):
    # Actualizar bridge con emoción actual
    if state.get("emotion"):
        profile = tone_bridge.update(
            label=state["emotion"]["label"],
            valence=state["emotion"]["valence"],
            arousal=state["emotion"]["arousal"]
        )
        state["tone_style"] = profile.style
        state["filler_hint"] = profile.filler_hint
    
    # Modulación con estilo apropiado
    style_prompt = get_style_prompt(state.get("tone_style", "neutral_support"))
    
    prompt = f"""Reformula con tono {style_prompt}.

Respuesta técnica:
{state['hard_response']}

Usuario ({state.get('emotion', {}).get('label', 'neutral')}):
{state['input']}

Reformula manteniendo datos técnicos."""
    
    lfm2 = model_pool.get("tiny")
    state["response"] = lfm2.generate(prompt)
    
    return state
```

---

## 📊 Estado Compartido en Graph

### Campos añadidos al State

```python
class State(TypedDict):
    # ... campos existentes ...
    
    # Layer1
    input_type: str              # "text" | "audio" | "image"
    audio_path: Optional[str]    # Path al audio si aplica
    emotion: Optional[Dict]      # Resultado de detect_emotion()
    
    # Layer3
    tone_style: Optional[str]    # "energetic_positive", etc.
    filler_hint: Optional[str]   # "match_energy_positive", etc.
```

---

## 🧪 Testing de Layers

### Tests Layer1
```python
def test_emotion_detection():
    """Verifica detección de emoción"""
    emotion = detect_emotion("test_audio_happy.wav")
    assert emotion["label"] in ["neutral", "happy", "sad", "angry", "fearful"]
    assert 0.0 <= emotion["valence"] <= 1.0
    assert 0.0 <= emotion["arousal"] <= 1.0
```

### Tests Layer2
```python
def test_tone_memory_persistence():
    """Verifica persistencia de memoria de tono"""
    tone_memory = ToneMemoryBuffer()
    tone_memory.append({"label": "happy", "valence": 0.8})
    
    # Reiniciar buffer (simula restart)
    tone_memory2 = ToneMemoryBuffer()
    recent = tone_memory2.recent(limit=1)
    
    assert len(recent) > 0
    assert recent[-1]["label"] == "happy"
```

### Tests Layer3
```python
def test_tone_bridge_smoothing():
    """Verifica smoothing de transiciones"""
    bridge = ToneStyleBridge(smoothing=0.25)
    
    # Entrada súbita de tristeza
    profile = bridge.update("sad", valence=0.2, arousal=0.4)
    
    # Valence no debe cambiar abruptamente (smoothing)
    assert 0.35 < profile.valence_avg < 0.5  # No salta a 0.2
```

---

## 🎯 KPIs de Layers

| KPI | Objetivo | Método |
|-----|----------|--------|
| Latencia emotion detection | <100ms | Profiling con audio test |
| Accuracy emotion | >75% | Validación con dataset etiquetado |
| Tone memory hit rate | >80% | % de queries con historial |
| Smoothing effectiveness | Δ<0.2 | Cambio máximo entre updates |

---

## 🚀 Roadmap de Integración

### Fase 2.1: Layer1 I/O ✅
- [x] Componentes existentes
- [ ] Integración en graph.py
- [ ] Tests de emotion detection

### Fase 2.2: Layer2 Memory
- [x] ToneMemoryBuffer implementado
- [ ] Integración con MCP
- [ ] Tests de persistencia

### Fase 2.3: Layer3 Fluidity
- [x] ToneBridge implementado
- [ ] Integración en modulación
- [ ] Tests de smoothing

### Fase 2.4: Integración Completa
- [ ] State extendido
- [ ] Nodos del grafo modificados
- [ ] Tests end-to-end

---

## 📚 Referencias

- Layer1 I/O: `core/layer1_io/`
- Layer2 Memory: `core/layer2_memory/`
- Layer3 Fluidity: `core/layer3_fluidity/`
- Graph: `core/graph.py`

---

**Conclusión**: La arquitectura de 3 layers permite a SARAi procesar audio con **contexto emocional**, mantener **memoria de tono** y producir **transiciones suaves** en las respuestas.
