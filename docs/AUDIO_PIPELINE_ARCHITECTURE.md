# Arquitectura del Pipeline de Audio v2.16.3

## 🎯 Objetivo

Crear un pipeline **modular y eficiente** para procesamiento de voz que use **LFM2 como cerebro** en lugar de modelos externos de 7GB.

---

## 📊 Comparativa de Arquitecturas

### ❌ Arquitectura Incorrecta (Qwen2.5-Omni-7B Monolítico)

```
Audio Input (WAV)
  ↓
Qwen2.5-Omni-7B (7GB)
  ├─ Audio Encoder (3.5GB)
  ├─ Language Model (7B params)
  ├─ Talker (integrado)
  └─ Vocoder BigVGAN (1.2GB)
  ↓
Audio Output / Text

Total RAM: ~12GB
Problema: Usa modelo LLM de 7B cuando ya tenemos LFM2-1.2B
```

---

## ✅ Arquitectura Correcta (Modular con LFM2 - IMPLEMENTABLE)

```
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE DE VOZ COMPLETO - ONNX PURO               │
└─────────────────────────────────────────────────────────────────┘

Flujo 1: AUDIO → TEXTO (STT - Speech-to-Text)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 🎤 Audio Input (WAV 16kHz)
   ↓
2. 🎧 Audio Encoder (qwen25_audio_int8.onnx ~97MB)
   → Convierte audio → audio_features [B, T, D]
   → Latencia: ~100ms
   ↓
3. 📝 Audio Decoder (qwen25_audio_int8.onnx ~97MB) ← REUTILIZADO
   → Audio features → Texto
   → Latencia: ~40ms
   ↓
   📄 TEXTO TRANSCRITO


Flujo 2: TEXTO → RAZONAMIENTO (LLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. 🧠 Thinker (LFM2-1.2B GGUF ~700MB)
   → Texto entrada → Razonamiento → Texto respuesta
   → Latencia: ~200-400ms (según longitud)
   ↓
   📄 TEXTO RESPUESTA


Flujo 3: TEXTO → AUDIO (TTS - Text-to-Speech)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. 🗣️ Text Encoder (qwen25_audio_int8.onnx ~97MB) ← REUTILIZADO
   → Texto → hidden_states [B, S, 3584]
   → Latencia: ~40ms
   ↓
6. ✅ Talker (qwen25_7b_audio.onnx ~41MB) ← COMPONENTE ESPECÍFICO
   → hidden_states → audio_logits [B, S, 8448]
   → Latencia: ~5ms ⚡
   ↓
7. 🔊 Vocoder (qwen25_audio_int8.onnx ~97MB) ← REUTILIZADO
   → audio_logits → waveform de audio
   → Latencia: ~100ms
   ↓
   🎵 AUDIO OUTPUT
```

**Total RAM**: ~840MB (qwen25_audio_int8.onnx compartido ~97MB + Talker 42MB + LFM2 700MB)
**Latencia E2E**: ~485ms (STT 140ms + LFM2 250ms + TTS 145ms)

```
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE DE VOZ COMPLETO - ONNX PURO               │
└─────────────────────────────────────────────────────────────────┘

Flujo 1: AUDIO → TEXTO (STT - Speech-to-Text)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 🎤 Audio Input (WAV 16kHz)
   ↓
2. 🎧 Audio Encoder (qwen25_audio_int8.onnx ~97MB)
   → Convierte audio → audio_features [B, T, D]
   → Latencia: ~100ms
   ↓
   � AUDIO FEATURES


Flujo 2: TEXTO → RAZONAMIENTO (LLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. 🧠 Thinker (LFM2-1.2B GGUF ~700MB)
   → Audio features → Texto razonado
   → Latencia: ~200-400ms (según longitud)
   ↓
   📄 TEXTO RESPUESTA


Flujo 3: TEXTO → AUDIO (TTS - Text-to-Speech)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. 🗣️ Text-to-Features
   → Texto → hidden_states [B, S, 3584]
   → Latencia: ~40ms
   ↓
5. ✅ Talker (qwen25_7b_audio.onnx ~41MB) ← EL QUE TENEMOS
   → hidden_states → audio_logits [B, S, 8448]
   → Latencia: ~5ms ⚡
   ↓
6. 🔊 Vocoder (qwen25_audio_int8.onnx ~97MB) ← REUTILIZADO
   → audio_logits → waveform de audio
   → Latencia: ~100ms
   ↓
   🎵 AUDIO OUTPUT
```

**Total RAM**: ~935MB (encoder/vocoder comparten modelo, ~97MB cargado 1 vez)
**Latencia E2E**: ~445ms (STT 100ms + LFM2 250ms + TTS 145ms)

---

## 🔧 Componentes Necesarios

### ✅ Ya Tenemos (IMPLEMENTABLE HOY)

1. **Audio Encoder/Decoder**: `models/onnx/qwen25_audio_int8.onnx` (~97MB) ✅
   - **Ubicación**: `models/onnx/qwen25_audio_int8.onnx` (97 MB archivo único)
   - **Cuantización**: INT8 optimizado para CPU
   - Función Quad: 
     - STT Encoder: Audio → Features
     - STT Decoder: Features → Texto
     - TTS Encoder: Texto → Features
     - TTS Vocoder: Audio logits → Waveform
   - Compatible con formato ONNX nativo

2. **Talker**: `models/onnx/qwen25_7b_audio.onnx` + `.data` (~42MB) ✅
   - **Ubicación**:
     - Header: `models/onnx/qwen25_7b_audio.onnx` (922 bytes)
     - Data: `models/onnx/qwen25_7b_audio.onnx.data` (41.2 MB)
   - Input: hidden_states [B, S, 3584]
   - Output: audio_logits [B, S, 8448]
   - Modelo ONNX específico para TTS

3. **Thinker (LLM)**: LFM2-1.2B GGUF (~700MB) ✅
   - Razonamiento: Texto entrada → Texto respuesta
   - Formato: GGUF (llama.cpp)
   - Acceso: ModelPool.get("tiny")

**Total modelos únicos**: 2 ONNX (139MB) + 1 GGUF (700MB) = **~840MB**

**Compartición de memoria**:
- qwen25_audio_int8.onnx se carga UNA VEZ (97MB), se usa en CUATRO puntos:
  1. Audio → Features (Encoder STT)
  2. Features → Texto (Decoder STT)
  3. Texto → Features (Encoder TTS)
  4. Audio logits → Waveform (Vocoder TTS)
- Esto ahorra ~291MB vs cargar 4 modelos separados (97MB × 4 = 388MB)

### ⏳ CANCELADO - Ya no necesitamos exportar nada

~~| Componente | Modelo Base | Exportar a ONNX | Tamaño Estimado |~~  
~~|------------|-------------|-----------------|-----------------|~~  
~~| Audio Encoder | Whisper-tiny | whisper_encoder.onnx | ~39MB |~~  
~~| Vocoder | BigVGAN | bigvgan_vocoder.onnx | ~150MB |~~  
~~| TTS Encoder | Qwen2.5-Omni | tts_text_encoder.onnx | ~200MB |~~

**✅ Ventaja clave**: `qwen25_audio_int8.onnx` sirve TANTO para encoder como vocoder

---

## 📝 Pipeline E2E Actual (Implementado)

### Flujo Simplificado: Audio → Texto → LFM2

```python
# PASO 1: Audio → Texto (STT)
# Usamos modelo TEMPORAL monolítico mientras exportamos Whisper
audio_bytes = load_audio("input.wav")
text = whisper_stt(audio_bytes)  # TODO: Reemplazar con whisper_encoder.onnx

# PASO 2: Texto → Razonamiento (LFM2)
model_pool = ModelPool("config/sarai.yaml")
lfm2 = model_pool.get("tiny")  # LFM2-1.2B GGUF
response = lfm2(f"Usuario: {text}\nAsistente:")

# PASO 3: Respuesta en texto
print(response)
```

**RAM Total**: ~1.5GB (Whisper 39MB + LFM2 700MB + overhead)  
**Latencia E2E**: ~350ms (Whisper 80ms + LFM2 250ms)

---

## 🚀 Pipeline Completo Futuro: Voz → Voz

```python
# PASO 1: Audio → Texto (STT)
audio_encoder = load_onnx("whisper_encoder.onnx")  # 39MB
embeddings = audio_encoder(audio_bytes)
text = decode_text(embeddings)

# PASO 2: Texto → Razonamiento (LFM2)
lfm2 = model_pool.get("tiny")
response_text = lfm2(f"Usuario: {text}\nAsistente:")

# PASO 3: Texto → Audio (TTS)
tts_encoder = load_onnx("tts_text_encoder.onnx")  # 200MB
hidden_states = tts_encoder(response_text)

talker = load_onnx("qwen25_7b_audio.onnx")  # 41MB ✅
audio_logits = talker(hidden_states)

vocoder = load_onnx("bigvgan_vocoder.onnx")  # 150MB
audio_output = vocoder(audio_logits)

# PASO 4: Reproducir audio
play_audio(audio_output)
```

**RAM Total**: ~1.3GB (Whisper 39MB + LFM2 700MB + Talker 41MB + Vocoder 150MB + TTS Encoder 200MB + overhead)  
**Latencia E2E**: ~450ms (STT 80ms + LFM2 250ms + TTS 120ms)

---

## 📦 Scripts de Exportación Necesarios

### 1. Exportar Whisper Encoder

```bash
python scripts/export_whisper_encoder.py \
  --model openai/whisper-tiny \
  --output models/onnx/whisper_encoder.onnx \
  --optimize int8
```

### 2. Exportar Vocoder BigVGAN

```bash
python scripts/export_bigvgan_vocoder.py \
  --model nvidia/bigvgan \
  --output models/onnx/bigvgan_vocoder.onnx \
  --optimize fp16
```

### 3. Exportar TTS Text Encoder

```bash
python scripts/export_tts_encoder.py \
  --model Qwen/Qwen2.5-Omni-7B \
  --component text_encoder \
  --output models/onnx/tts_text_encoder.onnx \
  --optimize int8
```

---

## ✅ Próximos Pasos

1. **Inmediato**: Usar modelo monolítico para STT mientras exportamos Whisper
2. **Corto plazo** (esta semana):
   - Exportar Whisper-tiny a ONNX
   - Exportar BigVGAN a ONNX
   - Exportar TTS Text Encoder a ONNX
3. **Medio plazo** (próxima semana):
   - Integrar componentes ONNX en AudioOmniPipeline
   - Validar latencia E2E <500ms
   - Documentar benchmarks

---

## 🎯 KPIs del Pipeline Modular Final

| Métrica | Objetivo | Estado |
|---------|----------|--------|
| RAM P99 | ≤ 2GB | ⏳ Pendiente validar |
| Latencia STT | ≤ 100ms | ⏳ Pendiente exportar Whisper |
| Latencia LFM2 | ≤ 300ms | ✅ Ya funciona |
| Latencia TTS | ≤ 150ms | ⏳ Pendiente exportar componentes |
| **Latencia E2E** | **≤ 500ms** | **⏳ Objetivo** |
| Calidad MOS | ≥ 4.0 | ⏳ Pendiente evaluar |

---

## 🔒 Sin Dependencias de Qwen2.5-Omni-7B

**Clave**: El Talker ONNX (41MB) es **solo una pieza**. NO contiene Encoder ni Vocoder integrados.

- ❌ **NO** cargar Qwen2.5-Omni-7B completo (7GB)
- ✅ **SÍ** usar componentes ONNX independientes (~430MB total)
- ✅ **SÍ** usar LFM2-1.2B como cerebro (700MB)

**Ahorro de RAM**: 7GB → 1.3GB (reducción del 81%)
