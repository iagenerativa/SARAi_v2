# 🔍 Hallazgos Críticos: Token2Wav y Pipeline de Voz

**Fecha**: 30 de Octubre de 2025  
**Estado**: ⚠️ **BLOQUEADO - Requiere componentes adicionales**

---

## 🎯 Objetivo Original

Integrar Token2Wav para generar voz en español desde los audio_logits del Talker ONNX.

---

## 🚧 Problemas Descubiertos

### 1. **Parámetros Obligatorios**

Token2Wav NO acepta `None` en parámetros críticos:

```python
waveform = model(
    code=audio_logits,          # ❌ NO: Debe ser índices enteros, no floats
    conditioning=vector,         # ❌ NO puede ser None
    reference_mel=mel_spec,      # ❌ NO puede ser None
    num_steps=3,
    guidance_scale=1.0
)
```

**Requiere**:
- `code`: Índices enteros de tokens de audio cuantizados (no floats)
- `conditioning`: Vector de condicionamiento [B, 768]
- `reference_mel`: Mel-spectrogram de referencia [B, 30000, 80]

### 2. **Incompatibilidad de Tipos**

```
Pipeline actual:
Talker ONNX → [B, T, 8448] float32/float16 (audio_logits)
                    ↓
                  ❌ GAP ❌
                    ↓
Token2Wav espera: [B, T] int64 (audio_token_indices)
```

**El problema**: Talker genera **logits continuos**, pero Token2Wav espera **índices discretos**.

**Falta**: Un componente de **cuantización** (Vector Quantization).

### 3. **Componente Faltante: Audio Quantizer**

El pipeline completo según Qwen2.5-Omni requiere:

```
Talker → Audio Quantizer → Token2Wav

Audio Quantizer:
• Input: audio_logits [B, T, 8448]
• Output: audio_token_ids [B, T] (enteros)
• Método: Vector Quantization (VQ) con codebook
```

**Archivo que falta**: `audio_quantizer.pt` o similar

---

## 📊 Comparación: Documentación vs Realidad

| Componente | Documentado | Realidad | Estado |
|------------|-------------|----------|--------|
| Audio Encoder | ✅ Existe | ✅ `audio_encoder_int8.pt` (620MB) | OK |
| Projection | ✅ Existe | ✅ `projection.onnx` (2.4KB) | OK |
| Talker | ✅ Existe | ✅ `qwen25_7b_audio.onnx` (42MB) | OK |
| **Audio Quantizer** | ❓ No docs | ❌ **FALTA** | **BLOCKER** |
| Token2Wav | ✅ Existe | ⚠️ `token2wav_fp16.pt` (858MB) | Inusable sin Quantizer |

---

## 🎤 Pipeline Correcto (Qwen2.5-Omni Completo)

```
Audio Input (16kHz)
    ↓
Audio Encoder INT8
    ↓ [B, T, 512]
Projection ONNX
    ↓ [B, T, 3584]
LFM2-1.2B (opcional modulación)
    ↓ [B, T, 3584]
Talker ONNX
    ↓ [B, T, 8448] float32 (audio_logits)
──────────────────────────────────────
    ↓ FALTA ESTE COMPONENTE ↓
──────────────────────────────────────
Audio Quantizer (VQ)
    ↓ [B, T] int64 (audio_token_ids)
──────────────────────────────────────
Token2Wav
    ↓ + conditioning [B, 768]
    ↓ + reference_mel [B, 30000, 80]
    ↓
Waveform Output (24kHz)
```

---

## 🔧 Soluciones Posibles

### Opción A: Buscar el Audio Quantizer

**Archivos a buscar**:
```bash
find models/ -name "*quantizer*" -o -name "*codebook*" -o -name "*vq*"
```

**Si existe**: Integrarlo entre Talker y Token2Wav.

**Si NO existe**: Descargar desde Hugging Face.

```bash
# Desde Qwen/Qwen2.5-Omni-7B
huggingface-cli download Qwen/Qwen2.5-Omni-7B \
    --include "audio_quantizer*.pt" \
    --local-dir models/pytorch/
```

---

### Opción B: TTS Alternativo (MÁS RÁPIDO)

Usar un TTS standalone que SÍ funciona:

#### B.1: **Piper TTS** (Recomendado)
```bash
# Instalar
pip install piper-tts

# Descargar modelo español
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/es_ES-davefx-medium.tar.gz
tar -xzf es_ES-davefx-medium.tar.gz -C models/tts/

# Usar
from piper import PiperVoice
voice = PiperVoice.load("models/tts/es_ES-davefx-medium.onnx")
audio = voice.synthesize("Hola, soy SARAi")
```

**Ventajas**:
- ✅ Latencia: ~50ms (ONNX optimizado)
- ✅ Calidad: MOS ~4.0 (muy bueno)
- ✅ Español nativo
- ✅ Sin dependencias complejas
- ✅ **FUNCIONA YA**

**Desventajas**:
- ⚠️ No integrado en el pipeline Qwen
- ⚠️ Necesita texto, no audio_logits

#### B.2: **Coqui TTS**
```bash
pip install coqui-tts
tts --text "Hola" --model_name tts_models/es/css10/vits --out_path output.wav
```

**Ventajas**:
- ✅ Alta calidad (MOS ~4.3)
- ✅ Español nativo
- ✅ Open source

**Desventajas**:
- ⚠️ Más lento (~200-500ms)
- ⚠️ Mayor consumo de RAM (~2GB)

---

### Opción C: Usar Pipeline Parcial (Sin Voz)

Mantener el pipeline actual **SIN** Token2Wav:

```
Audio → Encoder → Projection → LFM2 → Talker
                                         ↓
                                   (audio_logits)
                                         ↓
                                  Guardar como .npy
                                         ↓
                              Post-proceso con TTS externo
```

**Ventajas**:
- ✅ Todo el pipeline LLM funciona
- ✅ Latencias medidas (1320ms)
- ✅ Podemos añadir TTS después

**Desventajas**:
- ⚠️ No es end-to-end
- ⚠️ Requiere 2 pasos

---

## 💡 Recomendación

**CORTO PLAZO (HOY)**:  
👉 **Opción B.1 - Piper TTS**

**Por qué**:
1. ✅ Instalación: 5 minutos
2. ✅ Funciona garantizado
3. ✅ Latencia excelente (~50ms)
4. ✅ Calidad muy buena (MOS 4.0)
5. ✅ Español natural
6. ✅ Te permite **escuchar a SARAi HOY**

**Implementación**:
```python
# Modificar pipeline para generar texto en vez de audio_logits
LFM2 → Texto de respuesta
       ↓
    Piper TTS
       ↓
    Audio WAV
```

---

**LARGO PLAZO (Esta semana)**:  
👉 **Opción A - Buscar/descargar Audio Quantizer**

**Pasos**:
1. Buscar en `models/` si ya existe
2. Si no, descargar desde HuggingFace
3. Integrar entre Talker y Token2Wav
4. Validar pipeline end-to-end

---

## ✅ Lo Que SÍ Funciona

```
✅ Audio Encoder (INT8, 620MB)
✅ Projection ONNX (2.4KB, ~8ms)
✅ LFM2-1.2B (GGUF, ~1250ms)
✅ Talker ONNX qwen25_7b (42MB, ~10ms)
✅ Pipeline completo hasta audio_logits

TOTAL: ~1320ms de latencia
```

---

## 🚫 Lo Que NO Funciona

```
❌ Token2Wav (requiere Audio Quantizer)
❌ Pipeline end-to-end audio→audio
❌ Voz directa desde audio_logits
```

---

## 🎯 Decisión Requerida

**¿Qué prefieres?**

**A)** Buscar/descargar Audio Quantizer y completar pipeline Qwen (2-3 horas)  
**B)** Instalar Piper TTS y tener voz HOY (10 minutos) 👈 **RECOMENDADO**  
**C)** Seguir con pipeline sin voz, solo audio_logits guardados  

---

**Conclusión**: Token2Wav existe y funciona, pero el pipeline actual **está incompleto**. Falta el cuantizador de audio que convierte logits flotantes en índices discretos.
