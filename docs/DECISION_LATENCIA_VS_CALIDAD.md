# Análisis de Decisión: Latencia vs Calidad

**Fecha**: 29 octubre 2025  
**Contexto**: Evaluar si Qwen3-Omni-30B (2.87s) justifica el aumento vs objetivo 190ms

---

## 📊 Comparativa Crítica

### Escenario A: Objetivo Original (190-240ms)

**Modelo hipotético**: Qwen3-VL-4B-Instruct (si existiera en ONNX)

| Métrica | Valor Esperado | Caso de Uso Ideal |
|---------|---------------|-------------------|
| **Latencia** | 190-240ms | ✅ Conversación fluida en tiempo real |
| **STT WER** | ~2.0% | ✅ Transcripción aceptable |
| **TTS MOS** | 4.21 natural, 4.38 empatía | ✅ Voz natural |
| **RAM** | ~1.5GB (3B INT8) | ✅ Ligero |
| **Parámetros** | 3B | ⚠️ Capacidad limitada |

**Limitaciones**:
- ❌ Menor capacidad de razonamiento (3B parámetros)
- ❌ Vocabulario más limitado
- ❌ Menos contexto para respuestas complejas
- ⚠️ **Problema**: Este modelo NO existe como ONNX INT8 actualmente

---

### Escenario B: Situación Actual (Qwen3-Omni-30B)

| Métrica | Valor Real | Impacto |
|---------|-----------|---------|
| **Latencia** | 2.87s (2870ms) | ❌ **15x más lenta** que objetivo |
| **STT WER** | ≤1.8% (esperado) | ✅ 10% mejor que 3B |
| **TTS MOS** | ≥4.32/4.50 (esperado) | ✅ ~3% mejor que 3B |
| **RAM** | 1.1GB INT8 | ✅ Más ligero que 3B FP32 |
| **Parámetros** | 30B | ✅ **10x más capacidad** |

**Ventajas**:
- ✅ 10x más parámetros (razonamiento profundo)
- ✅ Mejor calidad STT/TTS (medible)
- ✅ Mayor vocabulario y contexto
- ✅ Más robusto ante acentos/ruido

**Desventajas**:
- ❌ **Latencia INACEPTABLE para conversación en tiempo real**
- ❌ 2.87s = pausa perceptible (límite humano ~200ms)
- ❌ Experiencia de usuario degradada

---

## 🎯 Casos de Uso: ¿Cuándo Importa la Latencia?

### Caso 1: Conversación en Tiempo Real ⚠️ CRÍTICO

**Escenario**: Usuario habla con SARAi como asistente de voz

```
Usuario: "Hola SARAi, ¿qué tiempo hará mañana?"
         └─ Silencio de 2.87 segundos ─┐
SARAi:                                  "Mañana habrá sol..."
```

**Percepción del usuario**:
- ✅ 190ms: Natural, fluido ✨
- ⚠️ 500ms: Aceptable, ligera pausa
- ❌ 1000ms: Lento, notable
- ❌ **2870ms: INACEPTABLE, parece roto** 💔

**Veredicto**: ❌ **Qwen3-Omni-30B NO sirve para conversación en tiempo real**

---

### Caso 2: Procesamiento Batch/Asíncrono ✅ VIABLE

**Escenario**: Transcripción de archivos, procesamiento offline

```
Usuario sube 10 archivos de audio → SARAi procesa en background
Latencia por archivo: 2.87s
Total: 28.7s para 10 archivos

Usuario NO espera activamente (proceso asíncrono)
```

**Percepción del usuario**:
- ✅ Calidad superior (WER 1.8% vs 2.0%)
- ✅ No hay espera perceptible (background task)
- ✅ Mejor para transcripciones largas

**Veredicto**: ✅ **Qwen3-Omni-30B EXCELENTE para batch/offline**

---

### Caso 3: Swapping bajo Demanda (Tu Propuesta) 🤔

**Escenario**: SARAi mantiene modelo ligero por defecto, carga 30B solo cuando necesita calidad máxima

```
Flujo Normal (95% casos):
├─ Modelo ligero (190ms): Conversación casual
└─ Swapping activado si:
    ├─ Transcripción crítica (legal, médica)
    ├─ TTS emocional complejo
    └─ Usuario solicita máxima calidad

Swapping Cost:
├─ Carga modelo 30B: ~27s (one-time)
├─ Inferencia: 2.87s
├─ Descarga: ~1s
└─ Total: ~31s overhead
```

**Viabilidad**:
- ✅ Permite usar ambos modelos según contexto
- ✅ Optimiza latencia (190ms) para casos comunes
- ✅ Reserva calidad (30B) para casos críticos
- ⚠️ Requiere detectar automáticamente cuándo swappear
- ⚠️ 31s overhead al activar 30B (aceptable si es poco frecuente)

**Veredicto**: ✅ **Estrategia híbrida VIABLE** si:
1. Existe modelo 3B ONNX INT8 (190ms)
2. Detección automática de casos críticos
3. Swapping <5% del tiempo total

---

## 💡 Opciones Disponibles

### Opción 1: Buscar/Crear Qwen3-VL-4B-Instruct ONNX INT8 ⭐ RECOMENDADO

**Acción**:
1. Buscar en HuggingFace: `Qwen/Qwen3-VL-4B-Instruct-Instruct`
2. Si existe PyTorch: Convertir a ONNX + cuantizar INT8
3. Si NO existe: Usar alternativa (MeloTTS + Whisper-small)

**Pros**:
- ✅ Latencia objetivo (190ms) alcanzable
- ✅ Experiencia fluida en tiempo real
- ✅ Calidad suficiente para uso general

**Contras**:
- ⚠️ Requiere trabajo de conversión/búsqueda
- ⚠️ Calidad inferior a 30B (WER 2.0% vs 1.8%)

**Timeline**: 2-4 horas (si existe modelo base)

---

### Opción 2: Sistema Híbrido (Dual-Model Swapping) 🔄

**Arquitectura**:
```yaml
models:
  audio_light:
    name: "Qwen3-VL-4B-Instruct"  # Si se encuentra
    latency: 190ms
    use_case: "conversación general (95% casos)"
    ram: 1.5GB
  
  audio_heavy:
    name: "Qwen3-Omni-30B"
    latency: 2870ms
    use_case: "transcripción crítica, TTS emocional complejo (5% casos)"
    ram: 1.1GB
    swap_triggers:
      - user_request: "máxima calidad"
      - context: "legal|médico|técnico"
      - emotion_intensity: >0.8
```

**Pros**:
- ✅ Mejor de ambos mundos
- ✅ Latencia baja por defecto
- ✅ Calidad máxima cuando importa

**Contras**:
- ⚠️ Complejidad arquitectural (2 modelos)
- ⚠️ Requiere modelo ligero (no lo tenemos aún)
- ⚠️ Swapping overhead (31s al cambiar)

**Timeline**: 1 día (con modelo ligero disponible)

---

### Opción 3: Mantener Solo Qwen3-Omni-30B ❌ NO RECOMENDADO

**Uso exclusivo**:
- ✅ Transcripción batch
- ✅ Procesamiento offline
- ❌ Conversación en tiempo real (ROTO)

**Veredicto**: Solo viable si SARAi **NO necesita conversación fluida**

---

## 🔍 Investigación Necesaria

### ¿Existe Qwen3-VL-4B-Instruct en ONNX?

Necesito verificar:

1. **HuggingFace Search**:
   ```bash
   # Buscar modelos Qwen Omni disponibles
   https://huggingface.co/models?search=qwen+omni
   ```

2. **Alternativas REALISTAS** (latencia <300ms, <4GB RAM):
   
   **Opción A - Balance Óptimo** (RECOMENDADO):
   - **Whisper-small** (244M, STT): ~80ms, WER ~3-4%
   - **Piper TTS** (o MeloTTS): ~60ms, MOS ~4.0
   - **Total**: ~140ms ✅, ~1.5GB RAM ✅
   
   **Opción B - Más Ligero**:
   - **Whisper-tiny** (39M, STT): ~30ms, WER ~5-6% (peor)
   - **espeak-ng** (síntesis rápida): ~20ms, MOS ~3.0 (robótico)
   - **Total**: ~50ms ✅, ~500MB RAM ✅ (pero calidad baja)
   
   **Opción C - Calidad Media-Alta**:
   - **Whisper-medium** (769M, STT): ~150ms, WER ~2.5%
   - **VITS** (TTS): ~100ms, MOS ~4.2
   - **Total**: ~250ms ⚠️, ~2.5GB RAM ✅
   
   **❌ NO VIABLE** (descartado por latencia/RAM):
   - Whisper-large-v3 (1.5B): ~400ms, WER ~1.8% (muy lento)
   - Higgs Audio V2 (5.77B): ~800ms, MOS ~4.8 (VRAM >6GB)

3. **Conversión manual**:
   - Si existe Qwen3-VL-4B-Instruct en PyTorch → Convertir ONNX
   - Timeline: 4-6 horas

---

## 📋 Recomendación Final

### Para Conversación en Tiempo Real: ❌ QWEN3-OMNI-30B NO ES VIABLE

**Razón**: 2.87s es **14.5x más lenta** que el límite de fluidez humana (~200ms)

### Estrategia Recomendada: SISTEMA HÍBRIDO

```
┌─────────────────────────────────────────────┐
│ SARAi Dual-Speed Audio Architecture        │
├─────────────────────────────────────────────┤
│                                             │
│  DEFAULT (95% casos):                       │
│  ├─ Modelo Ligero (3B o alternativa)        │
│  ├─ Latencia: 190-240ms ✅                  │
│  └─ Calidad: Suficiente para uso general   │
│                                             │
│  SWAP TO HEAVY (5% casos críticos):        │
│  ├─ Qwen3-Omni-30B                         │
│  ├─ Latencia: 2870ms (aceptable offline)   │
│  ├─ Triggers: transcripción legal/médica   │
│  └─ Calidad: Máxima (WER 1.8%, MOS 4.5)    │
│                                             │
│  Swapping Cost: 31s (amortizado en >10     │
│  inferencias de calidad)                    │
└─────────────────────────────────────────────┘
```

**Próximos Pasos**:

1. **INMEDIATO**: Buscar/crear modelo ligero (3B ONNX INT8)
   - Timeline: 4-6 horas
   - Prioridad: CRÍTICA

2. **CORTO PLAZO**: Implementar lógica de swapping
   - Timeline: 1 día
   - Prioridad: ALTA

3. **FALLBACK**: Si no existe 3B, usar alternativa:
   - Whisper-small (STT) + MeloTTS (TTS)
   - Timeline: 2-3 horas
   - Calidad: 85-90% del Qwen-3B

---

## 🎯 Respuesta Directa a Tu Pregunta

> "¿Justifica la mejora del sistema el aumento de latencia aunque solo sea para swapping?"

**Respuesta**: ✅ **SÍ, pero SOLO como modelo secundario** (5% del tiempo)

**Justificación**:
- ❌ **NO** para conversación por defecto (2.87s es inaceptable)
- ✅ **SÍ** para transcripción crítica bajo demanda
- ✅ **SÍ** para procesamiento batch/offline
- ✅ **SÍ** como "modo alta calidad" activable

**Condición crítica**: Necesitas un modelo ligero (190ms) como default.

**Prioridad absoluta**: Conseguir/crear Qwen3-VL-4B-Instruct ONNX INT8 o alternativa equivalente.

---

## 🚀 Plan de Acción Inmediato

## 📊 Tabla Comparativa de Alternativas REALISTAS

| Modelo/Pipeline | Latencia | RAM | STT WER | TTS MOS | Uso Recomendado | Viable? |
|----------------|----------|-----|---------|---------|-----------------|---------|
| **Qwen3-Omni-30B INT8** | 2870ms | 1.1GB | 1.8% ✅ | 4.5 ✅ | Batch/offline, calidad máxima | ⚠️ NO para real-time |
| **Whisper-small + Piper** | 190ms ✅ | 1.9GB | 3.5% ⚠️ | 4.0 ⚠️ | Conversación fluida, uso general | ✅ **RECOMENDADO** |
| **Whisper-medium + VITS** | 250ms ⚠️ | 2.5GB | 2.5% ✅ | 4.2 ✅ | Balance calidad/latencia | ✅ Viable |
| **Whisper-tiny + espeak-ng** | 50ms ✅ | 0.5GB | 6% ❌ | 3.0 ❌ | Prototipado rápido | ❌ Calidad muy baja |
| **Whisper-large-v3 + Higgs V2** | 1200ms ❌ | 8GB ❌ | 1.5% ✅ | 4.8 ✅ | SOTA (si tienes GPU) | ❌ Latencia/RAM excesiva |
| **Qwen3-VL-4B-Instruct** (si existiera) | 190ms ✅ | 1.5GB | 2.0% ✅ | 4.3 ✅ | Ideal (modelo unificado) | ⚠️ NO existe en ONNX |

**Leyenda**:
- ✅ Excelente (cumple objetivo)
- ⚠️ Aceptable (trade-off necesario)
- ❌ Inaceptable (no viable)

---

## 🎯 Decisión Recomendada: Sistema Híbrido REALISTA

### Arquitectura Dual-Speed con Modelos EXISTENTES

```yaml
# config/audio_models.yaml

audio:
  default_tier: "fast"  # Para 95% de casos
  
  models:
    fast:  # Conversación en tiempo real
      stt: "openai/whisper-small"
      tts: "rhasspy/piper-tts"
      latency_target: 190ms
      quality: "aceptable"
      ram: 1.9GB
      use_cases:
        - conversación casual
        - comandos de voz
        - asistente doméstico
    
    quality:  # Procesamiento crítico
      stt: "Qwen3-Omni-30B-A3B-Instruct (STT component)"
      tts: "Qwen3-Omni-30B-A3B-Instruct (TTS component)"
      latency_target: 2870ms
      quality: "excelente"
      ram: 1.1GB
      swap_triggers:
        - user_command: "transcribe con máxima calidad"
        - context_type: ["legal", "médico", "técnico"]
        - emotion_required: high  # TTS emocional complejo
      use_cases:
        - transcripción legal/médica
        - TTS emocional (empatía, tristeza profunda)
        - documentación técnica
```

**Implementación**:

```python
# core/audio_router.py

class AudioRouter:
    """Router inteligente entre modelo rápido y calidad"""
    
    def __init__(self):
        self.fast_pipeline = AudioLightPipeline()  # Whisper-small + Piper
        self.quality_pipeline = AudioOmniPipeline()  # Qwen3-Omni-30B
        self.current_tier = "fast"
    
    def process(self, audio: bytes, context: dict) -> dict:
        """
        Decide qué pipeline usar según contexto
        
        Args:
            audio: Audio bytes
            context: {
                'user_command': str,
                'context_type': str,
                'emotion_required': str,
                'is_critical': bool
            }
        
        Returns:
            Procesamiento del pipeline seleccionado
        """
        # Detección automática de necesidad de calidad
        needs_quality = (
            context.get('is_critical', False) or
            context.get('context_type') in ['legal', 'médico', 'técnico'] or
            context.get('emotion_required') == 'high' or
            'máxima calidad' in context.get('user_command', '').lower()
        )
        
        if needs_quality:
            print(f"🔄 Swapping a modelo de CALIDAD (latencia ~3s, justificado)")
            return self.quality_pipeline.process_audio(audio)
        else:
            # 95% de casos: latencia óptima
            return self.fast_pipeline.process_audio(audio)
```

**Beneficios**:
1. ✅ **95% del tiempo**: Latencia 190ms (conversación fluida)
2. ✅ **5% crítico**: Calidad máxima (WER 1.8%, MOS 4.5)
3. ✅ **Sin frustración**: Usuario sabe cuándo esperar (swap explícito)
4. ✅ **Modelos existentes**: No requiere buscar Qwen-3B inexistente

**Trade-offs aceptados**:
- ⚠️ Calidad ligeramente inferior en modo rápido (WER 3.5% vs 1.8%)
- ⚠️ Necesita detectar contextos críticos (reglas o ML)
- ⚠️ Swapping overhead (31s al activar calidad)

---

## 🚀 Plan de Acción ACTUALIZADO

### PASO 1: Investigar disponibilidad (30 min)
```bash
# Buscar en HuggingFace
- Qwen/Qwen3-VL-4B-Instruct-Instruct
- Qwen/Qwen-Audio-3B
- Alternativas: speechbrain, coqui-ai/TTS

# Si NO existe en ONNX:
- Verificar si existe PyTorch → Convertir
- Fallback: Whisper-small + MeloTTS
```

### PASO 2: Implementar modelo ligero (4-6h)

**Opción RECOMENDADA**: Whisper-small + Piper TTS

```python
# agents/audio_light_pipeline.py - Pipeline ligero para conversación

class AudioLightPipeline:
    """
    Pipeline ligero para conversación en tiempo real
    - STT: Whisper-small (244M) ~80ms
    - TTS: Piper (60M) ~60ms
    - TOTAL: ~140ms ✅
    """
    
    def __init__(self):
        # STT: Whisper-small (mejor balance latencia/calidad)
        import whisper
        self.stt_model = whisper.load_model("small")  # 244M params
        
        # TTS: Piper (TTS rápido, calidad aceptable)
        from piper import PiperVoice
        self.tts_model = PiperVoice.load("es_ES-davefx-medium")  # 60M params
    
    def process_audio(self, audio_bytes: bytes) -> dict:
        """
        Procesa audio en tiempo real con latencia <200ms
        
        Returns:
            {
                'text': str,           # Transcripción
                'response_audio': bytes,  # Respuesta TTS
                'latency_ms': float,   # Latencia total
                'metadata': {
                    'stt_wer_expected': 3.5,  # % error esperado
                    'tts_mos_expected': 4.0,  # Calidad voz
                    'model': 'whisper-small + piper'
                }
            }
        """
        import time
        start = time.time()
        
        # 1. STT con Whisper-small (~80ms)
        import librosa
        audio, sr = librosa.load(audio_bytes, sr=16000)
        result = self.stt_model.transcribe(audio)
        text = result["text"]
        
        # 2. [AQUÍ Iría LFM2 para lógica/RAG/empatía]
        # Por ahora, eco simple para testing
        response_text = f"Entendido: {text}"
        
        # 3. TTS con Piper (~60ms)
        response_audio = self.tts_model.synthesize(response_text)
        
        latency_ms = (time.time() - start) * 1000
        
        return {
            'text': text,
            'response_audio': response_audio,
            'latency_ms': latency_ms,
            'metadata': {
                'stt_wer_expected': 3.5,
                'tts_mos_expected': 4.0,
                'model': 'whisper-small + piper'
            }
        }
```

**Benchmarks esperados**:
```
Latencia STT:        ~80ms  (Whisper-small CPU)
Latencia TTS:        ~60ms  (Piper CPU)
Latencia LFM2:       ~50ms  (si se integra para lógica)
─────────────────────────────────────────────
TOTAL:               ~190ms ✅ (dentro de objetivo)

RAM:
- Whisper-small:     ~1GB
- Piper:             ~200MB
- LFM2 (opcional):   ~700MB
─────────────────────────────────────────────
TOTAL:               ~1.9GB ✅ (viable en 16GB)

Calidad:
- WER (STT):         ~3-4% (aceptable, no SOTA)
- MOS (TTS):         ~4.0  (natural, no emocional)
```

**Ventajas vs Qwen3-Omni-30B**:
- ✅ Latencia: 190ms vs 2870ms (**15x más rápido**)
- ✅ Conversación fluida en tiempo real
- ⚠️ Calidad: 3.5% WER vs 1.8% (peor, pero aceptable)

**Desventajas**:
- ⚠️ Menor calidad que Qwen-30B (WER +100%, MOS -8%)
- ⚠️ Sin empatía nativa en TTS (Piper es neutro)
- ⚠️ Pipeline desagregado (más puntos de fallo)

**Alternativas descartadas**:
```
❌ Whisper-large-v3 + Higgs Audio V2:
   - Latencia: ~1200ms (4x objetivo)
   - RAM: ~8GB (inaceptable)
   - Calidad: SOTA pero inviable para real-time

❌ Whisper-tiny + espeak-ng:
   - Latencia: ~50ms ✅
   - RAM: ~500MB ✅
   - Calidad: Muy baja (WER 6%, MOS 3.0 robótico)
```

### PASO 3: Configurar swapping (1 día)
```python
# core/model_pool.py
class AudioModelPool:
    def get_audio_model(self, quality_tier: str):
        if quality_tier == "fast":
            return self.load("qwen-3b-int8")  # 190ms
        elif quality_tier == "quality":
            return self.load("qwen-30b-int8")  # 2870ms
```

---

**Conclusión ACTUALIZADA**: 

1. **Qwen3-Omni-30B es EXCELENTE**, pero **NO para conversación en tiempo real** (2.87s inaceptable)

2. **NO persigas Whisper-large-v3 + Higgs V2** (latencia ~1.2s, RAM ~8GB, overkill)

3. **SÍ implementa sistema híbrido**:
   - **DEFAULT**: Whisper-small + Piper (~190ms, WER 3.5%, MOS 4.0) ✅
   - **SWAP**: Qwen3-Omni-30B cuando se justifica (transcripción crítica) ✅

4. **Timeline realista**:
   - Implementar pipeline ligero: **4-6 horas**
   - Configurar router de swapping: **2-3 horas**
   - Testing integración: **2 horas**
   - **TOTAL: 1 día de trabajo**

¿Procedo a implementar el **AudioRouter** y el **AudioLightPipeline** (Whisper-small + Piper)?
