# 🎙️ Pruebas de Conversación Real con SARAi

Este directorio contiene tests para evaluar SARAi en **conversación real** usando micrófono y altavoz.

---

## 📋 Objetivo

Medir en entorno real:

1. **Latencia E2E** (micrófono → respuesta en altavoz)
2. **Calidad de voz** (tono, expresividad, naturalidad)
3. **Precisión STT** (Word Error Rate)
4. **Calidad TTS** (Mean Opinion Score)
5. **Coherencia de respuestas LLM**

---

## 🚀 Scripts Disponibles

### 1. `test_voice_quick.py` - Test Rápido (RECOMENDADO para empezar)

**Características**:
- ✅ Setup mínimo (solo pyaudio, whisper, pyttsx3, LFM2)
- ✅ Grabación de 5 segundos por turno
- ✅ Métricas básicas de latencia
- ✅ Funciona sin ONNX

**Instalación**:
```bash
# Instalar dependencias
pip install pyaudio whisper pyttsx3 llama-cpp-python

# En Ubuntu/Debian (si pyaudio falla)
sudo apt-get install python3-pyaudio portaudio19-dev

# En macOS
brew install portaudio
pip install pyaudio
```

**Uso**:
```bash
# Test de 3 turnos (default)
python tests/test_voice_quick.py

# El script te pedirá hablar por 5 segundos en cada turno
# Espera la señal "🎤 Grabando 5s..." y habla claramente
# Ejemplo de conversación:
#   Turno 1: "Hola, ¿cómo estás?"
#   Turno 2: "¿Qué puedes hacer?"
#   Turno 3: "Gracias, adiós"
```

**Salida Esperada**:
```
🔧 Cargando Whisper...
✅ Whisper listo
🔧 Cargando LFM2...
✅ LFM2 listo (467ms)
✅ TTS listo

════════════════════════════════════════════════════════════
TURNO 1
════════════════════════════════════════════════════════════

🎤 Grabando 5s...
✅ Grabación completa
🔄 Transcribiendo...
✅ Transcripción: "Hola, ¿cómo estás?" (1200ms)
🤔 Razonando...
✅ Respuesta: "Hola! ¿En qué puedo ayudarte?" (890ms)
🔊 Hablando...
✅ TTS (350ms)

────────────────────────────────────────────────────────────
⏱️  LATENCIAS:
   STT:  1200ms
   LLM:   890ms
   TTS:   350ms
   E2E:  7500ms  (incluye 5s de grabación)
────────────────────────────────────────────────────────────
```

---

### 2. `test_real_voice_conversation.py` - Test Completo (Avanzado)

**Características**:
- ✅ Grabación con detección automática de silencio
- ✅ Soporte para pipeline ONNX (si está disponible)
- ✅ Métricas avanzadas (MOS, estadísticas)
- ✅ Logs JSON detallados
- ✅ Modo silencioso (sin reproducción de audio)

**Dependencias adicionales**:
```bash
pip install sounddevice scipy librosa
```

**Uso**:
```bash
# Test normal (con reproducción de audio)
python tests/test_real_voice_conversation.py --turns 5

# Modo silencioso (solo métricas, sin TTS)
python tests/test_real_voice_conversation.py --silent --turns 3

# Escenario predefinido
python tests/test_real_voice_conversation.py --scenario greeting
```

**Salida Esperada**:
```
🎙️  TEST DE CONVERSACIÓN REAL CON SARAi
════════════════════════════════════════════════════════════

Turnos máximos: 5
Presiona Ctrl+C para terminar en cualquier momento

════════════════════════════════════════════════════════════
TURNO 1
════════════════════════════════════════════════════════════

[1/5] Grabación de audio

🎤 Escuchando... (habla y luego haz silencio)
✅ Silencio detectado, finalizando grabación
📊 Grabado: 3.2s, 3500ms

[2/5] Transcripción (STT)
👤 Usuario: "Hola SARAi, ¿cómo estás?"
⏱️  STT: 1150ms

[3/5] Razonamiento (LLM)
🤖 SARAi: "Hola! Estoy funcionando correctamente. ¿En qué puedo ayudarte?"
⏱️  LLM: 920ms

[4/5] Síntesis de voz (TTS)
⏱️  TTS: 380ms
📊 Calidad estimada: {'mos_estimated': 3.8, 'quality': 'good'}

[5/5] Reproducción
🔊 Reproduciendo respuesta...

────────────────────────────────────────────────────────────
⏱️  LATENCIA E2E TOTAL: 5950ms
   ├─ Grabación: 3500ms
   ├─ STT: 1150ms
   ├─ LLM: 920ms
   └─ TTS: 380ms
────────────────────────────────────────────────────────────

¿Continuar conversación? (Enter=Sí, Ctrl+C=No)
```

**Resumen Final**:
```
📊 RESUMEN DE LA CONVERSACIÓN
════════════════════════════════════════════════════════════

🔢 Turnos completados: 5
⏰ Tiempo total: 32.5s

📈 LATENCIAS PROMEDIO:
  STT:  1180ms (±95ms)
  LLM:   905ms (±42ms)
  TTS:   365ms (±28ms)
  E2E:  5850ms (±320ms)

📊 LATENCIAS MIN/MAX:
  STT:  1050ms / 1320ms
  LLM:   850ms / 970ms
  TTS:   330ms / 410ms
  E2E:  5400ms / 6280ms

🎤 ANÁLISIS DE CALIDAD:
  MOS estimado: 3.75/5.0
  ✅ Calidad buena

════════════════════════════════════════════════════════════

💾 Log guardado: logs/voice_conversation_20251030_143022.json

📋 EVALUACIÓN FINAL:
✅ Latencia E2E buena: 5850ms < 6000ms

🎯 PRÓXIMOS PASOS SUGERIDOS:
  • Optimizar LLM (latencia alta: 905ms)
```

---

## 📊 Interpretación de Resultados

### Latencia E2E

| Latencia | Evaluación | Acción |
|----------|------------|--------|
| < 2s | ✅ Excelente | Listo para producción |
| 2-3s | ✅ Bueno | Aceptable para uso general |
| 3-5s | ⚠️ Aceptable | Considerar optimizaciones |
| > 5s | ❌ Alto | **Optimizar urgente** |

**Nota**: La latencia E2E incluye el tiempo de grabación. Restar ~3-5s para obtener latencia real de procesamiento.

### Componentes

**STT (Speech-to-Text)**:
- **Objetivo**: < 1000ms
- **Si > 1500ms**: Considerar Whisper tiny o modelo más rápido

**LLM (Razonamiento)**:
- **Objetivo**: < 500ms
- **Si > 1000ms**: Optimizar LFM2 (INT8, GPU, modelo más pequeño)

**TTS (Text-to-Speech)**:
- **Objetivo**: < 500ms
- **Si > 800ms**: Usar TTS más rápido (pyttsx3 → Coqui, VITS)

### Calidad de Voz (MOS)

| MOS | Calidad | Recomendación |
|-----|---------|---------------|
| 4.0-5.0 | ✅ Excelente | Listo |
| 3.5-4.0 | ✅ Buena | Aceptable |
| 3.0-3.5 | ⚠️ Aceptable | Mejorar si es posible |
| < 3.0 | ❌ Baja | **Cambiar TTS** |

---

## 🔍 Análisis de Expresión y Tono

### Escuchar Grabaciones

Los archivos de audio se guardan en:
```
state/voice_test_temp/
├── temp_input.wav      # Tu voz grabada
└── temp_tts.wav        # Respuesta de SARAi
```

### Evaluación Manual

Escucha las respuestas y evalúa:

1. **Naturalidad** (1-5):
   - 5: Suena humano, indistinguible
   - 4: Natural con pequeñas imperfecciones
   - 3: Robótico pero entendible
   - 2: Muy robótico
   - 1: Incomprensible

2. **Expresividad** (1-5):
   - 5: Transmite emociones claramente
   - 4: Algo de entonación emocional
   - 3: Monótono pero correcto
   - 2: Completamente plano
   - 1: Sin expresión

3. **Tono** (1-5):
   - 5: Tono apropiado al contexto
   - 4: Generalmente apropiado
   - 3: Neutral sin problemas
   - 2: Inapropiado ocasionalmente
   - 1: Tono incorrecto

### Ejemplo de Evaluación

```python
# Después de correr el test, escucha las grabaciones:

# 1. Escuchar tu voz
# state/voice_test_temp/temp_input.wav

# 2. Escuchar respuesta de SARAi
# state/voice_test_temp/temp_tts.wav

# 3. Evaluar:
evaluacion = {
    "naturalidad": 3,      # Robótico pero entendible
    "expresividad": 2,     # Muy monótono
    "tono": 4,             # Apropiado al contexto
    "comprension": 5,      # Perfectamente comprensible
    "velocidad": 4         # Velocidad adecuada
}

promedio = sum(evaluacion.values()) / len(evaluacion)
print(f"Evaluación promedio: {promedio}/5.0")
# Salida: Evaluación promedio: 3.6/5.0
```

---

## 🛠️ Troubleshooting

### Error: "No default input device available"

**Problema**: No hay micrófono detectado

**Solución**:
```bash
# Listar dispositivos de audio
python -c "import sounddevice as sd; print(sd.query_devices())"

# Configurar dispositivo por defecto
export AUDIODEV=hw:0,0  # Linux
```

### Error: "PortAudio not found"

**Problema**: pyaudio no instalado correctamente

**Solución**:
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio
```

### Latencia muy alta (> 10s)

**Posibles causas**:
1. LFM2 en CPU lento → Probar con GPU
2. Whisper modelo grande → Usar `whisper.load_model("tiny")`
3. TTS lento → Cambiar a pyttsx3 básico

**Optimización rápida**:
```python
# En test_voice_quick.py, cambiar:
self.stt = whisper.load_model("tiny")  # En lugar de "base"
```

### Audio distorsionado

**Problema**: TTS genera audio con ruido

**Solución**:
1. Reducir velocidad TTS: `self.tts.setProperty('rate', 120)`
2. Verificar volumen: `self.tts.setProperty('volume', 0.7)`
3. Probar TTS alternativo (Coqui, VITS)

---

## 📝 Logs y Reportes

### Ubicación de Logs

```
logs/voice_conversation_TIMESTAMP.json
```

### Formato del Log

```json
{
  "timestamp": "20251030_143022",
  "stats": {
    "stt": {
      "mean": 1180.5,
      "min": 1050.2,
      "max": 1320.8,
      "std": 95.3
    },
    "llm": {
      "mean": 905.2,
      "min": 850.1,
      "max": 970.5,
      "std": 42.1
    },
    "tts": {
      "mean": 365.7,
      "min": 330.2,
      "max": 410.3,
      "std": 28.4
    },
    "e2e": {
      "mean": 5850.3,
      "min": 5400.1,
      "max": 6280.5,
      "std": 320.2
    },
    "total_conversation_time_ms": 32500,
    "turns_completed": 5
  },
  "conversation": [
    {
      "turn": 1,
      "timestamp": "2025-10-30T14:30:22.123456",
      "user_text": "Hola, ¿cómo estás?",
      "response_text": "Hola! Estoy bien, gracias.",
      "latencies": {
        "recording": 3500,
        "stt": 1150,
        "llm": 920,
        "tts": 380,
        "total_e2e": 5950
      },
      "audio_quality": {
        "mos_estimated": 3.8,
        "quality": "good"
      }
    }
    // ... más turnos
  ]
}
```

### Análisis de Logs

```python
import json
import numpy as np

# Cargar log
with open("logs/voice_conversation_20251030_143022.json") as f:
    data = json.load(f)

# Análisis de latencias
e2e_latencies = [turn["latencies"]["total_e2e"] for turn in data["conversation"]]
print(f"E2E Promedio: {np.mean(e2e_latencies):.0f}ms")
print(f"E2E P95: {np.percentile(e2e_latencies, 95):.0f}ms")

# Análisis de calidad
mos_scores = [turn["audio_quality"]["mos_estimated"] for turn in data["conversation"]]
print(f"MOS Promedio: {np.mean(mos_scores):.2f}/5.0")
```

---

## 🎯 Mejoras Sugeridas (Basado en Resultados)

### Si Latencia E2E > 3s (sin contar grabación)

1. **Optimizar LLM**:
   ```bash
   # Probar modelo más pequeño
   # LFM2-1.2B → TinyLLM-0.5B
   
   # O usar GPU
   llm = Llama(
       model_path=str(llm_path),
       n_gpu_layers=32,  # Cargar capas en GPU
       n_ctx=512
   )
   ```

2. **Optimizar STT**:
   ```python
   # Usar Whisper tiny
   self.stt = whisper.load_model("tiny")
   
   # O Whisper.cpp (más rápido)
   # pip install whisper-cpp-python
   ```

3. **Optimizar TTS**:
   ```python
   # Usar TTS más rápido
   # pyttsx3 → Coqui TTS → StyleTTS2
   ```

### Si MOS < 3.5

1. **Mejorar TTS**:
   - Cambiar de pyttsx3 a Coqui TTS
   - Usar voces pre-entrenadas de mejor calidad
   - Ajustar sample rate (16kHz → 22.05kHz)

2. **Pipeline ONNX** (si no está activo):
   - Activar pipeline modular ONNX
   - Usar qwen25_7b_audio.onnx para mejor calidad

### Si STT tiene errores frecuentes

1. **Mejorar grabación**:
   - Usar micrófono de mejor calidad
   - Reducir ruido ambiente
   - Aumentar volumen de grabación

2. **Mejorar modelo STT**:
   ```python
   # Usar Whisper medium o large
   self.stt = whisper.load_model("medium")
   
   # O fine-tunar Whisper en tu voz
   ```

---

## 📋 Checklist de Evaluación

Después de correr los tests, completar:

### Latencia
- [ ] E2E < 3s (sin grabación)
- [ ] STT < 1s
- [ ] LLM < 1s
- [ ] TTS < 500ms

### Calidad
- [ ] MOS > 3.5
- [ ] Comprensión > 90%
- [ ] Sin distorsión de audio
- [ ] Velocidad de habla natural

### Expresividad
- [ ] Tono apropiado al contexto
- [ ] Algo de expresión emocional
- [ ] No completamente monótono

### Experiencia de Usuario
- [ ] Respuestas coherentes
- [ ] Sin interrupciones
- [ ] Audio claro y comprensible

**Si todos los checkmarks están marcados**: ✅ Listo para uso real!

**Si faltan algunos**: Revisar secciones de "Mejoras Sugeridas"

---

## 🚀 Próximos Pasos

1. **Ejecutar test rápido**:
   ```bash
   python tests/test_voice_quick.py
   ```

2. **Analizar resultados**:
   - Revisar latencias
   - Escuchar grabaciones en `state/voice_test_temp/`
   - Evaluar calidad de voz manualmente

3. **Documentar hallazgos**:
   - Crear issue en GitHub con métricas
   - Adjuntar grabaciones de ejemplo
   - Sugerir mejoras específicas

4. **Iterar**:
   - Implementar mejoras sugeridas
   - Re-ejecutar tests
   - Comparar con resultados anteriores

---

**Buena suerte con las pruebas! 🎙️**
