# 🚨 Full-Duplex: Problemas Críticos y Soluciones

## 📋 Problemas Identificados

### 1. ❌ Sistema Bloqueante (No Interactivo)

**Problema**: 
```python
# output_thread.py línea 369
self._wait_for_user_silence()  # BLOQUEA TODO
```

**Flujo actual**:
```
Usuario habla → [ESPERA 5s] → STT → [ESPERA silencio] → TTS → [BLOQUEA 1-2s] → Audio
                     ↑                      ↑                       ↑
                 BLOQUEANTE            BLOQUEANTE              BLOQUEANTE
```

**Impacto**:
- ❌ Conversación por turnos rígidos
- ❌ Silencios absurdos
- ❌ No puede interrumpir
- ❌ No parece natural

---

### 2. ❌ TTS Bloquea STT

**Problema**:
```python
# output_thread.py línea 372-374
audio_output = self.melo_tts.synthesize(response_text)  # 600-700ms BLOQUEO
# Durante este tiempo, InputThread NO puede procesar audio del usuario
```

**Consecuencia**:
- Usuario habla → Se pierde
- STT está congelado mientras TTS trabaja
- Primeras palabras del usuario desaparecen

---

### 3. ❌ Detección de Emociones Silenciosa

**Problema**:
```python
# input_thread.py línea 264
tone_result = self.audio_emotion.analyze(audio_data, sr=self.sample_rate)
# Si falla, no muestra error, solo pone "silencio"
```

**Debug necesario**:
```bash
# ¿Está cargado EmotionAudioLite?
# ¿Tiene el modelo descargado?
# ¿El audio tiene el formato correcto?
```

---

### 4. ❌ LoRA Ignora Experiencia

**Problema**:
```python
# lora_router.py - NO usa tone_result en la predicción
decision = self.lora_router.predict(embedding)  # Solo BERT, no tono
```

**Falta**:
- Integrar tono emocional en LoRA
- Usar ToneMemoryBuffer para contexto
- Adaptar respuesta según emoción detectada

---

### 5. ❌ Latencia STT Inicial

**Problema**:
- Vosk tarda ~1-2s en arrancar el recognizer
- Primeros chunks de audio se pierden
- Usuario tiene que repetir

---

## 🎯 Soluciones Propuestas

### FASE 1: Hacer Sistema No-Bloqueante (CRÍTICO)

#### 1.1 Eliminar `_wait_for_user_silence()` Bloqueante

**Antes**:
```python
self._wait_for_user_silence()  # Bloquea hasta 5s
audio_output = self.melo_tts.synthesize(response_text)  # Bloquea 600ms
```

**Después**:
```python
# TTS en thread separado, NO bloquea STT
threading.Thread(target=self._async_tts_and_play, args=(response_text,), daemon=True).start()
```

#### 1.2 TTS Asíncrono con Interrupción

```python
def _async_tts_and_play(self, text: str):
    """TTS en background, puede ser interrumpido si usuario habla"""
    
    # 1. Generar audio
    audio = self.melo_tts.synthesize(text)
    
    # 2. Reproducir en chunks pequeños (100ms)
    chunk_size = 4410  # 100ms a 44.1kHz
    for i in range(0, len(audio), chunk_size):
        
        # COMPROBAR: ¿Usuario empezó a hablar?
        if self.user_speaking:
            print("⚠️ Usuario interrumpió, deteniendo TTS")
            return  # ABORTAR reproducción
        
        # Reproducir chunk
        chunk = audio[i:i+chunk_size]
        self._play_audio_chunk(chunk)
```

**Beneficio**:
- ✅ Usuario puede interrumpir en cualquier momento
- ✅ STT siempre activo
- ✅ Conversación fluida

---

### FASE 2: Optimizar Latencias

#### 2.1 Pre-warm Vosk Recognizer

```python
# input_thread.py - load_components()
def load_components(self):
    self.vosk_stt = VoskSTTStreaming()
    self.vosk_session = VoskStreamingSession(self.vosk_stt)
    
    # PRE-WARM: Procesar 100ms de silencio
    dummy_audio = np.zeros(1600, dtype=np.float32)  # 100ms silencio
    self.vosk_session.feed_audio(dummy_audio)
    
    print("✅ Vosk pre-warmed, listo para audio real")
```

#### 2.2 TTS Streaming (Generación + Reproducción Simultánea)

```python
# MeloTTS no soporta streaming nativo, pero podemos:
# 1. Dividir texto en frases
# 2. Sintetizar primera frase
# 3. Mientras reproduce frase 1, sintetizar frase 2
# 4. Reducir latencia percibida ~50%
```

---

### FASE 3: Arreglar Detección de Emociones

#### 3.1 Debug EmotionAudioLite

```python
# input_thread.py - load_components()
def load_components(self):
    try:
        self.audio_emotion = EmotionAudioLite()
        
        # TEST: Audio de prueba
        test_audio = np.random.randn(16000).astype(np.float32) * 0.01
        test_result = self.audio_emotion.analyze(test_audio, sr=16000)
        
        print(f"✅ EmotionAudio funcionando: {test_result.label}")
        
    except Exception as e:
        print(f"❌ EmotionAudio FALLÓ: {e}")
        print("   Instalando dependencias...")
        # Fallback o instalación automática
```

#### 3.2 Logging de Emociones Detallado

```python
# input_thread.py - línea 264
tone_result = self.audio_emotion.analyze(audio_data, sr=self.sample_rate)

# AÑADIR DEBUG:
print(f"🎭 Emoción: {tone_result.label} ({tone_result.confidence:.2f})")
print(f"   Valence: {tone_result.valence:.2f}, Arousal: {tone_result.arousal:.2f}")
```

---

### FASE 4: Integrar Tono en LoRA

#### 4.1 Expandir Input de LoRA

**Antes**:
```python
decision = self.lora_router.predict(embedding)  # Solo BERT (768D)
```

**Después**:
```python
# Concatenar BERT + Tono emocional
tone_features = np.array([
    tone_result.valence,
    tone_result.arousal,
    tone_result.confidence
])

combined_input = np.concatenate([embedding, tone_features])  # 771D
decision = self.lora_router.predict(combined_input)
```

#### 4.2 Adaptar Respuesta según Emoción

```python
# Si usuario está triste → Respuesta más empática
if tone_result.label == "triste":
    # Activar estilo "empathy_support" en ToneBridge
    self.tone_bridge.set_active_style("empathy_support")

# Si usuario está enojado → Respuesta más calmada
elif tone_result.label == "enojado":
    self.tone_bridge.set_active_style("calming_neutral")
```

---

### FASE 5: Mejorar VAD (Voice Activity Detection)

#### 5.1 VAD Más Inteligente

**Problema actual**:
```python
# input_thread.py - línea 194
energy = np.sqrt(np.mean(audio_float32 ** 2))
if energy > self.vad_energy_threshold:  # Muy simple
```

**Mejora**:
```python
# Usar webrtcvad (más robusto)
import webrtcvad

vad = webrtcvad.Vad(2)  # Agresividad 0-3
is_speech = vad.is_speech(audio_bytes, sample_rate)
```

**Beneficio**:
- ✅ Detecta voz vs ruido de fondo
- ✅ Menos falsos positivos
- ✅ No corta por ruidos ambientales

---

## 📊 Comparativa: Antes vs Después

| Aspecto | Antes (Bloqueante) | Después (Asíncrono) |
|---------|-------------------|---------------------|
| **Interrupción** | ❌ Imposible | ✅ Inmediata |
| **Latencia respuesta** | 2-3s | 0.6-0.8s |
| **STT mientras TTS** | ❌ Bloqueado | ✅ Activo |
| **Emociones** | ❌ No funciona | ✅ Detectadas |
| **Naturalidad** | ⭐⭐ (2/5) | ⭐⭐⭐⭐⭐ (5/5) |

---

## 🚀 Plan de Implementación

### Prioridad 1 (HOY):
1. ✅ Ajustar VAD timeouts (YA HECHO)
2. 🔧 TTS asíncrono no-bloqueante
3. 🔧 Debug detección de emociones

### Prioridad 2 (MAÑANA):
4. 🔧 Pre-warm Vosk
5. 🔧 Integrar tono en LoRA
6. 🔧 Sistema de interrupción

### Prioridad 3 (OPCIONAL):
7. 🔧 TTS streaming por frases
8. 🔧 webrtcvad reemplazo
9. 🔧 Fillers mientras procesa

---

## 🎯 Objetivo Final

**Conversación Natural**:
```
Usuario: "Hola [pausa] ¿cómo estás?"
          ↓ (STT detecta en tiempo real)
SARAi:   [Empieza a responder ANTES de que termine]
         "Hola! Estoy bien, gracias por..."
          ↓
Usuario: [INTERRUMPE] "Espera, tengo una pregunta"
          ↓
SARAi:   [DETIENE audio inmediatamente]
         [PROCESA nueva pregunta]
```

**Características clave**:
- ✅ Respuesta rápida (< 1s percibido)
- ✅ Puede ser interrumpida
- ✅ Detecta emociones
- ✅ Adapta tono
- ✅ No bloquea STT
- ✅ Conversación fluida

---

**Próximo paso**: ¿Empezamos con TTS asíncrono + debug de emociones?
