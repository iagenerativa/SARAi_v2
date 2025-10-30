# 🎤 Solución Rápida: SARAi con Voz (Piper TTS)

**Fecha**: 30 de Octubre de 2025  
**Tiempo de implementación**: ~10 minutos  
**Estado**: ✅ **READY TO IMPLEMENT**

---

## 🎯 Plan

Usar **Piper TTS** (ONNX, ultra-rápido, español nativo) como generador de voz mientras investigamos el Audio Quantizer.

---

## 📦 Instalación

```bash
cd /home/noel/SARAi_v2

# 1. Instalar Piper
pip install piper-tts

# 2. Descargar modelo español (100MB)
mkdir -p models/tts
cd models/tts

wget https://github.com/rhasspy/piper/releases/download/v1.2.0/es_ES-davefx-medium.tar.gz
tar -xzf es_ES-davefx-medium.tar.gz

# Resultado:
# models/tts/es_ES-davefx-medium.onnx
# models/tts/es_ES-davefx-medium.onnx.json
```

---

## 🔧 Integración

### Crear `agents/piper_tts.py`

```python
"""
Piper TTS Agent para SARAi
Genera voz en español de alta calidad
"""

import wave
import numpy as np
from pathlib import Path


class PiperTTSEngine:
    """Motor TTS basado en Piper (ONNX)"""
    
    def __init__(self, model_path: str = None):
        try:
            from piper import PiperVoice
            
            if model_path is None:
                base_path = Path(__file__).parent.parent
                model_path = base_path / "models/tts/es_ES-davefx-medium.onnx"
            
            self.voice = PiperVoice.load(str(model_path))
            self.sample_rate = 22050  # Piper usa 22kHz
            
        except ImportError:
            raise ImportError(
                "piper-tts no instalado. Ejecuta: pip install piper-tts"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Modelo no encontrado en {model_path}. "
                "Descarga con: wget https://github.com/rhasspy/piper/releases/download/v1.2.0/es_ES-davefx-medium.tar.gz"
            )
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Genera audio desde texto
        
        Args:
            text: Texto en español
        
        Returns:
            audio: numpy array (float32, [-1, 1])
        """
        # Piper retorna audio directamente
        audio = self.voice.synthesize(text)
        
        # Convertir a numpy si es necesario
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        
        return audio
    
    def synthesize_to_file(self, text: str, output_path: str) -> float:
        """
        Genera audio y guarda en archivo WAV
        
        Returns:
            latency_ms: Latencia en milisegundos
        """
        import time
        
        start = time.perf_counter()
        audio = self.synthesize(text)
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Guardar como WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return latency_ms


# Test standalone
if __name__ == "__main__":
    import sys
    
    print("🎤 Test Piper TTS\n")
    
    engine = PiperTTSEngine()
    print("✅ Motor cargado\n")
    
    # Test 1: Frase corta
    text1 = "Hola, soy SARAi, tu asistente de voz en español."
    print(f"Texto 1: {text1}")
    latency = engine.synthesize_to_file(text1, "test_piper_1.wav")
    print(f"✓ Audio generado en {latency:.1f}ms")
    print(f"  Archivo: test_piper_1.wav\n")
    
    # Test 2: Frase técnica
    text2 = "El pipeline de voz tiene una latencia de mil trescientos veinte milisegundos."
    print(f"Texto 2: {text2}")
    latency = engine.synthesize_to_file(text2, "test_piper_2.wav")
    print(f"✓ Audio generado en {latency:.1f}ms")
    print(f"  Archivo: test_piper_2.wav\n")
    
    print("🔊 Escucha los audios:")
    print("  aplay test_piper_1.wav")
    print("  aplay test_piper_2.wav")
```

---

### Modificar `tests/test_voice_realtime.py`

Cambiar de Token2Wav a Piper TTS:

```python
# En load_components(), reemplazar Token2Wav con:

# 4. Piper TTS (Voz Española)
print(f"{C.C}[4/4] Piper TTS (Voz Española)...{C.E}", end=' ')
from agents.piper_tts import PiperTTSEngine

self.components['tts'] = PiperTTSEngine()
print(f"{C.G}✓{C.E}")

# En process_turn(), reemplazar sección Token2Wav con:

# 6. TTS (Generación de Voz desde Texto)
print(f"{C.C}[5/5] Piper TTS → Voz en Español...{C.E}", end=' ')
start = time.perf_counter()

# Generar audio desde el texto de LFM2
output_file = f"logs/audio_output_{timestamp}.wav"
latency_tts = self.components['tts'].synthesize_to_file(
    response_text,  # Del LFM2
    output_file
)

latencies['tts'] = latency_tts
print(f"{C.G}{latencies['tts']:.1f}ms{C.E}")
print(f"{C.M}   🔊 Audio guardado: {output_file}{C.E}")
```

---

## 🎯 Pipeline Modificado

```
Audio Input (Micrófono)
    ↓
Features Sintéticas [1, 100, 512]
    ↓ [~8ms]
Projection ONNX [1, 100, 3584]
    ↓ [~1250ms]
LFM2-1.2B → TEXTO de respuesta
    ↓ [~50ms]
Piper TTS → Audio WAV
    ↓
🔊 Voz en Español
```

**Latencia Total**: ~1310ms (casi igual que con Token2Wav)

---

## ⚡ Ventajas de Piper

| Aspecto | Piper TTS | Token2Wav (bloqueado) |
|---------|-----------|----------------------|
| **Latencia** | ~50ms | ~50ms (si funcionara) |
| **Calidad** | MOS 4.0 | MOS 4.21 |
| **Instalación** | 1 comando | Ya instalado |
| **Funciona HOY** | ✅ SÍ | ❌ NO (falta quantizer) |
| **Español** | ✅ Nativo | ✅ Nativo |
| **Tamaño** | 100MB | 858MB |
| **Dependencias** | Mínimas | Complejas |

---

## 📋 Checklist de Implementación

```bash
# 1. Instalar Piper
[ ] pip install piper-tts

# 2. Descargar modelo español
[ ] cd models && mkdir tts
[ ] wget https://github.com/rhasspy/piper/releases/download/v1.2.0/es_ES-davefx-medium.tar.gz
[ ] tar -xzf es_ES-davefx-medium.tar.gz

# 3. Crear agent
[ ] Crear agents/piper_tts.py

# 4. Test standalone
[ ] python agents/piper_tts.py
[ ] aplay test_piper_1.wav

# 5. Integrar en pipeline
[ ] Modificar tests/test_voice_realtime.py

# 6. Probar end-to-end
[ ] python tests/test_voice_realtime.py
[ ] Hablar → Escuchar respuesta
```

---

## 🎉 Resultado Esperado

```
TURNO 1:
🎤 Grabando 5s...
✓ Grabación completa

[1/5] Features: 1.4ms
[2/5] Projection: 7.6ms
[3/5] LFM2: 1276.0ms
[4/5] Talker: 16.8ms (descartado por ahora)
[5/5] Piper TTS: 52.3ms ✅

🔊 Audio guardado: logs/audio_output_20251030_154522.wav

📊 Resultados:
   Procesamiento: 1352.1ms
   Total E2E: 6352.1ms

💬 Respuesta LLM:
   "Hola, estoy bien. ¿En qué puedo ayudarte hoy?"

🔊 REPRODUCIR: aplay logs/audio_output_20251030_154522.wav
```

---

## 💡 Migración Futura a Token2Wav

Cuando encontremos/descarguemos el Audio Quantizer:

1. Integrar Quantizer entre Talker y Token2Wav
2. Cambiar `agents/piper_tts.py` por `agents/token2wav_agent.py`
3. Mantener Piper como fallback

**Código modular**: El cambio será de 10 líneas, sin afectar el resto.

---

**¿Procedemos con Piper TTS?** 🎤
