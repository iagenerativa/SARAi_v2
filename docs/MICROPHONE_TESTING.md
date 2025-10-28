# 🎤 Testing Interactivo con Micrófono

## Descripción

SARAi v2.11 incluye tests interactivos que permiten validar la detección de idioma y emociones usando tu **micrófono real**.

## 📋 Requisitos

### Dependencias Python

```bash
pip install pyaudio scipy
```

### Hardware

- Micrófono funcional
- Permisos de acceso al micrófono (Linux: verificar con `arecord -l`)

## 🚀 Uso Rápido

### Opción 1: Script Helper (RECOMENDADO)

```bash
# Test de detección de idioma
python scripts/test_microphone.py --audio-router

# Test de detección emocional
python scripts/test_microphone.py --emotion

# Ambos tests
python scripts/test_microphone.py --all
```

### Opción 2: Pytest Directo

```bash
# Test de idioma
pytest tests/test_audio_router.py::TestLanguageDetector::test_detect_with_real_microphone -s

# Test de emoción
pytest tests/test_emotion_modulator.py::TestEmotionModulationIntegration::test_emotion_detection_with_real_microphone -s

# Ambos (filtro por keyword)
pytest tests/ -k real_microphone -s
```

## 📖 Guía de Tests

### Test 1: Detección de Idioma 🌍

**Objetivo**: Validar que el `LanguageDetector` identifica correctamente el idioma hablado.

**Proceso**:
1. El sistema grabará **5 segundos** de audio
2. Habla en cualquier idioma (español, inglés, francés, etc.)
3. El sistema mostrará el idioma detectado
4. Confirma si es correcto (y/n)

**Idiomas Soportados**:
- 🇪🇸 Español (es)
- 🇬🇧 English (en)
- 🇫🇷 Français (fr)
- 🇩🇪 Deutsch (de)
- 🇯🇵 日本語 (ja)
- 🇵🇹 Português (pt)
- 🇮🇹 Italiano (it)
- 🇷🇺 Русский (ru)
- 🇨🇳 中文 (zh)
- 🇸🇦 العربية (ar)
- 🇮🇳 हिन्दी (hi)
- 🇰🇷 한국어 (ko)

**Ejemplo de Output**:

```
🎤 TEST INTERACTIVO: Detección de idioma con micrófono
================================================================

📋 Configuración:
   - Duración: 5 segundos
   - Sample rate: 16000 Hz
   - Channels: 1

▶️  Presiona ENTER para comenzar a grabar...

🔴 GRABANDO... (habla en cualquier idioma)
██████████ ✅ Grabación completada

🔍 Detectando idioma...

================================================================
✅ RESULTADO: EN
================================================================

🌍 Idioma detectado: English (en)

❓ ¿Es correcto? (y/n): y
✅ Test PASSED - Detección correcta
```

### Test 2: Detección Emocional 🎭

**Objetivo**: Validar que el `EmotionModulator` identifica correctamente el estado emocional del hablante.

**Proceso**:
1. El sistema grabará **3 segundos** de audio
2. Expresa una emoción clara (alegría, tristeza, enojo, etc.)
3. El sistema analizará las características acústicas
4. Mostrará la emoción detectada + scores + features
5. Confirma si es correcto (y/n)

**Emociones Detectables**:
- 😊 HAPPY (feliz, alegre)
- 😢 SAD (triste, deprimido)
- 😠 ANGRY (enojado, furioso)
- 😨 FEARFUL (miedo, nervioso)
- 😮 SURPRISED (sorprendido)
- 🤢 DISGUSTED (asco)
- 😌 CALM (tranquilo, sereno)
- 🤩 EXCITED (emocionado, excitado)
- 😐 NEUTRAL (neutro)

**Ejemplos de Frases**:

```python
# Feliz
"¡Estoy muy contento hoy! Todo va genial"

# Triste
"Me siento mal... esto me deprime"

# Enojado
"¡Esto es frustrante! ¡No puede ser!"

# Tranquilo
"Todo está bien, estoy relajado"
```

**Ejemplo de Output**:

```
🎭 TEST INTERACTIVO: Detección de emoción con micrófono
======================================================================

📋 Instrucciones:
   1. Grabarás 3 segundos de audio
   2. Expresa una emoción clara (alegría, tristeza, enojo, etc.)
   3. El sistema detectará tu estado emocional

▶️  Presiona ENTER cuando estés listo...

🔴 GRABANDO... (expresa una emoción)
   💡 Ejemplos:
      - Feliz: 'Estoy muy contento hoy!'
      - Triste: 'Me siento mal...'
      - Enojado: '¡Esto es frustrante!'
      - Tranquilo: 'Todo está bien'

████████ ✅ Grabación completada

🔍 Analizando emoción del audio...

======================================================================
📊 RESULTADOS DE ANÁLISIS EMOCIONAL
======================================================================

🎭 Emoción Primaria: 😊 HAPPY
   Intensidad: 0.78 / 1.00
   Confianza: 0.65 / 1.00

📈 Top 3 Scores:
   1. happy        ██████████████████████████ 0.520
   2. excited      ████████████ 0.280
   3. neutral      ████ 0.200

🔊 Características Acústicas:
   Energía promedio: 0.3245
   Energía máxima: 0.8912
   Desviación std: 0.2134
   Zero-crossing rate: 0.0876

======================================================================
❓ ¿La detección es correcta? (y/n): y

✅ Test PASSED - Detección emocional correcta
   Emoción detectada: happy
   Confianza: 65.0%
```

## 🐛 Troubleshooting

### Error: "No module named 'pyaudio'"

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Fedora/RHEL
sudo dnf install portaudio-devel
pip install pyaudio
```

### Error: "ALSA lib ... cannot open shared library"

```bash
# Linux: Verificar permisos de audio
groups $USER  # Debe incluir 'audio'

# Si no está:
sudo usermod -a -G audio $USER
# Logout y login de nuevo
```

### Error: "Input overflowed"

Esto es normal si el buffer del micrófono se llena. El test continuará correctamente.

### Micrófono no detectado

```bash
# Listar dispositivos de audio
arecord -l

# Probar grabación manual
arecord -d 3 -f cd test.wav
aplay test.wav
```

## 📊 Interpretación de Resultados

### Detección de Idioma

**Confianza alta** (>0.8): El modelo está muy seguro del idioma detectado
**Confianza media** (0.5-0.8): Detección correcta pero con incertidumbre
**Confianza baja** (<0.5): Puede ser necesario más audio o contexto

### Detección Emocional

**Fase 1 (Actual)**: Heurísticas básicas
- Precisión esperada: ~60-70%
- Basado en energía, ZCR, keywords

**Fase 2 (Próximo)**: Modelo pre-entrenado (emoDBert)
- Precisión objetivo: ≥85%
- MOS Empatía: ≥4.0/5.0

### Características Acústicas

| Feature | Rango | Interpretación |
|---------|-------|----------------|
| Energía promedio | 0.0-1.0 | Volumen general (alto = excited/angry) |
| Energía máxima | 0.0-1.0 | Picos de volumen (alto = énfasis) |
| Desviación std | 0.0-0.5 | Variabilidad (alto = angry/fearful) |
| ZCR | 0.0-0.3 | Pitch/excitación (alto = excited) |

## 🎯 Casos de Uso

### Desarrollo

```bash
# Validar cambios en LanguageDetector
python scripts/test_microphone.py --audio-router

# Validar cambios en EmotionModulator
python scripts/test_microphone.py --emotion
```

### Testing de Regresión

```bash
# Ejecutar ambos tests antes de merge
python scripts/test_microphone.py --all
```

### Demostración

```bash
# Demo para stakeholders
python scripts/test_microphone.py --all
```

## 📚 Referencias

- **Whisper**: https://github.com/openai/whisper
- **fastText LID**: https://fasttext.cc/docs/en/language-identification.html
- **Emotion Detection**: Ekman, P. (1992). "An argument for basic emotions"

## 🔗 Archivos Relacionados

- `tests/test_audio_router.py`: Test de detección de idioma
- `tests/test_emotion_modulator.py`: Test de detección emocional
- `agents/audio_router.py`: Implementación LanguageDetector
- `agents/emotion_modulator.py`: Implementación EmotionModulator
- `scripts/test_microphone.py`: Script helper

---

**Autor**: SARAi Team  
**Versión**: v2.11  
**Fecha**: 2025-10-28
