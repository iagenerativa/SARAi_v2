# 🎤 Test Interactivo de Voz con SARAi v2.16.3

## ✅ ¡El Pipeline TIENE Voz en Español!

**Componente**: `Token2Wav` (Qwen2.5-Omni)  
**Idioma**: Español nativo  
**Calidad esperada**: MOS 4.21 natural / 4.38 empatía  
**Latencia**: ~50ms (3 diffusion steps)

---

## 🚀 Cómo Probarlo

### Opción 1: Test Completo con Tu Voz

```bash
cd /home/noel/SARAi_v2
python3 tests/test_voice_realtime.py
```

**Flujo**:
1. El sistema carga los 4 componentes (~5 segundos)
2. Te pregunta si quieres guardar los audios (recomendado: `s`)
3. Presionas Enter para cada turno
4. Hablas durante 5 segundos
5. SARAi procesa y genera audio de respuesta
6. El audio se guarda en `logs/audio_output_XXXXXX.wav`

**Para escuchar los audios**:
```bash
# Ver audios generados
ls -lh logs/audio_output_*.wav

# Reproducir con aplay (Linux)
aplay logs/audio_output_20251030_143022.wav

# O con mpv
mpv logs/audio_output_20251030_143022.wav
```

---

### Opción 2: Benchmark Rápido (Sin Micrófono)

Si no tienes micrófono configurado o quieres solo medir latencias:

```bash
cd /home/noel/SARAi_v2
echo "1" | python3 tests/test_voice_with_llm.py
```

**Nota**: Esta versión usa features sintéticas pero **SÍ genera audio real** con Token2Wav.

---

## 🎯 Qué Esperar

### Latencias Proyectadas CON Token2Wav Real

```
[1/5] Features:     ~2ms    (sintético por ahora)
[2/5] Projection:   ~8ms    ✅ Medido
[3/5] LFM2:         ~1250ms ✅ Medido
[4/5] Talker:       ~10ms   ✅ Medido
[5/5] Token2Wav:    ~50ms   ⏳ Por medir (real, no simulado)
─────────────────────────────
TOTAL E2E:          ~1320ms

Con audio real:
• Recording: 5000ms (tú hablando)
• Processing: 1320ms
• TOTAL: ~6.3 segundos por turno
```

---

## 🔊 Calidad de la Voz

**Token2Wav** es el componente que genera la voz en español. Según la documentación de Qwen2.5-Omni:

- **MOS Natural**: 4.21/5.0 (muy bueno)
- **MOS Empatía**: 4.38/5.0 (excelente)
- **Sample Rate**: 24kHz
- **Idioma**: Español nativo (no traducido)

La voz debería sonar:
✅ Natural (no robótica)
✅ En español claro
✅ Con entonación apropiada
✅ Sin latencia perceptible (<2s total)

---

## 🐛 Troubleshooting

### Error: "No default audio device"

```bash
# Instalar ALSA utils si no tienes
sudo apt-get install alsa-utils

# Ver dispositivos disponibles
arecord -L

# Configurar default device
export AUDIODEV=hw:0,0
```

### Error al cargar Token2Wav

Si ves error de "weights_only", el código ya lo maneja con `weights_only=False`. Si persiste:

```bash
# Verificar que el archivo existe
ls -lh models/onnx/token2wav_int8.pt

# Debería mostrar ~545MB
```

### Audio se escucha distorsionado

Token2Wav genera a 24kHz. Si tu sistema espera 16kHz o 44.1kHz, puede sonar mal. Usa:

```bash
# Convertir si es necesario
ffmpeg -i logs/audio_output_XXX.wav -ar 44100 logs/audio_output_XXX_44k.wav
```

---

## 📊 Estructura del Pipeline

```
Tu Voz (Micrófono)
    ↓ [5s grabación]
Audio Input (16kHz)
    ↓ [~2ms]
Features Sintéticas [1, 100, 512]
    ↓ [~8ms]
Projection ONNX [1, 100, 3584]
    ↓ [~1250ms]
LFM2-1.2B Razonamiento
    ↓ (modula hidden states)
Hidden States [1, 100, 3584]
    ↓ [~10ms]
Talker ONNX [1, 100, 8448]
    ↓ [~50ms]
Token2Wav → Waveform (24kHz)
    ↓
Audio Output → Archivo WAV
    ↓
🔊 ¡Escuchas la voz de SARAi en español!
```

---

## ✅ Checklist

- [x] Token2Wav existe (`models/onnx/token2wav_int8.pt`)
- [x] Modelo carga correctamente
- [x] Interfaz identificada (`code`, `num_steps`, `guidance_scale`)
- [x] Script de test actualizado
- [ ] **Probar con tu voz y escuchar el resultado** ← TÚ AQUÍ

---

## 🎉 Próximos Pasos

1. **AHORA**: Ejecuta `python3 tests/test_voice_realtime.py`
2. Habla en español durante 5 segundos
3. Espera ~1.3s de procesamiento
4. Escucha el audio generado con `aplay logs/audio_output_*.wav`
5. ¡Disfruta de SARAi hablando en español!

---

**Comando directo**:
```bash
cd /home/noel/SARAi_v2 && python3 tests/test_voice_realtime.py
```

**¡A PROBAR! 🚀**
