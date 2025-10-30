# Guía de Benchmark: Qwen3-VL-4B-Instruct vs Qwen3-Omni-30B

**Fecha**: 29 octubre 2025  
**Objetivo**: Validar empíricamente si Qwen3-VL-4B-Instruct alcanza latencia <240ms

---

## 🎯 Hipótesis a Validar

### Teoría

| Modelo | Parámetros | Latencia Esperada | WER Esperado | MOS Esperado |
|--------|------------|-------------------|--------------|--------------|
| **Qwen3-VL-4B-Instruct** | 3B | **190-240ms** ✅ | 2.0% | 4.21/4.38 |
| **Qwen3-Omni-30B** | 30B | **2870ms** ❌ | 1.8% | 4.32/4.50 |

**Pregunta crítica**: ¿La latencia teórica de 3B (~190ms) se cumple en la práctica?

---

## 📥 Paso 1: Obtener el Modelo ONNX

### Opción A: Descargar desde HuggingFace (si existe)

```bash
# Buscar modelo en HuggingFace
https://huggingface.co/models?search=Qwen3-VL-4B-Instruct

# Si existe versión ONNX directa:
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct-ONNX \
    --local-dir models/onnx/ \
    --include "*.onnx" "*.onnx.data"
```

### Opción B: Convertir desde PyTorch (si solo existe PyTorch)

```bash
# 1. Descargar modelo PyTorch
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct-Instruct \
    --local-dir models/pytorch/Qwen3-VL-4B-Instruct/

# 2. Convertir a ONNX
python scripts/convert_to_onnx.py \
    --model models/pytorch/Qwen3-VL-4B-Instruct/ \
    --output models/onnx/Qwen3-VL-4B-Instruct.onnx \
    --optimize

# 3. Cuantizar a INT8 (opcional, para mejorar latencia)
python scripts/quantize_int8.py \
    --model models/onnx/Qwen3-VL-4B-Instruct.onnx \
    --output models/onnx/Qwen3-VL-4B-Instruct-int8.onnx
```

### Opción C: Archivo compartido

Si lo tienes de otra fuente:

```bash
# Copiar a la ubicación correcta
cp /path/to/Qwen3-VL-4B-Instruct.onnx models/onnx/
cp /path/to/Qwen3-VL-4B-Instruct.onnx.data models/onnx/  # Si existe archivo .data

# Verificar integridad
ls -lh models/onnx/Qwen3-VL-4B-Instruct*
```

---

## 🧪 Paso 2: Ejecutar Benchmark Individual

### Test del Modelo 3B (cuando lo tengas)

```bash
cd /home/noel/SARAi_v2

# Benchmark completo
python scripts/benchmark_audio_latency.py \
    --model models/onnx/Qwen3-VL-4B-Instruct.onnx \
    --iterations 20

# Output esperado:
# ⏱️  Tiempos de Carga:
#    Carga modelo:  XX.XXs
#    Warmup (avg):  XX.XXs
# 
# ⚡ Latencia de Inferencia:
#    P50 promedio:  XXX.Xms  <-- ESTO ES CRÍTICO
#    P99 promedio:  XXX.Xms
# 
# 🎯 Objetivo: <240ms
#    ✅ CUMPLE o ❌ NO CUMPLE
```

### Test del Modelo 30B (baseline actual)

```bash
# Para comparación
python scripts/benchmark_audio_latency.py \
    --model models/onnx/agi_audio_core_int8.onnx \
    --iterations 20

# Resultado esperado: P50 ~2870ms ❌
```

---

## 📊 Paso 3: Comparación Directa

### Benchmark Lado a Lado

```bash
python scripts/benchmark_audio_latency.py --compare \
    --model-a models/onnx/Qwen3-VL-4B-Instruct.onnx \
    --model-b models/onnx/agi_audio_core_int8.onnx \
    --iterations 20
```

**Output esperado**:

```
📊 TABLA COMPARATIVA
================================================================================

Métrica                   Modelo A             Modelo B             Diferencia
--------------------------------------------------------------------------------
Modelo                    Qwen3-VL-4B-Instruct      agi_audio_core_int8  N/A
Carga (s)                 XX.XX                26.88                ⬇️ -X.XX (-X%)
Warmup (s)                XX.XX                47.35                ⬇️ -X.XX (-X%)
Latencia P50 (ms)         XXX.X                2872.0               ⬇️ -XXXX (-XX%) 🎯
Latencia P99 (ms)         XXX.X                XXXX.X               ⬇️ -XXXX (-XX%)
Cumple <240ms             ✅ SÍ/❌ NO          ❌ NO                N/A
Throughput (inf/s)        X.XX                 0.35                 ⬆️ +X.XX (+XX%)

================================================================================

🎯 VEREDICTO:
   ✅/❌ Modelo A (Qwen3-VL-4B-Instruct) CUMPLE/NO CUMPLE objetivo (<240ms)
   ❌ Modelo B (agi_audio_core_int8) NO cumple
   → Recomendación: ...
```

---

## ✅ Paso 4: Decisión Basada en Datos

### Escenario A: 3B CUMPLE (<240ms) ✅

**Si P50 de Qwen3-VL-4B-Instruct < 240ms**:

```
✅ DECISIÓN: Usar Qwen3-VL-4B-Instruct como modelo principal

Próximos pasos:
1. Integrar Qwen3-VL-4B-Instruct.onnx en audio_omni_pipeline.py
2. Validar calidad (WER/MOS) con audio real
3. Mantener Qwen3-Omni-30B como modelo opcional de máxima calidad
4. Implementar sistema híbrido:
   - DEFAULT: Qwen3-VL-4B-Instruct (latencia óptima)
   - SWAP: Qwen3-Omni-30B (calidad máxima bajo demanda)

Timeline: 4-6 horas de integración
```

### Escenario B: 3B NO CUMPLE (>240ms) ❌

**Si P50 de Qwen3-VL-4B-Instruct > 240ms**:

```
❌ DECISIÓN: Modelo 3B tampoco es viable para real-time

Próximos pasos:
1. Implementar pipeline desagregado:
   - Whisper-small (STT): ~80ms
   - Piper TTS: ~60ms
   - Total: ~140ms ✅

2. Mantener ambos Qwen como opciones de calidad:
   - Qwen3-VL-4B-Instruct: Calidad media (batch)
   - Qwen3-Omni-30B: Calidad máxima (crítico)

3. Arquitectura tri-tier:
   - FAST: Whisper-small + Piper (~140ms)
   - QUALITY: Qwen3-VL-4B-Instruct (~XXXms, si <1s)
   - MAX: Qwen3-Omni-30B (~2870ms)

Timeline: 1-2 días de implementación
```

### Escenario C: 3B MEJORA pero NO SUFICIENTE (240-500ms) ⚠️

**Si 240ms < P50 < 500ms**:

```
⚠️ DECISIÓN: Modelo 3B es mejor que 30B pero aún lento

Análisis:
- ¿Es aceptable 300-400ms de latencia? (límite perceptible)
- Usuario puede tolerar leve pausa vs pipeline desagregado
- Calidad integral (STT+TTS unificado) vs componentes separados

Opciones:
A) Aceptar latencia 240-500ms si calidad justifica
B) Usar 3B como tier medio + pipeline ligero para real-time
C) Optimizar 3B más (cuantización agresiva, pruning)

Timeline: Decisión subjetiva basada en UX testing
```

---

## 🔍 Métricas a Capturar

### Durante el Benchmark

1. **Latencia P50/P90/P99**: Distribución de tiempos de respuesta
2. **Carga del modelo**: Tiempo inicial (one-time cost)
3. **Warmup**: Compilación de kernels (one-time)
4. **Throughput**: Inferencias por segundo
5. **RAM usage**: Consumo de memoria (usar `htop` en paralelo)

### Post-Benchmark (con audio real)

6. **WER (Word Error Rate)**: Calidad de transcripción
7. **MOS (Mean Opinion Score)**: Calidad de TTS (requiere evaluadores)
8. **Robustez ante ruido**: Test con audio ruidoso
9. **Empatía en TTS**: Capacidad de prosodia emocional

---

## 📝 Template de Reporte de Resultados

```markdown
# Resultados Benchmark: Qwen3-VL-4B-Instruct

**Fecha**: [Fecha del test]
**Hardware**: i7 quad-core, 16GB RAM, CPU-only
**ONNX Runtime**: [versión]

## Modelo Testeado

- Nombre: Qwen3-VL-4B-Instruct
- Path: models/onnx/Qwen3-VL-4B-Instruct.onnx
- Tamaño: [X.X GB]
- Formato: ONNX [FP32/INT8]

## Resultados de Latencia

| Métrica | Valor | Objetivo | ✅/❌ |
|---------|-------|----------|-------|
| Carga modelo | XX.XXs | N/A | - |
| Warmup | XX.XXs | N/A | - |
| **Latencia P50** | **XXX.Xms** | **<240ms** | **✅/❌** |
| Latencia P90 | XXX.Xms | <300ms | ✅/❌ |
| Latencia P99 | XXX.Xms | <400ms | ✅/❌ |
| Throughput | X.XX inf/s | >4 inf/s | ✅/❌ |

## Comparación vs Qwen3-Omni-30B

| Métrica | 3B | 30B | Mejora |
|---------|----|----|--------|
| Latencia P50 | XXXms | 2870ms | -XX% |
| RAM | X.XGB | 1.1GB | +/-X% |
| Calidad (esperada) | WER 2.0% | WER 1.8% | -10% |

## Decisión

[✅ VIABLE / ❌ NO VIABLE / ⚠️ REQUIERE ANÁLISIS]

**Justificación**:
[Explicar decisión basada en datos]

**Próximos pasos**:
1. [Acción 1]
2. [Acción 2]
3. [Acción 3]
```

---

## 🚀 Checklist de Validación

Antes de tomar decisión final:

- [ ] Modelo ONNX descargado y verificado
- [ ] Benchmark de latencia ejecutado (≥20 iteraciones)
- [ ] Comparación con Qwen3-Omni-30B realizada
- [ ] RAM usage medido (<4GB objetivo)
- [ ] Latencia P50 < 240ms validada (o documentada si no cumple)
- [ ] Test con audio real (no solo dummy input)
- [ ] Decisión documentada en reporte
- [ ] Timeline de implementación definido

---

## 💡 Tips de Optimización (si latencia es borderline)

Si Qwen3-VL-4B-Instruct está cerca pero no cumple (<300ms):

### 1. Cuantización Agresiva

```bash
# INT8 si está en FP32
python scripts/quantize_int8.py --model Qwen3-VL-4B-Instruct.onnx

# O incluso INT4 (pérdida de calidad mínima en algunos modelos)
python scripts/quantize_int4.py --model Qwen3-VL-4B-Instruct.onnx
```

**Ganancia esperada**: -20-30% latencia

### 2. Graph Optimization Extra

```python
# En el benchmark, añadir:
sess_options.add_session_config_entry("session.disable_prepacking", "0")
sess_options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
```

**Ganancia esperada**: -5-10% latencia

### 3. Reducir Resolución de Audio

```python
# Si modelo acepta menor sample rate
# 16kHz → 8kHz (solo si calidad no se degrada mucho)
```

**Ganancia esperada**: -15-25% latencia

---

## ❓ FAQ

**P: ¿Qué hago si el modelo no existe en ONNX?**

R: Opciones:
1. Contactar a Qwen/Alibaba para versión ONNX oficial
2. Convertir desde PyTorch (requiere código del modelo)
3. Usar alternativa (Whisper-small + Piper)

**P: ¿240ms es un límite estricto?**

R: No, es guideline. 190-300ms es aceptable para conversación. >500ms ya se siente lento.

**P: ¿Y si 3B tampoco cumple?**

R: Implementar pipeline desagregado (Whisper + Piper) garantiza <200ms.

**P: ¿Puedo usar GPU para acelerar?**

R: Sí, pero entonces la comparación con 30B debe ser también en GPU (fair comparison).

---

**Siguiente paso**: Cuando tengas el modelo ONNX, ejecuta:

```bash
python scripts/benchmark_audio_latency.py \
    --model models/onnx/Qwen3-VL-4B-Instruct.onnx \
    --iterations 20
```

Y comparte los resultados. Tomaremos la decisión basada en **datos reales**, no teoría. 🎯
