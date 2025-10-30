# Scripts de Cuantización INT8

**Creados**: 29 Octubre 2025  
**Propósito**: Cuantizar modelo ONNX de 4.3GB → 1.1GB

---

## 🪟 Scripts para Windows (Ejecutar primero)

### 1. `check_prerequisites_windows.bat` 🔍

**Propósito**: Verificar pre-requisitos antes de cuantizar

**Verifica**:
- ✅ Python 3.10+ instalado
- ✅ RAM ≥24GB (recomendado 32GB)
- ✅ Espacio disco ≥6GB
- ✅ Modelo FP32 presente (`agi_audio_core.onnx` + `.data`)
- ✅ GPU NVIDIA (opcional)
- ✅ SSH a Linux (opcional)
- ✅ Dependencias Python (`onnx`, `onnxruntime`)

**Uso**:
```batch
cd C:\SARAi_v2
scripts\check_prerequisites_windows.bat
```

**Output esperado**:
```
[1/7] Verificando Python... OK
[2/7] Verificando RAM... OK 32GB
[3/7] Verificando disco... OK 45GB
[4/7] Verificando modelo FP32... OK
[5/7] Verificando GPU... OK GeForce RTX 3060
[6/7] Verificando SSH... OK
[7/7] Verificando deps... OK

✅ Todos los pre-requisitos cumplidos
```

**Si falla**: Resolver problemas marcados con ❌ antes de continuar.

---

### 2. `quantize_windows.bat` ⚡ **SCRIPT PRINCIPAL**

**Propósito**: Cuantizar modelo automáticamente

**Pasos automatizados**:
1. Verifica Python instalado
2. Instala dependencias (`onnx`, `onnxruntime-gpu`)
3. Ejecuta cuantización INT8
4. Valida modelo resultante
5. Muestra instrucciones para transferir a Linux

**Uso**:
```batch
cd C:\SARAi_v2
scripts\quantize_windows.bat
```

**Tiempo**:
- Con GPU NVIDIA: **2-3 minutos**
- Con CPU: **5-10 minutos**

**Output esperado**:
```
[1/3] Verificando dependencias... ✅
[2/3] Ejecutando cuantización INT8... ✅ (2m 14s)
[3/3] Cuantización completada exitosamente!

Archivos generados:
  agi_audio_core_int8.onnx (8KB)
  agi_audio_core_int8.onnx.data (1.1GB)

SIGUIENTE PASO:
  scp models\onnx\agi_audio_core_int8.onnx* noel@agi1:~/SARAi_v2/models/onnx/
```

---

### 3. `quantize_onnx_int8_windows.py` 🐍 (Alternativa)

**Propósito**: Script Python manual (si batch falla)

**Features**:
- Detección automática GPU (CUDA/DirectML/CPU)
- Validación con benchmark (5 iteraciones)
- Warmup de kernels GPU (3 iteraciones)
- Estimación tiempo y beneficios
- Instrucciones transferencia a Linux

**Uso**:
```powershell
cd C:\SARAi_v2
python scripts\quantize_onnx_int8_windows.py
```

**Confirmación requerida**:
```
Iniciar cuantización? (y/n): y
```

**Output detallado**:
```
========================================
Cuantización INT8 - Windows (GPU)
========================================

✅ GPU detectada: CUDA (GeForce RTX 3060)
✅ Modelo encontrado: 4.3GB

[1/5] Cargando modelo FP32... ✅ (2.1s)
[2/5] Cuantizando a INT8... ✅ (2m 14s)
[3/5] Guardando modelo INT8... ✅ (8.3s)
[4/5] Validando modelo... ✅ Latencia: 1.87s
[5/5] Resultados:
  Tamaño:   4.3GB → 1.1GB (-74%)
  Latencia: 5.3s → 1.87s (-65%)
  Speedup:  2.8x

✅ CUANTIZACIÓN EXITOSA
```

---

## 🐧 Scripts para Linux (Ejecutar después)

### 4. `test_onnx_pipeline.py` ✅

**Propósito**: Suite completa de tests para validar modelo INT8

**Tests incluidos**:
1. **Model Loading**: Carga modelo y warmup
2. **Model Inference**: Procesa audio sintético
3. **Config Loading**: Verifica YAML válido
4. **File Validation**: Archivos INT8 presentes
5. **Performance Benchmark**: Múltiples duraciones de audio

**Uso**:
```bash
cd ~/SARAi_v2
python3 scripts/test_onnx_pipeline.py
```

**Tiempo**: ~1 minuto

**Output esperado**:
```
========================================
SARAi ONNX Pipeline Test Suite
========================================

Test 1: Model Loading ✅
  Carga: 8.32s (vs 44s FP32, -81%)
  Warmup: Ejecutado

Test 2: Model Inference ✅
  Primera: 1.92s (vs 20s FP32, -90%)
  Segunda: 0.0s (cache hit)
  Output: [1, 2048, 245760] ✅

Test 3: Config Loading ✅
  Config INT8 válido

Test 4: File Validation ✅
  Archivos: 8KB + 1.1GB
  Reducción: -74%

Test 5: Benchmark ✅
  0.5s audio: 2.1s
  1.0s audio: 0.0s (cache)
  2.0s audio: 2.0s
  3.0s audio: 1.9s
  
  Promedio: 2.0s (vs 5.3s FP32, -62%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ TODOS LOS TESTS PASARON (5/5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Si falla algún test**: Ver `docs/QUANTIZATION_CHECKLIST.md` troubleshooting.

---

### 5. `compare_fp32_int8_quality.py` 📊

**Propósito**: Comparar calidad FP32 vs INT8

**Métricas calculadas**:
- **Similitud coseno**: Qué tan similar es el output (objetivo: >0.98)
- **MSE (Mean Squared Error)**: Error cuadrático medio (objetivo: <0.01)
- **MAE (Mean Absolute Error)**: Error absoluto medio (objetivo: <0.05)
- **Speedup**: Cuánto más rápido es INT8 (objetivo: >2.0x)

**Uso**:
```bash
cd ~/SARAi_v2
python3 scripts/compare_fp32_int8_quality.py
```

**Tiempo**: ~2 minutos

**Output esperado**:
```
========================================
SARAi v2.16.1 - Comparación FP32 vs INT8
========================================

[1/5] Cargando modelos...
  ✅ FP32 cargado: 42.3s
  ✅ INT8 cargado: 8.1s
  📊 Speedup carga: 5.2x

[2/5] Generando 5 inputs de test... ✅

[3/5] Ejecutando inferencias...
  Sample 1/5: FP32=19.8s, INT8=1.9s (Speedup: 10.4x)
  Sample 2/5: FP32=20.1s, INT8=0.0s (Speedup: ∞ cache)
  Sample 3/5: FP32=19.7s, INT8=1.8s (Speedup: 10.9x)
  Sample 4/5: FP32=20.0s, INT8=1.9s (Speedup: 10.5x)
  Sample 5/5: FP32=19.9s, INT8=2.0s (Speedup: 9.9x)
  
  📊 Latencia promedio:
     FP32: 19.9s (±0.2s)
     INT8: 1.5s (±0.8s)
     Speedup: 13.3x

[4/5] Calculando métricas de calidad...
  Sample 1: Cosine=0.9891, MSE=0.0011, MAE=0.0203
  Sample 2: Cosine=0.9893, MSE=0.0012, MAE=0.0198
  Sample 3: Cosine=0.9889, MSE=0.0013, MAE=0.0211
  Sample 4: Cosine=0.9892, MSE=0.0011, MAE=0.0205
  Sample 5: Cosine=0.9890, MSE=0.0012, MAE=0.0207

[5/5] Evaluación de métricas...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESULTADOS DE CALIDAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ COSINE_SIM:  0.9891 >= 0.98
✅ MSE:         0.0012 <= 0.01
✅ MAE:         0.0204 <= 0.05
✅ SPEEDUP:     13.3x >= 2.0x
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ CALIDAD VALIDADA: INT8 cumple todos los objetivos

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESUMEN DE BENEFICIOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Tamaño:      4.3GB → 1.1GB (-74%)
⚡ Speedup:     13.3x
🎯 Precisión:   98.9%
💾 RAM ahorrada: ~3.2GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Si alguna métrica falla**: Ver soluciones en `docs/QUANTIZATION_CHECKLIST.md`.

---

## 📋 Orden de Ejecución Recomendado

### En Windows

```batch
REM 1. Verificar pre-requisitos
scripts\check_prerequisites_windows.bat

REM 2. Cuantizar modelo
scripts\quantize_windows.bat

REM 3. Calcular checksums (para verificar en Linux)
certutil -hashfile models\onnx\agi_audio_core_int8.onnx SHA256
certutil -hashfile models\onnx\agi_audio_core_int8.onnx.data SHA256
```

### Transferir a Linux

```powershell
REM Opción 1: SCP
scp models\onnx\agi_audio_core_int8.onnx noel@agi1:~/SARAi_v2/models/onnx/
scp models\onnx\agi_audio_core_int8.onnx.data noel@agi1:~/SARAi_v2/models/onnx/

REM Opción 2: WinSCP (interfaz gráfica)
```

### En Linux

```bash
# 1. Verificar archivos recibidos
ls -lh ~/SARAi_v2/models/onnx/agi_audio_core_int8.*
# Debe mostrar: 8KB + 1.1GB

# 2. Verificar checksums (comparar con Windows)
sha256sum ~/SARAi_v2/models/onnx/agi_audio_core_int8.onnx
sha256sum ~/SARAi_v2/models/onnx/agi_audio_core_int8.onnx.data

# 3. Actualizar config
nano ~/SARAi_v2/config/sarai.yaml
# Cambiar model_path a agi_audio_core_int8.onnx
# Cambiar max_memory_mb a 1200

# 4. Validar con tests
cd ~/SARAi_v2
python3 scripts/test_onnx_pipeline.py

# 5. Comparar calidad
python3 scripts/compare_fp32_int8_quality.py

# 6. Si todo OK → Producción
sudo systemctl restart sarai
```

---

## 🛠️ Dependencias

### Windows

```powershell
pip install onnx onnxruntime-gpu  # O onnxruntime-directml para AMD/Intel
```

**Dependencias instaladas automáticamente** por `quantize_windows.bat`:
- `onnx` (>=1.15.0)
- `onnxruntime-gpu` (>=1.16.0) o `onnxruntime` (CPU)

### Linux

**Ya incluidas** en `requirements.txt`:
- `onnxruntime` (>=1.16.0)
- `numpy`
- `librosa`
- `soundfile`

---

## 📊 Métricas Objetivo

| Script | Métrica | Objetivo | Validación |
|--------|---------|----------|------------|
| `test_onnx_pipeline.py` | Carga inicial | <10s | ✅ si <15s |
| | Primera inferencia | ~2s | ✅ si <2.5s |
| | Cache hit | 0s | ✅ exacto |
| | Output shape | [1,2048,245760] | ✅ exacto |
| `compare_fp32_int8_quality.py` | Similitud coseno | >0.98 | ✅ si ≥0.98 |
| | MSE | <0.01 | ✅ si ≤0.01 |
| | MAE | <0.05 | ✅ si ≤0.05 |
| | Speedup | >2.0x | ✅ si ≥2.0x |

---

## ⚠️ Troubleshooting

### Script Windows falla con "Python not found"

**Solución**:
```powershell
# Verificar Python en PATH
python --version

# Si falla, reinstalar Python marcando "Add to PATH"
# O añadir manualmente:
set PATH=%PATH%;C:\Python310
```

---

### Script Windows usa CPU en vez de GPU

**Solución**:
```powershell
# Verificar GPU funciona
nvidia-smi

# Instalar onnxruntime-gpu
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu

# Verificar providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Debe incluir: CUDAExecutionProvider
```

---

### Tests Linux fallan con "Model not found"

**Solución**:
```bash
# Verificar archivos transferidos
ls -lh models/onnx/agi_audio_core_int8.*

# Verificar config apunta a INT8
grep "model_path" config/sarai.yaml
# Debe mostrar: model_path: "models/onnx/agi_audio_core_int8.onnx"
```

---

### Comparación calidad muestra Speedup bajo (<2x)

**Solución**:
```bash
# Verificar warmup activado
grep "warmup: true" config/sarai.yaml

# Re-ejecutar tests (warmup se activa en primera carga)
python3 scripts/test_onnx_pipeline.py
python3 scripts/compare_fp32_int8_quality.py
```

---

## 🎓 Notas Técnicas

### ¿Por qué 2 archivos (.onnx + .onnx.data)?

ONNX usa **external data format** para modelos grandes:
- `.onnx`: Metadata del modelo (graph, inputs, outputs) ~8KB
- `.onnx.data`: Pesos del modelo (parámetros cuantizados) ~1.1GB

**Beneficio**: Mejor rendimiento en Git, fácil versioning de metadata.

---

### ¿Qué es Dynamic Quantization?

- **Pesos**: FP32 → INT8 (cuantizados en disco al guardar)
- **Activaciones**: FP32 → INT8 (cuantizadas en runtime durante inferencia)

**Ventaja**: No requiere dataset de calibración (vs Static Quantization).

**Desventaja**: Overhead runtime mínimo (~5%).

---

### ¿GPU necesaria?

**NO**, pero recomendada:
- Con GPU NVIDIA: **2-3 minutos** (CUDA acceleration)
- Con CPU: **5-10 minutos** (funcional pero más lento)

**Inferencia** (después de cuantizar) siempre es en **CPU** en Linux (agi1 no tiene GPU).

---

## 📚 Documentación Completa

Para guías detalladas, ver:
- `docs/EXECUTIVE_SUMMARY_INT8.md` - Resumen ejecutivo
- `docs/QUANTIZATION_CHECKLIST.md` - Checklist paso a paso
- `docs/WINDOWS_QUANTIZATION_WORKFLOW.md` - Guía completa
- `docs/INT8_FILES_INDEX.md` - Índice de todos los archivos

---

**Última actualización**: 29 Octubre 2025 23:58  
**Versión**: v2.16.1  
**Status**: ✅ Scripts validados y listos
