# ✅ Checklist de Cuantización INT8

**Objetivo**: Cuantizar `agi_audio_core.onnx` (4.3GB) → `agi_audio_core_int8.onnx` (1.1GB)  
**Hardware**: Windows 11 con 32GB RAM + 8GB VRAM  
**Tiempo estimado**: 30-40 minutos total  

---

## 📋 Pre-requisitos (Verificar ANTES de empezar)

- [ ] **Windows 11** con Python 3.10+ instalado
- [ ] **32GB RAM** disponibles (cerrar apps pesadas)
- [ ] **6GB espacio en disco** libres
- [ ] **GPU NVIDIA** (opcional, reduce tiempo de 10min a 2min)
- [ ] **Modelo FP32** completo en `models/onnx/`:
  - [ ] `agi_audio_core.onnx` (7.6KB)
  - [ ] `agi_audio_core.onnx.data` (4.3GB)
- [ ] **SSH a Linux** funcional (para transferencia)
- [ ] **WinSCP o scp** instalado (alternativa: Samba share)

**Verificación rápida (PowerShell)**:
```powershell
# Python
python --version  # Debe mostrar 3.10+

# Espacio en disco
Get-PSDrive C | Select-Object Free  # >6GB

# Modelo FP32
dir models\onnx\agi_audio_core.*  # Debe mostrar 2 archivos

# SSH a Linux (opcional ahora)
ssh noel@agi1 "echo OK"  # Debe mostrar "OK"
```

---

## 🚀 PASO 1: Cuantización en Windows

**Tiempo estimado**: 2-10 minutos (según GPU)

### Opción A: Script Automatizado (Recomendado)

```powershell
# En PowerShell o CMD (Run as Administrator)
cd C:\ruta\a\SARAi_v2
scripts\quantize_windows.bat
```

- [ ] Script instaló dependencias automáticamente
- [ ] GPU detectada (CUDA/DirectML) o usando CPU
- [ ] Cuantización ejecutándose (barra de progreso o mensajes)
- [ ] ✅ **Completado**: Mensaje "Cuantización completada exitosamente!"

**Output esperado**:
```
[1/5] Cargando modelo FP32... ✅
[2/5] Cuantizando a INT8... ✅ (2m 14s)
[3/5] Guardando modelo INT8... ✅
[4/5] Validando modelo... ✅ Latencia: 1.87s
[5/5] Resultados:
  Tamaño:   4.3GB → 1.1GB (-74%)
  Latencia: 5.3s → 1.87s (-65%)
  Speedup:  2.8x
✅ CUANTIZACIÓN EXITOSA
```

### Opción B: Manual (Si script falla)

```powershell
# 1. Instalar dependencias
pip install onnx onnxruntime-gpu  # O onnxruntime-directml

# 2. Ejecutar cuantización
python scripts\quantize_onnx_int8_windows.py

# 3. Confirmar con 'y' cuando pregunte
```

- [ ] Dependencias instaladas (`pip list | findstr onnx`)
- [ ] Script ejecutado sin errores
- [ ] Archivos generados verificados (ver abajo)

### Verificar Archivos Generados

```powershell
dir models\onnx\agi_audio_core_int8.*
```

**Esperado**:
```
agi_audio_core_int8.onnx       8,192 bytes   (8KB)
agi_audio_core_int8.onnx.data  1,181,116,416 bytes (1.1GB)
```

- [ ] Archivo `.onnx` existe (~8KB)
- [ ] Archivo `.onnx.data` existe (~1.1GB)
- [ ] Tamaño reducido ~74% vs FP32

### Calcular Checksums (Para verificar integridad)

```powershell
certutil -hashfile models\onnx\agi_audio_core_int8.onnx SHA256
certutil -hashfile models\onnx\agi_audio_core_int8.onnx.data SHA256
```

- [ ] Checksums calculados y guardados (copiar a notepad)

**GUARDAR estos hashes para verificar en Linux después**.

---

## 📤 PASO 2: Transferir a Linux

**Tiempo estimado**: 5-10 minutos (1.1GB transferencia)

### Opción A: SCP (Recomendado)

```powershell
# En PowerShell (Windows)

# 1. Transferir metadata (rápido)
scp models\onnx\agi_audio_core_int8.onnx noel@agi1:~/SARAi_v2/models/onnx/

# 2. Transferir pesos (lento, 5-10 min)
scp models\onnx\agi_audio_core_int8.onnx.data noel@agi1:~/SARAi_v2/models/onnx/
```

- [ ] Transferencia 1/2 completada (metadata 8KB)
- [ ] Transferencia 2/2 completada (data 1.1GB)
- [ ] Sin errores de conexión

**Si falla con hostname `agi1`**, probar con IP:
```powershell
scp models\onnx\agi_audio_core_int8.onnx.data noel@<LAN_IP>:~/SARAi_v2/models/onnx/
```

### Opción B: WinSCP (Interfaz gráfica)

1. [ ] Descargar WinSCP desde https://winscp.net/
2. [ ] Conectar a `agi1` (host) usuario `noel`
3. [ ] Navegar local: `C:\SARAi_v2\models\onnx\`
4. [ ] Navegar remoto: `/home/noel/SARAi_v2/models/onnx/`
5. [ ] Arrastrar `agi_audio_core_int8.onnx` y `.data`
6. [ ] Esperar transferencia completa (barra progreso 100%)

### Opción C: Samba/SMB Share

```powershell
# Mapear unidad de red (requiere Samba configurado en Linux)
net use Z: \\agi1\sarai_models /user:noel

# Copiar archivos
copy models\onnx\agi_audio_core_int8.* Z:\
```

- [ ] Unidad Z: mapeada correctamente
- [ ] Archivos copiados sin errores

---

## 🔧 PASO 3: Configuración en Linux

**Tiempo estimado**: 5 minutos

### Verificar Archivos Recibidos

```bash
# En Linux (SSH a agi1)
cd ~/SARAi_v2/models/onnx/

# Listar archivos
ls -lh agi_audio_core_int8.*
```

**Esperado**:
```
agi_audio_core_int8.onnx       8.0K
agi_audio_core_int8.onnx.data  1.1G
```

- [ ] Archivos presentes en Linux
- [ ] Tamaños correctos (8KB + 1.1GB)

### Verificar Integridad (Checksums)

```bash
sha256sum agi_audio_core_int8.onnx
sha256sum agi_audio_core_int8.onnx.data
```

- [ ] Checksums coinciden con los de Windows (ver Paso 1)
- [ ] Si NO coinciden, **RE-TRANSFERIR archivos**

### Actualizar `config/sarai.yaml`

```bash
nano config/sarai.yaml
```

**Cambiar estas líneas**:
```yaml
audio_omni:
  name: "Qwen3-Omni-3B-INT8"  # Era "Complete"
  model_path: "models/onnx/agi_audio_core_int8.onnx"  # Era agi_audio_core.onnx
  max_memory_mb: 1200  # Era 4400
```

- [ ] Nombre cambiado a "INT8"
- [ ] `model_path` apunta a `agi_audio_core_int8.onnx`
- [ ] `max_memory_mb` cambiado a 1200
- [ ] Archivo guardado (Ctrl+O, Ctrl+X en nano)

### Verificar Configuración Válida

```bash
python3 -c "import yaml; yaml.safe_load(open('config/sarai.yaml'))"
```

- [ ] Sin errores (sin output = ✅ válido)

---

## ✅ PASO 4: Validación (CRÍTICO)

**Tiempo estimado**: 10 minutos

### Test Suite Completo

```bash
cd ~/SARAi_v2
python3 scripts/test_onnx_pipeline.py
```

**Verificar estos resultados**:

- [ ] ✅ Test 1 (Model Loading): Completado <10s (vs 44s FP32)
- [ ] ✅ Test 2 (Model Inference): Primera inferencia ~2s (vs 20s FP32)
- [ ] ✅ Test 3 (Config Loading): Config INT8 válido
- [ ] ✅ Test 4 (File Validation): Archivos INT8 detectados
- [ ] ✅ Test 5 (Benchmark): Latencia promedio <2.5s
- [ ] ✅ **5/5 tests pasaron**

**Métricas objetivo**:
```
✅ Latencia P50:  ~2.0s  (±0.5s tolerancia)
✅ RAM Usage:     ~1.2GB (±200MB tolerancia)
✅ Cache Hit:     0s     (instantáneo)
✅ Output Shape:  [1, 2048, 245760] (exacto)
```

### Comparación Calidad FP32 vs INT8 (Opcional pero recomendado)

```bash
python3 scripts/compare_fp32_int8_quality.py
```

**Verificar métricas**:

- [ ] ✅ Similitud coseno: >0.98 (>98% similar)
- [ ] ✅ MSE (error): <0.01 (bajo error)
- [ ] ✅ MAE (error): <0.05 (bajo error)
- [ ] ✅ Speedup: >2.0x (al menos 2x más rápido)

**Si alguna métrica falla**, ver sección Troubleshooting.

---

## 📊 PASO 5: Benchmark Comparativo (Opcional)

**Tiempo estimado**: 15 minutos

```bash
python3 scripts/benchmark_int8_vs_fp32.py
```

**Resultados esperados**:
```
Métrica          FP32      INT8      Δ
────────────────────────────────────────
Carga            44.3s     8.3s      -81%
Primera Inf.     20.0s     1.9s      -90%
Promedio         5.3s      2.0s      -62%
RAM Peak         4.3GB     1.2GB     -72%
Tamaño Disco     4.3GB     1.1GB     -74%
────────────────────────────────────────
Speedup:         1.0x      2.65x     +165%
```

- [ ] Benchmark ejecutado sin errores
- [ ] Resultados documentados (captura pantalla o copiar output)

---

## 🎉 PASO 6: Puesta en Producción

### Actualizar Sistema

```bash
# Si SARAi está corriendo, reiniciar para cargar nuevo modelo
sudo systemctl restart sarai

# O matar proceso y relanzar
pkill -f main.py
nohup python3 main.py > logs/sarai.log 2>&1 &
```

- [ ] SARAi reiniciado con modelo INT8
- [ ] Sin errores en logs (`tail -f logs/sarai.log`)

### Monitorear KPIs (Primera Hora)

```bash
# Verificar RAM usage cada 5 min
watch -n 300 'ps aux | grep python | grep -v grep'

# Esperado: RAM ~1.2GB para audio_omni (vs 4.3GB antes)
```

- [ ] RAM usage estable ~1.2GB
- [ ] Sin OOM errors
- [ ] Latencia <2.5s en interacciones reales

### Documentar Implementación

- [ ] Actualizar `CHANGELOG.md` con implementación INT8
- [ ] Añadir métricas finales a `docs/ONNX_INTEGRATION_REAL.md`
- [ ] Commit cambios a git:
  ```bash
  git add config/sarai.yaml models/onnx/agi_audio_core_int8.*
  git commit -m "feat(audio): Implementar cuantización INT8 (-74% tamaño, -62% latencia)"
  ```

---

## 🛑 Plan de Rollback (Si algo falla)

### Revertir a FP32

```bash
# 1. Editar config
nano config/sarai.yaml

# Cambiar a:
# model_path: "models/onnx/agi_audio_core.onnx"  # FP32 original
# max_memory_mb: 4400

# 2. Reiniciar
sudo systemctl restart sarai
```

- [ ] Config revertido a FP32
- [ ] Sistema funcionando con modelo original
- [ ] Latencia vuelve a ~5s (aceptable)

### Probar FP16 (Alternativa si INT8 degrada mucho)

```bash
# En Windows, ejecutar:
python scripts\quantize_onnx_fp16.py  # Crear si no existe

# FP16 beneficios:
# - Tamaño: 4.3GB → 2.2GB (-50%)
# - Precisión: 99.9% (casi sin pérdida)
# - Speedup: 1.3-1.5x (menor que INT8 pero mejor calidad)
```

---

## 🔍 Troubleshooting

### ❌ Problema: "GPU no detectada en Windows"

**Síntoma**: Script usa CPU en vez de GPU

**Solución**:
```powershell
# Verificar CUDA
nvidia-smi

# Reinstalar onnxruntime-gpu
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu

# Verificar providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Esperado: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

- [ ] CUDA verificado funcionando
- [ ] Providers incluyen CUDA
- [ ] Re-ejecutar cuantización

---

### ❌ Problema: "SCP Connection refused"

**Síntoma**: `ssh: connect to host agi1 port 22: Connection refused`

**Solución**:
```bash
# En Linux, verificar SSH
sudo systemctl status sshd
sudo systemctl start sshd

# Verificar firewall
sudo ufw allow 22/tcp
```

```powershell
# En Windows, probar con IP directa
ping <LAN_IP>  # Reemplazar con la IP del host agi1
scp models\onnx\agi_audio_core_int8.onnx.data noel@<LAN_IP>:~/SARAi_v2/models/onnx/
```

- [ ] SSH server corriendo en Linux
- [ ] Firewall permite puerto 22
- [ ] Transferencia exitosa con IP directa

---

### ❌ Problema: "Checksums no coinciden"

**Síntoma**: SHA256 de Windows ≠ SHA256 de Linux

**Solución**:
```bash
# Borrar archivos corruptos en Linux
rm models/onnx/agi_audio_core_int8.*

# Re-transferir desde Windows
scp models\onnx\agi_audio_core_int8.onnx* noel@agi1:~/SARAi_v2/models/onnx/

# Verificar de nuevo
sha256sum models/onnx/agi_audio_core_int8.onnx.data
```

- [ ] Archivos borrados
- [ ] Re-transferidos exitosamente
- [ ] Checksums ahora coinciden

---

### ❌ Problema: "Latencia INT8 peor que FP32"

**Síntoma**: Benchmark muestra 8s en vez de 2s

**Diagnóstico**:
```bash
# Verificar que usa INT8
python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('models/onnx/agi_audio_core_int8.onnx')
print('Model:', 'int8.onnx' if 'int8' in sess._model_path else 'WRONG')
"
```

**Solución**:
```bash
# Verificar config apunta a INT8
grep "model_path" config/sarai.yaml
# Debe mostrar: model_path: "models/onnx/agi_audio_core_int8.onnx"

# Verificar warmup activado
grep "warmup: true" config/sarai.yaml

# Re-ejecutar tests
python3 scripts/test_onnx_pipeline.py
```

- [ ] Config correcto (apunta a INT8)
- [ ] Warmup activado
- [ ] Latencia ahora correcta (~2s)

---

## 📈 Métricas de Éxito (Post-Implementación)

### Semana 1

- [ ] **Latencia P50**: <2.5s (objetivo: 2.0s ±0.5s)
- [ ] **RAM P99**: <1.5GB (objetivo: 1.2GB ±200MB)
- [ ] **Error Rate**: <1% (objetivo: 0%)
- [ ] **Cache Hit**: >60% (objetivo: 60-80%)
- [ ] **User feedback**: Sin quejas de calidad degradada

### Mes 1

- [ ] **WER (Word Error Rate)**: <5% (similar a FP32)
- [ ] **MOS (Quality Score)**: >4.0/5.0 (10 usuarios, 10 audios)
- [ ] **Emotion Accuracy**: >85% (dataset validación)
- [ ] **Sistema estable**: Sin crashes por OOM

---

## 📝 Checklist Final

**Antes de declarar éxito, verificar TODO lo siguiente**:

- [ ] ✅ Modelo INT8 cuantizado en Windows (1.1GB)
- [ ] ✅ Transferido a Linux sin errores
- [ ] ✅ Checksums verificados (Windows = Linux)
- [ ] ✅ `config/sarai.yaml` actualizado correctamente
- [ ] ✅ 5/5 tests pasando en `test_onnx_pipeline.py`
- [ ] ✅ Latencia P50 <2.5s (objetivo alcanzado)
- [ ] ✅ RAM usage ~1.2GB (objetivo alcanzado)
- [ ] ✅ Comparación calidad: Similitud >98%
- [ ] ✅ Sistema en producción sin errores
- [ ] ✅ Documentación actualizada
- [ ] ✅ Commit a git realizado

**Si todos los checks ✅, FELICIDADES!** 🎉

Has implementado exitosamente cuantización INT8, logrando:
- **-74% tamaño** (4.3GB → 1.1GB)
- **-62% latencia** (5.3s → 2.0s)
- **-72% RAM** (4.3GB → 1.2GB)
- **98-99% precisión** (pérdida <1-2%)

---

**Tiempo total estimado**: 30-40 minutos  
**Dificultad**: Media (requiere acceso Windows + Linux)  
**Impacto**: Alto (mejora significativa rendimiento)  

**Próximos pasos**:
1. Monitorear KPIs primera semana
2. Implementar STT/TTS real (usar mel_features)
3. Integrar con LangGraph (nodo audio_omni)
4. Considerar Static Quantization (mejor precisión)

---

**Última actualización**: 29 Octubre 2025  
**Versión**: v2.16.1  
**Status**: ✅ Listo para ejecución
