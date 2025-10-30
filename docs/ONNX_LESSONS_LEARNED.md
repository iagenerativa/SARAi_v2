# ❌ ONNX Exportation - Lecciones Aprendidas

## 📋 Contexto

El usuario solicitó exportar MeloTTS a formato ONNX para aprovechar ONNX Runtime y sus optimizaciones (AVX/AVX2/AVX512) en CPU.

## 🔬 Intentos Realizados

### 1. Exportación ONNX Directa del Modelo Completo

**Script**: `scripts/export_melo_to_onnx.py`

**Resultado**: ❌ FALLIDO

```python
Error: SynthesizerTrn.forward() missing 3 required positional arguments: 
'language', 'bert', and 'ja_bert'
```

**Causa**: 
- MeloTTS usa arquitectura compleja (BERT + HiFi-GAN)
- Método `forward()` tiene firma no estándar
- Requiere múltiples entradas condicionales (idioma, BERT embeddings)

### 2. Exportación Parcial (Solo Vocoder)

**Resultado**: ❌ FALLIDO

```python
Error: Given groups=1, weight of size [512, 192, 7], 
expected input[1, 80, 100] to have 192 channels, but got 80 channels instead
```

**Causa**:
- Vocoder (HiFi-GAN) espera entradas procesadas específicas
- Mel-spectrogram tiene dimensionalidad incorrecta
- Componentes no pueden separarse fácilmente

### 3. ONNX Runtime como Backend de PyTorch

**Script**: `agents/melo_tts_onnx.py`

**Resultado**: ⚠️ PARCIALMENTE FUNCIONAL

- Configuración de ONNX Runtime exitosa
- Optimizaciones de threading aplicadas
- **PERO**: No hay mejora significativa sin modelo exportado
- PyTorch sigue siendo el backend de inferencia

### 4. melotts-onnx (Paquete PyPI)

**Instalación**: `pip3 install melotts-onnx`

**Resultado**: ❌ NO VIABLE

**Problemas**:

1. **Modelos no disponibles**:
   ```python
   Error: Can not found the configuration file "models/melo_onnx_models/configuration.json"
   ```
   - Modelos ONNX pre-entrenados NO están en HuggingFace
   - Repositorios esperados no existen:
     - `BarryWang/MeloTTS-ONNX-ES` ❌
     - `myshell-ai/MeloTTS-ES-ONNX` ❌
     - `melotts/onnx-es` ❌

2. **Conflictos de dependencias**:
   ```bash
   melotts-onnx instaló:
   - numpy==2.0.2 → Incompatible con scipy (requiere < 2.0)
   - tokenizers==0.20.0 → Incompatible con transformers (requiere < 0.14)
   ```

3. **Desinstalación necesaria**:
   ```bash
   pip3 uninstall -y melotts-onnx
   pip3 install "numpy<2.0"
   pip3 install "tokenizers>=0.11.1,!=0.11.3,<0.14"
   ```

## 🎯 Conclusión

### ¿Por qué ONNX NO funcionó?

1. **Complejidad del modelo**:
   - Arquitectura multi-componente (Text Encoder + Vocoder)
   - Inputs dinámicos y condicionales
   - No diseñado para exportación ONNX

2. **Falta de modelos pre-entrenados**:
   - ONNX requiere modelos ya exportados
   - MeloTTS oficial solo distribuye checkpoints PyTorch
   - Comunidad no mantiene versiones ONNX

3. **Beneficio marginal**:
   - Con optimizaciones PyTorch: 0.5-0.7s latencia
   - ONNX teórico (si funcionara): ~0.3-0.5s
   - **Mejora esperada**: 20-30% vs 93% con preload
   - **Riesgo**: Alta complejidad, baja estabilidad

### ¿Qué SÍ funcionó?

✅ **Optimizaciones PyTorch**:
- Precarga: **93.9% mejora**
- Velocidad 1.3x: **29% mejora**
- Caché: **100% mejora** (hits)
- **Total**: 0.5-0.7s latencia (suficiente para tiempo real)

## 📚 Lecciones para el Futuro

### Cuándo considerar ONNX:

✅ **Modelos simples**:
- Single-input, single-output
- Forward pass estándar
- Sin dependencias complejas

✅ **Modelos oficialmente soportados**:
- BERT, ResNet, YOLO (modelos estándar)
- Checkpoints ONNX disponibles en Model Hub

✅ **Necesidad crítica de velocidad**:
- Latencia actual inaceptable (> 5s)
- Hardware sin GPU
- Batch processing a gran escala

### Cuándo NO usar ONNX:

❌ **Modelos experimentales/complejos**:
- Arquitecturas personalizadas
- Multi-stage pipelines
- Inputs dinámicos

❌ **PyTorch ya es suficientemente rápido**:
- < 1s latencia objetivo
- Hardware moderno (AVX2+)
- Optimizaciones simples disponibles

❌ **Falta de recursos oficiales**:
- Sin modelos ONNX pre-entrenados
- Sin documentación de exportación
- Comunidad pequeña

## 🔧 Alternativas Efectivas

### Si se necesita MÁS velocidad en el futuro:

1. **Cuantización PyTorch INT8**:
   ```python
   model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```
   - Mejora esperada: 20-30%
   - Compatible con PyTorch actual
   - Sin exportación necesaria

2. **JIT Compilation**:
   ```python
   model = torch.jit.script(model)
   # o
   model = torch.jit.trace(model, example_input)
   ```
   - Mejora esperada: 10-15%
   - Optimiza graph computation
   - Sin cambios en API

3. **TorchScript + Mobile Deployment**:
   - Para edge devices (Raspberry Pi, etc.)
   - Menor footprint de memoria
   - Latencia comparable a ONNX

4. **Modelo más pequeño**:
   - MeloTTS tiene variantes (1.2B, 600M, etc.)
   - Menos parámetros = menos latencia
   - Trade-off calidad vs velocidad

## 📊 Comparativa Final

| Método | Latencia | Complejidad | Estabilidad | Recomendado |
|--------|----------|-------------|-------------|-------------|
| PyTorch base | 1.5s | Baja | ✅ Alta | ❌ |
| PyTorch optimizado | 0.6s | Baja | ✅ Alta | ✅ **SÍ** |
| ONNX exportado | N/A | ⚠️ Muy alta | ❌ Baja | ❌ |
| melotts-onnx | N/A | Media | ❌ Baja | ❌ |
| INT8 Quant | ~0.4s | Media | ✅ Media | ⚠️ Futuro |
| JIT Compile | ~0.5s | Baja | ✅ Alta | ⚠️ Futuro |

## ✅ Decisión Final

**MANTENER PyTorch con optimizaciones actuales**

**Razones**:
1. ✅ Latencia objetivo cumplida (< 1s)
2. ✅ Código estable y mantenible
3. ✅ Compatible con actualizaciones de MeloTTS
4. ✅ Sin dependencias problemáticas
5. ✅ Todos los tests pasando

**Si en el futuro se necesita más velocidad**:
- Probar cuantización INT8 primero
- Evaluar JIT compilation
- Considerar modelo más pequeño

**NO reintentar ONNX a menos que**:
- MyShell.ai publique checkpoints ONNX oficiales
- Aparezca soporte oficial en MeloTTS
- Latencia actual se vuelva inaceptable (> 2s)

---
**Autor**: SARAi Development Team  
**Fecha**: 30 de octubre de 2025  
**Conclusión**: ❌ ONNX no es viable para MeloTTS | ✅ PyTorch optimizado es suficiente
