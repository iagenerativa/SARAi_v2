# Modelos GGUF de SARAi v2.11

Este documento detalla los modelos GGUF utilizados por SARAi, sus fuentes oficiales y especificaciones.

---

## 📦 Modelos Principales

### 1. SOLAR-10.7B-Instruct (Expert Tier)

**Propósito**: Razonamiento técnico profundo, respuestas detalladas.

- **Nombre completo**: SOLAR-10.7B-Instruct-v1.0
- **Cuantización**: Q4_K_M (4-bit)
- **Tamaño**: ~6GB
- **RAM requerida**: 
  - `expert_short` (n_ctx=512): ~4.8GB
  - `expert_long` (n_ctx=2048): ~6GB
- **Repositorio HuggingFace**: 
  ```
  hf.co/solxxcero/SOLAR-10.7B-Instruct-v1.0-Q4_K_M-GGUF
  ```
- **Archivo específico**: `Q4_K_M` (solar-10.7b-instruct-v1.0.Q4_K_M.gguf)

**Descarga manual**:
```bash
huggingface-cli download \
  solxxcero/SOLAR-10.7B-Instruct-v1.0-Q4_K_M-GGUF \
  --include "*.gguf" \
  --local-dir models/gguf/
```

**Contexto dinámico**: Este modelo se carga UNA SOLA VEZ con diferentes `n_ctx` según la necesidad:
- Queries cortas (<400 chars): `n_ctx=512` → 4.8GB RAM
- Queries largas (>400 chars): `n_ctx=2048` → 6GB RAM

**Características**:
- Upscaled Depth-Upscaled Architecture
- Fine-tuned para instrucciones
- Excelente en razonamiento técnico
- Latencia CPU: ~30-60s (según n_ctx)

---

### 2. LFM2-1.2B (Tiny Tier)

**Propósito**: Respuestas rápidas, soft-skills, modulación emocional.

- **Nombre completo**: LFM2-1.2B
- **Cuantización**: Q4_K_M (4-bit)
- **Tamaño**: ~700MB
- **RAM requerida**: ~700MB (n_ctx=2048)
- **Repositorio HuggingFace**: 
  ```
  hf.co/LiquidAI/LFM2-1.2B-GGUF
  ```
- **Archivo específico**: `Q4_K_M` (lfm2-1.2b.Q4_K_M.gguf)

**Descarga manual**:
```bash
huggingface-cli download \
  LiquidAI/LFM2-1.2B-GGUF \
  --include "*.gguf" \
  --local-dir models/gguf/
```

**Características**:
- Liquid Foundation Models 2.0
- Optimizado para edge devices
- Excelente latencia en CPU (~5-10s)
- Usado para:
  - Modulación de tono (empático/neutral/urgente)
  - Respuestas rápidas soft-skills
  - Refinamiento de respuestas técnicas (híbrido)

---

### 3. Qwen2.5-Omni-7B (Multimodal - Opcional)

**Propósito**: Procesamiento de audio y visión.

- **Nombre completo**: Qwen2.5-Omni-7B
- **Cuantización**: Q4_K_M (4-bit)
- **Tamaño**: ~4GB
- **RAM requerida**: ~4GB (n_ctx=2048)
- **Repositorio HuggingFace**: 
  ```
  hf.co/Qwen/Qwen2.5-Omni-7B-GGUF
  ```
- **Archivo específico**: `Q4_K_M` (qwen2.5-omni-7b.Q4_K_M.gguf)

**Descarga manual**:
```bash
huggingface-cli download \
  Qwen/Qwen2.5-Omni-7B-GGUF \
  --include "*.gguf" \
  --local-dir models/gguf/
```

**Características**:
- Procesamiento nativo de audio (STT + TTS integrado)
- Detección de emoción en voz
- Multimodal: texto + audio + visión
- Se descarga/descarga dinámicamente según RAM disponible

---

## 🔧 Descarga Automatizada

SARAi incluye un script automatizado para descargar todos los modelos:

```bash
python scripts/download_gguf_models.py
```

**Opciones**:
```bash
# Solo modelos requeridos (SOLAR + LFM2)
python scripts/download_gguf_models.py --required-only

# Incluir multimodal
python scripts/download_gguf_models.py --include-multimodal

# Verificar integridad sin descargar
python scripts/download_gguf_models.py --verify-only
```

---

## 📊 Comparación de Modelos

| Modelo | Tamaño | RAM (n_ctx) | Latencia P50 | Uso Principal |
|--------|--------|-------------|--------------|---------------|
| SOLAR-10.7B | 6GB | 4.8-6GB | 30-60s | Razonamiento técnico |
| LFM2-1.2B | 700MB | 700MB | 5-10s | Soft-skills, modulación |
| Qwen2.5-Omni-7B | 4GB | 4GB | 15-25s | Audio/visión |

---

## 🛠️ Conversión de Modelos (Avanzado)

Si necesitas convertir un modelo HuggingFace a GGUF:

```bash
# Instalar llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convertir modelo
python convert.py /path/to/huggingface/model \
  --outtype q4_K_M \
  --outfile /path/to/output.gguf
```

**Nota**: Los modelos listados arriba ya están cuantizados y listos para usar. La conversión manual solo es necesaria si usas modelos custom.

---

## 🔒 Verificación de Integridad

Para verificar que los modelos descargados son correctos:

```python
# scripts/verify_gguf.py
from pathlib import Path
from llama_cpp import Llama

def verify_gguf(path: str):
    """Carga el GGUF y verifica que es válido"""
    try:
        model = Llama(model_path=path, n_ctx=512, verbose=False)
        result = model("Test", max_tokens=10)
        print(f"✅ {path} is valid")
        return True
    except Exception as e:
        print(f"❌ {path} failed: {e}")
        return False

# Verificar todos los modelos
verify_gguf("models/gguf/solar-10.7b.gguf")
verify_gguf("models/gguf/lfm2-1.2b.gguf")
```

---

## 📚 Referencias

- **SOLAR**: https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0
- **LFM2**: https://huggingface.co/LiquidAI/LFM2-1.2B
- **Qwen2.5-Omni**: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
- **GGUF Format**: https://github.com/ggerganov/llama.cpp
- **llama-cpp-python**: https://github.com/abetlen/llama-cpp-python

---

**Última actualización**: 2025-10-28  
**Versión SARAi**: v2.11  
**Estado**: Modelos verificados y funcionales
