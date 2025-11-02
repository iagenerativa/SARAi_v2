# Unified Model Wrapper - Guía Completa v2.14

**Fecha**: 1 de noviembre de 2025  
**Versión**: v2.14  
**Estado**: ✅ PRODUCCIÓN

---

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Filosofía](#filosofía)
3. [Arquitectura](#arquitectura)
4. [Backends Soportados](#backends-soportados)
5. [Uso Básico](#uso-básico)
6. [Configuración](#configuración)
7. [Ejemplos Avanzados](#ejemplos-avanzados)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Migración desde model_pool](#migración-desde-model_pool)

---

## Introducción

El **Unified Model Wrapper** es la capa de abstracción universal para todos los modelos de SARAi v2.14. Permite usar cualquier modelo (GGUF, Transformers, Multimodal, APIs) con una interfaz única basada en LangChain.

### ¿Por qué Unified Wrapper?

**Antes v2.13** (model_pool):
```python
# Código específico por backend
from llama_cpp import Llama
solar = Llama(model_path="...", n_ctx=512, ...)

from transformers import AutoModelForCausalLM
lfm2 = AutoModelForCausalLM.from_pretrained("...", load_in_4bit=True, ...)
```

**Después v2.14** (Unified Wrapper):
```python
# Una interfaz para TODOS los modelos
from core.unified_model_wrapper import get_model

solar = get_model("solar_short")  # GGUF o Ollama
lfm2 = get_model("lfm2")           # GGUF local
embeddings = get_model("embeddings")  # Embeddings
qwen = get_model("qwen3_vl")       # Multimodal

# TODOS usan la misma API
response = solar.invoke("¿Qué es Python?")
```

---

## Filosofía

### Principio Fundamental

> **"SARAi no debe conocer sus modelos. Solo debe invocar capacidades."**

- **YAML define** → `config/models.yaml` es la única fuente de verdad
- **LangChain orquesta** → Pipelines LCEL declarativos
- **Wrapper abstrae** → Backends intercambiables sin cambiar código

### Ventajas

| Aspecto | Antes (model_pool) | Después (Unified Wrapper) |
|---------|-------------------|---------------------------|
| **Agregar modelo** | Modificar código Python | Solo editar YAML |
| **Cambiar backend** | Reescribir lógica | Cambiar 1 línea en YAML |
| **Testing** | Mocks complejos | Integración real |
| **Migración GPU** | Reescribir todo | Cambiar `backend: "gguf"` → `backend: "transformers"` |
| **APIs cloud** | Código custom | `backend: "openai_api"` |

---

## Arquitectura

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────┐
│              config/models.yaml                     │
│           (Fuente única de verdad)                  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           ModelRegistry (Factory)                   │
│  ┌────────┬──────────┬──────────┬──────────┐       │
│  │  GGUF  │Transform │Multimodal│  Ollama  │       │
│  ├────────┼──────────┼──────────┼──────────┤       │
│  │OpenAI  │Embedding │ PyTorch  │  Config  │       │
│  │  API   │(Vectors) │Checkpoint│ (System) │       │
│  └────────┴──────────┴──────────┴──────────┘       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│        UnifiedModelWrapper (Interface)              │
│         LangChain Runnable + LCEL                   │
└─────────────────────────────────────────────────────┘
```

### Clases Principales

1. **`UnifiedModelWrapper`** (Clase base abstracta)
   - Interfaz: LangChain `Runnable`
   - Métodos: `invoke()`, `ainvoke()`, `stream()`, `batch()`
   - Lifecycle: `_load_model()`, `unload()`, `_ensure_loaded()`

2. **`ModelRegistry`** (Factory + Singleton)
   - Carga `config/models.yaml`
   - Crea wrappers según backend
   - Cache automático (misma instancia si re-solicitado)

3. **Backend Wrappers** (Implementaciones específicas)
   - `GGUFModelWrapper`: llama-cpp-python
   - `TransformersModelWrapper`: HuggingFace 4-bit
   - `MultimodalModelWrapper`: Qwen3-VL, Qwen-Omni
   - `OllamaModelWrapper`: API local
   - `OpenAIAPIWrapper`: GPT-4, Claude, Gemini
   - `EmbeddingModelWrapper`: EmbeddingGemma-300M
   - `PyTorchCheckpointWrapper`: TRM, MCP (futuro)

---

## Backends Soportados

### 1. GGUF (CPU Optimizado)

**Uso**: Modelos cuantizados para CPU con `llama-cpp-python`

```yaml
# config/models.yaml
lfm2:
  name: "LiquidAI-LFM2-1.2B"
  backend: "gguf"
  model_path: "models/cache/lfm2/lfm2-1.2b.Q4_K_M.gguf"
  n_ctx: 2048
  n_threads: 6
  temperature: 0.8
```

```python
# Uso
lfm2 = get_model("lfm2")
response = lfm2.invoke("Explica Python en 50 palabras")
```

### 2. Transformers (GPU con 4-bit)

**Uso**: Modelos HuggingFace con cuantización automática

```yaml
solar_gpu:
  name: "SOLAR-10.7B (GPU)"
  backend: "transformers"
  repo_id: "upstage/SOLAR-10.7B-Instruct-v1.0"
  load_in_4bit: true
  device_map: "auto"
```

```python
solar = get_model("solar_gpu")
response = solar.invoke("¿Cómo funciona la IA?")
```

### 3. Multimodal (Visión + Audio)

**Uso**: Modelos Qwen para imagen y audio

```yaml
qwen3_vl:
  name: "Qwen3-VL-4B-Instruct"
  backend: "multimodal"
  repo_id: "Qwen/Qwen3-VL-4B-Instruct"
  supports_images: true
  supports_video: true
```

```python
qwen = get_model("qwen3_vl")

# Procesar imagen
response = qwen.invoke({
    "text": "Describe esta imagen",
    "image": "path/to/image.jpg"
})
```

### 4. Ollama (API Local)

**Uso**: Modelos servidos por Ollama

```yaml
solar_short:
  name: "SOLAR-10.7B (Ollama)"
  backend: "ollama"
  api_url: "${OLLAMA_BASE_URL}"  # Resuelve env var
  model_name: "hf.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF:Q4_K_M"
```

```python
solar = get_model("solar_short")
response = solar.invoke("Hola, ¿cómo estás?")
```

### 5. OpenAI API (Cloud)

**Uso**: GPT-4, Claude, Gemini

```yaml
gpt4:
  name: "GPT-4 Turbo"
  backend: "openai_api"
  api_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model_name: "gpt-4-turbo-preview"
```

```python
gpt4 = get_model("gpt4")
response = gpt4.invoke("Explica cuántica")
```

### 6. Embedding (Vectores Semánticos)

**Uso**: EmbeddingGemma-300M para clasificación

```yaml
embeddings:
  name: "EmbeddingGemma-300M"
  backend: "embedding"
  repo_id: "google/embeddinggemma-300m-qat-q4_0-unquantized"
  embedding_dim: 768
```

```python
embeddings = get_model("embeddings")

# Single embedding
vector = embeddings.invoke("SARAi es una AGI local")
print(vector.shape)  # (768,)

# Batch
vectors = embeddings.batch_encode(["texto1", "texto2", "texto3"])
print(vectors.shape)  # (3, 768)
```

### 7. PyTorch Checkpoint (Sistema Interno)

**Uso**: TRM, MCP (componentes internos)

```yaml
trm_classifier:
  name: "TRM-Dual-7M"
  backend: "pytorch_checkpoint"
  checkpoint_path: "models/trm_classifier/checkpoint.pth"
  device: "cpu"
```

### 8. Config (Configuraciones)

**Uso**: No son modelos, solo configuración

```yaml
legacy_mappings:
  backend: "config"
  expert: solar_long
  tiny: lfm2
```

---

## Uso Básico

### 1. Cargar Modelo

```python
from core.unified_model_wrapper import get_model

# Forma simple
solar = get_model("solar_short")

# Con validación
try:
    model = get_model("modelo_inexistente")
except ValueError as e:
    print(f"Error: {e}")
```

### 2. Invocar (Síncrono)

```python
# Texto simple
response = solar.invoke("¿Qué es Python?")

# Con configuración
response = solar.invoke(
    "Explica IA",
    config={"temperature": 0.9, "max_tokens": 200}
)
```

### 3. Invocar (Asíncrono)

```python
import asyncio

async def main():
    response = await solar.ainvoke("Hola")
    print(response)

asyncio.run(main())
```

### 4. Streaming

```python
for chunk in solar.stream("Cuenta del 1 al 10"):
    print(chunk, end="", flush=True)
```

### 5. Batch Processing

```python
queries = [
    "¿Qué es Python?",
    "¿Qué es JavaScript?",
    "¿Qué es Rust?"
]

responses = solar.batch(queries)
for q, r in zip(queries, responses):
    print(f"Q: {q}\nA: {r}\n")
```

### 6. Listar Modelos Disponibles

```python
from core.unified_model_wrapper import list_available_models

models = list_available_models()
print(models)
# ['solar_short', 'solar_long', 'lfm2', 'qwen3_vl', 'embeddings', ...]
```

---

## Configuración

### Estructura de `models.yaml`

```yaml
nombre_modelo:
  # Metadata
  name: "Nombre Humano del Modelo"
  type: "text" | "multimodal" | "embedding" | "classifier" | "orchestrator"
  backend: "gguf" | "transformers" | "multimodal" | "ollama" | "openai_api" | "embedding" | "pytorch_checkpoint" | "config"
  
  # Configuración específica del backend
  # (ver sección de cada backend)
  
  # Gestión de memoria (opcional)
  load_on_demand: true | false  # Lazy loading
  priority: 1-10  # Prioridad de descarga (10 = máxima)
  max_memory_mb: 4096  # Límite de RAM
```

### Variables de Entorno

El wrapper resuelve automáticamente `${VARIABLE}`:

```yaml
api_url: "${OLLAMA_BASE_URL}"  # → http://<OLLAMA_HOST>:11434
api_key: "${OPENAI_API_KEY}"   # → sk-...
model_name: "${SOLAR_MODEL_NAME}"  # → solar-10.7b-q4_k_m
```

**Fallback**: Si la variable no existe, usa un valor por defecto o lanza error.

---

## Ejemplos Avanzados

### 1. Pipeline LCEL con Unified Wrapper

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Template
prompt = ChatPromptTemplate.from_template(
    "Responde en {idioma}: {pregunta}"
)

# Pipeline
chain = prompt | solar | StrOutputParser()

# Invocar
response = chain.invoke({
    "idioma": "español",
    "pregunta": "¿Qué es la IA?"
})
```

### 2. Fallback Chain

```python
from langchain_core.runnables import RunnableBranch

# Fallback: solar → lfm2 → error
chain = solar.with_fallbacks([
    get_model("lfm2"),
])

response = chain.invoke("Hola")
```

### 3. Multimodal con Imagen

```python
qwen = get_model("qwen3_vl")

# Desde archivo
response = qwen.invoke({
    "text": "¿Qué aparece en esta imagen?",
    "image": "diagrams/arquitectura.png"
})

# Desde base64
import base64
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = qwen.invoke({
    "text": "Describe esto",
    "image": f"data:image/jpeg;base64,{image_b64}"
})
```

### 4. Embeddings para Clasificación

```python
embeddings = get_model("embeddings")

# Clasificar intención
query = "¿Cómo instalar Python?"
categories = [
    "pregunta técnica",
    "saludo",
    "despedida",
    "consulta general"
]

# Generar embeddings
query_emb = embeddings.invoke(query)
category_embs = embeddings.batch_encode(categories)

# Similitud coseno
import numpy as np
similarities = np.dot(category_embs, query_emb)
best_match = categories[np.argmax(similarities)]

print(f"Categoría: {best_match}")  # → "pregunta técnica"
```

### 5. Gestión Manual de Memoria

```python
from core.unified_model_wrapper import ModelRegistry

registry = ModelRegistry()

# Ver modelos cargados
loaded = registry.get_loaded_models()
print(loaded)  # ['solar_short']

# Descargar manualmente
registry.unload_model("solar_short")

# Descargar TODOS
registry.unload_all()
```

---

## API Reference

### `get_model(name: str) -> UnifiedModelWrapper`

Obtiene wrapper para modelo especificado.

**Args:**
- `name`: Nombre del modelo (key en `models.yaml`)

**Returns:**
- Instancia de `UnifiedModelWrapper` (o subclase)

**Raises:**
- `ValueError`: Si el modelo no existe en config

**Ejemplo:**
```python
solar = get_model("solar_short")
```

---

### `list_available_models() -> List[str]`

Lista todos los modelos disponibles en `models.yaml`.

**Returns:**
- Lista de nombres de modelos

**Ejemplo:**
```python
models = list_available_models()
# ['solar_short', 'solar_long', 'lfm2', ...]
```

---

### `UnifiedModelWrapper.invoke(input, config=None) -> str`

Ejecuta el modelo de forma síncrona.

**Args:**
- `input`: `str`, `dict` (multimodal), o `List[BaseMessage]` (LangChain)
- `config`: Dict opcional con parámetros de generación

**Returns:**
- `str`: Respuesta del modelo (o `np.ndarray` para embeddings)

**Ejemplo:**
```python
response = model.invoke(
    "Explica Python",
    config={"temperature": 0.9, "max_tokens": 200}
)
```

---

### `UnifiedModelWrapper.ainvoke(input, config=None) -> str`

Versión asíncrona de `invoke()`.

**Ejemplo:**
```python
response = await model.ainvoke("Hola")
```

---

### `UnifiedModelWrapper.stream(input, config=None) -> Iterator[str]`

Genera respuesta en streaming.

**Ejemplo:**
```python
for chunk in model.stream("Cuenta hasta 10"):
    print(chunk, end="")
```

---

### `UnifiedModelWrapper.batch(inputs: List) -> List[str]`

Procesa múltiples inputs en batch.

**Ejemplo:**
```python
responses = model.batch(["query1", "query2", "query3"])
```

---

### `UnifiedModelWrapper.unload()`

Descarga el modelo de memoria manualmente.

**Ejemplo:**
```python
model.unload()
```

---

## Troubleshooting

### Error: "Model not found in config"

**Causa**: El modelo no existe en `models.yaml`

**Solución**:
```python
# Ver modelos disponibles
from core.unified_model_wrapper import list_available_models
print(list_available_models())

# O agregar modelo a config/models.yaml
```

---

### Error: "Ollama server not available"

**Causa**: Servidor Ollama no está corriendo o URL incorrecta

**Solución**:
```bash
# Verificar Ollama
curl http://<OLLAMA_HOST>:11434/api/tags

# O configurar variable de entorno
export OLLAMA_BASE_URL="http://localhost:11434"
```

---

### Error: "sentence-transformers not installed"

**Causa**: Dependencia faltante para embeddings

**Solución**:
```bash
pip install sentence-transformers
# O usar implementación directa (ya en v2.14)
```

---

### Modelo carga muy lento

**Causa**: Primera carga de modelo grande

**Solución**:
- Usar `load_on_demand: false` en config para precarga
- Reducir `n_ctx` para modelos GGUF
- Usar cuantización más agresiva (Q4_K_M → Q2_K)

---

### RAM se agota

**Causa**: Demasiados modelos cargados

**Solución**:
```python
# Ver modelos cargados
registry = ModelRegistry()
loaded = registry.get_loaded_models()

# Descargar los que no usas
for model in loaded:
    if model not in ["lfm2"]:  # Mantener solo esenciales
        registry.unload_model(model)
```

---

## Migración desde model_pool

### Antes (v2.13 - model_pool)

```python
from core.model_pool import ModelPool

pool = ModelPool(config)
solar = pool.get("expert")
response = solar.generate(
    prompt="Hola",
    temperature=0.7,
    max_tokens=100
)
pool.release("expert")
```

### Después (v2.14 - Unified Wrapper)

```python
from core.unified_model_wrapper import get_model

solar = get_model("solar_long")  # legacy mapping: expert → solar_long
response = solar.invoke(
    "Hola",
    config={"temperature": 0.7, "max_tokens": 100}
)
# Auto-release (garbage collection automático)
```

### Legacy Mappings

Para compatibilidad temporal, existe mapeo automático:

```yaml
# config/models.yaml
legacy_mappings:
  backend: "config"
  expert: solar_long
  expert_short: solar_short
  expert_long: solar_long
  tiny: lfm2
  multimodal: qwen3_vl
```

```python
# Funciona con nombres legacy
expert = get_model("expert")  # → Resuelve a solar_long
tiny = get_model("tiny")      # → Resuelve a lfm2
```

**Recomendación**: Migrar a nombres nuevos para claridad.

---

## Mejores Prácticas

### 1. YAML sobre Código

❌ **No hacer**:
```python
# Hard-coded en Python
solar = Llama(model_path="/path/to/model", n_ctx=512, ...)
```

✅ **Hacer**:
```yaml
# config/models.yaml
solar_custom:
  backend: "gguf"
  model_path: "/path/to/model"
  n_ctx: 512
```
```python
solar = get_model("solar_custom")
```

---

### 2. Usar Config para Parámetros Runtime

❌ **No hacer**:
```python
# Hard-coded temperature
response = model.invoke("texto")
```

✅ **Hacer**:
```python
# Flexible via config
response = model.invoke(
    "texto",
    config={"temperature": 0.9, "max_tokens": 200}
)
```

---

### 3. Lazy Loading para Modelos Grandes

```yaml
qwen3_vl:
  load_on_demand: true  # Solo carga cuando se necesita
  priority: 7  # Descarga primero si hay presión de RAM
```

---

### 4. Validación de Input

```python
def safe_invoke(model, input_text):
    try:
        return model.invoke(input_text)
    except Exception as e:
        logger.error(f"Error en {model.name}: {e}")
        return "Lo siento, ocurrió un error."
```

---

## Roadmap v2.15+

- [ ] **Backend PyTorchCheckpoint real** (actualmente solo config)
- [ ] **Auto-tuning de parámetros** basado en hardware
- [ ] **Distributed inference** (multi-GPU)
- [ ] **Model versioning** en YAML
- [ ] **Telemetría integrada** (latencia, RAM, tokens/s)
- [ ] **Hot-reload de config** sin reinicio

---

## Contribuir

Para agregar un nuevo backend:

1. Crear clase en `core/unified_model_wrapper.py`:
   ```python
   class MiBackendWrapper(UnifiedModelWrapper):
       def _load_model(self) -> Any:
           # Tu lógica
           pass
       
       def _invoke_sync(self, input, config) -> str:
           # Tu lógica
           pass
   ```

2. Registrar en `ModelRegistry.get_model()`:
   ```python
   elif backend == "mi_backend":
       wrapper = MiBackendWrapper(name, config)
   ```

3. Agregar tests en `tests/test_unified_wrapper_integration.py`

4. Documentar en este archivo

---

## Licencia

MIT License - SARAi v2.14  
© 2025 iagenerativa

---

**Última actualización**: 1 de noviembre de 2025  
**Autor**: GitHub Copilot + SARAi Team  
**Estado**: ✅ PRODUCCIÓN
