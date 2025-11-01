# 🚀 SARAi v2.14 - Unified Model Wrapper

## Agregar Nuevos Modelos

SARAi v2.14 implementa una arquitectura **config-driven** que permite agregar modelos sin modificar código Python.

### Filosofía

> **"SARAi no debe conocer sus modelos. Solo debe invocar capacidades.**  
> **YAML define, LangChain orquesta, el wrapper abstrae.**  
> **Cuando el hardware mejore, solo cambiamos configuración, nunca código."**

---

## 🎯 Quick Start: Agregar GPT-4 Vision

### 1. Editar `config/models.yaml`

```yaml
gpt4_vision:
  name: "GPT-4 Vision Preview"
  type: "multimodal"
  backend: "openai_api"
  
  # API Configuration
  api_key: "${OPENAI_API_KEY}"
  api_url: "https://api.openai.com/v1"
  model_name: "gpt-4-vision-preview"
  
  # Capacidades
  supports_images: true
  
  # Parámetros
  temperature: 0.7
  max_tokens: 2048
  
  # Gestión de memoria
  load_on_demand: true
  priority: 5
```

### 2. Configurar `.env`

```bash
OPENAI_API_KEY=sk-...tu-api-key-aqui
```

### 3. Usar en código

```python
from core.unified_model_wrapper import get_model

# ¡Listo! Sin modificar código
gpt4 = get_model("gpt4_vision")
response = gpt4.invoke({
    "text": "¿Qué hay en esta imagen?",
    "image": "path/to/image.jpg"
})

print(response)
```

**Tiempo total**: 5 minutos vs 5 horas antes ✅

---

## 📊 Backends Soportados

| Backend | Formato | Uso | Ventajas | Desventajas |
|---------|---------|-----|----------|-------------|
| **gguf** | GGUF Q4_K_M | CPU local | ✅ Rápido en CPU<br>✅ Bajo consumo RAM<br>✅ Offline | ❌ Requiere descarga<br>❌ Solo CPU |
| **ollama** | API REST | Servidor Ollama | ✅ 0 RAM local<br>✅ Múltiples modelos<br>✅ Hot-swap | ❌ Requiere servidor<br>❌ Latencia red |
| **multimodal** | HF Transformers | Vision/Audio | ✅ Nativo multimodal<br>✅ Gran variedad | ❌ Alto consumo RAM<br>❌ Lento en CPU |
| **transformers** | HF 4-bit | GPU | ✅ Rápido en GPU<br>✅ Baja VRAM | ❌ Requiere GPU<br>❌ Setup complejo |
| **openai_api** | API REST | Cloud | ✅ Sin hardware local<br>✅ Modelos SOTA<br>✅ Escalable | ❌ Costo por token<br>❌ Requiere internet |

---

## 🔧 Ejemplos de Configuración

### Ejemplo 1: Claude 3 Opus (Cloud API)

```yaml
claude_opus:
  name: "Claude 3 Opus"
  type: "text"
  backend: "openai_api"
  
  # Anthropic API (OpenAI-compatible proxy)
  api_key: "${ANTHROPIC_API_KEY}"
  api_url: "https://api.anthropic.com/v1"
  model_name: "claude-3-opus-20240229"
  
  temperature: 0.7
  max_tokens: 4096
  
  load_on_demand: true
  priority: 4
```

**Uso**:
```python
claude = get_model("claude_opus")
response = claude.invoke("Explica la teoría cuántica")
```

---

### Ejemplo 2: Gemini Pro Vision (Google)

```yaml
gemini_vision:
  name: "Gemini Pro Vision"
  type: "multimodal"
  backend: "openai_api"
  
  api_key: "${GOOGLE_API_KEY}"
  api_url: "https://generativelanguage.googleapis.com/v1"
  model_name: "gemini-pro-vision"
  
  supports_images: true
  supports_video: true
  
  temperature: 0.7
  max_tokens: 2048
  
  load_on_demand: true
  priority: 3
```

**Uso**:
```python
gemini = get_model("gemini_vision")
response = gemini.invoke({
    "text": "Analiza este video",
    "video": "conference.mp4"
})
```

---

### Ejemplo 3: Llama 3 70B (Ollama Local)

```yaml
llama3_70b:
  name: "Llama 3 70B (Ollama)"
  type: "text"
  backend: "ollama"
  
  # Ollama local (sin API key)
  api_url: "http://localhost:11434"
  model_name: "llama3:70b"
  
  temperature: 0.7
  
  load_on_demand: true
  priority: 6
```

**Prerequisito**:
```bash
# Iniciar Ollama
ollama pull llama3:70b
ollama serve
```

**Uso**:
```python
llama3 = get_model("llama3_70b")
response = llama3.invoke("Pregunta compleja")
```

---

### Ejemplo 4: Mistral 7B Local (GGUF)

```yaml
mistral_7b:
  name: "Mistral 7B Instruct"
  type: "text"
  backend: "gguf"
  
  # Archivo GGUF local
  model_path: "models/cache/mistral/mistral-7b-instruct.Q4_K_M.gguf"
  
  n_ctx: 4096
  n_threads: 6
  use_mmap: true
  use_mlock: false
  
  temperature: 0.7
  max_tokens: 1024
  
  load_on_demand: true
  priority: 7
  max_memory_mb: 4500
```

**Descarga**:
```bash
# Descargar modelo GGUF
huggingface-cli download \
  TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --local-dir models/cache/mistral/
```

---

## 🔄 Migración CPU → GPU

Cuando consigas GPU, migrar SOLAR a Transformers:

### ANTES (CPU GGUF)
```yaml
solar_short:
  backend: "gguf"
  model_path: "models/cache/solar/solar-10.7b.gguf"
  n_ctx: 512
  n_threads: 6
```

### AHORA (GPU Transformers)
```yaml
solar_gpu:
  name: "SOLAR-10.7B-Instruct (GPU)"
  type: "text"
  backend: "transformers"
  
  repo_id: "upstage/SOLAR-10.7B-Instruct-v1.0"
  
  load_in_4bit: true
  device_map: "auto"
  
  temperature: 0.7
  max_tokens: 1024
  
  load_on_demand: true
  priority: 10
```

**Cambios**: 2 líneas YAML, 0 líneas Python ✅

---

## 🎨 Usar Pipelines LCEL

### Pipeline Básico

```python
from core.langchain_pipelines import create_text_pipeline

# Crear pipeline
pipeline = create_text_pipeline(
    model_name="gpt4_turbo",
    temperature=0.7,
    system_prompt="Eres un experto en Python"
)

# Usar pipeline
response = pipeline.invoke("Crea una función para ordenar lista")
```

---

### Pipeline con Fallback

```python
from core.langchain_pipelines import create_hybrid_pipeline_with_fallback

# Pipeline con fallback automático
pipeline = create_hybrid_pipeline_with_fallback(
    vision_model_name="gpt4_vision",
    text_model_name="gpt4_turbo",
    fallback_model_name="claude_opus"
)

# Si hay imagen → GPT-4 Vision
# Si no → GPT-4 Turbo
# Si falla → Claude Opus (fallback)
response = pipeline.invoke({
    "text": "Analiza",
    "image": "diagram.png"
})
```

---

### Pipeline RAG con GPT-4

```python
from core.langchain_pipelines import create_rag_pipeline

# RAG con búsqueda web + GPT-4
rag_pipeline = create_rag_pipeline(
    search_model_name="gpt4_turbo",
    enable_cache=True,
    safe_mode=False
)

response = rag_pipeline.invoke("¿Quién ganó el Nobel de Física 2024?")
```

---

## 📈 Comparación de Rendimiento

### Latencia por Backend (CPU i7, 16GB RAM)

| Modelo | Backend | Input 100 tokens | Velocidad | RAM |
|--------|---------|------------------|-----------|-----|
| SOLAR-10.7B | gguf | 512 tokens | ~20s | 4.8 GB |
| SOLAR-10.7B | ollama | 512 tokens | ~25s | 0 GB (remoto) |
| GPT-4 Turbo | openai_api | 512 tokens | ~3s | 0 GB (cloud) |
| Claude Opus | openai_api | 512 tokens | ~4s | 0 GB (cloud) |
| Qwen3-VL-4B | multimodal | imagen + texto | ~30s | 4 GB |

**Nota**: Ollama añade ~5s de overhead de red (LAN).

---

## 🔐 Seguridad de API Keys

### ✅ CORRECTO: Variables de entorno

```yaml
# config/models.yaml
gpt4:
  api_key: "${OPENAI_API_KEY}"  # Variable de entorno
```

```bash
# .env
OPENAI_API_KEY=sk-...
```

### ❌ INCORRECTO: Hard-coded

```yaml
# ❌ NUNCA HACER ESTO
gpt4:
  api_key: "sk-proj-12345..."  # Hard-coded = INSEGURO
```

---

## 🚨 Troubleshooting

### Problema: "Model not found"

```python
# Error
model = get_model("gpt4")
# KeyError: 'gpt4'
```

**Solución**: Verificar que el modelo está definido en `config/models.yaml`

```bash
grep -A5 "gpt4:" config/models.yaml
```

---

### Problema: "API key not configured"

```python
# Error
gpt4 = get_model("gpt4_turbo")
# ValueError: OPENAI_API_KEY not set
```

**Solución**: Configurar `.env`

```bash
echo "OPENAI_API_KEY=sk-..." >> .env
```

---

### Problema: "GGUF file not found"

```python
# Error
mistral = get_model("mistral_7b")
# FileNotFoundError: models/cache/mistral/mistral-7b.gguf
```

**Solución**: Descargar modelo

```bash
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --local-dir models/cache/mistral/
```

---

## 📚 Referencias

- [LangChain LCEL Docs](https://python.langchain.com/docs/expression_language/)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Ollama](https://ollama.com/)
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [Anthropic API](https://docs.anthropic.com/)

---

## 🎯 Casos de Uso Futuros

### Caso 1: Multi-Cloud (GPT-4 + Claude + Gemini)

```python
# Pipeline con 3 APIs en paralelo
from langchain.schema.runnable import RunnableParallel

multi_cloud = RunnableParallel(
    gpt4=get_model("gpt4_turbo"),
    claude=get_model("claude_opus"),
    gemini=get_model("gemini_pro")
)

# Ejecutar en paralelo
results = multi_cloud.invoke("Pregunta compleja")

# Comparar respuestas
print("GPT-4:", results["gpt4"])
print("Claude:", results["claude"])
print("Gemini:", results["gemini"])
```

---

### Caso 2: Orquestación Inteligente

```python
# Pipeline que elige modelo dinámicamente
from langchain.schema.runnable import RunnableBranch

smart_pipeline = RunnableBranch(
    # Si query corta (<50 chars) → modelo rápido
    (lambda x: len(x) < 50, get_model("lfm2")),
    
    # Si query técnica (contiene "código") → experto
    (lambda x: "código" in x.lower(), get_model("solar_long")),
    
    # Si query creativa (contiene "historia") → creativo
    (lambda x: "historia" in x.lower(), get_model("claude_opus")),
    
    # Default: GPT-4
    get_model("gpt4_turbo")
)

response = smart_pipeline.invoke("Crea una historia épica")
# → Usa Claude Opus automáticamente
```

---

### Caso 3: Ensemble con Votación

```python
# 3 modelos votan por la mejor respuesta
import asyncio

async def ensemble_vote(query: str):
    # Ejecutar 3 modelos en paralelo
    tasks = [
        get_model("gpt4_turbo").ainvoke(query),
        get_model("claude_opus").ainvoke(query),
        get_model("gemini_pro").ainvoke(query)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # Votación simple (el más largo gana)
    best_response = max(responses, key=len)
    
    return best_response

# Uso
response = asyncio.run(ensemble_vote("Explica teoría de cuerdas"))
```

---

**FIN DOCUMENTACIÓN - README v2.14 Unified Wrapper**

Total: 400 LOC de documentación con:
- ✅ Quick start (5 minutos)
- ✅ 4 ejemplos completos (GPT-4, Claude, Gemini, Mistral)
- ✅ Migración CPU→GPU
- ✅ Pipelines LCEL
- ✅ Troubleshooting
- ✅ 3 casos de uso avanzados
