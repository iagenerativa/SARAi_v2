# 🎯 RESUMEN EJECUTIVO: Unified Model Wrapper - Arquitectura Universal

**Fecha**: 31 Octubre 2025  
**Versión**: v2.14 REDISEÑO DEFINITIVO  
**Propósito**: Capa de abstracción que permita evolucionar SARAi sin refactorización masiva

---

## 💡 Visión del Usuario (Clarificada)

> "las capacidades multimodales no sólo están en la parte de QWEN3, quiero que el wrapper 
> **potencie TODAS las capacidades** que tiene SARAi y que me permita ofrecerle **nuevas capacidades**, 
> aprovechando la potencia de **langchain** para poder hacer nuevas implementaciones de modelos 
> **sin tener que hacer un gran desarrollo** y así preparamos a SARAi para poder **evolucionar en el futuro**, 
> cuando tenga menos limitaciones de hardware."

---

## ✅ Solución: Unified Model Wrapper

### Arquitectura en 3 Capas

```
┌─────────────────────────────────────────────────────────┐
│         CAPA 1: Abstracción Universal (Core)            │
│  UnifiedModelWrapper (LangChain Runnable interface)     │
│  - Text, Audio, Vision, Multimodal                      │
│  - Backend-agnostic (GGUF, Transformers, API, gRPC)     │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│         CAPA 2: Backend Implementations                 │
│  - GGUFModelWrapper (llama-cpp-python)                  │
│  - TransformersModelWrapper (HuggingFace 4-bit)         │
│  - MultimodalModelWrapper (Qwen3-VL, Qwen-Omni)         │
│  - OllamaModelWrapper (API local)                       │
│  - OpenAIAPIWrapper (GPT-4, Claude, Gemini, etc.)       │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│         CAPA 3: LangChain Pipelines (LCEL)              │
│  - create_text_pipeline() → simple text gen             │
│  - create_vision_pipeline() → multimodal                │
│  - create_hybrid_pipeline_with_fallback() → resilience  │
│  - create_video_conference_pipeline() → complex flow    │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Principios Arquitectónicos

### 1. **Configuración > Código** (Config-Driven)
- Modelos definidos en `config/models.yaml`
- Agregar modelo nuevo = 6 líneas YAML, 0 líneas Python
- Cambiar backend = editar YAML, sin tocar código

### 2. **LangChain Native**
- Interfaz `Runnable` universal
- Composición con LCEL (`|` operator)
- Streaming automático
- Async nativo

### 3. **Evolución Sin Refactorización**
- CPU → GPU = cambiar `backend:` en config
- Agregar GPT-4 = entry en YAML + API key
- Hardware upgrade = 0 cambios de código

### 4. **Modularidad Phoenix**
- Cada backend es un wrapper independiente
- Factory pattern automático
- Fallback integrado (GGUF → Transformers → API)

---

## 📐 Componentes Principales

### 1. `core/unified_model_wrapper.py` (500 LOC)

**Clases base**:
```python
class UnifiedModelWrapper(Runnable, ABC):
    """LangChain Runnable interface para TODOS los modelos"""
    def invoke(input: Union[str, Dict, List[BaseMessage]]) -> Any
    def ainvoke(...) -> Any  # Async
    def unload() -> None     # Liberar memoria
```

**Backend implementations**:
- `GGUFModelWrapper`: llama-cpp-python (CPU optimized)
- `TransformersModelWrapper`: HuggingFace 4-bit (GPU)
- `MultimodalModelWrapper`: Vision + Audio support
- `OllamaModelWrapper`: API local (futuro)
- `OpenAIAPIWrapper`: Cloud APIs (GPT-4, Claude, Gemini)

**Registry**:
```python
class ModelRegistry:
    """Carga modelos desde config/models.yaml"""
    def get_model(name: str) -> UnifiedModelWrapper
    def list_models() -> List[str]
    def unload_all()
```

---

### 2. `config/models.yaml` (100 LOC)

**Estructura**:
```yaml
solar_short:
  name: "SOLAR-10.7B-Instruct"
  type: "text"
  backend: "gguf"
  model_path: "models/gguf/solar.gguf"
  n_ctx: 512
  load_on_demand: false  # Siempre en memoria
  priority: 10

qwen3_vl:
  name: "Qwen3-VL-4B"
  type: "multimodal"
  backend: "transformers"
  repo_id: "Qwen/Qwen3-VL-4B"
  supports_images: true
  supports_video: true
  load_on_demand: true
  priority: 7

# Futuro: Solo descomentar cuando tengas API key
# gpt4_vision:
#   name: "gpt-4-vision-preview"
#   backend: "openai_api"
#   api_key: "${OPENAI_API_KEY}"
#   supports_images: true
```

---

### 3. `core/langchain_pipelines.py` (300 LOC)

**Pipelines LCEL**:

```python
# Pipeline 1: Text simple
pipeline = get_model("solar_short") | StrOutputParser()
response = pipeline.invoke("¿Qué es Python?")

# Pipeline 2: Vision
pipeline = get_model("qwen3_vl") | StrOutputParser()
response = pipeline.invoke({
    "text": "Describe",
    "image": "screenshot.jpg"
})

# Pipeline 3: Hybrid con fallback
pipeline = RunnableBranch(
    (has_image, get_model("qwen3_vl")),  # Si imagen → vision
    get_model("solar_long")               # Else → text-only
) | StrOutputParser()

# Pipeline 4: Video conference (multi-step)
pipeline = (
    RunnableParallel(
        visual=lambda x: process_frames(x["frames"]),
        audio=lambda x: transcribe(x["audio"])
    )
    | ChatPromptTemplate.from_template("Resume: {visual} {audio}")
    | get_model("solar_long")
    | StrOutputParser()
)
```

---

## ✅ Beneficios vs. Arquitectura Anterior

| Aspecto | Antes (model_pool.py) | Ahora (Unified Wrapper) |
|---------|----------------------|------------------------|
| **Agregar modelo** | Modificar Python (~100 LOC) | YAML entry (6 líneas) |
| **Cambiar backend** | Refactorizar _load() | Cambiar 1 palabra en YAML |
| **LangChain** | Wrapper manual | Runnable nativo |
| **Composición** | Código imperativo | LCEL declarativo (\|) |
| **Fallback** | Try-except manual | RunnableBranch integrado |
| **Async** | Thread hacks | ainvoke() nativo |
| **Streaming** | Complejo | Automático |
| **Multimodal** | Hard-coded | Unified interface |
| **Future-proof** | Cambios masivos | Config-driven (0 código) |

---

## 🚀 Casos de Uso: Evolutibilidad

### Caso 1: Agregar GPT-4 Vision (cuando tengas API key)

**Código Python**: 0 líneas  
**Configuración**: 
```yaml
# config/models.yaml
gpt4_vision:
  backend: "openai_api"
  api_key: "${OPENAI_API_KEY}"
```

**Uso inmediato**:
```python
gpt4 = get_model("gpt4_vision")
response = gpt4.invoke({"text": "Analiza", "image": "img.jpg"})
```

---

### Caso 2: Migrar SOLAR a GPU (cuando mejore hardware)

**ANTES (CPU)**:
```yaml
solar_short:
  backend: "gguf"
  model_path: "models/gguf/solar.gguf"
```

**DESPUÉS (GPU)**:
```yaml
solar_short:
  backend: "transformers"
  repo_id: "upstage/SOLAR-10.7B-Instruct-v1.0"
```

**Código Python**: SIN CAMBIOS (interfaz unificada)

---

### Caso 3: Video conferencia con Gemini Pro

```yaml
# Solo agregar en models.yaml
gemini_vision:
  backend: "openai_api"
  api_url: "https://generativelanguage.googleapis.com/v1"
  api_key: "${GOOGLE_API_KEY}"
```

```python
# Pipeline sin cambios, solo swap
pipeline = create_video_conference_pipeline()
# Internamente usa gemini en vez de qwen3_vl
```

---

## 📊 Plan de Implementación

### Fase 1: Core Wrapper (4h)
- [ ] `core/unified_model_wrapper.py` (500 LOC)
  * Clase base `UnifiedModelWrapper`
  * `GGUFModelWrapper`, `MultimodalModelWrapper`
  * `ModelRegistry` con factory

- [ ] `config/models.yaml` (100 LOC)
  * Modelos actuales (SOLAR, LFM2, Qwen3-VL)

- [ ] Tests `tests/test_unified_wrapper.py` (200 LOC)

**Tiempo**: 4 horas

---

### Fase 2: LangChain Pipelines (3h)
- [ ] `core/langchain_pipelines.py` (300 LOC)
  * `create_text_pipeline()`
  * `create_vision_pipeline()`
  * `create_hybrid_pipeline_with_fallback()`
  * `create_video_conference_pipeline()`

- [ ] Tests `tests/test_pipelines.py` (150 LOC)

**Tiempo**: 3 horas

---

### Fase 3: Integración Graph (2h)
- [ ] Refactorizar `core/graph.py`
  * Nodos LangChain (no funciones imperativas)
  * Migrar `model_pool` → `ModelRegistry`

**Tiempo**: 2 horas

---

### Fase 4: Documentación (1h)
- [ ] README: "Agregar nuevos modelos"
- [ ] Ejemplos pipelines LCEL

**Tiempo**: 1 hora

---

**TOTAL IMPLEMENTACIÓN**: ~10 horas

**ROI**: 
- Agregar GPT-4 en futuro: 5 minutos vs 5 horas antes
- Migrar a GPU: 10 minutos vs 20 horas antes
- Mantener compatibilidad: Automática vs manual

---

## 🎯 KPIs de esta Arquitectura

| KPI | Objetivo | Método de Medición |
|-----|----------|-------------------|
| **LOC nuevos modelos** | ≤10 líneas | Agregar GPT-4: 6 líneas YAML |
| **Tiempo agregar modelo** | ≤10 min | Config + restart |
| **Compatibilidad LangChain** | 100% | Runnable interface completo |
| **Fallback automático** | ✅ | RunnableBranch integrado |
| **RAM overhead** | 0 GB | Misma gestión que model_pool |
| **Latencia overhead** | ≤5% | Abstracción ligera |
| **Tests coverage** | ≥90% | Pytest con mocks |

---

## 💬 Recomendación Final

**Implementar Unified Model Wrapper AHORA** por estas razones:

1. ✅ **Cumple visión del usuario**: Evolución sin refactorización
2. ✅ **LangChain native**: Aprovecha toda la potencia del framework
3. ✅ **Future-proof**: GPT-4, Gemini, cualquier modelo futuro
4. ✅ **Config-driven**: YAML > código Python
5. ✅ **Tiempo acotado**: 10 horas vs beneficios a largo plazo
6. ✅ **No bloquea otras fases**: Se integra con FASE 3, 4, 5

**Siguiente acción sugerida**:
```
¿Procedo con implementación del Unified Model Wrapper?

Fase 1 (4h): Core wrapper + Registry
Fase 2 (3h): LangChain pipelines LCEL
Fase 3 (2h): Integración en graph.py
Fase 4 (1h): Documentación

Total: 10h de trabajo enfocado
```

---

**Mantra Definitivo v2.14**:  
_"SARAi no debe conocer sus modelos. Solo debe invocar capacidades.  
YAML define, LangChain orquesta, el wrapper abstrae.  
Cuando el hardware mejore, solo cambiamos configuración, nunca código."_
