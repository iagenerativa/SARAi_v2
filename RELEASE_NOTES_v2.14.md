# SARAi v2.14.0 - Unified Wrapper + VisCoder2

**Fecha de Release**: 1 de noviembre de 2025  
**Tag**: v2.14.0  
**Rama**: master  
**Commit**: a007d2a

---

## 🎉 Resumen Ejecutivo

v2.14 introduce una capa de abstracción unificada (Unified Wrapper) para 8 backends de modelos LLM, más la integración de VisCoder2-7B como especialista de programación vía Ollama. **Sin breaking changes**, mantiene 100% compatibilidad con v2.12 (Phoenix Skills) y v2.13 (Layer Architecture).

### Logros Clave

✅ **Overhead <5% validado**: Wrapper -3.87% (más rápido que API directa) en Ollama, 2-3% en Embeddings  
✅ **Cache 36× más rápida**: Embeddings 2.2s → 61ms en llamadas repetidas  
✅ **VisCoder2 integrado**: Generación de código 19% más rápida que SOLAR  
✅ **Tests 100%**: 13/13 wrapper + tests específicos VisCoder2  
✅ **Documentación completa**: 1,200 LOC (guía + ejemplos + benchmark)

---

## 📦 Componentes Principales

### 1. Unified Model Wrapper

**Archivo**: `core/unified_model_wrapper.py`

**8 Backends soportados**:
1. **gguf** - llama-cpp-python (CPU)
2. **transformers** - HuggingFace 4-bit
3. **multimodal** - Qwen3-VL, Qwen-Omni
4. **ollama** - SOLAR, VisCoder2 (servidor remoto)
5. **openai_api** - GPT-4, Claude, Gemini
6. **embedding** - EmbeddingGemma-300M
7. **pytorch_checkpoint** - TRM, MCP
8. **config** - System configs

**API Unificada**:
```python
from core.unified_model_wrapper import get_model

# Uso simple
model = get_model("solar_short")
response = model.invoke("¿Qué es la IA?")

# Async
response = await model.ainvoke("Explica Python")

# Streaming
for chunk in model.stream("Cuenta hasta 10"):
    print(chunk, end="")

# Batch
responses = model.batch(["Query 1", "Query 2"])
```

**Beneficios**:
- Config-driven: YAML único (`config/models.yaml`)
- Backend switching sin cambios de código
- Cache automático por modelo
- Error handling robusto

### 2. VisCoder2-7B Integration

**Modelo**: `hf.co/mradermacher/VisCoder2-7B-GGUF:Q4_K_M`  
**Backend**: Ollama (servidor compartido con SOLAR)  
**Especialización**: Generación de código

**Configuración**:
```yaml
# config/models.yaml
viscoder2:
  backend: "ollama"
  api_url: "${OLLAMA_BASE_URL}"
  model_name: "${VISCODER2_MODEL_NAME}"
  n_ctx: 4096
  temperature: 0.3
  specialty: "code_generation"
```

```python
# core/skill_configs.py
PROGRAMMING_SKILL = SkillConfig(
    name="programming",
    preferred_model="viscoder2",
    temperature=0.3,
    max_tokens=3072
)
```

**Rendimiento**:
- Latencia: 5.66s (simple), 11.08s (complejo)
- vs SOLAR: 19% más rápido en inferencia de código
- Calidad: Código limpio con docstrings y manejo de errores

---

## 📊 KPIs Validados

### Benchmark Overhead (Objetivo: <5%)

| Componente | Método | Overhead Real | Resultado |
|------------|--------|---------------|-----------|
| **Ollama (SOLAR)** | Direct API vs Wrapper | **-3.87%** | ✅ Wrapper MÁS RÁPIDO |
| **Embeddings** | Direct AutoModel vs Wrapper | **2-3%** | ✅ Primera inferencia |
| **Cache Embeddings** | Segunda llamada | **36× speedup** | ✅ 2,200ms → 61ms |

**Metodología**: 5 iteraciones + 1 warmup, servidor Ollama real (<OLLAMA_HOST>:11434)

### Tests

| Categoría | Tests | Estado |
|-----------|-------|--------|
| Unified Wrapper | 13/13 | ✅ PASSING |
| VisCoder2 Integration | 4/4 | ✅ PASSING |
| **Total** | **17/17** | **✅ 100%** |

### Documentación

| Archivo | LOC | Propósito |
|---------|-----|-----------|
| `docs/UNIFIED_WRAPPER_GUIDE.md` | 850 | Guía completa de uso |
| `examples/unified_wrapper_examples.py` | 450 | 15 ejemplos prácticos |
| `BENCHMARK_WRAPPER_OVERHEAD_v2.14.md` | 400 | Metodología y resultados |
| `STATUS_v2.14_FINAL.md` | 650 | Estado completo del sistema |
| **Total** | **2,350** | **Documentación** |

---

## 🚀 Archivos Nuevos/Modificados

### Creados (16 archivos)

**Documentación**:
- `docs/UNIFIED_WRAPPER_GUIDE.md`
- `BENCHMARK_WRAPPER_OVERHEAD_v2.14.md`
- `STATUS_v2.14_FINAL.md`
- `STATUS_EMBEDDINGS_INTEGRATION_v2.14.md`
- `STATUS_FASE3_v2.14_ACTUAL.md`
- `COMMIT_SUMMARY_v2.14.md`
- `SESSION_SUMMARY_01NOV2025.md`
- `docs/EXTERNAL_MODELS_DECISION_v2.14.md`
- `docs/EXTERNAL_MODELS_STRATEGY.md`
- `docs/SPECIALIZED_MODELS_DEPLOYMENT_NOTE_v2.14.md`
- `docs/SPECIALIZED_MODELS_IMPLEMENTATION_v2.14.md`

**Código**:
- `examples/unified_wrapper_examples.py`
- `scripts/benchmark_wrapper_overhead.py`
- `scripts/download_viscoder2.py`

**Tests**:
- `tests/test_unified_wrapper_integration.py`
- `tests/test_viscoder2_integration.py`

### Modificados (7 archivos)

- `.env` - Añadido `VISCODER2_MODEL_NAME`
- `config/models.yaml` - Entry `viscoder2` (backend ollama)
- `core/skill_configs.py` - `PROGRAMMING_SKILL` → viscoder2
- `core/unified_model_wrapper.py` - Backend implementations
- `pytest.ini` - Test configuration
- `README.md` - Sección Unified Wrapper + KPIs v2.14
- `.github/copilot-instructions.md` - Novedades v2.14–v2.18
- `tests/test_unified_wrapper.py` - Extended tests

**Total cambios**: 6,324 insertions, 82 deletions

---

## 🔧 Cómo Usar

### Instalación

```bash
# Clonar repo
git clone https://github.com/iagenerativa/SARAi_v2.git
cd SARAi_v2

# Checkout v2.14.0
git checkout v2.14.0

# Configurar .env
cp .env.example .env
# Editar OLLAMA_BASE_URL y VISCODER2_MODEL_NAME

# Instalar dependencias
pip install -e .
```

### Validar Instalación

```bash
# Test del wrapper
pytest tests/test_unified_wrapper_integration.py -v

# Test de VisCoder2
pytest tests/test_viscoder2_integration.py -v

# Benchmark (opcional)
python scripts/benchmark_wrapper_overhead.py
```

### Uso Básico

```python
# 1. Unified Wrapper
from core.unified_model_wrapper import get_model

solar = get_model("solar_short")
response = solar.invoke("Explica la cuantización Q4_K_M")
print(response)

# 2. VisCoder2 para código
viscoder = get_model("viscoder2")
code = viscoder.invoke("Escribe una función Python para calcular factorial")
print(code)

# 3. Via Skills (automático)
from core.graph import create_sarai_graph

graph = create_sarai_graph()
result = graph.invoke({"input": "Crea una clase Python para TODO list"})
# Automáticamente usa VisCoder2 por keywords "clase Python"
```

---

## 🐛 Issues Conocidos

1. **Env vars en scripts standalone**: Warnings si no se carga `.env` (fallbacks funcionan)
2. **Ollama timeout**: Si servidor remoto no responde en 120s, falla (configurable)
3. **VisCoder2 temperatura fija**: 0.3 hardcoded en config (cambiar en `models.yaml`)

**Workarounds**: Documentados en `docs/UNIFIED_WRAPPER_GUIDE.md` sección Troubleshooting

---

## 📈 Breaking Changes

**Ninguno**. v2.14 es 100% compatible con:
- v2.12 (Phoenix Skills)
- v2.13 (Layer Architecture)
- Código existente usando `core/model_pool.py` (wrapper NO reemplaza pool)

---

## 🎯 Próximos Pasos

### Hito 2 — v2.16 Omni-Loop (Planeado)

**Timeline**: 1–2 semanas  
**Enfoque**: Skills containerizados (draft, image, lora_trainer)  
**KPIs objetivo**:
- RAM P99 ≤ 9.9 GB
- Latency P50 ≤ 7.9s
- Auto-corrección ≥ 68%

Ver `ROADMAP_v2.16_OMNI_LOOP.md` para detalles.

### Hito 3 — v2.17 Capas 2–4 (Planeado)

**Timeline**: 1 semana  
**Enfoque**: Memoria (RAG), Fluidez (fillers), Orquestación (LoRA)  
**Estado**: Capa 1 completa, ver `STATUS_LAYER1_v2.17.md`

### Hito 4 — v2.18 Multiprocessing (Planeado)

**Timeline**: 1 semana  
**Enfoque**: TRUE full-duplex sin GIL  
**Beneficio**: STT -60%, interrupciones <10ms

---

## 🙏 Agradecimientos

- **llama-cpp-python**: Backend GGUF eficiente
- **Ollama**: Servidor compartido para SOLAR + VisCoder2
- **HuggingFace**: Modelos y transformers library
- **Comunidad VisCoder2**: Modelo especializado de código

---

## 📝 Cómo Citar

```bibtex
@software{sarai_v2_14,
  title = {SARAi v2.14: Unified Model Wrapper + VisCoder2},
  author = {SARAi Development Team},
  year = {2025},
  month = {11},
  version = {2.14.0},
  url = {https://github.com/iagenerativa/SARAi_v2/releases/tag/v2.14.0}
}
```

---

**Mantenedores**: SARAi Dev Team  
**Licencia**: Ver LICENSE_GUIDE.md  
**Soporte**: GitHub Issues
