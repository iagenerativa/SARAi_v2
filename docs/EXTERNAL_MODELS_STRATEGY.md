# 🔍 Estrategia de Modelos Externos - Análisis Crítico

**Fecha**: 2025-11-01  
**Contexto**: Replanteamiento de modelos externos para tareas específicas  
**Objetivo**: Validar si LLaMarketing-8B, Qwen2.5-Coder-7B, y SOLAR-10.7B son necesarios

---

## 🎯 Modelos Propuestos (Evaluación)

### 1. LLaMarketing-8B (Marketing Especializado)

| Aspecto | Análisis |
|---------|----------|
| **Propósito** | Generación de contenido de marketing (copywriting, ads, campañas) |
| **Tamaño** | 8B parámetros (~5 GB GGUF Q4_K_M) |
| **Caso de uso** | Marketing copy, social media, SEO content |
| **Alternativa** | SOLAR-10.7B + skill_marketing (prompt especializado) |
| **Justificación** | ❓ ¿SARAi necesita ser experto en marketing? |

**🔴 RECOMENDACIÓN**: **NO incluir** (fuera del scope de SARAi v2.14)

**Razones**:
1. SARAi es un asistente **técnico + emocional**, no de marketing
2. SOLAR-10.7B ya cubre generación de texto profesional
3. 5 GB adicionales de RAM sin justificación técnica
4. **Filosofía Phoenix v2.12**: Skills son prompts, no modelos separados

**Si realmente necesario** (futuro):
- Implementar `skill_marketing` con prompt especializado en SOLAR
- Temperatura alta (0.9) para creatividad
- System prompt: _"Eres un experto en marketing digital..."_

---

### 2. Qwen2.5-Coder-7B (Desarrollo/Código)

| Aspecto | Análisis |
|---------|----------|
| **Propósito** | Generación de código, debugging, code review |
| **Tamaño** | 7B parámetros (~4.5 GB GGUF Q4_K_M) |
| **Caso de uso** | Autocompletado, generación de funciones, refactoring |
| **Alternativa** | SOLAR-10.7B + skill_programming (ya implementado) |
| **Justificación** | ⚠️ Podría mejorar calidad de código, pero a qué costo? |

**🟡 RECOMENDACIÓN**: **CONDICIONAL** (solo si benchmark valida mejora >15%)

**Razones PRO**:
- Especialización real: Qwen2.5-Coder entrenado específicamente en código
- Podría superar a SOLAR en tasks de programación
- 4.5 GB manejables con ModelPool

**Razones CONTRA**:
- SOLAR-10.7B ya es muy competente en código (10.7B > 7B en general)
- **Filosofía Phoenix v2.12**: `skill_programming` ya existe
- Añade complejidad: ¿cuándo usar Coder vs SOLAR?

**Criterio de decisión**:
```python
# Benchmark necesario ANTES de incluir
benchmark_solar_programming = benchmark_code_quality(
    model="solar_short",
    skill="programming",
    queries=PROGRAMMING_QUERIES
)

benchmark_coder = benchmark_code_quality(
    model="qwen25_coder",
    queries=PROGRAMMING_QUERIES
)

if benchmark_coder["accuracy"] > benchmark_solar_programming["accuracy"] * 1.15:
    # Solo si mejora >15%, justifica la inclusión
    add_model("qwen25_coder")
else:
    # SOLAR + skill_programming es suficiente
    pass
```

**Implementación tentativa** (comentada):
```yaml
# qwen25_coder:
#   name: "Qwen2.5-Coder-7B-Instruct"
#   type: "text"
#   backend: "ollama"  # O "gguf" si local
#   
#   # Solo para skill_programming
#   skill_override: "programming"  # Reemplaza SOLAR en este skill
#   
#   api_url: "${OLLAMA_BASE_URL}"
#   model_name: "qwen2.5-coder:7b-instruct"
#   
#   n_ctx: 2048
#   temperature: 0.3  # Determinista para código
#   max_tokens: 1024
#   
#   load_on_demand: true
#   priority: 7
#   max_memory_mb: 0  # Ollama remoto
```

---

### 3. SOLAR-10.7B (Razonamiento General)

| Aspecto | Análisis |
|---------|----------|
| **Propósito** | Modelo general hard-skills (ya implementado) |
| **Tamaño** | 10.7B parámetros (~6 GB GGUF) |
| **Caso de uso** | Razonamiento técnico, explicaciones, análisis |
| **Status** | ✅ **YA IMPLEMENTADO** en v2.14 |
| **Backend** | Ollama (192.168.0.251:11434) |

**✅ RECOMENDACIÓN**: **MANTENER** (ya operacional)

**Razones**:
- Core del sistema de hard-skills
- 0 RAM local (servidor Ollama remoto)
- Altamente competente en razonamiento general
- 10.7B es sweet-spot (ni muy grande ni muy pequeño)

**Configuración actual** (validada):
```yaml
solar_short:
  backend: "ollama"
  api_url: "${OLLAMA_BASE_URL}"
  model_name: "${SOLAR_MODEL_NAME}"
  n_ctx: 512
  priority: 9

solar_long:
  backend: "ollama"
  api_url: "${OLLAMA_BASE_URL}"
  model_name: "${SOLAR_MODEL_NAME}"
  n_ctx: 2048
  priority: 8
```

**Variables .env**:
```bash
OLLAMA_BASE_URL=http://192.168.0.251:11434
SOLAR_MODEL_NAME=una-solar-10.7b-instruct-v1.0-q4
```

---

## 📊 Análisis de RAM (Peor Caso)

### Escenario Actual v2.14 (SOLO modelos core)

| Modelo | Backend | RAM Local | Estado |
|--------|---------|-----------|--------|
| **LFM2-1.2B** | GGUF local | ~700 MB | Siempre en memoria |
| **SOLAR-10.7B** | Ollama remoto | 0 MB | On-demand |
| **Qwen3-VL-4B** | Multimodal local | ~3.5 GB | On-demand |
| **Total P99** | - | **~4.2 GB** | ✅ |

**Margen disponible**: 16 GB - 4.2 GB = **11.8 GB libre** ✅

---

### Escenario Propuesto (con modelos externos)

| Modelo | Backend | RAM Local | Justificación |
|--------|---------|-----------|---------------|
| LFM2-1.2B | GGUF local | ~700 MB | Core soft-skills |
| SOLAR-10.7B | Ollama remoto | 0 MB | Core hard-skills |
| Qwen3-VL-4B | Multimodal local | ~3.5 GB | Vision |
| **LLaMarketing-8B** | Ollama remoto | 0 MB | ❌ NO justificado |
| **Qwen2.5-Coder-7B** | Ollama remoto | 0 MB | ⚠️ Condicional |
| **Total P99** | - | **~4.2 GB** | ✅ |

**Análisis**:
- Si todos los modelos externos usan **Ollama remoto** → 0 impacto RAM
- PERO: Añade **complejidad de routing** sin beneficio claro
- **Filosofía Phoenix violada**: Skills deben ser configs, no modelos

---

## 🎯 Recomendación Final

### ✅ INCLUIR en .env (Mínimo Viable)

```bash
# ============================================
# MODELOS EXTERNOS - OLLAMA REMOTO
# ============================================

# Servidor Ollama (desarrollo/producción)
OLLAMA_BASE_URL=http://192.168.0.251:11434

# Modelo Expert (hard-skills)
SOLAR_MODEL_NAME=una-solar-10.7b-instruct-v1.0-q4

# NOTA: LFM2 es local GGUF, no usa Ollama
```

**Total modelos externos**: **1** (SOLAR-10.7B)

---

### 🔵 OPCIONAL en .env (Futuro, si benchmarks lo justifican)

```bash
# ============================================
# MODELOS ESPECIALIZADOS (OPCIONALES)
# ============================================

# Qwen2.5-Coder (solo si benchmark valida >15% mejora)
# Descomentar cuando:
#   1. Benchmark programming accuracy (SOLAR vs Coder) ejecutado
#   2. Mejora >15% en code generation
#   3. Validación manual de calidad de código
# QWEN_CODER_MODEL_NAME=qwen2.5-coder:7b-instruct

# LLaMarketing (NO RECOMENDADO - fuera de scope)
# SARAi es asistente técnico+emocional, no de marketing
# Si necesario, usar SOLAR + skill_marketing (prompt especializado)
# LLAMARKETING_MODEL_NAME=llamarketing-8b
```

---

### ❌ NO INCLUIR (Violación de Filosofía Phoenix)

- **LLaMarketing-8B**: Fuera del scope técnico de SARAi
- **Modelos de marketing**: Implementar como skills (prompts), no modelos separados
- **Modelos genéricos**: SOLAR-10.7B ya cubre razonamiento general

---

## 📝 Criterios de Decisión para Futuros Modelos Externos

### ✅ SÍ incluir cuando:

1. **Especialización demostrable**: Benchmark valida >15% mejora vs SOLAR
2. **Caso de uso específico**: No cubierto por skills existentes
3. **RAM-neutral**: Ollama remoto (0 RAM local) o justifica el costo
4. **Filosofía preservada**: Skills siguen siendo prompts, modelo es engine

### ❌ NO incluir cuando:

1. **Skill puede hacerlo**: Prompt especializado en SOLAR suficiente
2. **Fuera de scope**: Marketing, finanzas, legal (SARAi = técnico+emocional)
3. **Mejora marginal**: <15% de mejora vs SOLAR
4. **Complejidad injustificada**: Routing complejo sin beneficio claro

---

## 🚀 Plan de Acción Propuesto

### Fase 1: v2.14 (AHORA) - Minimalista

1. ✅ Mantener SOLAR-10.7B como único modelo externo
2. ✅ .env con solo `OLLAMA_BASE_URL` y `SOLAR_MODEL_NAME`
3. ✅ Skills implementados como prompts (programming, diagnosis, etc.)

**Archivo .env propuesto**:
```bash
# SARAi v2.14 - Configuración de Modelos Externos
OLLAMA_BASE_URL=http://192.168.0.251:11434
SOLAR_MODEL_NAME=una-solar-10.7b-instruct-v1.0-q4
```

**Total complejidad**: Mínima ✅  
**Total RAM local**: 4.2 GB ✅  
**Filosofía Phoenix**: Preservada ✅

---

### Fase 2: v2.15 (FUTURO) - Validación Condicional

Si y solo si benchmarks lo justifican:

1. ⚠️ Benchmark SOLAR vs Qwen2.5-Coder en programming tasks
2. ⚠️ Validar mejora >15% en code quality
3. ⚠️ Implementar routing condicional: `skill_programming` → Coder
4. ⚠️ Añadir a .env solo si validación positiva

**Criterio de éxito**:
```python
# Ejecutar ANTES de incluir Qwen2.5-Coder
make benchmark VERSION=v2.14_solar
make benchmark VERSION=v2.15_coder
make benchmark-compare OLD=v2.14_solar NEW=v2.15_coder

# SI programming_accuracy mejora >15% → Incluir
# SI mejora <15% → Mantener SOLAR + skill_programming
```

---

### Fase 3: v2.16+ (LARGO PLAZO) - Evaluación Continua

- Revisar nuevos modelos especializados cada 6 meses
- Aplicar criterios de decisión estrictos
- Priorizar filosofía Phoenix sobre especialización

---

## 💡 Filosofía Final

**Mantra v2.14 Modelos Externos**:

> _"Un modelo especializado debe probar con benchmarks que es >15% mejor  
> que SOLAR + skill_config. Si no lo prueba, es solo ruido.  
> SARAi prefiere la simplicidad documentada sobre la complejidad especulativa."_

**Regla de oro**:
```
Nuevo modelo externo = Benchmark obligatorio ANTES de incluir
Sin benchmark = Sin inclusión
```

**Prioridades**:
1. 🥇 Filosofía Phoenix (skills = configs)
2. 🥈 Simplicidad operacional
3. 🥉 Especialización validada (solo si benchmark lo prueba)

---

## 📊 Resumen Ejecutivo

| Modelo | Incluir v2.14? | Razón | Alternativa |
|--------|----------------|-------|-------------|
| **SOLAR-10.7B** | ✅ SÍ | Core hard-skills, ya operacional | N/A |
| **LLaMarketing-8B** | ❌ NO | Fuera de scope técnico | SOLAR + skill_marketing |
| **Qwen2.5-Coder-7B** | 🟡 CONDICIONAL | Solo si benchmark >15% mejora | SOLAR + skill_programming |

**Decisión final v2.14**: 
- `.env` con **solo SOLAR-10.7B** (minimalista)
- Evaluar Qwen2.5-Coder en v2.15 **solo si benchmark lo justifica**
- NO incluir LLaMarketing (violación de scope)

---

**Creado**: 2025-11-01  
**Versión**: v2.14  
**Status**: Propuesta para validación  
**Next**: Crear .env minimalista con solo SOLAR-10.7B
