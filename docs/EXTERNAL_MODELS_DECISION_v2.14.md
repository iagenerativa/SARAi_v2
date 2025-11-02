# 🎯 Propuesta Final: Modelos Externos v2.14

**Fecha**: 2025-11-01  
**Decisión**: Minimalista con path de evolución claro

---

## ✅ DECISIÓN FINAL: Solo SOLAR-10.7B

### Modelos a Incluir en .env

| Modelo | Incluir? | Backend | Justificación | Variables .env |
|--------|----------|---------|---------------|----------------|
| **SOLAR-10.7B** | ✅ **SÍ** | Ollama | Core hard-skills, ya operacional | `OLLAMA_BASE_URL`<br>`SOLAR_MODEL_NAME` |
| LLaMarketing-8B | ❌ NO | - | Fuera de scope (SARAi = técnico+emocional) | - |
| Qwen2.5-Coder-7B | 🟡 FUTURO | Ollama | Solo si benchmark >15% mejora | `QWEN_CODER_MODEL_NAME`<br>(comentado) |

**Total modelos externos v2.14**: **1** (SOLAR-10.7B)

---

## 📋 Variables .env Propuestas

### Sección 1: Modelos Externos (NUEVO en v2.14)

```bash
# ============================================================================
# MODELOS EXTERNOS - OLLAMA REMOTE
# ============================================================================

# Servidor Ollama (SOLAR-10.7B)
OLLAMA_BASE_URL=http://<OLLAMA_HOST>:11434
SOLAR_MODEL_NAME=una-solar-10.7b-instruct-v1.0-q4

# FUTURO v2.15+ (comentado, requiere benchmark previo)
# Qwen2.5-Coder: Descomentar SOLO SI benchmark valida >15% mejora
# QWEN_CODER_MODEL_NAME=qwen2.5-coder:7b-instruct
```

### Sección 2: Audio Engine (v2.11 - MANTENER)

```bash
# ============================================================================
# AUDIO ENGINE
# ============================================================================

AUDIO_ENGINE=omni3b
LANGUAGES=es,en,fr,de,ja
```

### Sección 3: Seguridad (v2.11 - MANTENER)

```bash
# ============================================================================
# SECURITY & AUDIT
# ============================================================================

HMAC_SECRET_KEY=sarai-v2.14-change-in-production
```

---

## 🚀 Path de Evolución

### v2.14 (AHORA)
```
.env:
  OLLAMA_BASE_URL=...
  SOLAR_MODEL_NAME=...
  
models.yaml:
  solar_short: backend=ollama
  solar_long: backend=ollama
  lfm2: backend=gguf (local)

Total modelos externos: 1 (SOLAR)
Filosofía Phoenix: ✅ Preservada
```

### v2.15 (FUTURO - SI benchmark lo valida)
```
.env:
  OLLAMA_BASE_URL=...
  SOLAR_MODEL_NAME=...
  QWEN_CODER_MODEL_NAME=...  # NUEVO, solo si benchmark >15% mejora
  
models.yaml:
  solar_short: backend=ollama
  solar_long: backend=ollama
  qwen25_coder: backend=ollama  # NUEVO
  lfm2: backend=gguf (local)

core/mcp.py:
  if skill == "programming" and qwen_coder_available:
      return "qwen25_coder"
  else:
      return "solar_short"

Total modelos externos: 2 (SOLAR + Qwen2.5-Coder)
Condición: Benchmark programming_accuracy(Coder) > programming_accuracy(SOLAR) * 1.15
```

### v2.16+ (LARGO PLAZO)
```
Evaluación continua cada 6 meses:
  - Nuevos modelos especializados (FinQwen, MedLlama, etc.)
  - Criterio estricto: Benchmark >15% mejora O nuevo caso de uso crítico
  - Filosofía Phoenix: Skills = prompts, modelos = engines (cuando necesario)
```

---

## 📊 Análisis de Trade-offs

### Opción A: SOLO SOLAR-10.7B (RECOMENDADA v2.14)

| Aspecto | Evaluación |
|---------|------------|
| **Simplicidad** | ✅ Máxima (1 modelo externo) |
| **RAM** | ✅ 0 GB local (Ollama remoto) |
| **Latencia** | ✅ ~25s P50 (validado) |
| **Versatilidad** | ✅ 10.7B parámetros = muy competente |
| **Especialización** | ⚠️ Buena, no excelente en código |
| **Complejidad config** | ✅ Mínima (2 variables .env) |
| **Filosofía Phoenix** | ✅ Preservada |

**Veredicto**: **IDEAL para v2.14** ✅

---

### Opción B: SOLAR + Qwen2.5-Coder (FUTURO v2.15)

| Aspecto | Evaluación |
|---------|------------|
| **Simplicidad** | ⚠️ Media (2 modelos externos) |
| **RAM** | ✅ 0 GB local (Ollama remoto) |
| **Latencia** | ⚠️ Similar (~25s P50) |
| **Versatilidad** | ✅ Excelente (SOLAR general + Coder especializado) |
| **Especialización código** | 🎯 Potencialmente mejor (requiere benchmark) |
| **Complejidad config** | ⚠️ Mayor (3 variables .env + routing) |
| **Filosofía Phoenix** | ⚠️ Riesgo si no se valida con benchmark |

**Veredicto**: **CONSIDERAR para v2.15** si benchmark lo justifica ⚠️

**Condición obligatoria**:
```python
# ANTES de incluir Qwen2.5-Coder
benchmark_result = compare_models(
    model_a="solar_short",
    model_b="qwen25_coder",
    skill="programming",
    queries=PROGRAMMING_BENCHMARK_QUERIES
)

if benchmark_result["accuracy_improvement"] > 0.15:  # >15%
    print("✅ Incluir Qwen2.5-Coder en v2.15")
else:
    print("❌ Mantener SOLAR + skill_programming")
```

---

### Opción C: SOLAR + Qwen2.5-Coder + LLaMarketing (NO RECOMENDADA)

| Aspecto | Evaluación |
|---------|------------|
| **Simplicidad** | ❌ Baja (3 modelos externos) |
| **RAM** | ✅ 0 GB local (Ollama remoto) |
| **Scope alignment** | ❌ Marketing fuera de scope técnico SARAi |
| **Complejidad config** | ❌ Alta (4+ variables .env) |
| **Filosofía Phoenix** | ❌ Violada (skill_marketing debería ser prompt) |
| **Justificación** | ❌ SARAi = asistente técnico+emocional, NO marketing |

**Veredicto**: **NO INCLUIR** ❌

---

## 🎯 Archivo .env Propuesto para v2.14

```bash
# ============================================================================
# SARAi v2.14 - Configuración de Modelos Externos
# ============================================================================

# ----------------------------------------------------------------------------
# MODELOS EXTERNOS - OLLAMA REMOTE
# ----------------------------------------------------------------------------

# Servidor Ollama (SOLAR-10.7B hard-skills)
OLLAMA_BASE_URL=http://<OLLAMA_HOST>:11434
SOLAR_MODEL_NAME=una-solar-10.7b-instruct-v1.0-q4

# FUTURO v2.15+ (comentado, requiere benchmark previo)
# Qwen2.5-Coder: Descomentar SOLO SI:
#   1. Benchmark programming valida >15% mejora vs SOLAR
#   2. Routing condicional implementado en core/mcp.py
#   3. Validación manual de calidad de código positiva
# QWEN_CODER_MODEL_NAME=qwen2.5-coder:7b-instruct

# LLaMarketing: NO RECOMENDADO
# Razón: SARAi es asistente técnico+emocional, NO de marketing
# Alternativa: Usar SOLAR + skill_marketing (prompt especializado)
# LLAMARKETING_MODEL_NAME=llamarketing-8b

# ----------------------------------------------------------------------------
# AUDIO ENGINE (v2.11+)
# ----------------------------------------------------------------------------

AUDIO_ENGINE=omni3b
LANGUAGES=es,en,fr,de,ja

# ----------------------------------------------------------------------------
# SEGURIDAD (v2.11+)
# ----------------------------------------------------------------------------

HMAC_SECRET_KEY=sarai-v2.14-change-in-production

# ----------------------------------------------------------------------------
# RUNTIME
# ----------------------------------------------------------------------------

RUNTIME_BACKEND=cpu
N_THREADS=6
MAX_RAM_GB=12
```

---

## 📝 Checklist de Implementación

### v2.14 (AHORA)

- [x] Análisis estratégico de modelos externos (`docs/EXTERNAL_MODELS_STRATEGY.md`)
- [ ] Actualizar `.env.example` con sección de modelos externos
- [ ] Validar `config/models.yaml` tiene SOLAR correctamente configurado
- [ ] Documentar en README.md la filosofía de modelos externos
- [ ] Commit: "config: modelos externos v2.14 - solo SOLAR minimalista"

### v2.15 (FUTURO)

- [ ] Implementar benchmark de programming (SOLAR vs Qwen2.5-Coder)
- [ ] Ejecutar benchmark con queries reales de código
- [ ] Analizar resultados: mejora >15%?
- [ ] SI >15%: Añadir Qwen2.5-Coder a models.yaml y .env
- [ ] SI <15%: Mantener SOLAR + skill_programming
- [ ] Documentar decisión en benchmark results

---

## 💡 Filosofía Final

**Mantra v2.14 Modelos Externos**:

> _"SARAi mantiene complejidad mínima. Un modelo nuevo debe probar con  
> benchmarks que aporta >15% de mejora. Sin prueba, es solo ruido.  
> SOLAR-10.7B es el core. Todo lo demás es condicional."_

**Reglas**:
1. **Default**: SOLAR-10.7B para hard-skills (10.7B >> 7B-8B en general)
2. **Especialización**: Solo si benchmark >15% mejora O caso de uso único
3. **Scope**: SARAi = técnico + emocional. NO marketing, finanzas, legal.
4. **Filosofía Phoenix**: Skills = prompts. Modelos = engines (cuando necesario).

---

## ✅ Resumen Ejecutivo

**v2.14 AHORA**:
- `.env` con solo `OLLAMA_BASE_URL` + `SOLAR_MODEL_NAME`
- 1 modelo externo (SOLAR-10.7B)
- Filosofía Phoenix preservada ✅

**v2.15 FUTURO**:
- Evaluar Qwen2.5-Coder **solo si** benchmark >15% mejora
- LLaMarketing NO incluir (fuera de scope)

**Decisión tomada**: **Minimalismo con path de evolución claro** ✅

---

**Creado**: 2025-11-01  
**Versión**: v2.14  
**Status**: Propuesta validada  
**Next**: ¿Actualizamos `.env.example` con esta configuración?
