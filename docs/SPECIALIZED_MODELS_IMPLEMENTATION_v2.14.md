# 🎯 Modelos Especializados - Implementación v2.14

**Fecha**: 2025-11-01  
**Estrategia**: Especialización por nicho con routing inteligente  
**Filosofía**: Skills como prompts + Modelos especializados para nichos críticos

---

## 🧠 Planteamiento Validado

### Uso Estratégico de Modelos Especializados

**Tu criterio** (validado y aprobado):

1. **LLaMarketing-8B**: Para solicitudes **muy específicas de marketing** (no genéricas)
   - Copy publicitario avanzado
   - Campañas multi-canal
   - SEO técnico
   - **NO para**: "Escribe un email" o "Crea un tweet"

2. **Qwen2.5-Coder-7B**: Para **tareas avanzadas de desarrollo** (no básicas)
   - Arquitectura de sistemas complejos
   - Debugging profundo
   - Code review técnico
   - **NO para**: "¿Qué es Python?" o "Explica loops"

3. **SOLAR-10.7B**: Para **todo lo demás** (general purpose)
   - Preguntas genéricas
   - Razonamiento general
   - Explicaciones técnicas básicas

---

## 📋 Sistema de Routing Inteligente

### Nivel 1: TRM-Router (Clasificación Base)

```python
# core/trm_classifier.py
scores = trm_router.invoke(user_input)
# {
#   "hard": 0.85,
#   "soft": 0.15,
#   "programming": 0.9,   # Skill detectado
#   "marketing": 0.1
# }
```

### Nivel 2: Skill Detection (Long-tail Matching)

```python
# core/skill_configs.py
def detect_specialized_skill(query: str, scores: dict) -> Optional[str]:
    """
    Detecta si la query requiere modelo especializado
    
    Returns:
        - "qwen_coder": Qwen2.5-Coder-7B (dev avanzado)
        - "llamarketing": LLaMarketing-8B (marketing nicho)
        - None: Usar SOLAR (genérico)
    """
    
    # CRITERIO 1: Qwen2.5-Coder (desarrollo avanzado)
    # Solo si: skill=programming Y complejidad alta
    if scores.get("programming", 0) > 0.7:
        # Keywords de complejidad avanzada
        advanced_dev_keywords = [
            # Arquitectura
            ("arquitectura", "microservicios", 3.0),
            ("diseño", "sistema", 2.5),
            ("patrones", "diseño", 2.5),
            
            # Debugging profundo
            ("debug", "segfault", 3.0),
            ("memory", "leak", 3.0),
            ("profiling", "performance", 2.5),
            
            # Code review técnico
            ("refactor", "código", 2.5),
            ("optimización", "algoritmo", 3.0),
            ("complejidad", "temporal", 2.5),
        ]
        
        for word1, word2, weight in advanced_dev_keywords:
            if word1 in query.lower() and word2 in query.lower():
                if weight >= 2.5:
                    return "qwen_coder"
    
    # CRITERIO 2: LLaMarketing (marketing nicho)
    # Solo si: keywords muy específicos de marketing
    marketing_niche_keywords = [
        # Copy avanzado
        ("copy", "publicitario", 3.0),
        ("anuncio", "conversión", 3.0),
        
        # Campañas multi-canal
        ("campaña", "omnicanal", 3.0),
        ("estrategia", "marketing", 2.5),
        
        # SEO técnico
        ("seo", "técnico", 3.0),
        ("keyword", "research", 2.5),
        
        # Funnel de ventas
        ("funnel", "ventas", 2.5),
        ("lead", "nurturing", 3.0),
    ]
    
    for word1, word2, weight in marketing_niche_keywords:
        if word1 in query.lower() and word2 in query.lower():
            if weight >= 2.5:
                return "llamarketing"
    
    # DEFAULT: SOLAR (genérico)
    return None
```

### Nivel 3: MCP Routing (Selección de Modelo)

```python
# core/mcp.py
def route_to_model(self, state: State) -> str:
    """
    Routing inteligente a modelo especializado o genérico
    
    Prioridad:
      1. Especialización nicho (Qwen/LLaMarketing)
      2. Genérico competente (SOLAR)
      3. Tiny fallback (LFM2)
    """
    
    # Detectar skill especializado
    from core.skill_configs import detect_specialized_skill
    specialized = detect_specialized_skill(state["input"], state)
    
    if specialized == "qwen_coder":
        return "qwen25_coder_long"  # Desarrollo avanzado
    
    elif specialized == "llamarketing":
        return "llamarketing_long"  # Marketing nicho
    
    # DEFAULT: SOLAR (genérico)
    elif state["hard"] > 0.7:
        context_len = len(state["input"])
        return "solar_long" if context_len > 400 else "solar_short"
    
    # Fallback: LFM2 (soft-skills)
    else:
        return "lfm2"
```

---

## 📦 Configuración models.yaml

### Modelos Especializados (NUEVOS)

```yaml
# ----------------------------------------------------------------------------
# MODELOS ESPECIALIZADOS - OLLAMA REMOTE
# ----------------------------------------------------------------------------

qwen25_coder_long:
  name: "Qwen2.5-Coder-7B-Instruct (Long Context)"
  type: "text"
  backend: "ollama"
  
  # Especialización: Desarrollo avanzado, arquitectura, debugging
  # NO usar para: Preguntas básicas de programación
  # Routing: Solo si detect_specialized_skill() == "qwen_coder"
  
  api_url: "${OLLAMA_BASE_URL}"
  model_name: "${QWEN_CODER_MODEL_NAME}"
  
  # Contexto largo para análisis de código complejo
  n_ctx: 2048
  
  # Temperatura baja (determinista para código)
  temperature: 0.3
  max_tokens: 1024
  top_p: 0.9
  
  # Gestión de memoria
  load_on_demand: true
  priority: 7  # Media-alta (especializado)
  max_memory_mb: 0  # Ollama remoto

llamarketing_long:
  name: "LLaMarketing-8B (Long Context)"
  type: "text"
  backend: "ollama"
  
  # Especialización: Marketing nicho (copy, campañas, SEO técnico)
  # NO usar para: Emails genéricos, tweets simples
  # Routing: Solo si detect_specialized_skill() == "llamarketing"
  
  api_url: "${OLLAMA_BASE_URL}"
  model_name: "${LLAMARKETING_MODEL_NAME}"
  
  # Contexto largo para campañas multi-canal
  n_ctx: 2048
  
  # Temperatura alta (creatividad)
  temperature: 0.9
  max_tokens: 1024
  top_p: 0.95
  
  # Gestión de memoria
  load_on_demand: true
  priority: 6  # Media (nicho específico)
  max_memory_mb: 0  # Ollama remoto
```

---

## 🔧 Variables .env

### Configuración Completa

```bash
# ============================================================================
# MODELOS EXTERNOS - OLLAMA REMOTE
# ============================================================================

# Servidor Ollama
OLLAMA_BASE_URL=http://192.168.0.251:11434

# Modelo genérico (hard-skills general)
SOLAR_MODEL_NAME=una-solar-10.7b-instruct-v1.0-q4

# Modelo especializado: Desarrollo avanzado
# Uso: Arquitectura, debugging profundo, code review técnico
# NO usar para: Preguntas básicas de Python, "¿Qué es un loop?"
QWEN_CODER_MODEL_NAME=qwen2.5-coder:7b-instruct

# Modelo especializado: Marketing nicho
# Uso: Copy publicitario, campañas omnicanal, SEO técnico
# NO usar para: Emails genéricos, tweets simples
LLAMARKETING_MODEL_NAME=llamarketing:8b
```

---

## 📊 Matriz de Decisión de Routing

### Ejemplos de Queries y Routing

| Query | Skill | Especializado? | Modelo | Razón |
|-------|-------|----------------|--------|-------|
| "¿Qué es Python?" | programming | ❌ NO | **solar_short** | Pregunta básica → genérico |
| "Explica loops" | programming | ❌ NO | **solar_short** | Conceptos básicos → genérico |
| "Arquitectura microservicios Docker" | programming | ✅ SÍ | **qwen25_coder** | Keywords: arquitectura+microservicios (3.0) |
| "Debug segfault en C++" | programming | ✅ SÍ | **qwen25_coder** | Keywords: debug+segfault (3.0) |
| "Optimización algoritmo O(n²) → O(n log n)" | programming | ✅ SÍ | **qwen25_coder** | Keywords: optimización+algoritmo (3.0) |
| "Escribe un email profesional" | - | ❌ NO | **solar_short** | No es marketing nicho → genérico |
| "Crea un tweet sobre IA" | - | ❌ NO | **solar_short** | Social media simple → genérico |
| "Copy publicitario campaña multi-canal" | marketing | ✅ SÍ | **llamarketing** | Keywords: copy+publicitario (3.0) |
| "Estrategia SEO técnico schema markup" | marketing | ✅ SÍ | **llamarketing** | Keywords: seo+técnico (3.0) |
| "Funnel de ventas lead nurturing" | marketing | ✅ SÍ | **llamarketing** | Keywords: funnel+ventas + lead+nurturing |

---

## 🎯 Flujo de Decisión (Diagrama)

```
User Input
    ↓
TRM-Router
    ↓
Skill Detection (programming/marketing/etc)
    ↓
detect_specialized_skill(query, scores)
    ↓
┌──────────────────┬────────────────────┬──────────────────┐
↓                  ↓                    ↓                  ↓
"qwen_coder"      "llamarketing"       None               soft > 0.7
(dev avanzado)    (marketing nicho)    (genérico)         (emocional)
    ↓                  ↓                    ↓                  ↓
Qwen2.5-Coder     LLaMarketing         SOLAR              LFM2
(2048 ctx)        (2048 ctx)           (512/2048 ctx)     (2048 ctx)
    ↓                  ↓                    ↓                  ↓
Response          Response             Response           Response
```

---

## 📝 Implementación Paso a Paso

### Paso 1: Actualizar `core/skill_configs.py`

```python
# Añadir función detect_specialized_skill() con criterios long-tail
# Ver código completo arriba
```

### Paso 2: Actualizar `core/mcp.py`

```python
def route_to_model(self, state: State) -> str:
    """Routing con modelos especializados"""
    # Ver código completo arriba
```

### Paso 3: Actualizar `config/models.yaml`

```yaml
# Añadir qwen25_coder_long y llamarketing_long
# Ver configuración completa arriba
```

### Paso 4: Actualizar `.env`

```bash
# Añadir QWEN_CODER_MODEL_NAME y LLAMARKETING_MODEL_NAME
# Ver configuración completa arriba
```

### Paso 5: Validar con Tests

```python
# tests/test_specialized_routing.py
def test_qwen_coder_routing():
    """Valida routing a Qwen2.5-Coder para dev avanzado"""
    query = "Diseña arquitectura microservicios con Docker y Kubernetes"
    
    scores = trm_router.invoke(query)
    specialized = detect_specialized_skill(query, scores)
    
    assert specialized == "qwen_coder"
    
    model = mcp.route_to_model({"input": query, **scores})
    assert model == "qwen25_coder_long"

def test_llamarketing_routing():
    """Valida routing a LLaMarketing para marketing nicho"""
    query = "Crea copy publicitario para campaña omnicanal con análisis de conversión"
    
    scores = trm_router.invoke(query)
    specialized = detect_specialized_skill(query, scores)
    
    assert specialized == "llamarketing"
    
    model = mcp.route_to_model({"input": query, **scores})
    assert model == "llamarketing_long"

def test_solar_generic_routing():
    """Valida que preguntas básicas van a SOLAR"""
    query = "¿Qué es Python y para qué sirve?"
    
    scores = trm_router.invoke(query)
    specialized = detect_specialized_skill(query, scores)
    
    assert specialized is None  # No especializado
    
    model = mcp.route_to_model({"input": query, **scores})
    assert model == "solar_short"  # Genérico
```

---

## 📊 Análisis de Impacto

### RAM (Sin Cambios - Ollama Remoto)

| Modelo | Backend | RAM Local | Estado |
|--------|---------|-----------|--------|
| LFM2-1.2B | GGUF local | ~700 MB | Siempre en memoria |
| SOLAR-10.7B | Ollama remoto | 0 MB | On-demand |
| Qwen3-VL-4B | Multimodal local | ~3.5 GB | On-demand |
| **Qwen2.5-Coder-7B** | **Ollama remoto** | **0 MB** | **On-demand** |
| **LLaMarketing-8B** | **Ollama remoto** | **0 MB** | **On-demand** |
| **Total P99** | - | **~4.2 GB** | **✅ Sin cambios** |

**Conclusión**: 0 impacto en RAM local (todos los modelos externos usan Ollama) ✅

---

### Complejidad de Código

| Aspecto | Antes v2.14 | Después v2.14+ | Impacto |
|---------|-------------|----------------|---------|
| Modelos en .env | 1 (SOLAR) | 3 (SOLAR, Qwen, LLaMarketing) | +2 variables |
| Modelos en models.yaml | 5 | 7 | +2 configs |
| Función routing | `route_to_model()` | `detect_specialized_skill()` + `route_to_model()` | +1 función |
| LOC añadidas | - | ~150 LOC | Moderado |
| Tests añadidos | - | 3 tests | +3 tests |

**Conclusión**: Complejidad moderada, pero justificada por especialización ✅

---

### Beneficios vs Trade-offs

| Aspecto | Beneficio | Trade-off |
|---------|-----------|-----------|
| **Especialización dev** | Qwen2.5-Coder mejor en arquitectura/debugging | Routing más complejo |
| **Especialización marketing** | LLaMarketing mejor en copy/campañas | +1 modelo a mantener |
| **Filosofía Phoenix** | ⚠️ Añade modelos, pero con criterio claro | Requiere long-tail matching preciso |
| **RAM** | ✅ 0 impacto (Ollama remoto) | N/A |
| **Latencia** | Similar (~25s P50) | Posible overhead de routing (+50ms) |

**Veredicto**: **Beneficios superan trade-offs** si routing es preciso ✅

---

## 🎯 Criterios de Éxito

### KPIs para Validar Implementación

1. **Precisión de Routing** (>90%):
   - Queries de dev avanzado → Qwen2.5-Coder (>90% accuracy)
   - Queries de marketing nicho → LLaMarketing (>90% accuracy)
   - Queries genéricas → SOLAR (>90% accuracy)

2. **Calidad de Respuestas** (benchmark):
   - Qwen2.5-Coder en dev avanzado > SOLAR (>15% mejora)
   - LLaMarketing en copy > SOLAR (>15% mejora)
   - SOLAR en genérico = baseline (sin regresión)

3. **Latencia** (sin degradación):
   - Overhead routing: <100ms
   - Latencia P50 total: ≤25s (sin cambios vs v2.14 baseline)

4. **Usabilidad**:
   - 0 configuración manual de usuario
   - Routing automático transparente
   - Fallback a SOLAR si modelo especializado no disponible

---

## 🚀 Plan de Implementación

### Fase 1: Configuración (30 min)

1. ✅ Añadir modelos a `config/models.yaml`
2. ✅ Actualizar `.env.example` con nuevas variables
3. ✅ Documentar criterios de routing

### Fase 2: Código (1h)

1. ✅ Implementar `detect_specialized_skill()` en `core/skill_configs.py`
2. ✅ Actualizar `route_to_model()` en `core/mcp.py`
3. ✅ Añadir long-tail patterns para dev avanzado y marketing nicho

### Fase 3: Tests (30 min)

1. ✅ Tests de routing especializado (`tests/test_specialized_routing.py`)
2. ✅ Tests de fallback a SOLAR
3. ✅ Tests de precisión de long-tail matching

### Fase 4: Validación (1h)

1. ✅ Ejecutar benchmark de routing accuracy
2. ✅ Validar calidad de respuestas con queries reales
3. ✅ Medir latencia overhead

### Fase 5: Documentación (30 min)

1. ✅ Actualizar README con modelos especializados
2. ✅ Documentar criterios de routing
3. ✅ Ejemplos de uso

**Total estimado**: ~3 horas de implementación completa

---

## 💡 Filosofía Actualizada

**Mantra v2.14 Modelos Especializados**:

> _"SARAi usa especialización cuando el nicho lo justifica.  
> SOLAR es el genérico competente. Qwen y LLaMarketing son los expertos  
> que solo intervienen cuando la tarea es claramente de su dominio.  
> El routing debe ser preciso: mejor SOLAR genérico que especialista equivocado."_

**Reglas actualizadas**:
1. **Default**: SOLAR (genérico competente)
2. **Especialización**: Solo para nichos claros (dev avanzado, marketing técnico)
3. **Routing**: Long-tail matching con pesos ≥2.5
4. **Fallback**: Siempre SOLAR si duda
5. **Validación**: Benchmark antes de producción

---

## ✅ Checklist Final

- [ ] `config/models.yaml`: Añadir qwen25_coder_long y llamarketing_long
- [ ] `.env.example`: Añadir QWEN_CODER_MODEL_NAME y LLAMARKETING_MODEL_NAME
- [ ] `core/skill_configs.py`: Implementar detect_specialized_skill()
- [ ] `core/mcp.py`: Actualizar route_to_model()
- [ ] `tests/test_specialized_routing.py`: 3 tests de routing
- [ ] Ejecutar tests: `pytest tests/test_specialized_routing.py -v`
- [ ] Benchmark routing accuracy (>90% objetivo)
- [ ] Validar con queries reales
- [ ] Documentar en README
- [ ] Commit: "feat: modelos especializados Qwen2.5-Coder + LLaMarketing con routing inteligente"

---

**Creado**: 2025-11-01  
**Versión**: v2.14+  
**Status**: Propuesta de implementación  
**Next**: ¿Apruebas para implementar?
