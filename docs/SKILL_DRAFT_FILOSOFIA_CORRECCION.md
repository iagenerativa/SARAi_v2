# 🔍 Análisis y Corrección de Filosofía: skill_draft

**Fecha**: 31 Octubre 2025  
**Versión objetivo**: v2.16 Omni-Loop  
**Status**: ❌ **FILOSOFÍA INCORRECTA** → Requiere corrección

---

## ❌ Problema Identificado

### Concepto Erróneo Actual

En los roadmaps v2.15 y v2.16, `skill_draft` se describe como:

> "Draft LLM containerizado que reemplaza llama.cpp local para reducir latencia de 6s → 0.5s"

**Implementación descrita**:
```python
# INCORRECTO (según roadmap actual)
draft_client = pool.get_skill_client("draft")  # Container gRPC
response_pb = draft_client.Generate(request, timeout=10.0)
```

**Modelo asociado**: Qwen3-VL-4B-Instruct (3.3 GB) en container separado

### ⚠️ Por qué es INCORRECTO

1. **Viola el principio Phoenix v2.12**:
   - Phoenix establece que skills NO son modelos separados
   - Phoenix usa prompts especializados sobre SOLAR/LFM2 existentes
   - `skill_draft` con Qwen3-VL-4B rompe esta filosofía

2. **Complejidad innecesaria**:
   - Añade un LLM completo (3.3 GB) solo para drafts
   - Requiere gRPC overhead + Docker networking
   - Duplica capacidad que ya tiene LFM2-1.2B

3. **RAM P99 violation**:
   - Qwen3-VL-4B: 3.3 GB en container
   - SOLAR + LFM2 + Qwen3-VL = ~11.8 GB
   - Supera límite de 12 GB si hay concurrencia

4. **No sigue el patrón Skills Phoenix**:
   - Financial skill: prompt sobre SOLAR ✅
   - Programming skill: prompt sobre SOLAR ✅
   - Draft skill: LLM completo ❌ **INCOHERENTE**

---

## ✅ Filosofía Correcta: skill_draft como Prompt

### Concepto Correcto

`skill_draft` debe ser un **prompt especializado** aplicado a **LFM2-1.2B** (tiny agent) con configuración optimizada para drafts rápidos.

### Principios Rectores

1. **Phoenix Consistency**: Todos los skills son configuraciones de prompting, NO modelos
2. **Resource Efficiency**: Reutilizar LFM2 ya cargado en memory
3. **Latency**: Optimizar con n_ctx reducido y temperatura específica
4. **Simplicity**: Sin Docker, sin gRPC, solo prompt configuration

---

## 🎯 Implementación Correcta

### Archivo: `core/skill_configs.py`

Añadir skill_draft a la lista de skills existentes:

```python
# core/skill_configs.py
SKILLS = {
    # ... skills existentes (programming, creative, etc.) ...
    
    "draft": {
        "name": "draft",
        "temperature": 0.9,  # Alta creatividad para drafts variados
        "system_prompt": """You are a rapid draft generator specialized in creating quick, coherent initial responses.

Your role:
- Generate concise, well-structured first drafts
- Focus on clarity over perfection
- Maintain consistent tone and style
- Prepare content for refinement in subsequent iterations

Guidelines:
- Keep responses between 50-150 tokens
- Use simple, direct language
- Avoid over-elaboration
- Create a solid foundation for iteration""",
        
        "keywords": [
            "draft", "borrador", "iteración", "refinamiento"
        ],
        
        "longtail_patterns": [
            ("draft", "inicial", 3.0),
            ("borrador", "rápido", 2.5),
            ("iteración", "primera", 2.5)
        ],
        
        "agent_type": "tiny",  # ✅ USA LFM2, NO modelo separado
        
        "config_overrides": {
            "n_ctx": 512,      # Contexto reducido para velocidad
            "max_tokens": 150,  # Limitar longitud de draft
            "stop_sequences": ["</draft>", "\n\n\n"],
            "cache_prompt": True  # Reutilizar prompt en iteraciones
        }
    }
}
```

### Archivo: `core/omni_loop.py`

Modificar para usar skill_draft como prompt config:

```python
# core/omni_loop.py
from core.mcp import detect_and_apply_skill
from core.model_pool import get_model_pool

class OmniLoopEngine:
    def _run_iteration(
        self, 
        prompt: str, 
        iteration: int,
        previous_response: Optional[str]
    ) -> LoopIteration:
        """
        Ejecuta una iteración del loop con skill_draft (CORRECTED v2.16)
        
        FILOSOFÍA CORRECTA: skill_draft es un PROMPT sobre LFM2, NO un modelo separado
        """
        import time
        start = time.perf_counter()
        
        # ✅ CORRECCIÓN: Aplicar skill_draft como prompt config
        skill_config = detect_and_apply_skill("draft inicial", agent_type="tiny")
        
        # Construir prompt con contexto previo
        full_prompt = skill_config['system_prompt'] + "\n\n" + prompt
        
        if previous_response:
            full_prompt += f"""\n\n[Previous draft]
{previous_response}

[Improve and refine]"""
        
        # Obtener LFM2 del model pool (ya cargado)
        model_pool = get_model_pool()
        lfm2 = model_pool.get("tiny")
        
        # Generar draft con config especializada
        response = lfm2.generate(
            full_prompt,
            temperature=skill_config['temperature'],
            max_tokens=skill_config['config_overrides']['max_tokens'],
            n_ctx=skill_config['config_overrides']['n_ctx']
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Calcular confidence
        confidence = self._calculate_confidence(response, prompt)
        corrected = previous_response is not None and response != previous_response
        
        return LoopIteration(
            iteration=iteration,
            response=response,
            confidence=confidence,
            corrected=corrected,
            latency_ms=latency_ms
        )
```

### Beneficios de la Corrección

| Aspecto | Versión Incorrecta | Versión Correcta | Mejora |
|---------|-------------------|------------------|--------|
| **RAM** | +3.3 GB (Qwen3-VL) | +0 GB (LFM2 ya cargado) | -3.3 GB |
| **Latencia** | 0.5s (gRPC overhead) | 0.3-0.4s (directo) | -25% |
| **Complejidad** | Docker + gRPC + Protobuf | Solo prompt config | -90% código |
| **Filosofía** | Viola Phoenix | Sigue Phoenix | ✅ Coherente |
| **Mantenimiento** | 2 sistemas (host + container) | 1 sistema (host) | -50% surface |

---

## 📊 Comparación Arquitectónica

### ❌ Arquitectura Incorrecta (Roadmap actual)

```
┌─────────────────────────────────────┐
│         Omni-Loop Engine            │
│  (core/omni_loop.py)                │
└──────────────┬──────────────────────┘
               ↓ gRPC call
┌──────────────┴──────────────────────┐
│   skill_draft Container              │
│   • Docker runtime overhead          │
│   • gRPC server (50ms latency)       │
│   • Qwen3-VL-4B (3.3 GB)            │
│   • Network serialization            │
└─────────────────────────────────────┘

TOTAL: ~500ms latency, +3.3GB RAM
```

### ✅ Arquitectura Correcta (Phoenix v2.12)

```
┌─────────────────────────────────────┐
│         Omni-Loop Engine            │
│  (core/omni_loop.py)                │
└──────────────┬──────────────────────┘
               ↓ detect_and_apply_skill("draft")
┌──────────────┴──────────────────────┐
│   skill_draft Config                 │
│   • Prompt especializado             │
│   • temperature=0.9                  │
│   • n_ctx=512, max_tokens=150        │
└──────────────┬──────────────────────┘
               ↓ apply to LFM2
┌──────────────┴──────────────────────┐
│   LFM2-1.2B (Tiny Agent)            │
│   • Ya cargado en ModelPool          │
│   • 700 MB RAM (compartido)          │
│   • Sin overhead de red              │
└─────────────────────────────────────┘

TOTAL: ~300-400ms latency, +0GB RAM
```

---

## 🔧 Plan de Corrección

### Fase 1: Actualizar Documentación ✅

- [x] Crear `docs/SKILL_DRAFT_FILOSOFIA_CORRECCION.md` (este documento)
- [ ] Actualizar `ROADMAP_v2.16_OMNI_LOOP.md` eliminando referencias a container
- [ ] Actualizar `ROADMAP_v2.15_SENTIENCE.md` removiendo skill-draft Docker

### Fase 2: Implementar skill_draft Correctamente

- [ ] Añadir skill_draft a `core/skill_configs.py` (30 min)
- [ ] Modificar `core/omni_loop.py` para usar detect_and_apply_skill() (1h)
- [ ] Eliminar referencias a gRPC draft_client (15 min)
- [ ] Actualizar `core/model_pool.py` si tiene get_skill_client("draft") (30 min)

### Fase 3: Tests

- [ ] Crear `tests/test_skill_draft.py` (1h)
  - Test: draft detection con keywords
  - Test: prompt application con LFM2
  - Test: latency < 500ms
  - Test: no extra RAM usage

- [ ] Actualizar `tests/test_omni_loop.py` (30 min)
  - Verificar que usa skill_draft como prompt
  - No usa gRPC client
  - Latency mejora vs versión container

### Fase 4: Cleanup

- [ ] Eliminar `docker-compose.sentience.yml` skill-draft service (5 min)
- [ ] Eliminar Dockerfile.skill-draft si existe (5 min)
- [ ] Eliminar protobuf definitions de draft si existen (10 min)

**Tiempo total estimado**: ~4 horas

---

## 🎓 Lecciones Aprendidas

### ❌ Anti-patrones a Evitar

1. **Skill como LLM separado**:
   - Incrementa complejidad
   - Viola filosofía Phoenix
   - Aumenta RAM usage innecesariamente

2. **Containerización prematura**:
   - Docker útil para SANDBOXING (firejail, sql injection)
   - Docker NO útil para simple prompt variation

3. **gRPC para llamadas locales**:
   - Overhead de red innecesario
   - Serialización/deserialización añade latencia
   - Solo justificado en arquitecturas distribuidas

### ✅ Patrones Correctos

1. **Skills como configuraciones**:
   - Mantener coherencia con Phoenix v2.12
   - Reutilizar modelos existentes (SOLAR/LFM2)
   - Prompt engineering > Nuevos modelos

2. **Optimización por configuración**:
   - `n_ctx` reducido para velocidad
   - `max_tokens` limitado para drafts
   - `temperature` alta para creatividad

3. **Containerización selectiva**:
   - Usar Docker SOLO para skills peligrosos:
     * SQL (injection risk)
     * Bash (command execution)
     * Network (DDoS potential)
   - NO usar Docker para:
     * Prompts especializados
     * Variaciones de temperatura
     * Contexto reducido

---

## 📝 Mantra v2.16 Corregido

**ANTES** (Incorrecto):
> "Omni-Loop usa skill_draft containerizado via gRPC para reducir latencia de iteraciones"

**DESPUÉS** (Correcto):
> "Omni-Loop usa skill_draft como prompt especializado sobre LFM2 existente, manteniendo coherencia con filosofía Phoenix donde skills son configuraciones, NO modelos separados"

---

## 🚀 Próximos Pasos

1. **Revisar TODOS los roadmaps** que mencionen skill_draft containerizado
2. **Actualizar copilot-instructions.md** con filosofía correcta
3. **Implementar skill_draft según patrón Phoenix** (4h estimadas)
4. **Documentar en PLAN_MAESTRO** la corrección realizada

---

## 🎯 Conclusión

`skill_draft` debe ser tratado exactamente igual que `programming`, `creative`, `financial`, etc.:
- ✅ Prompt especializado
- ✅ Configuración de temperatura
- ✅ Aplicado sobre modelo existente (LFM2)
- ✅ Sin overhead de infraestructura

**Filosofía**: "Skills no son modelos, son estrategias de prompting"

---

**Autor**: Sistema de consolidación SARAi  
**Validación**: Pendiente implementación correcta  
**Prioridad**: ALTA (afecta coherencia arquitectónica)
