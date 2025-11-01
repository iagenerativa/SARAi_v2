# 🎯 Skills System - Quick Reference

## Decisión Rápida: ¿Qué Skill Usar?

```
Query con "código"/"función"        → Programming (temp=0.3, SOLAR)
Query con "problema"/"error"        → Diagnosis (temp=0.4, SOLAR)
Query con "inversión"/"ROI"         → Financial (temp=0.5, SOLAR)
Query con "crea"/"historia"         → Creative (temp=0.9, LFM2)
Query con "razonamiento"/"lógica"   → Reasoning (temp=0.6, SOLAR)
Query con "arquitectura"/"roadmap"  → CTO (temp=0.5, SOLAR)
Query con "kubernetes"/"monitoring" → SRE (temp=0.4, SOLAR)
```

## Uso en Código

### Detección Automática
```python
from core.skill_configs import match_skill_by_keywords

skill = match_skill_by_keywords("Escribe código Python")
# → skill.name == "programming"
```

### Aplicación con MCP
```python
from core.mcp import detect_and_apply_skill

result = detect_and_apply_skill(query, model_name="solar")
if result:
    prompt = result["full_prompt"]
    params = result["generation_params"]
    model = result["preferred_model"]  # "solar" o "lfm2"
```

### Listar Skills Disponibles
```python
from core.skill_configs import list_skills, get_skills_info

skills = list_skills()  
# → ["programming", "diagnosis", "financial", ...]

info = get_skills_info()  
# → {skill_name: {description, keywords, temperature, ...}}
```

## Parámetros por Skill

| Skill | Temp | Tokens | Modelo | Uso |
|-------|------|--------|--------|-----|
| programming | 0.3 | 3072 | SOLAR | Código preciso |
| diagnosis | 0.4 | 2560 | SOLAR | Análisis sistemático |
| financial | 0.5 | 2048 | SOLAR | Métricas de negocio |
| creative | 0.9 | 3584 | LFM2 | Storytelling |
| reasoning | 0.6 | 2560 | SOLAR | Problemas complejos |
| cto | 0.5 | 2048 | SOLAR | Decisiones técnicas |
| sre | 0.4 | 2048 | SOLAR | Ops & reliability |

## Ejemplos de Prompts Generados

### Programming Skill
```
You are an expert programming assistant with deep knowledge of multiple 
programming languages, design patterns, and optimization techniques.

User: Implementa quicksort en Python
Assistant:
```

### Creative Skill
```
You are a highly creative assistant specializing in idea generation, 
storytelling, and innovative thinking.

User: Crea una historia sobre un robot
Assistant:
```

## Testing

```bash
# Ejecutar tests completos
python3 -m pytest tests/test_skill_configs.py -v

# Test específico
python3 -m pytest tests/test_skill_configs.py::TestSkillUtilityFunctions::test_match_skill_by_keywords_programming -v
```

## Arquitectura

```
Input Query
    ↓
match_skill_by_keywords()  ← Detección automática
    ↓
SkillConfig
    ├─ name: "programming"
    ├─ system_prompt: "You are an expert..."
    ├─ temperature: 0.3
    ├─ max_tokens: 3072
    └─ preferred_model: "solar"
    ↓
build_prompt(query)  ← Construye prompt completo
    ↓
SOLAR/LFM2 + Specialized Prompt
    ↓
Optimized Response
```

## Principios de Diseño

1. **Skills ≠ Modelos**: Skills son configuraciones de prompting
2. **Reutilización**: SOLAR/LFM2 se adaptan con prompts
3. **Detección Automática**: Keywords permiten routing transparente
4. **Optimización por Dominio**: Cada skill tiene parámetros específicos
5. **Zero RAM Overhead**: No se cargan modelos adicionales

## Referencias

- Implementación: `core/skill_configs.py`
- Tests: `tests/test_skill_configs.py`
- Docs: `docs/SKILLS_SYSTEM_v2.12.md`
- Progreso: `PROGRESO_SKILLS_31102025.md`

---
**SARAi v2.12 - Skills System**  
_Especialización sin complejidad_
