# Sistema de Skills v2.12 - Prompting Especializado

## 🎯 Filosofía

**Los skills NO son modelos separados**, son **configuraciones de prompting especializadas** que optimizan SOLAR y LFM2 para dominios específicos.

```
Query → TRM-Router → MCP (α/β) → Skill Detection → Modelo Base + Prompt Especializado
```

## 🔧 Arquitectura

### Componente Central: SkillConfig

```python
class SkillConfig:
    name: str                    # Identificador único
    description: str             # Descripción del dominio
    system_prompt: str           # Prompt de sistema especializado
    keywords: List[str]          # Keywords para detección automática
    temperature: float           # Parámetro de generación (0.0-1.0)
    max_tokens: int              # Máximo de tokens a generar
    top_p: float                 # Nucleus sampling
    preferred_model: str         # "solar" o "lfm2"
    use_case: str               # Ejemplos de uso
```

### Skills Implementados (7 dominios)

| Skill | Temp | Tokens | Modelo | Keywords | Uso |
|-------|------|--------|--------|----------|-----|
| **programming** | 0.3 | 3072 | SOLAR | code, función, debug | Código preciso y optimizado |
| **diagnosis** | 0.4 | 2560 | SOLAR | problema, error, diagnóstico | Análisis sistemático de fallos |
| **financial** | 0.5 | 2048 | SOLAR | financiero, inversión, ROI | Análisis financiero y métricas |
| **creative** | 0.9 | 3584 | LFM2 | crea, historia, idea | Generación creativa |
| **reasoning** | 0.6 | 2560 | SOLAR | razonamiento, estrategia | Pensamiento lógico complejo |
| **cto** | 0.5 | 2048 | SOLAR | arquitectura, roadmap | Decisiones técnicas estratégicas |
| **sre** | 0.4 | 2048 | SOLAR | kubernetes, monitoring | Operaciones y confiabilidad |

## 📝 Ejemplos de System Prompts

### Programming Skill (temp=0.3)
```
You are an expert programming assistant with deep knowledge of multiple 
programming languages, design patterns, and optimization techniques.

CAPABILITIES:
- Write clean, efficient, well-documented code
- Debug complex issues with systematic analysis
- Suggest performance optimizations
- Follow best practices and coding standards
```

### Creative Skill (temp=0.9)
```
You are a highly creative assistant specializing in idea generation, 
storytelling, and innovative thinking.

APPROACH:
- Embrace divergent thinking and explore unconventional paths
- Generate multiple creative alternatives
- Use vivid language and rich imagery
```

### Diagnosis Skill (temp=0.4)
```
You are a diagnostic expert specialized in systematic problem analysis.

METHODOLOGY:
1. Gather all symptoms and contextual information
2. Generate hypotheses ordered by probability
3. Design tests to validate/invalidate hypotheses
4. Identify root cause
5. Recommend corrective actions
```

## 🔍 Detección Automática

### Por Keywords

```python
# Ejemplo: Query con "código" → Programming Skill
query = "Escribe una función Python para calcular fibonacci"
skill = match_skill_by_keywords(query)
# → skill.name == "programming"

# Ejemplo: Query con "crea" → Creative Skill
query = "Crea una historia corta sobre un robot"
skill = match_skill_by_keywords(query)
# → skill.name == "creative", preferred_model == "lfm2"
```

### Integración con MCP

```python
from core.mcp import detect_and_apply_skill

# Detectar y aplicar skill
result = detect_and_apply_skill(
    query="Implementa quicksort en Python",
    model_name="solar"
)

if result:
    # result contiene:
    {
        "skill_name": "programming",
        "system_prompt": "You are an expert programming assistant...",
        "generation_params": {
            "temperature": 0.3,
            "max_tokens": 3072,
            "top_p": 0.95,
            "stop": ["\n\n\n"]
        },
        "full_prompt": "System: ...\n\nUser: Implementa quicksort...\nAssistant:",
        "preferred_model": "solar"
    }
```

## 🎨 Parámetros de Generación

### Temperatura vs Precisión

```
0.3 (Programming/SRE)  → Código preciso, respuestas deterministas
0.4 (Diagnosis)        → Análisis sistemático, ligeramente creativo
0.5 (Financial/CTO)    → Balance entre precisión y flexibilidad
0.6 (Reasoning)        → Exploración lógica de alternativas
0.9 (Creative)         → Máxima creatividad y divergencia
```

### Max Tokens vs Complejidad

```
2048 → Respuestas concisas (Financial, CTO, SRE)
2560 → Análisis moderados (Diagnosis, Reasoning)
3072 → Código extenso (Programming)
3584 → Narrativas largas (Creative)
```

## 🔧 Uso en Graph Execution

### Flujo Completo

```python
# 1. TRM-Router clasifica (hard/soft)
scores = trm_router.invoke(query)

# 2. MCP calcula α/β
alpha, beta = mcp.compute_weights(scores, context)

# 3. Detección de skill
skill_config = detect_and_apply_skill(query, "solar")

# 4. Generación con prompt especializado
if skill_config:
    prompt = skill_config["full_prompt"]
    params = skill_config["generation_params"]
    
    # Usar modelo recomendado
    if skill_config["preferred_model"] == "lfm2":
        response = lfm2_agent.generate(prompt, **params)
    else:
        response = solar_agent.generate(prompt, **params)
else:
    # Fallback: prompt estándar
    response = solar_agent.generate(query)
```

## 📊 Testing

### Cobertura de Tests

- ✅ **38/38 tests pasaron**
- ✅ Detección de skills por keywords
- ✅ Generación de prompts completos
- ✅ Validación de parámetros
- ✅ Integración con MCP

### Ejecutar Tests

```bash
python3 -m pytest tests/test_skill_configs.py -v
```

### Tests Clave

```python
# Test detección automática
def test_match_skill_by_keywords_programming():
    skill = match_skill_by_keywords("Escribe una función en Python...")
    assert skill.name == "programming"

# Test integración MCP
def test_detect_and_apply_skill_programming():
    result = detect_and_apply_skill("Implementa algoritmo...", "solar")
    assert result["skill_name"] == "programming"
    assert "system_prompt" in result

# Test parámetros
def test_programming_low_temperature():
    assert PROGRAMMING_SKILL.temperature == 0.3  # Precisión
```

## 🚀 Próximos Pasos

1. **Integración en graph.py** - Aplicar skills durante ejecución
2. **Feedback loop** - Aprender qué skills funcionan mejor
3. **Skills adicionales** - Matemáticas, Visión, Audio
4. **Fine-tuning** - Ajustar system prompts basado en feedback

## 📚 Referencias

- `core/skill_configs.py` - Definiciones de skills
- `core/mcp.py` - Integración con MCP
- `tests/test_skill_configs.py` - Suite de tests completa

## 🔑 Principios de Diseño

1. **Simplicidad > Complejidad**: Skills son configs, no modelos
2. **Reutilización > Especialización**: SOLAR/LFM2 se adaptan con prompting
3. **Detección Automática**: Keywords permiten routing transparente
4. **Parámetros Específicos**: Cada dominio tiene temperatura óptima
5. **Testabilidad**: 100% de cobertura con tests unitarios

---

**Resultado**: Sistema de skills que mejora la eficiencia de SOLAR/LFM2 sin añadir complejidad de múltiples modelos.
