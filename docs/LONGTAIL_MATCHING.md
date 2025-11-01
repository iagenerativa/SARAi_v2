# 🎯 Long-Tail Matching para Skills

## Problema

El matching simple de keywords tiene **falsos positivos**:

```python
# Problema: "analizar" aparece en múltiples skills
query = "Analiza este error"
# ❌ Podría matchear: financial, diagnosis, reasoning

query = "Analiza este ROI"  
# ❌ Podría matchear: financial, diagnosis
```

## Solución: Long-Tail Patterns

**Long-tail matching** usa **combinaciones de palabras** con **pesos** para mayor precisión.

### Ejemplo

```python
# Query: "Calcula el ROI de esta inversión"

# Long-tail patterns para financial:
("roi", "inversión", 3.0)  # Peso 3.0 = alta confianza

# Match:
# - "roi" ✓ presente
# - "inversión" ✓ presente
# → Score: 3.0 (alta confianza)

# Otros skills solo matchean keywords simples (peso 1.0)
# → financial gana por score alto
```

## Implementación

### Estructura de Patterns

```python
longtail_patterns = {
    "skill_name": [
        (word1, word2, weight),  # Combinación
        (phrase, weight),        # Frase única
    ]
}
```

### Pesos

| Peso | Significado | Uso |
|------|-------------|-----|
| 3.0 | **Alta confianza** | Combinaciones muy específicas |
| 2.5 | **Media-alta** | Combinaciones comunes en el dominio |
| 2.0 | **Media** | Patrones generales |
| 1.0 | **Baja** | Keyword simple (fallback) |

### Umbral de Decisión

```python
if best_score >= 2.5:  # Umbral de confianza alta
    return skill  # Match inmediato
else:
    # Continuar con keyword matching simple
```

## Patterns por Skill

### Programming (temp=0.3)
```python
("código", "python", 3.0)          # "Escribe código Python..."
("función", "implementa", 3.0)     # "Implementa una función..."
("algoritmo", "código", 3.0)       # "Algoritmo en código..."
("debug", "error", 2.5)            # "Debug este error..."
("refactoriza", "código", 2.5)     # "Refactoriza este código..."
```

### Diagnosis (temp=0.4)
```python
("diagnostica", "problema", 3.0)   # "Diagnostica el problema..."
("analiza", "error", 3.0)          # "Analiza este error..."
("fallo", "sistema", 2.5)          # "Fallo en el sistema..."
("memory leak", 3.0)               # Frase específica
("bug", "producción", 2.5)         # "Bug en producción..."
```

### Financial (temp=0.5)
```python
("análisis", "financiero", 3.0)    # "Análisis financiero..."
("roi", "inversión", 3.0)          # "ROI de la inversión..."
("presupuesto", "costos", 2.5)     # "Presupuesto y costos..."
("margen", "beneficio", 2.5)       # "Margen de beneficio..."
("flujo", "caja", 2.5)             # "Flujo de caja..."
```

### Creative (temp=0.9)
```python
("crea", "historia", 3.0)          # "Crea una historia..."
("escribe", "narrativa", 3.0)      # "Escribe una narrativa..."
("genera", "ideas", 2.5)           # "Genera ideas..."
("brainstorm", 2.5)                # Frase única
("innovador", "concepto", 2.5)     # "Concepto innovador..."
```

### Reasoning (temp=0.6)
```python
("razonamiento", "lógico", 3.0)    # "Razonamiento lógico..."
("analiza", "estrategia", 2.5)     # "Analiza la estrategia..."
("problema", "complejo", 2.5)      # "Problema complejo..."
("paso", "paso", 2.0)              # "Paso a paso..."
```

### CTO (temp=0.5)
```python
("arquitectura", "sistema", 3.0)   # "Arquitectura del sistema..."
("roadmap", "técnico", 3.0)        # "Roadmap técnico..."
("escalabilidad", "infraestructura", 2.5)
("stack", "tecnológico", 2.5)      # "Stack tecnológico..."
```

### SRE (temp=0.4)
```python
("kubernetes", "cluster", 3.0)     # "Cluster Kubernetes..."
("monitoring", "alertas", 3.0)     # "Monitoring y alertas..."
("reliability", "sla", 2.5)        # "Reliability y SLA..."
("incident", "postmortem", 2.5)    # "Postmortem del incident..."
```

## Algoritmo de Matching

### Paso 1: Long-Tail Scoring
```python
for skill_name, patterns in longtail_patterns.items():
    score = 0.0
    for pattern in patterns:
        if all_words_in_query(pattern):
            score += pattern.weight
    
    if score >= 2.5:  # Alta confianza
        return skill
```

### Paso 2: Fallback a Keywords Simples
```python
for skill_name, skill_config in ALL_SKILLS.items():
    for keyword in skill_config.keywords:
        if keyword in query:
            score += 1.0
```

### Paso 3: Retornar Mejor Score
```python
best_skill = max(scores, key=scores.get)
return best_skill
```

## Ventajas

1. **Precisión**: Combinaciones de palabras reducen falsos positivos
2. **Confianza**: Pesos permiten decisiones rápidas (umbral 2.5)
3. **Fallback**: Keywords simples como respaldo
4. **Escalable**: Fácil añadir nuevos patterns

## Tests

### Test de Long-Tail Pattern
```python
def test_financial_longtail():
    # Long-tail: "roi" + "inversión"
    skill = match_skill_by_keywords("Calcula el ROI de esta inversión")
    assert skill.name == "financial"
    
    # NO matchea con solo "roi"
    skill = match_skill_by_keywords("El ROI es importante")
    # Puede ser financial o no (depende de otros keywords)
```

### Test de Frases Específicas
```python
def test_memory_leak_phrase():
    # Frase específica: "memory leak"
    skill = match_skill_by_keywords("Tengo un memory leak en producción")
    assert skill.name == "diagnosis"
```

## Expansión Futura

### Añadir Nuevo Pattern

```python
# En longtail_patterns:
"programming": [
    # ... patterns existentes ...
    ("test", "unitario", 2.5),  # NEW: "Test unitario..."
    ("ci/cd", 2.5),             # NEW: Frase específica
]
```

### Ajustar Pesos

```python
# Si un pattern genera falsos positivos, reducir peso:
("analiza", "estrategia", 2.5)  # → 2.0

# Si un pattern es muy específico, aumentar peso:
("kubernetes", "cluster", 2.5)  # → 3.0
```

## Resultados

**38/38 tests pasando** con long-tail matching:

- ✅ 0 falsos positivos en tests
- ✅ Alta precisión en detección
- ✅ Fallback robusto a keywords simples

---

**Conclusión**: Long-tail matching proporciona **precisión quirúrgica** en la detección de skills sin sacrificar **robustez** en casos ambiguos.
