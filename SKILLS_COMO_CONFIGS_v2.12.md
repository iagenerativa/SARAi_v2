# 🔧 Skills como Configuraciones Especializadas (Estrategia Corregida v2.12)

## ❌ Malentendido Original

Interpreté "Skills MoE" como:
- Cargar 6 LLMs diferentes (CodeLlama, Mistral, FinGPT, etc)
- +4.8 GB RAM
- Gestión compleja de modelos

## ✅ Realidad del Sistema

Los **skills son CONFIGURACIONES**, no modelos:

```yaml
skills:
  programming:
    model: "solar"  # ← Usa SOLAR, no CodeLlama
    temperature: 0.2  # Preciso para código
    system_prompt: |
      Eres un experto en programación. Responde con:
      - Código limpio y comentado
      - Explicaciones técnicas precisas
      - Ejemplos ejecutables
    domains: ["código", "programación", "python", "debugging"]
    
  diagnosis:
    model: "solar"  # ← Usa SOLAR, no Mistral
    temperature: 0.3  # Diagnóstico preciso
    system_prompt: |
      Eres un experto en diagnóstico de sistemas. Analiza:
      - Logs de error sistemáticamente
      - Causa raíz (RCA)
      - Pasos de resolución
    domains: ["diagnóstico", "error", "logs", "sistema"]
```

## 🎯 Arquitectura Correcta

### Capa 1: TRM-Router con Heads Especializados

El TRM clasifica con **6 heads** (no solo hard/soft):

```python
class TRMClassifierSpecialized(nn.Module):
    def __init__(self):
        super().__init__()
        # ... código recursivo existente ...
        
        # 6 cabezas especializadas
        self.head_programming = nn.Linear(256, 1)
        self.head_diagnosis = nn.Linear(256, 1)
        self.head_finance = nn.Linear(256, 1)
        self.head_creative = nn.Linear(256, 1)
        self.head_reasoning = nn.Linear(256, 1)
        self.head_general = nn.Linear(256, 1)  # Fallback
    
    def forward(self, x_embedding):
        # ... recursión TRM ...
        
        return {
            "programming": torch.sigmoid(self.head_programming(y)).item(),
            "diagnosis": torch.sigmoid(self.head_diagnosis(y)).item(),
            "finance": torch.sigmoid(self.head_finance(y)).item(),
            "creative": torch.sigmoid(self.head_creative(y)).item(),
            "reasoning": torch.sigmoid(self.head_reasoning(y)).item(),
            "general": torch.sigmoid(self.head_general(y)).item()
        }
```

### Capa 2: SkillConfigs (Prompts + Parámetros)

```python
# core/skill_configs.py
SKILL_CONFIGS = {
    "programming": {
        "model": "solar",  # Usa SOLAR existente
        "temperature": 0.2,
        "max_tokens": 512,
        "system_prompt": """Eres un experto en programación con 15 años de experiencia.

DIRECTRICES:
- Código limpio, idiomático y bien documentado
- Explicaciones técnicas precisas sin simplificaciones
- Ejemplos ejecutables con manejo de errores
- Referencia buenas prácticas (PEP-8, SOLID, etc)
- Si hay múltiples soluciones, menciona trade-offs

FORMATO DE RESPUESTA:
1. Explicación breve (2-3 líneas)
2. Código con comentarios
3. Ejemplo de uso
4. Posibles edge cases""",
        "domains": ["código", "programación", "python", "javascript", "debugging"]
    },
    
    "diagnosis": {
        "model": "solar",
        "temperature": 0.3,
        "max_tokens": 512,
        "system_prompt": """Eres un experto en diagnóstico de sistemas y análisis de causa raíz.

METODOLOGÍA:
1. Recolección de síntomas y evidencia
2. Análisis sistemático (logs, métricas, estado)
3. Hipótesis ordenadas por probabilidad
4. Pasos de verificación concretos
5. Solución con prevención futura

FORMATO DE RESPUESTA:
## Diagnóstico
- Síntoma principal: ...
- Evidencia: ...

## Causa Raíz Probable
...

## Pasos de Resolución
1. ...
2. ...""",
        "domains": ["diagnóstico", "error", "logs", "sistema", "fallo"]
    },
    
    "finance": {
        "model": "solar",
        "temperature": 0.4,
        "max_tokens": 512,
        "system_prompt": """Eres un analista financiero certificado (CFA Level II).

ENFOQUE:
- Análisis cuantitativo riguroso
- Citas de métricas estándar (ROI, IRR, P/E, etc)
- Consideración de riesgos y volatilidad
- Contexto macroeconómico cuando relevante
- Disclaimers de inversión apropiados

FORMATO:
## Análisis
- Métricas clave: ...
- Tendencia: ...

## Recomendación
...

## Riesgos
...""",
        "domains": ["finanzas", "inversión", "mercado", "roi"]
    },
    
    "creative": {
        "model": "lfm2",  # Usa LFM2 para creatividad
        "temperature": 0.9,  # Alta creatividad
        "max_tokens": 512,
        "system_prompt": """Eres un escritor creativo galardonado.

DIRECTRICES:
- Narrativa envolvente con detalles sensoriales
- Desarrollo de personajes profundo
- Diálogos naturales y distintivos
- Uso creativo del lenguaje (metáforas, símiles)
- Estructura narrativa sólida (setup, conflicto, resolución)

TONO: Adaptable según petición (poético, humorístico, dramático, etc)""",
        "domains": ["historia", "cuento", "poema", "narrativa"]
    },
    
    "reasoning": {
        "model": "solar",
        "temperature": 0.5,
        "max_tokens": 512,
        "system_prompt": """Eres un experto en razonamiento lógico y pensamiento crítico.

METODOLOGÍA:
- Descomponer problemas complejos en pasos
- Identificar premisas y supuestos
- Aplicar lógica deductiva/inductiva
- Detectar falacias lógicas
- Verificar coherencia interna

FORMATO:
## Análisis del Problema
...

## Razonamiento Paso a Paso
1. Premisa: ...
2. Inferencia: ...
3. Conclusión: ...

## Verificación
...""",
        "domains": ["razonamiento", "lógica", "análisis", "paso a paso"]
    },
    
    "general": {
        "model": "solar",
        "temperature": 0.7,
        "max_tokens": 512,
        "system_prompt": """Eres un asistente útil, preciso y conciso.

Responde de forma:
- Clara y estructurada
- Técnicamente correcta
- Adaptada al nivel del usuario
- Honesta sobre limitaciones""",
        "domains": []  # Fallback genérico
    }
}
```

### Capa 3: MCP con Skill Selection

```python
# core/mcp.py (NUEVO MÉTODO)
def select_skill_and_generate(self, trm_scores: dict, user_input: str) -> str:
    """
    Selecciona skill basado en scores TRM y ejecuta con config especializada
    
    Args:
        trm_scores: {"programming": 0.85, "diagnosis": 0.12, ...}
        user_input: Consulta del usuario
    
    Returns:
        Respuesta generada con el skill apropiado
    """
    # 1. Seleccionar skill de mayor score
    best_skill = max(trm_scores.items(), key=lambda x: x[1])
    skill_name, confidence = best_skill
    
    # 2. Obtener configuración del skill
    skill_config = SKILL_CONFIGS[skill_name]
    
    # 3. Construir prompt especializado
    specialized_prompt = f"""{skill_config['system_prompt']}

# Consulta del Usuario
{user_input}

# Instrucciones
Responde según las directrices de {skill_name} con nivel de confianza {confidence:.2%}."""
    
    # 4. Seleccionar modelo apropiado
    model_name = skill_config['model']  # "solar" o "lfm2"
    model = self.model_pool.get(model_name)
    
    # 5. Generar con parámetros especializados
    response = model.generate(
        specialized_prompt,
        temperature=skill_config['temperature'],
        max_tokens=skill_config['max_tokens']
    )
    
    # 6. Log para feedback
    self.feedback_logger.log_skill_usage(
        skill=skill_name,
        confidence=confidence,
        model=model_name,
        input=user_input,
        output=response
    )
    
    return response
```

## 📊 Comparación de Estrategias

| Aspecto | ❌ Skills como LLMs | ✅ Skills como Configs |
|---------|---------------------|------------------------|
| **Modelos en RAM** | 7 (SOLAR + 6 skills) | 2 (SOLAR + LFM2) |
| **RAM total** | ~11.6 GB | ~900 MB |
| **Latencia** | Alta (carga modelo) | Baja (mismo modelo) |
| **Especialización** | Limitada al LLM | Alta (prompts expertos) |
| **Mantenibilidad** | Compleja (6 GGUFs) | Simple (1 YAML) |
| **Aprendizaje TRM** | No | Sí (scores → mejor routing) |

## 🎯 Plan de Implementación Corregido

### T1.1-FINAL: Skill Configs + TRM Heads (8h)

**Archivos**:
- `core/skill_configs.py`: Diccionario SKILL_CONFIGS (NEW)
- `core/trm_classifier.py`: Añadir 6 heads especializados (+30 LOC)
- `core/mcp.py`: Método `select_skill_and_generate()` (+60 LOC)

**Tests**:
- `test_skill_configs.py`: Validar que cada skill tiene system_prompt, temperature
- `test_trm_specialized.py`: Verificar que TRM clasifica correctamente (>85% accuracy)
- `test_mcp_skill_selection.py`: E2E con prompt especializado

**Dataset de entrenamiento**:
```python
# scripts/generate_skill_dataset.py
import random

def generate_skill_queries():
    """Genera 10K queries sintéticas con SOLAR pre-clasificadas"""
    
    queries = {
        "programming": [
            "¿Cómo implemento un decorador en Python?",
            "Explica el patrón Observer con código",
            "Debug este error: AttributeError en línea 42"
        ],
        "diagnosis": [
            "Mi servidor devuelve 502 Bad Gateway",
            "Docker container reinicia constantemente",
            "RAM al 98% pero sin proceso culpable"
        ],
        # ... 10K total
    }
    
    # Generar embeddings con EmbeddingGemma
    # Guardar en data/trm_skill_dataset.npz
```

### T1.2-FINAL: Entrenar TRM con Skill Heads (10h)

```python
# scripts/train_trm_skills.py
def train_trm_specialized():
    """
    Entrena TRM-Router con 6 heads especializados
    Dataset: 10K queries (data/trm_skill_dataset.npz)
    Target accuracy: >85% por skill
    """
    
    # Arquitectura: Shared trunk + 6 independent heads
    # Loss: Multi-label BCE (una query puede activar múltiples skills)
    # Optimizer: AdamW con lr=1e-4
    # Epochs: 50 con early stopping
```

### T1.3-FINAL: MCP Skill Selection (6h)

- Integrar `select_skill_and_generate()` en `core/graph.py`
- Routing: TRM scores → Skill config → SOLAR/LFM2 con prompt especializado
- Feedback loop: Ajustar TRM basado en calidad de respuestas

## 🔍 Ejemplo de Flujo Completo

**Input**: "¿Cómo optimizo esta consulta SQL lenta?"

**1. TRM-Router**:
```python
scores = {
    "programming": 0.72,  # SQL es programación
    "diagnosis": 0.65,    # "lenta" sugiere diagnóstico
    "finance": 0.05,
    "creative": 0.02,
    "reasoning": 0.35,
    "general": 0.20
}
# Selecciona: programming (0.72)
```

**2. Skill Config**:
```python
skill = SKILL_CONFIGS["programming"]
# temperature: 0.2 (precisión)
# system_prompt: "Eres experto en programación..."
```

**3. Prompt Especializado**:
```
Eres un experto en programación con 15 años de experiencia.

DIRECTRICES:
- Código limpio, idiomático y bien documentado
[...]

# Consulta del Usuario
¿Cómo optimizo esta consulta SQL lenta?

# Instrucciones
Responde según directrices de programming con nivel de confianza 72%.
```

**4. SOLAR genera con temp=0.2**:
```
## Análisis de Optimización SQL

1. **Indexación**:
   - Crea índices en columnas de WHERE/JOIN
   [código específico]

2. **EXPLAIN ANALYZE**:
   ```sql
   EXPLAIN ANALYZE
   SELECT ...
   ```
   
[respuesta técnica precisa]
```

## ✅ Beneficios de esta Estrategia

1. **RAM**: 900 MB vs 11.6 GB (ahorro 92%)
2. **Especialización**: Prompts expertos >> LLMs genéricos de 7B
3. **Latencia**: Sin carga de modelos (SOLAR ya en RAM)
4. **Aprendizaje**: TRM mejora routing con feedback
5. **Mantenibilidad**: Editar YAML >> reentrenar LLMs

## 📦 Archivos Nuevos/Modificados

**CREAR**:
- ✅ `core/skill_configs.py` (+180 LOC)
- ✅ `scripts/generate_skill_dataset.py` (+200 LOC)
- ✅ `scripts/train_trm_skills.py` (+150 LOC)
- ✅ `tests/test_skill_configs.py` (+120 LOC)

**MODIFICAR**:
- ✅ `core/trm_classifier.py`: +6 heads (+30 LOC)
- ✅ `core/mcp.py`: `select_skill_and_generate()` (+60 LOC)
- ✅ `core/graph.py`: Integrar skill selection (+25 LOC)

**NO MODIFICAR**:
- ❌ `core/model_pool.py`: NO skills_cache (usa SOLAR/LFM2 existentes)
- ❌ `config/sarai.yaml`: Skills quedan como referencia, no se cargan

**Total**: +765 LOC (vs +970 LOC incorrecto)

---

**Conclusión**: Los skills SON PROMPTS ESPECIALIZADOS + CONFIGS, no modelos separados. El TRM aprende a routear, el MCP selecciona el skill, y SOLAR/LFM2 generan con contexto especializado. 🎯
