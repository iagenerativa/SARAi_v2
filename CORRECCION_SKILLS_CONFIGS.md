# ✅ CORRECCIÓN FINAL: Skills como Configs (31 Oct 2025)

## 🎯 Entendimiento Correcto

Los **skills NO son modelos LLM separados**, son **CONFIGURACIONES** que optimizan SOLAR/LFM2 para tareas específicas.

### Ejemplo Real del Sistema

```yaml
# config/sarai.yaml
skills:
  programming:
    model: "solar"  # ← USA SOLAR EXISTENTE (no descarga CodeLlama)
    temperature: 0.2  # Precisión para código
    max_tokens: 512
    system_prompt: |
      Eres un experto en programación con 15 años de experiencia.
      
      DIRECTRICES:
      - Código limpio, idiomático y bien documentado
      - Explicaciones técnicas precisas sin simplificaciones
      - Ejemplos ejecutables con manejo de errores
      - Referencia buenas prácticas (PEP-8, SOLID, etc)
      
      FORMATO DE RESPUESTA:
      1. Explicación breve (2-3 líneas)
      2. Código con comentarios
      3. Ejemplo de uso
      4. Posibles edge cases
    domains: ["código", "programación", "python", "javascript", "debugging"]
```

## 🏗️ Arquitectura Correcta

```
┌──────────────────────────────────────────────────────────────┐
│  Input: "¿Cómo optimizo esta consulta SQL lenta?"            │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  TRM-Router (6 heads especializados)                         │
│  - programming: 0.85 ⭐ GANADOR                              │
│  - diagnosis: 0.45                                           │
│  - finance: 0.05                                             │
│  - creative: 0.02                                            │
│  - reasoning: 0.35                                           │
│  - general: 0.20                                             │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  MCP.select_skill_and_generate()                             │
│  1. Selecciona: programming (score más alto)                 │
│  2. Carga: SKILL_CONFIGS["programming"]                      │
│  3. Construye prompt especializado:                          │
│     "Eres experto en programación... [directrices]...        │
│      ¿Cómo optimizo SQL?"                                    │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  SOLAR-10.7B (YA en memoria)                                 │
│  - temperature: 0.2 (precisión)                              │
│  - max_tokens: 512                                           │
│  - Genera con contexto de "experto en programación"          │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Respuesta Especializada:                                    │
│                                                               │
│  ## Análisis de Optimización SQL                             │
│                                                               │
│  1. **Indexación**:                                          │
│     CREATE INDEX idx_user_created ON users(created_at);      │
│                                                               │
│  2. **EXPLAIN ANALYZE**:                                     │
│     [análisis técnico detallado]                             │
│                                                               │
│  [respuesta técnica precisa, estilo experto]                 │
└──────────────────────────────────────────────────────────────┘
```

## 📊 Comparación de Estrategias

| Aspecto | ❌ Skills MoE (Incorrecto) | ✅ Skill Configs (Correcto) |
|---------|----------------------------|----------------------------|
| **Concepto** | 6 LLMs especializados | 6 configuraciones (prompts) |
| **Modelos en RAM** | 8 (SOLAR + LFM2 + 6 skills) | 2 (SOLAR + LFM2) |
| **RAM total** | ~11.6 GB | ~900 MB |
| **Latencia** | Alta (carga modelo) | Baja (modelo ya cargado) |
| **Especialización** | Limitada al LLM | Profunda (prompts expertos) |
| **Ejemplos** | CodeLlama, Mistral, FinGPT | Prompt experto programación |
| **Mantenibilidad** | Compleja (6 GGUFs) | Simple (editar YAML) |
| **Aprendizaje TRM** | No | Sí (mejora routing) |
| **LOC** | +970 | +765 (-205 neto) |

## 🎯 Plan de Implementación

### Semana 1 - Tickets Corregidos

#### T1.1-FINAL: Skill Configs + TRM Heads (8h)

**Archivos NUEVOS**:
- `core/skill_configs.py` (+180 LOC)
  ```python
  SKILL_CONFIGS = {
      "programming": {
          "model": "solar",
          "temperature": 0.2,
          "system_prompt": "Eres experto en programación...",
          "domains": ["código", "python", "debugging"]
      },
      # ... 5 más
  }
  ```

- `scripts/generate_skill_dataset.py` (+200 LOC)
  - Genera 10K queries sintéticas clasificadas
  - Categories: programming, diagnosis, finance, creative, reasoning, general

- `scripts/train_trm_skills.py` (+150 LOC)
  - Entrena TRM-Router con 6 heads
  - Target accuracy: >85%

- `tests/test_skill_configs.py` (+120 LOC)
  - Valida estructura de configs
  - Tests por skill (6 × 2 = 12 tests)

**Archivos MODIFICADOS**:
- `core/trm_classifier.py` (+30 LOC)
  ```python
  class TRMClassifierSpecialized(nn.Module):
      def __init__(self):
          # ... recursión existente ...
          self.head_programming = nn.Linear(256, 1)
          self.head_diagnosis = nn.Linear(256, 1)
          self.head_finance = nn.Linear(256, 1)
          self.head_creative = nn.Linear(256, 1)
          self.head_reasoning = nn.Linear(256, 1)
          self.head_general = nn.Linear(256, 1)
  ```

- `core/mcp.py` (+60 LOC)
  ```python
  def select_skill_and_generate(self, trm_scores: dict, user_input: str) -> str:
      # 1. Seleccionar skill de mayor score
      best_skill = max(trm_scores.items(), key=lambda x: x[1])
      skill_name, confidence = best_skill
      
      # 2. Cargar config
      config = SKILL_CONFIGS[skill_name]
      
      # 3. Construir prompt especializado
      specialized_prompt = f"{config['system_prompt']}\n\n{user_input}"
      
      # 4. Generar con SOLAR/LFM2
      model = self.model_pool.get(config['model'])
      response = model.generate(
          specialized_prompt,
          temperature=config['temperature'],
          max_tokens=config['max_tokens']
      )
      
      return response
  ```

- `core/graph.py` (+25 LOC)
  - Integrar `select_skill_and_generate()` en nodos

**Total**: +765 LOC

#### T1.2-FINAL: MCP Skill Selection (6h)

**Objetivo**: Integrar skill selection en LangGraph

**Archivos**:
- `core/graph.py` (+40 LOC): Nodos especializados
- `core/feedback.py` (+25 LOC): Log skill metadata
- `tests/test_graph_skills.py` (+150 LOC): 6 tests E2E

**Total**: +215 LOC

#### T1.3-FINAL: MCP Continuous Learning (6h)

**Objetivo**: Ajustar skill routing basado en feedback

**Archivos**:
- `scripts/tune_trm_skills.py` (+100 LOC): Reentrenamiento nocturno
- `tests/test_mcp_learning.py` (+80 LOC): Tests de adaptación

**Total**: +180 LOC

### Resumen de Código

**REVERTIR (Skills MoE incorrecto)**:
- `core/model_pool.py`: get_skill(), skills_cache (-150 LOC)
- `core/mcp.py`: execute_skills_moe() (-110 LOC)
- `tests/test_model_pool_skills.py`: Borrar (-320 LOC)
- `tests/test_mcp_skills.py`: Borrar (-300 LOC)
- **Subtotal**: -880 LOC

**CREAR (Skill Configs correcto)**:
- T1.1-FINAL: +765 LOC
- T1.2-FINAL: +215 LOC
- T1.3-FINAL: +180 LOC
- **Subtotal**: +1,160 LOC

**Balance neto**: +280 LOC (simplificación + especialización)

## 🎓 Lecciones Aprendidas

### ❌ Malentendido inicial
"Skills MoE = cargar CodeLlama, Mistral, FinGPT..."

### ✅ Realidad del sistema
"Skills = prompts expertos + configuraciones óptimas para SOLAR/LFM2"

### 💡 Principio clave
> **Prompts especializados + TRM inteligente + 2 LLMs buenos >> 6 LLMs genéricos de 7B**

### 🎯 Beneficios reales

1. **RAM**: 900 MB vs 11.6 GB (ahorro 92%)
2. **Latencia**: Sin carga de modelos (SOLAR ya en memoria)
3. **Especialización**: Prompts de experto >> LLM genérico fine-tuned
4. **Mantenibilidad**: Editar YAML >> gestionar 6 GGUFs
5. **Aprendizaje**: TRM mejora routing con feedback

## ✅ Próximos Pasos Inmediatos

### Mañana (1 Nov):

1. **Revertir código incorrecto** (~2h):
   ```bash
   # Eliminar get_skill() de model_pool.py
   # Eliminar execute_skills_moe() de mcp.py
   # Borrar test_model_pool_skills.py
   # Borrar test_mcp_skills.py
   git add -A
   git commit -m "refactor: Revertir Skills MoE (eran configs, no LLMs)"
   ```

2. **Crear skill_configs.py** (~2h):
   - Diccionario con 6 configs especializados
   - System prompts expertos por dominio
   - Parámetros óptimos (temperature, max_tokens)

3. **Modificar TRM-Router** (~2h):
   - Añadir 6 heads especializados
   - Forward pass retorna 6 scores

4. **Tests básicos** (~2h):
   - Validar estructura de configs
   - Test por skill (12 total)

**ETA**: 8h (día completo)

---

**Firma**: SARAi Development Team  
**Fecha**: 31 Octubre 2025  
**Versión**: v2.12-alpha (estrategia corregida)  
**Status**: Skills como Configs ✅ CORRECTO
