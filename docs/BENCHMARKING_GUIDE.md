# 📊 Sistema de Benchmarking SARAi - Guía de Uso

## Filosofía

> **"No optimizamos lo que no medimos.**  
> **Cada fase debe probar con números que es mejor que la anterior."**

Este sistema te permite:
- ✅ Medir KPIs reales (latencia, RAM, precisión)
- ✅ Comparar versiones automáticamente
- ✅ Validar que cada fase es mejor que la anterior
- ✅ Guardar histórico de evolución

---

## 🚀 Quick Start

### 1. Benchmark de Versión Actual

```bash
# Al finalizar implementación de v2.14
make benchmark VERSION=v2.14
```

**Output**:
```
🚀 Running SARAi Benchmark Suite - v2.14
============================================================
📊 Benchmark: Latency Text Short
  Query: '¿Qué es Python?...' → 2.3s
  Query: 'Explica recursividad...' → 2.1s
  ...

📊 Benchmark: Memory Usage
  Base: 0.5 GB
  Text: 5.2 GB
  Vision: 8.9 GB
  P99: 10.8 GB

📊 Benchmark: Classification Accuracy
  Hard precision: 0.87
  Soft precision: 0.79
  
✅ Benchmark complete: v2.14
💾 Results saved to: benchmarks/results/benchmark_v2.14_20251101_143022.json
```

---

### 2. Comparar Con Versión Anterior

```bash
# Comparar v2.14 vs v2.13
make benchmark-compare OLD=v2.13 NEW=v2.14
```

**Output**:
```
================================================================================
📊 SARAi Version Comparison: v2.13 → v2.14
================================================================================

✅ IMPROVEMENTS:
  • latency_text_short_p50:
    2.8 → 2.3 (-17.9%)
  • memory_p99:
    12.1 GB → 10.8 GB (-10.7%)
  • code_loc:
    1022 → 380 (-62.8%)

❌ REGRESSIONS:
  (none)

📈 SUMMARY:
  Total Improvements: 3
  Total Regressions: 0
  Net Improvement: 3

🎉 Overall: VERSION IMPROVED ✅
================================================================================
```

---

### 3. Ver Histórico

```bash
make benchmark-history
```

**Output**:
```
📚 Benchmark History (5 results):
  • benchmark_v2.10_20251020_103045.json
  • benchmark_v2.11_20251022_141523.json
  • benchmark_v2.12_20251025_092034.json
  • benchmark_v2.13_20251028_154512.json
  • benchmark_v2.14_20251101_143022.json
```

---

### 4. Benchmark Rápido (Debug)

```bash
# Solo latencia y RAM (útil para debug rápido)
make benchmark-quick VERSION=v2.14
```

**Output** (JSON):
```json
{
  "latency_short": {
    "p50": 2.3,
    "p95": 2.8,
    "p99": 3.1,
    "mean": 2.4,
    "samples": 5
  },
  "memory": {
    "base_gb": 0.5,
    "text_gb": 5.2,
    "vision_gb": 8.9,
    "p99_gb": 10.8
  }
}
```

---

## 📋 KPIs Medidos

### Latencia

| KPI | Descripción | Objetivo |
|-----|-------------|----------|
| **text_short P50** | Mediana de queries cortas (5 queries) | ≤ 3s |
| **text_long P50** | Mediana de queries largas (2 queries) | ≤ 25s |
| **rag P50** | Mediana de queries RAG con web search | ≤ 30s |
| **P95/P99** | Percentiles altos (outliers) | Monitorear |

### Memoria

| KPI | Descripción | Objetivo |
|-----|-------------|----------|
| **base_gb** | RAM sin modelos cargados | Baseline |
| **text_gb** | RAM con LFM2 cargado | ~5 GB |
| **vision_gb** | RAM con Qwen3-VL cargado | ~9 GB |
| **p99_gb** | Pico máximo de RAM | **≤ 12 GB** |

### Precisión

| KPI | Descripción | Objetivo |
|-----|-------------|----------|
| **hard_precision** | Clasificación técnica (TRM) | ≥ 0.85 |
| **soft_precision** | Clasificación emocional (TRM) | ≥ 0.75 |
| **skills_precision** | Detección de skills (v2.12) | ≥ 0.90 |

### Complejidad de Código

| KPI | Descripción | Objetivo |
|-----|-------------|----------|
| **graph_loc** | Líneas de código en graph.py | Menos es mejor |
| **nesting_max** | Nivel máximo de anidación | ≤ 2 |
| **try_except_count** | Bloques try-except | Menos es mejor |

### Otros

| KPI | Descripción | Objetivo |
|-----|-------------|----------|
| **cold_start** | Tiempo de carga de modelos | ≤ 2s |

---

## 🔄 Flujo de Trabajo Típico

### Escenario 1: Nueva Fase Implementada

```bash
# 1. Implementar v2.15
# ... código ...

# 2. Ejecutar benchmark
make benchmark VERSION=v2.15

# 3. Comparar con v2.14
make benchmark-compare OLD=v2.14 NEW=v2.15

# 4. Si hay mejoras → commit
# Si hay regresiones → investigar
```

---

### Escenario 2: Optimización Iterativa

```bash
# Antes de optimización
make benchmark VERSION=v2.14-baseline

# Aplicar optimización
# ... código ...

# Después de optimización
make benchmark VERSION=v2.14-optimized

# Comparar
make benchmark-compare OLD=v2.14-baseline NEW=v2.14-optimized

# ¿Mejora > 5%? → Keeper
# ¿Mejora < 5%? → Revertir (no vale la pena la complejidad)
```

---

### Escenario 3: Validación de Regresión

```bash
# Después de cambio grande (ej: refactorización graph.py)
make benchmark VERSION=v2.14-after-refactor

# Comparar con versión estable anterior
make benchmark-compare OLD=v2.13-stable NEW=v2.14-after-refactor

# Validar que:
# ✅ Latencia no aumentó > 10%
# ✅ RAM no aumentó > 5%
# ✅ Precisión no bajó
# ✅ Código LOC disminuyó (si es refactorización)
```

---

## 📊 Ejemplo Real: v2.13 → v2.14

### Resultados v2.13 (Baseline)

```json
{
  "version": "v2.13",
  "latency": {
    "text_short": {"p50": 2.8, "p99": 3.5},
    "text_long": {"p50": 28.5, "p99": 35.2},
    "rag": {"p50": 32.1, "p99": 40.5}
  },
  "memory": {
    "p99_gb": 12.1
  },
  "accuracy": {
    "classification": {"hard_precision": 0.87, "soft_precision": 0.79}
  },
  "other": {
    "code_complexity": {"graph_loc": 1022, "nesting_max": 5}
  }
}
```

### Resultados v2.14 (Unified Wrapper + LCEL)

```json
{
  "version": "v2.14",
  "latency": {
    "text_short": {"p50": 2.3, "p99": 2.9},
    "text_long": {"p50": 26.2, "p99": 32.8},
    "rag": {"p50": 29.5, "p99": 37.1}
  },
  "memory": {
    "p99_gb": 10.8
  },
  "accuracy": {
    "classification": {"hard_precision": 0.87, "soft_precision": 0.79}
  },
  "other": {
    "code_complexity": {"graph_loc": 380, "nesting_max": 1}
  }
}
```

### Comparación Automática

```
✅ IMPROVEMENTS:
  • latency_text_short_p50: 2.8s → 2.3s (-17.9%) ⬇️
  • latency_text_long_p50: 28.5s → 26.2s (-8.1%) ⬇️
  • latency_rag_p50: 32.1s → 29.5s (-8.1%) ⬇️
  • memory_p99: 12.1 GB → 10.8 GB (-10.7%) ⬇️
  • code_loc: 1022 → 380 (-62.8%) ⬇️
  • nesting_max: 5 → 1 (-80%) ⬇️

❌ REGRESSIONS: (none)

📈 SUMMARY:
  Total Improvements: 6
  Total Regressions: 0
  Net Improvement: 6

🎉 Overall: VERSION IMPROVED ✅
```

**Conclusión**: v2.14 es **objetivamente mejor** que v2.13 en todos los aspectos.

---

## 🛠️ Personalización

### Agregar Nuevos KPIs

Editar `tests/benchmark_suite.py`:

```python
def benchmark_custom_metric(self) -> Dict[str, float]:
    """Tu métrica personalizada"""
    # ... implementación ...
    return {"custom_score": 0.95}

# Agregar en run_all()
self.results["custom"] = self.benchmark_custom_metric()
```

### Cambiar Queries de Test

Editar `BENCHMARK_QUERIES` en `tests/benchmark_suite.py`:

```python
BENCHMARK_QUERIES = {
    "text_short": [
        "Tu query 1",
        "Tu query 2",
        # ...
    ],
}
```

---

## 📈 Visualización (Futuro)

**Próximas versiones** incluirán:

```bash
# Dashboard web con gráficas
make benchmark-dashboard

# Exportar a CSV para Excel/Google Sheets
make benchmark-export-csv

# Integración con Grafana
make benchmark-grafana
```

---

## 🎯 Best Practices

### 1. Benchmark al Finalizar Cada Fase

```bash
# SIEMPRE ejecutar benchmark al completar una fase
git checkout -b feature/v2.15
# ... implementar ...
make benchmark VERSION=v2.15 --save
git commit -m "feat: v2.15 implementation + benchmark"
```

### 2. Comparar Solo Versiones Estables

```bash
# ✅ CORRECTO
make benchmark-compare OLD=v2.13 NEW=v2.14

# ❌ INCORRECTO (comparar WIP con stable)
make benchmark-compare OLD=v2.13 NEW=v2.14-wip-broken
```

### 3. Guardar Benchmarks en Git

```bash
# Incluir benchmarks en commits
git add benchmarks/results/benchmark_v2.14_*.json
git commit -m "chore: add v2.14 benchmark results"
```

### 4. Validar Antes de Merge

```bash
# Pre-merge checklist:
# ✅ Benchmark ejecutado
# ✅ Comparación vs versión anterior OK
# ✅ Sin regresiones críticas (latencia +20%, RAM +15%)
# ✅ Al menos 1 mejora significativa
```

---

## 🚨 Troubleshooting

### Error: "No benchmark found for vX.XX"

**Causa**: No existe benchmark guardado para esa versión.

**Solución**:
```bash
# Ejecutar benchmark primero
make benchmark VERSION=v2.13 --save

# Luego comparar
make benchmark-compare OLD=v2.13 NEW=v2.14
```

---

### Error: "Model not found"

**Causa**: Versión especificada no coincide con código actual.

**Solución**:
```bash
# Asegurarse de estar en el branch correcto
git checkout v2.14

# Ejecutar benchmark
make benchmark VERSION=v2.14
```

---

### Latencias Anormalmente Altas

**Causa**: CPU bajo carga, otros procesos, caché fría.

**Solución**:
```bash
# Ejecutar 2-3 veces y promediar
make benchmark VERSION=v2.14 --save
sleep 60
make benchmark VERSION=v2.14 --save
sleep 60
make benchmark VERSION=v2.14 --save

# Usar el resultado más consistente
```

---

## 📚 Referencias

- `tests/benchmark_suite.py` - Código fuente del benchmark
- `benchmarks/results/` - Directorio de resultados guardados
- `Makefile` - Targets de benchmark (líneas 820-860)

---

**Mantra del Benchmarking**:

_"Cada fase debe ser objetivamente mejor que la anterior._  
_Si no puedes medirlo, no puedes afirmar que mejoraste._  
_Los números no mienten."_

---

**Última actualización**: 1 Noviembre 2025 (v2.14)
