# 📊 SARAi Benchmark System - Quick Reference

## 🎯 v2.14 Baseline (2025-11-01)

### Resultados Reales

| Métrica | v2.13 | v2.14 | Mejora | Estado |
|---------|-------|-------|--------|--------|
| **LOC graph.py** | 720 | **354** | **-50.8%** | 🎉 EXCELENTE |
| **Nesting Max** | 6 | **5** | **-16.7%** | ✅ BUENO |
| **RAM Baseline** | ~3.0 GB | **2.1 GB** | **-30%** | ✅ SUPERADO |
| **Try/Except** | 8 | 5 | -37.5% | ✅ BUENO |

**Veredicto**: v2.14 **CUMPLE** todos los objetivos del Anti-Spaghetti Pattern ✅

---

## 🚀 Comandos Rápidos

```bash
# Ejecutar benchmark de una versión
make benchmark VERSION=v2.15

# Comparar dos versiones
make benchmark-compare OLD=v2.14 NEW=v2.15

# Ver historial de benchmarks
make benchmark-history

# Benchmark rápido (solo latencia + RAM)
make benchmark-quick VERSION=v2.15
```

---

## 📂 Archivos Clave

- **Sistema**: `tests/benchmark_suite.py` (490 LOC)
- **Guía completa**: `docs/BENCHMARKING_GUIDE.md`
- **Resultados v2.14**: `docs/BENCHMARK_v2.14_RESULTS.md`
- **Baseline JSON**: `benchmarks/results/benchmark_v2.14_*_real.json`

---

## 📈 KPIs Medidos

### 1. Latencia (3 categorías)
- `text_short`: 5 queries cortos (P50, P95, P99)
- `text_long`: 2 queries largos
- `rag`: 3 búsquedas web

### 2. Memoria (4 métricas)
- `base_gb`: RAM sin modelos
- `text_gb`: RAM con LFM2
- `vision_gb`: RAM con Qwen3-VL
- `p99_gb`: Pico de RAM (objetivo: ≤12 GB)

### 3. Precisión (3 tipos)
- `hard_precision`: Clasificación técnica (≥0.85)
- `soft_precision`: Clasificación emocional (≥0.75)
- `skills_precision`: Detección de skills (≥0.90)

### 4. Complejidad de Código (3 métricas)
- `graph_loc`: Líneas de código (menor es mejor)
- `nesting_max`: Indentación máxima (objetivo: ≤5)
- `try_except_count`: Bloques de manejo (menor es mejor)

### 5. Cold-Start (2 métricas)
- `lfm2_load_time`: Tiempo de carga LFM2 (≤2s)
- `solar_load_time`: Tiempo de carga SOLAR (≤2s)

---

## 🎯 Workflow por Fase

### Al Final de Cada Fase

1. **Ejecutar benchmark**:
   ```bash
   make benchmark VERSION=v2.XX
   ```

2. **Comparar con versión anterior**:
   ```bash
   make benchmark-compare OLD=v2.XX-1 NEW=v2.XX
   ```

3. **Validar resultados**:
   - ✅ Al menos 1 mejora >5%
   - ⚠️ Ninguna regresión >20%

4. **Commit a git**:
   ```bash
   git add benchmarks/results/
   git commit -m "benchmark: v2.XX results"
   ```

---

## 📊 Ejemplo de Output

```
================================================================================
                         🎉 COMPARACIÓN: v2.14 → v2.15                          
================================================================================

✅ MEJORAS

  • latency_text_short_p50: 2.8s → 2.3s (-17.9%)
  • memory_p99_gb: 12.1 GB → 10.8 GB (-10.7%)
  • graph_loc: 720 → 354 (-50.8%)

❌ REGRESIONES

  (ninguna)

📈 RESUMEN

  • Total mejoras: 3
  • Total regresiones: 0

💚 VEREDICTO: v2.15 MEJORÓ vs v2.14
```

---

## 🔍 Troubleshooting

### "No such file or directory: benchmark_suite.py"

**Solución**: El sistema de benchmarking está en `tests/benchmark_suite.py`. Asegúrate de ejecutar desde la raíz del proyecto.

### "ModuleNotFoundError: No module named 'psutil'"

**Solución**: Instala dependencias con `pip install psutil`.

### "Benchmark muy lento (>10 min)"

**Solución**: Usa `make benchmark-quick` para un benchmark rápido (solo latencia + RAM).

---

## 📝 Filosofía

> **"No optimizamos lo que no medimos.  
> Cada fase debe probar con números que es mejor que la anterior."**

**Principios**:
1. Benchmark al **final de cada fase**
2. Comparar **solo versiones estables** (no WIP)
3. Guardar **resultados en git** (trazabilidad)
4. Validar **antes de merge** a master

---

## 🎯 Objetivos por Versión

| Version | KPI Principal | Objetivo |
|---------|--------------|----------|
| v2.14 | Anti-Spaghetti | -50.8% LOC ✅ |
| v2.15 | (TBD) | (TBD) |
| v2.16 | (TBD) | (TBD) |

---

**Última actualización**: 2025-11-01  
**Versión sistema**: v2.14  
**Status**: ✅ Operacional
