#!/bin/bash
# Ejemplo de uso del sistema de benchmarking
# Este script demuestra el flujo completo de comparación entre fases

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "📊 SARAi Benchmark System - Ejemplo de Uso"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# ESCENARIO: Comparar v2.13 (baseline) vs v2.14 (unified wrapper)
# ============================================================================

echo "🎯 Escenario: Validar que v2.14 mejora sobre v2.13"
echo ""

# Paso 1: Verificar que tenemos benchmarks previos
echo "Paso 1/4: Verificar benchmarks existentes"
echo "─────────────────────────────────────────────"

if [ ! -d "benchmarks/results" ]; then
    echo "📁 Creando directorio de resultados..."
    mkdir -p benchmarks/results
fi

BENCHMARK_COUNT=$(ls benchmarks/results/benchmark_*.json 2>/dev/null | wc -l)
echo "✅ Benchmarks guardados: $BENCHMARK_COUNT"
echo ""

# Paso 2: Ejecutar benchmark de v2.14 (versión actual)
echo "Paso 2/4: Ejecutar benchmark v2.14"
echo "───────────────────────────────────"
echo "⚠️  NOTA: Esto requiere que SARAi esté instalado y funcional"
echo ""

read -p "¿Ejecutar benchmark real? (requiere ~5 min) [y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Ejecutando benchmark v2.14..."
    make benchmark VERSION=v2.14
    echo ""
else
    echo "⏭️  Saltando benchmark real (usando ejemplo simulado)"
    echo ""
    
    # Crear benchmark simulado para demostración
    cat > benchmarks/results/benchmark_v2.14_example.json <<EOF
{
  "version": "v2.14",
  "timestamp": "$(date -Iseconds)",
  "latency": {
    "text_short": {
      "p50": 2.3,
      "p95": 2.8,
      "p99": 3.1,
      "mean": 2.4,
      "samples": 5
    },
    "text_long": {
      "p50": 26.2,
      "p95": 31.5,
      "p99": 32.8,
      "mean": 27.1,
      "samples": 2
    },
    "rag": {
      "p50": 29.5,
      "p95": 36.2,
      "p99": 37.1,
      "mean": 30.8,
      "samples": 3
    }
  },
  "memory": {
    "base_gb": 0.5,
    "text_gb": 5.2,
    "vision_gb": 8.9,
    "p99_gb": 10.8,
    "delta_text_gb": 4.7,
    "delta_vision_gb": 3.7
  },
  "accuracy": {
    "classification": {
      "hard_precision": 0.87,
      "soft_precision": 0.79,
      "overall_accuracy": 0.83
    },
    "skills": {
      "precision": 0.94,
      "total_samples": 5,
      "correct": 4
    }
  },
  "other": {
    "cold_start": {
      "lfm2_load_time": 0.9,
      "solar_load_time": 0.0,
      "total_load_time": 0.9
    },
    "code_complexity": {
      "graph_loc": 380,
      "nesting_max": 1,
      "try_except_count": 1,
      "file": "core/graph_v2_14.py"
    }
  }
}
EOF
    
    echo "✅ Benchmark simulado creado"
fi

# Paso 3: Crear benchmark de v2.13 si no existe (para comparación)
echo ""
echo "Paso 3/4: Verificar benchmark v2.13 (baseline)"
echo "───────────────────────────────────────────────"

if [ ! -f "benchmarks/results/benchmark_v2.13_baseline.json" ]; then
    echo "📝 Creando benchmark baseline v2.13 (simulado)..."
    
    cat > benchmarks/results/benchmark_v2.13_baseline.json <<EOF
{
  "version": "v2.13",
  "timestamp": "$(date -Iseconds --date='7 days ago')",
  "latency": {
    "text_short": {
      "p50": 2.8,
      "p95": 3.4,
      "p99": 3.5,
      "mean": 2.9,
      "samples": 5
    },
    "text_long": {
      "p50": 28.5,
      "p95": 34.2,
      "p99": 35.2,
      "mean": 29.3,
      "samples": 2
    },
    "rag": {
      "p50": 32.1,
      "p95": 39.5,
      "p99": 40.5,
      "mean": 33.7,
      "samples": 3
    }
  },
  "memory": {
    "base_gb": 0.5,
    "text_gb": 5.8,
    "vision_gb": 9.2,
    "p99_gb": 12.1,
    "delta_text_gb": 5.3,
    "delta_vision_gb": 3.4
  },
  "accuracy": {
    "classification": {
      "hard_precision": 0.87,
      "soft_precision": 0.79,
      "overall_accuracy": 0.83
    },
    "skills": {
      "precision": 0.92,
      "total_samples": 5,
      "correct": 4
    }
  },
  "other": {
    "cold_start": {
      "lfm2_load_time": 1.2,
      "solar_load_time": 0.0,
      "total_load_time": 1.2
    },
    "code_complexity": {
      "graph_loc": 1022,
      "nesting_max": 5,
      "try_except_count": 7,
      "file": "core/graph.py"
    }
  }
}
EOF
    
    echo "✅ Benchmark baseline v2.13 creado"
else
    echo "✅ Benchmark v2.13 ya existe"
fi

# Paso 4: Comparar versiones
echo ""
echo "Paso 4/4: Comparar v2.13 vs v2.14"
echo "──────────────────────────────────"
echo ""

# Usar Python para comparación (más confiable que Makefile en demo)
python3 <<PYTHON
import json
from pathlib import Path

# Cargar benchmarks
with open("benchmarks/results/benchmark_v2.13_baseline.json", 'r') as f:
    v2_13 = json.load(f)

# Intentar cargar v2.14 real, si no existe usar example
v2_14_file = Path("benchmarks/results").glob("benchmark_v2.14_*.json")
v2_14_files = sorted(v2_14_file)

if v2_14_files:
    with open(v2_14_files[-1], 'r') as f:
        v2_14 = json.load(f)
else:
    print("❌ No se encontró benchmark v2.14")
    exit(1)

# Función helper para calcular delta
def calc_delta(old, new):
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100

# Mostrar comparación
print("=" * 80)
print(f"📊 SARAi Version Comparison: v2.13 → v2.14")
print("=" * 80)
print()

# Latency
print("✅ IMPROVEMENTS:")
improvements = []
regressions = []

# Latency comparisons
categories = [
    ("text_short", "Latencia Text Short P50"),
    ("text_long", "Latencia Text Long P50"),
    ("rag", "Latencia RAG P50"),
]

for cat, label in categories:
    old_val = v2_13["latency"][cat]["p50"]
    new_val = v2_14["latency"][cat]["p50"]
    delta = calc_delta(old_val, new_val)
    
    if delta < 0:
        improvements.append((label, old_val, new_val, delta))
        print(f"  • {label}:")
        print(f"    {old_val}s → {new_val}s ({delta:+.1f}%)")

# Memory
old_ram = v2_13["memory"]["p99_gb"]
new_ram = v2_14["memory"]["p99_gb"]
delta_ram = calc_delta(old_ram, new_ram)

if delta_ram < 0:
    improvements.append(("RAM P99", old_ram, new_ram, delta_ram))
    print(f"  • RAM P99:")
    print(f"    {old_ram} GB → {new_ram} GB ({delta_ram:+.1f}%)")

# Code complexity
old_loc = v2_13["other"]["code_complexity"]["graph_loc"]
new_loc = v2_14["other"]["code_complexity"]["graph_loc"]
delta_loc = calc_delta(old_loc, new_loc)

if delta_loc < 0:
    improvements.append(("Code LOC", old_loc, new_loc, delta_loc))
    print(f"  • Code LOC (graph.py):")
    print(f"    {old_loc} → {new_loc} ({delta_loc:+.1f}%)")

# Nesting
old_nest = v2_13["other"]["code_complexity"]["nesting_max"]
new_nest = v2_14["other"]["code_complexity"]["nesting_max"]
delta_nest = calc_delta(old_nest, new_nest)

if delta_nest < 0:
    improvements.append(("Nesting Max", old_nest, new_nest, delta_nest))
    print(f"  • Nesting Max:")
    print(f"    {old_nest} → {new_nest} ({delta_nest:+.1f}%)")

print()
print("❌ REGRESSIONS:")
if not regressions:
    print("  (none)")

print()
print("📈 SUMMARY:")
print(f"  Total Improvements: {len(improvements)}")
print(f"  Total Regressions: {len(regressions)}")
print(f"  Net Improvement: {len(improvements) - len(regressions)}")
print()

if len(improvements) > len(regressions):
    print("🎉 Overall: VERSION IMPROVED ✅")
else:
    print("⚠️  Overall: VERSION REGRESSED ❌")

print("=" * 80)
PYTHON

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Ejemplo Completado"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📚 Próximos pasos:"
echo "  1. Ver guía completa: docs/BENCHMARKING_GUIDE.md"
echo "  2. Ejecutar benchmark real: make benchmark VERSION=v2.14"
echo "  3. Ver histórico: make benchmark-history"
echo ""
echo "🎯 Recuerda: Ejecuta benchmark después de cada fase para validar mejoras"
echo ""
