#!/usr/bin/env bash
# scripts/detect_hardware.sh
# Detección automática de hardware para llama.cpp Hybrid Strategy
# Retorna: STRATEGY (avx512-16t | avx2-8t | avx2-blas | generic)

set -euo pipefail

# ============================================================================
# PASO 1: Detectar CPU Model, Cores, Threads, RAM
# ============================================================================

CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
CPU_CORES=$(nproc)
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')

# Flags de CPU
HAS_AVX512=$(grep -q avx512f /proc/cpuinfo && echo "true" || echo "false")
HAS_AVX2=$(grep -q avx2 /proc/cpuinfo && echo "true" || echo "false")
HAS_FMA=$(grep -q fma /proc/cpuinfo && echo "true" || echo "false")
HAS_F16C=$(grep -q f16c /proc/cpuinfo && echo "true" || echo "false")

# ============================================================================
# PASO 2: Lógica de Selección de Estrategia
# ============================================================================

STRATEGY="generic"  # Fallback por defecto

if [[ "$HAS_AVX512" == "true" ]] && [[ $CPU_CORES -ge 12 ]]; then
    # Alta gama: Ryzen 9, Xeon W, i9-12900K
    STRATEGY="avx512-16t"
    OPTIMAL_THREADS=16

elif [[ "$HAS_AVX2" == "true" ]] && [[ $CPU_CORES -ge 8 ]]; then
    # Media-alta: Ryzen 5, i7-12700K
    STRATEGY="avx2-8t"
    OPTIMAL_THREADS=8

elif [[ "$HAS_AVX2" == "true" ]] && [[ "$HAS_FMA" == "true" ]] && [[ $TOTAL_RAM_GB -ge 8 ]]; then
    # Media-baja: i5 Skylake, i3-10110U (CRÍTICO: OpenBLAS es salvavidas)
    STRATEGY="avx2-blas"
    OPTIMAL_THREADS=$((CPU_CORES - 1))  # N-1 para evitar saturación
    
    # Detectar arquitectura específica para OpenBLAS
    if grep -q "6th Gen" <<< "$CPU_MODEL" || grep -qi "skylake" <<< "$CPU_MODEL"; then
        ARCH_HINT="skylake"
    elif grep -qi "haswell\|4th Gen" <<< "$CPU_MODEL"; then
        ARCH_HINT="haswell"
    else
        ARCH_HINT="core2"  # Genérico conservador
    fi

elif [[ $TOTAL_RAM_GB -ge 8 ]]; then
    # Básico: cualquier x86-64 con 8GB+
    STRATEGY="generic"
    OPTIMAL_THREADS=$((CPU_CORES > 2 ? CPU_CORES - 1 : 1))

else
    # Ultra-low-end: <8GB RAM → no recomendado para SARAi
    echo "⚠️  ADVERTENCIA: RAM insuficiente ($TOTAL_RAM_GB GB < 8 GB mínimo)" >&2
    echo "⚠️  SARAi v2.16 requiere mínimo 8GB para SOLAR-10.7B Q4_K_M" >&2
    STRATEGY="generic"
    OPTIMAL_THREADS=1
fi

# ============================================================================
# PASO 3: Output (JSON para fácil parsing)
# ============================================================================

cat << EOF
{
  "strategy": "$STRATEGY",
  "cpu_model": "$CPU_MODEL",
  "cpu_cores": $CPU_CORES,
  "optimal_threads": ${OPTIMAL_THREADS:-4},
  "total_ram_gb": $TOTAL_RAM_GB,
  "has_avx512": $HAS_AVX512,
  "has_avx2": $HAS_AVX2,
  "has_fma": $HAS_FMA,
  "has_f16c": $HAS_F16C,
  "arch_hint": "${ARCH_HINT:-generic}"
}
EOF

# También exporta como variables de entorno (para Makefile)
echo "export STRATEGY=$STRATEGY" >&2
echo "export OPTIMAL_THREADS=${OPTIMAL_THREADS:-4}" >&2
echo "export ARCH_HINT=${ARCH_HINT:-generic}" >&2

# ============================================================================
# PASO 4: Mensaje de Estrategia Seleccionada
# ============================================================================

case "$STRATEGY" in
    avx512-16t)
        echo "🏎️  Estrategia: AVX512-16T (Alta gama, +60% velocidad)" >&2
        ;;
    avx2-8t)
        echo "🚗 Estrategia: AVX2-8T (Media gama, +40% velocidad)" >&2
        ;;
    avx2-blas)
        echo "🚙 Estrategia: AVX2-BLAS (CPU legacy con OpenBLAS, +30-50% velocidad)" >&2
        echo "   CPU detectada: $CPU_MODEL (${ARCH_HINT:-generic})" >&2
        echo "   Threads óptimos: $OPTIMAL_THREADS (N-1 para evitar saturación)" >&2
        ;;
    generic)
        echo "🚶 Estrategia: GENERIC (Zero-Compile portable, rendimiento base)" >&2
        ;;
esac
