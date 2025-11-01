#!/bin/bash
# verify_consolidation.sh - Validación pre-commit para SARAi v2.12
# Uso: ./scripts/verify_consolidation.sh

set -e  # Exit on error

echo "🔍 Verificación de consolidación SARAi v2.12"
echo "=============================================="
echo ""

# 1. Verificar que estamos en el directorio correcto
if [ ! -f "main.py" ]; then
    echo "❌ Error: Ejecutar desde el directorio raíz de SARAi_v2"
    exit 1
fi

# 2. Verificar archivos modificados existen
echo "📁 Verificando archivos modificados..."
files=(
    "core/model_pool.py"
    "core/mcp.py"
    "config/sarai.yaml"
    "tests/test_model_pool_skills.py"
    "tests/test_mcp_skills.py"
    "pytest.ini"
    "PROGRESO_31102025.md"
    "SEMANA1_TICKETS.md"
    "STATUS_31102025.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (FALTA)"
        exit 1
    fi
done
echo ""

# 3. Verificar sintaxis Python
echo "🐍 Verificando sintaxis Python..."
python3 -m py_compile core/model_pool.py
python3 -m py_compile core/mcp.py
python3 -m py_compile tests/test_model_pool_skills.py
python3 -m py_compile tests/test_mcp_skills.py
echo "  ✅ Sintaxis correcta"
echo ""

# 4. Verificar pytest.ini tiene markers
echo "🏷️  Verificando pytest.ini..."
if grep -q "markers" pytest.ini && \
   grep -q "integration:" pytest.ini && \
   grep -q "slow:" pytest.ini; then
    echo "  ✅ Markers configurados"
else
    echo "  ❌ pytest.ini incompleto"
    exit 1
fi
echo ""

# 5. Ejecutar tests unitarios (sin integration/slow)
echo "🧪 Ejecutando tests unitarios..."
python3 -m pytest tests/test_model_pool_skills.py tests/test_mcp_skills.py \
    -v -m "not integration and not slow" \
    --tb=short \
    -x  # Stop on first failure
echo ""

# 6. Verificar que config/sarai.yaml tiene skills
echo "⚙️  Verificando configuración de skills..."
if grep -q "skills:" config/sarai.yaml && \
   grep -q "programming:" config/sarai.yaml && \
   grep -q "max_concurrent_skills:" config/sarai.yaml; then
    echo "  ✅ Skills configurados"
else
    echo "  ❌ config/sarai.yaml incompleto"
    exit 1
fi
echo ""

# 7. Contar LOC añadidas
echo "📊 Estadísticas de cambios..."
core_pool_lines=$(wc -l < core/model_pool.py)
core_mcp_lines=$(wc -l < core/mcp.py)
test_pool_lines=$(wc -l < tests/test_model_pool_skills.py)
test_mcp_lines=$(wc -l < tests/test_mcp_skills.py)
total_lines=$((test_pool_lines + test_mcp_lines))

echo "  • core/model_pool.py: $core_pool_lines líneas"
echo "  • core/mcp.py: $core_mcp_lines líneas"
echo "  • tests/test_model_pool_skills.py: $test_pool_lines líneas"
echo "  • tests/test_mcp_skills.py: $test_mcp_lines líneas"
echo "  • Total tests: $total_lines líneas"
echo ""

# 8. Verificar PROGRESO_31102025.md actualizado
echo "📝 Verificando documentación..."
if grep -q "✅ COMPLETADO" PROGRESO_31102025.md && \
   grep -q "T1.1" PROGRESO_31102025.md && \
   grep -q "T1.2" PROGRESO_31102025.md; then
    echo "  ✅ PROGRESO_31102025.md actualizado"
else
    echo "  ❌ PROGRESO_31102025.md incompleto"
    exit 1
fi
echo ""

# 9. Resumen final
echo "═══════════════════════════════════════════"
echo "✅ VERIFICACIÓN COMPLETADA CON ÉXITO"
echo "═══════════════════════════════════════════"
echo ""
echo "Listo para commit con:"
echo ""
echo "  git add core/model_pool.py core/mcp.py config/sarai.yaml"
echo "  git add tests/test_model_pool_skills.py tests/test_mcp_skills.py pytest.ini"
echo "  git add PROGRESO_31102025.md SEMANA1_TICKETS.md STATUS_31102025.md"
echo "  git commit -m \"feat(v2.12): Implementar Skills MoE con LRU y routing dinámico\""
echo ""
echo "Progreso: 2/5 tickets (40%), 23/32 tests (72%), 970 LOC"
echo ""
