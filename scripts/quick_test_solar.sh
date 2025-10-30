#!/usr/bin/env bash
# Quick Test: Solo SOLAR-10.7B para validar setup
# Usage: ./scripts/quick_test_solar.sh

set -e

echo "🚀 SARAi v2.16 - Quick Test SOLAR-10.7B"
echo "========================================"
echo ""

# 1. Verificar Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama no encontrado"
    echo "   Instalar: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi
echo "✅ Ollama disponible"

# 2. Verificar modelo SOLAR (versión oficial optimizada)
echo "📦 Verificando modelo SOLAR-10.7B..."
if ! ollama list | grep -q "solar:10.7b"; then
    echo "⏳ Descargando SOLAR-10.7B-Instruct-v1.0 (versión oficial)..."
    ollama pull solar:10.7b
fi

ollama list | grep "solar:10.7b"

# 3. Test rápido
echo ""
echo "🧪 Test de inferencia SOLAR..."
echo "Prompt: '¿Qué es backpropagation en deep learning?'"
echo ""

time ollama run solar:10.7b "Explica backpropagation en deep learning en máximo 100 palabras. Incluye: función de pérdida, gradiente, regla de la cadena." --verbose

echo ""
echo "✅ Test completado"
echo ""
echo "📊 Si la respuesta incluye conceptos técnicos correctos,"
echo "   SOLAR está listo para producción."
echo ""
echo "Next steps:"
echo "  1. Pull LFM2: ollama pull hf.co/LiquidAI/LFM2-1.2B-GGUF:Q4_K_M"
echo "  2. Pull Qwen-Omni: ollama pull rockn/Qwen2.5-Omni-7B-Q4_K_M"
echo "  3. Pull Draft: ollama pull qwen2.5:0.5b"
echo "  4. Full test: make test-production"
