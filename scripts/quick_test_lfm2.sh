#!/bin/bash
# Quick test para LFM2-1.2B (Tiny Tier - Soft Skills)
# v2.16 Omni-Loop Testing

set -e

echo "================================"
echo "LFM2-1.2B QUICK TEST"
echo "================================"
echo ""

# 1. Verificar Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Error: Ollama no está instalado"
    exit 1
fi

echo "✅ Ollama encontrado: $(which ollama)"
echo ""

# 2. Verificar modelo LFM2
echo "📦 Verificando modelo LFM2-1.2B..."
if ! ollama list | grep -q "LFM2"; then
    echo "⏳ Descargando LFM2-1.2B-GGUF:Q4_K_M..."
    ollama pull hf.co/LiquidAI/LFM2-1.2B-GGUF:Q4_K_M
fi

ollama list | grep LFM2
echo ""

# 3. Test de inferencia con SOFT SKILLS prompt
echo "🧠 Test de inferencia (Soft Skills - Empathy)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PROMPT="Estoy muy frustrado porque mi código no funciona y llevo 3 horas intentándolo. No sé qué más hacer."

echo "📝 Prompt (empathy test):"
echo "\"$PROMPT\""
echo ""
echo "💬 Respuesta LFM2:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Medir tiempo de ejecución
time ollama run hf.co/LiquidAI/LFM2-1.2B-GGUF:Q4_K_M "$PROMPT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 4. Validación de respuesta empática
echo "🔍 Validación de soft skills..."
RESPONSE=$(ollama run hf.co/LiquidAI/LFM2-1.2B-GGUF:Q4_K_M "$PROMPT" 2>/dev/null)

# Palabras clave de empatía esperadas
EMPATHY_KEYWORDS=("entiendo" "frustración" "ayudar" "paso a paso" "respirar" "pausa")
FOUND=0

for keyword in "${EMPATHY_KEYWORDS[@]}"; do
    if echo "$RESPONSE" | grep -iq "$keyword"; then
        echo "  ✅ Keyword encontrado: '$keyword'"
        FOUND=$((FOUND + 1))
    fi
done

echo ""
if [ $FOUND -ge 2 ]; then
    echo "✅ LFM2 muestra empatía (≥2 keywords empáticos)"
else
    echo "⚠️  Empatía baja ($FOUND/6 keywords)"
fi

echo ""
echo "================================"
echo "NEXT STEPS:"
echo "================================"
echo "1. Esperar Qwen-Omni-7B download"
echo "2. Run: make test-production"
echo "3. Analizar stability report"
echo ""
