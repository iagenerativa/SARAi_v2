#!/bin/bash
# Script de instalación para SARAi v2 en sistema sin GPU

set -e

echo "🚀 Instalando SARAi v2..."
echo "================================"

# Verificar Python 3.10+
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if (( $(echo "$python_version < 3.10" | bc -l) )); then
    echo "❌ Error: Se requiere Python 3.10 o superior"
    echo "   Versión actual: $python_version"
    exit 1
fi
echo "✅ Python $python_version detectado"

# Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno
source venv/bin/activate

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install -r requirements.txt

# Crear directorios necesarios
echo "📁 Creando directorios..."
mkdir -p models/cache/{solar,lfm2,embeddings,qwen}
mkdir -p models/trm_classifier
mkdir -p models/mcp
mkdir -p logs
mkdir -p data

# Generar dataset inicial (opcional)
read -p "¿Generar dataset sintético para TRM? (s/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo "📝 Generando dataset..."
    python scripts/generate_synthetic_data.py --samples 1000 --output data/trm_training.json
fi

echo ""
echo "================================"
echo "✅ Instalación completada"
echo ""
echo "📖 Próximos pasos:"
echo "   1. Activa el entorno: source venv/bin/activate"
echo "   2. Ejecuta SARAi: python main.py"
echo ""
echo "💡 Comandos útiles:"
echo "   - Ver estadísticas: python main.py --stats"
echo "   - Entrenar TRM: python scripts/train_trm.py --data data/trm_training.json"
echo ""
echo "⚠️  Nota: Los modelos se descargarán automáticamente en el primer uso (~25GB)"
echo "================================"
