#!/usr/bin/env bash
# skills/protos/generate_pb.sh - Genera código Python desde skill.proto

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_FILE="$SCRIPT_DIR/skill.proto"
OUT_DIR="$SCRIPT_DIR"

echo "🔨 Generando código Python desde skill.proto..."

# Verificar que protoc esté instalado
if ! command -v protoc &> /dev/null; then
    echo "❌ Error: protoc no encontrado"
    echo "Instalar con:"
    echo "  sudo apt-get install -y protobuf-compiler"
    echo "  o"
    echo "  brew install protobuf"
    exit 1
fi

# Verificar versión de protoc (debe ser >=3.19)
PROTOC_VERSION=$(protoc --version | awk '{print $2}')
echo "protoc versión: $PROTOC_VERSION"

# Generar código Python
python -m grpc_tools.protoc \
    -I"$SCRIPT_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_FILE"

# Verificar archivos generados
if [ -f "$OUT_DIR/skill_pb2.py" ] && [ -f "$OUT_DIR/skill_pb2_grpc.py" ]; then
    echo "✅ Archivos generados:"
    echo "   - skill_pb2.py ($(wc -l < "$OUT_DIR/skill_pb2.py") líneas)"
    echo "   - skill_pb2_grpc.py ($(wc -l < "$OUT_DIR/skill_pb2_grpc.py") líneas)"
else
    echo "❌ Error: No se generaron los archivos"
    exit 1
fi

# Crear __init__.py si no existe
if [ ! -f "$OUT_DIR/__init__.py" ]; then
    cat > "$OUT_DIR/__init__.py" <<'EOF'
"""
skills/protos - Protocolo gRPC para Skills-as-Services v2.12

Generado automáticamente desde skill.proto
"""

from .skill_pb2 import (
    ExecuteRequest,
    ExecuteResponse,
    HealthRequest,
    HealthResponse,
    StatsRequest,
    StatsResponse,
)

from .skill_pb2_grpc import (
    SkillServiceStub,
    SkillServiceServicer,
    add_SkillServiceServicer_to_server,
)

__all__ = [
    "ExecuteRequest",
    "ExecuteResponse",
    "HealthRequest",
    "HealthResponse",
    "StatsRequest",
    "StatsResponse",
    "SkillServiceStub",
    "SkillServiceServicer",
    "add_SkillServiceServicer_to_server",
]
EOF
    echo "✅ __init__.py creado"
fi

echo ""
echo "Para usar en código:"
echo "  from skills.protos import ExecuteRequest, SkillServiceStub"
echo "  stub = SkillServiceStub(grpc.insecure_channel('localhost:50051'))"
echo "  response = stub.Execute(ExecuteRequest(query='SELECT 1'))"
