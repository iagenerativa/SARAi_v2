#!/usr/bin/env bash
# skills/generate_grpc_stubs.sh - Genera código Python desde skills.proto
# ═══════════════════════════════════════════════════════════════════════════
# REQUISITOS:
#   - protoc (protobuf compiler) ≥3.19
#   - grpcio-tools (pip install grpcio-tools)
# ═══════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_FILE="$SCRIPT_DIR/skills.proto"
OUT_DIR="$SCRIPT_DIR"

echo "🔨 Generando stubs gRPC desde skills.proto..."

# ──────────────────────────────────────────────────────────────────────────
# 1. Verificar protoc
# ──────────────────────────────────────────────────────────────────────────
if ! command -v protoc &> /dev/null; then
    echo "❌ Error: protoc no encontrado"
    echo ""
    echo "Instalar con:"
    echo "  Ubuntu/Debian: sudo apt-get install -y protobuf-compiler"
    echo "  macOS:         brew install protobuf"
    echo "  Fedora:        sudo dnf install protobuf-compiler"
    exit 1
fi

PROTOC_VERSION=$(protoc --version | awk '{print $2}')
echo "✓ protoc versión: $PROTOC_VERSION"

# ──────────────────────────────────────────────────────────────────────────
# 2. Verificar grpcio-tools
# ──────────────────────────────────────────────────────────────────────────
if ! python3 -c "import grpc_tools" &> /dev/null; then
    echo "❌ Error: grpcio-tools no instalado"
    echo ""
    echo "Instalar con:"
    echo "  pip install grpcio-tools"
    exit 1
fi

echo "✓ grpcio-tools instalado"

# ──────────────────────────────────────────────────────────────────────────
# 3. Generar código Python
# ──────────────────────────────────────────────────────────────────────────
python3 -m grpc_tools.protoc \
    --proto_path="$SCRIPT_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_FILE"

# ──────────────────────────────────────────────────────────────────────────
# 4. Verificar archivos generados
# ──────────────────────────────────────────────────────────────────────────
EXPECTED_FILES=(
    "$OUT_DIR/skills_pb2.py"
    "$OUT_DIR/skills_pb2_grpc.py"
)

for file in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Archivo no generado: $(basename "$file")"
        exit 1
    fi
done

echo "✅ Stubs generados exitosamente:"
echo "   - skills_pb2.py      ($(wc -l < "$OUT_DIR/skills_pb2.py") líneas)"
echo "   - skills_pb2_grpc.py ($(wc -l < "$OUT_DIR/skills_pb2_grpc.py") líneas)"

# ──────────────────────────────────────────────────────────────────────────
# 5. Crear __init__.py para importaciones limpias
# ──────────────────────────────────────────────────────────────────────────
cat > "$OUT_DIR/__init__.py" <<'EOF'
"""
skills - Skills-as-Services v2.12 Phoenix

Stubs gRPC generados automáticamente desde skills.proto
"""

# Importar mensajes (request/response)
from .skills_pb2 import (
    InferRequest,
    InferResponse,
    HealthCheckRequest,
    HealthCheckResponse,
    MetricsRequest,
    MetricsResponse,
)

# Importar servicios (stub client + servicer server)
from .skills_pb2_grpc import (
    SkillServiceStub,
    SkillServiceServicer,
    add_SkillServiceServicer_to_server,
)

__all__ = [
    # Mensajes
    "InferRequest",
    "InferResponse",
    "HealthCheckRequest",
    "HealthCheckResponse",
    "MetricsRequest",
    "MetricsResponse",
    # Servicios
    "SkillServiceStub",
    "SkillServiceServicer",
    "add_SkillServiceServicer_to_server",
]

__version__ = "2.12.0"
EOF

echo "✅ __init__.py creado"

# ──────────────────────────────────────────────────────────────────────────
# 6. Ejemplo de uso
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ Stubs listos para usar"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "EJEMPLO (Cliente):"
echo "  from skills import SkillServiceStub, InferRequest"
echo "  import grpc"
echo ""
echo "  channel = grpc.insecure_channel('localhost:50051')"
echo "  stub = SkillServiceStub(channel)"
echo "  response = stub.Infer(InferRequest(prompt='SELECT 1', max_tokens=64))"
echo "  print(response.text)"
echo ""
echo "EJEMPLO (Servidor):"
echo "  from skills import SkillServiceServicer, add_SkillServiceServicer_to_server"
echo "  import grpc"
echo ""
echo "  class MySkill(SkillServiceServicer):"
echo "      def Infer(self, request, context):"
echo "          return InferResponse(text='...')"
echo ""
echo "  server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))"
echo "  add_SkillServiceServicer_to_server(MySkill(), server)"
echo "  server.add_insecure_port('0.0.0.0:50051')"
echo "  server.start()"
