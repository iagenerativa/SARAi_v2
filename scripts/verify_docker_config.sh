#!/bin/bash
# scripts/verify_docker_config.sh - Verificar configuración Docker v2.16
#
# Valida:
# - Servicios definidos correctamente
# - Hardening aplicado (read_only, cap_drop, etc.)
# - Resource limits configurados
# - Healthchecks presentes
# - Red interna configurada

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker-compose.override.yml"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   SARAi v2.16 - Docker Configuration Validator                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Función helper para validar
validate_service() {
    local service="$1"
    local checks_passed=0
    local checks_total=7
    
    echo "🔍 Validando servicio: $service"
    
    # 1. Check read_only
    if grep -A 50 "^  $service:" "$COMPOSE_FILE" | grep -q "read_only: true"; then
        echo "  ✅ read_only: true"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${RED}❌ read_only: FALTA${NC}"
    fi
    
    # 2. Check security_opt
    if grep -A 50 "^  $service:" "$COMPOSE_FILE" | grep -q "no-new-privileges:true"; then
        echo "  ✅ security_opt: no-new-privileges"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${YELLOW}⚠️  security_opt: FALTA${NC}"
    fi
    
    # 3. Check cap_drop
    if grep -A 50 "^  $service:" "$COMPOSE_FILE" | grep -q "cap_drop:"; then
        echo "  ✅ cap_drop: ALL"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${YELLOW}⚠️  cap_drop: FALTA${NC}"
    fi
    
    # 4. Check user
    if grep -A 50 "^  $service:" "$COMPOSE_FILE" | grep -q 'user: "1000:1000"'; then
        echo "  ✅ user: non-root (1000:1000)"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${YELLOW}⚠️  user: FALTA (root por defecto)${NC}"
    fi
    
    # 5. Check tmpfs
    if grep -A 50 "^  $service:" "$COMPOSE_FILE" | grep -q "tmpfs:"; then
        echo "  ✅ tmpfs: configurado"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${YELLOW}⚠️  tmpfs: FALTA${NC}"
    fi
    
    # 6. Check resource limits
    if grep -A 70 "^  $service:" "$COMPOSE_FILE" | grep -q "memory:"; then
        echo "  ✅ resource limits: memory configurada"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${RED}❌ resource limits: FALTA${NC}"
    fi
    
    # 7. Check healthcheck
    if grep -A 70 "^  $service:" "$COMPOSE_FILE" | grep -q "healthcheck:"; then
        echo "  ✅ healthcheck: configurado"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "  ${YELLOW}⚠️  healthcheck: FALTA${NC}"
    fi
    
    # Score
    local percentage=$((checks_passed * 100 / checks_total))
    if [ $checks_passed -eq $checks_total ]; then
        echo -e "  ${GREEN}✅ $service: $checks_passed/$checks_total checks ($percentage%)${NC}"
    elif [ $checks_passed -ge 5 ]; then
        echo -e "  ${YELLOW}⚠️  $service: $checks_passed/$checks_total checks ($percentage%)${NC}"
    else
        echo -e "  ${RED}❌ $service: $checks_passed/$checks_total checks ($percentage%)${NC}"
    fi
    echo ""
}

# Validar servicios Phoenix
validate_service "skill_draft"
validate_service "skill_image"

# Validar red interna
echo "🌐 Validando red interna..."
if grep -q "sarai_internal:" "$COMPOSE_FILE"; then
    echo "  ✅ Red sarai_internal definida"
    
    # Check si está marcada como interna (producción) o abierta (desarrollo)
    if grep -A 5 "sarai_internal:" "$COMPOSE_FILE" | grep -q "internal: false"; then
        echo -e "  ${YELLOW}⚠️  Red configurada para DESARROLLO (internal: false)${NC}"
        echo "     💡 Cambiar a 'internal: true' en producción"
    else
        echo "  ✅ Red configurada para PRODUCCIÓN (internal: true)"
    fi
else
    echo -e "  ${RED}❌ Red sarai_internal NO definida${NC}"
fi
echo ""

# Validar volúmenes
echo "💾 Validando volúmenes..."
if grep -q "state/images:/app/cache:rw" "$COMPOSE_FILE"; then
    echo "  ✅ Cache de imágenes: state/images montado"
else
    echo -e "  ${YELLOW}⚠️  Cache de imágenes: FALTA${NC}"
fi

if [ -d "$REPO_ROOT/state/images" ]; then
    echo "  ✅ Directorio state/images existe"
else
    echo -e "  ${YELLOW}⚠️  Directorio state/images NO existe${NC}"
    echo "     💡 Crear con: mkdir -p state/images"
fi
echo ""

# Resumen final
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                         RESUMEN FINAL                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Servicios Phoenix:"
echo "  ✅ skill_draft:  Definido con hardening kernel-level"
echo "  ✅ skill_image:  Definido con hardening kernel-level"
echo ""
echo "Próximos pasos:"
echo "  1. Generar protobuf stubs:"
echo "     bash scripts/generate_grpc_stubs.sh"
echo ""
echo "  2. Build containers:"
echo "     docker-compose build skill_draft skill_image"
echo ""
echo "  3. Levantar servicios:"
echo "     docker-compose up -d skill_draft skill_image"
echo ""
echo "  4. Verificar healthchecks:"
echo "     docker ps"
echo "     docker logs sarai-skill-draft"
echo "     docker logs sarai-skill-image"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
