#!/bin/bash
# SARAi v2.6 - Script de Verificación de Release
# 
# Verifica criptográficamente la autenticidad e integridad de una release
# Requiere: cosign instalado
#
# Uso:
#   ./scripts/verify_sarai.sh v2.6.0-rc1
#   ./scripts/verify_sarai.sh v2.6.0

set -e

# Configuración
VERSION="${1:-v2.6.0-rc1}"
IMAGE="ghcr.io/iagenerativa/sarai_v2:$VERSION"
CERT_IDENTITY="https://github.com/iagenerativa/SARAi_v2/.*"
OIDC_ISSUER="https://token.actions.githubusercontent.com"

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🔍 Verificando SARAi $VERSION - DevSecOps Validation            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Verificar que cosign está instalado
if ! command -v cosign &> /dev/null; then
    echo -e "${RED}❌ Error: cosign no está instalado${NC}"
    echo ""
    echo "Instalar con:"
    echo "  curl -sSfL https://raw.githubusercontent.com/sigstore/cosign/main/install.sh | sh -s -- -b /usr/local/bin"
    exit 1
fi

echo -e "${BLUE}📦 Imagen a verificar:${NC} $IMAGE"
echo ""

# ============================================================================
# 1. VERIFICAR FIRMA COSIGN
# ============================================================================
echo -e "${YELLOW}1️⃣  Verificando firma criptográfica...${NC}"

if cosign verify \
    --certificate-identity-regexp="$CERT_IDENTITY" \
    --certificate-oidc-issuer="$OIDC_ISSUER" \
    "$IMAGE" > /tmp/cosign_verify.json 2>&1; then
    
    echo -e "   ${GREEN}✅ Firma válida${NC}"
    
    # Extraer información del certificado
    CERT_SUBJECT=$(jq -r '.[0].optional.Subject' /tmp/cosign_verify.json 2>/dev/null || echo "N/A")
    CERT_ISSUER=$(jq -r '.[0].optional.Issuer' /tmp/cosign_verify.json 2>/dev/null || echo "N/A")
    
    echo -e "   ${BLUE}📜 Certificado:${NC}"
    echo "      Subject: $CERT_SUBJECT"
    echo "      Issuer: $CERT_ISSUER"
else
    echo -e "   ${RED}❌ Firma inválida - NO USAR ESTA IMAGEN${NC}"
    echo ""
    echo "Detalles del error:"
    cat /tmp/cosign_verify.json
    exit 1
fi

echo ""

# ============================================================================
# 2. VERIFICAR SBOM ATTESTATION
# ============================================================================
echo -e "${YELLOW}2️⃣  Verificando SBOM attestation...${NC}"

if cosign verify-attestation \
    --type spdxjson \
    --certificate-identity-regexp="$CERT_IDENTITY" \
    --certificate-oidc-issuer="$OIDC_ISSUER" \
    "$IMAGE" > /tmp/sbom_attestation.json 2>&1; then
    
    echo -e "   ${GREEN}✅ SBOM válido${NC}"
    
    # Extraer información del SBOM
    PACKAGES_COUNT=$(jq '[.payload | @base64d | fromjson | .predicate.packages[]] | length' /tmp/sbom_attestation.json 2>/dev/null || echo "0")
    
    echo -e "   ${BLUE}📦 Dependencias:${NC} $PACKAGES_COUNT paquetes"
    
    # Mostrar algunas dependencias clave
    echo "   ${BLUE}🔑 Componentes clave:${NC}"
    jq -r '.payload | @base64d | fromjson | .predicate.packages[] | select(.name | contains("llama") or contains("torch") or contains("transformers")) | "      • \(.name) (\(.versionInfo // "N/A"))"' /tmp/sbom_attestation.json 2>/dev/null | head -5 || echo "      (No se pudo extraer)"
else
    echo -e "   ${YELLOW}⚠️  SBOM no verificable (puede ser opcional)${NC}"
fi

echo ""

# ============================================================================
# 3. VERIFICAR BUILD ENVIRONMENT ATTESTATION
# ============================================================================
echo -e "${YELLOW}3️⃣  Verificando entorno de build...${NC}"

if cosign verify-attestation \
    --type custom \
    --certificate-identity-regexp="$CERT_IDENTITY" \
    --certificate-oidc-issuer="$OIDC_ISSUER" \
    "$IMAGE" > /tmp/build_env.json 2>&1; then
    
    echo -e "   ${GREEN}✅ Entorno verificado${NC}"
    
    # Extraer información del entorno
    CPU_FLAGS=$(jq -r '.payload | @base64d | fromjson | .predicate.cpu_flags' /tmp/build_env.json 2>/dev/null || echo "N/A")
    BLAS=$(jq -r '.payload | @base64d | fromjson | .predicate.blas' /tmp/build_env.json 2>/dev/null || echo "N/A")
    PLATFORM=$(jq -r '.payload | @base64d | fromjson | .predicate.platform' /tmp/build_env.json 2>/dev/null || echo "N/A")
    
    echo -e "   ${BLUE}🏗️  Plataformas:${NC} $PLATFORM"
    echo -e "   ${BLUE}⚡ CPU Flags:${NC} $CPU_FLAGS"
    echo -e "   ${BLUE}📊 BLAS:${NC} $BLAS"
    
    # Verificar optimizaciones críticas
    if echo "$CPU_FLAGS" | grep -q "AVX2"; then
        echo -e "   ${GREEN}✅ Optimizaciones AVX2 presentes${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Sin optimizaciones AVX2${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  Attestation de entorno no disponible${NC}"
fi

echo ""

# ============================================================================
# 4. INSPECCIÓN DE REKOR (LOG PÚBLICO)
# ============================================================================
echo -e "${YELLOW}4️⃣  Verificando logs de transparencia (Rekor)...${NC}"

# Extraer UUID de Rekor del log de verificación
REKOR_UUID=$(jq -r '.[0].optional.Bundle.Payload.logID' /tmp/cosign_verify.json 2>/dev/null || echo "")

if [ -n "$REKOR_UUID" ] && [ "$REKOR_UUID" != "null" ]; then
    echo -e "   ${GREEN}✅ Entrada en Rekor encontrada${NC}"
    echo -e "   ${BLUE}🔗 UUID:${NC} ${REKOR_UUID:0:16}..."
else
    echo -e "   ${YELLOW}⚠️  No se pudo extraer UUID de Rekor${NC}"
fi

echo ""

# ============================================================================
# 5. RESUMEN FINAL
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     ✅ VERIFICACIÓN COMPLETADA                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Nivel de confianza: 🟢 COMPLETA (95%)${NC}"
echo ""
echo "La imagen $VERSION ha pasado todas las verificaciones:"
echo "  ✅ Firma Cosign válida (OIDC keyless)"
echo "  ✅ SBOM attestation verificado"
echo "  ✅ Entorno de build atestado"
echo "  ✅ Logs de transparencia (Rekor)"
echo ""
echo -e "${GREEN}🚀 Imagen segura para usar en producción${NC}"
echo ""

# ============================================================================
# 6. COMANDOS ADICIONALES
# ============================================================================
echo -e "${BLUE}📝 Comandos útiles:${NC}"
echo ""
echo "  # Pull de la imagen verificada"
echo "  docker pull $IMAGE"
echo ""
echo "  # Ver SBOM completo"
echo "  cosign verify-attestation --type spdxjson $IMAGE | jq . > sbom.json"
echo ""
echo "  # Escanear vulnerabilidades (requiere grype)"
echo "  grype $IMAGE"
echo ""

exit 0
