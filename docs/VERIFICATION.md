# 🔐 Verificación de Releases Firmados

SARAi v2.6+ incluye firma criptográfica y SBOM (Software Bill of Materials) para cada release. Este documento explica cómo verificar la autenticidad e integridad de las releases.

## 🎯 ¿Por qué verificar?

La verificación garantiza que:
- ✅ La imagen Docker proviene del repositorio oficial
- ✅ No ha sido modificada por un atacante
- ✅ Contiene exactamente las dependencias documentadas
- ✅ Fue construida en un entorno conocido (GitHub Actions)

**Sin verificación**, cualquier release es un acto de fe.

## 📋 Prerequisitos

```bash
# Instalar Cosign (herramienta de firma de Sigstore)
curl -sSfL https://raw.githubusercontent.com/sigstore/cosign/main/install.sh | sh -s -- -b /usr/local/bin

# Verificar instalación
cosign version
```

## 🔍 Verificación Básica

### 1. Verificar Firma de la Imagen

```bash
# Verificar que la imagen fue construida por el repositorio oficial
cosign verify \
  --certificate-identity-regexp="https://github.com/iagenerativa/SARAi_v2/.*" \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  ghcr.io/iagenerativa/sarai_v2:v2.6.0

# Salida esperada:
# ✅ Verified OK
# Certificate subject: https://github.com/iagenerativa/SARAi_v2/.github/workflows/release.yml@refs/tags/v2.6.0
```

**Si la verificación falla** → Imagen comprometida, NO ejecutar.

### 2. Verificar SBOM (Dependencias)

```bash
# Obtener y verificar el SBOM atestado
cosign verify-attestation \
  --type spdxjson \
  --certificate-identity-regexp="https://github.com/iagenerativa/SARAi_v2/.*" \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  ghcr.io/iagenerativa/sarai_v2:v2.6.0 | jq . > sbom.json

# Ver componentes principales
jq '.payload | @base64d | fromjson | .predicate.packages[] | select(.name | contains("llama")) | {name, version}' sbom.json
```

**Ejemplo de salida**:
```json
{
  "name": "llama-cpp-python",
  "version": "0.2.90"
}
```

### 3. Verificar Entorno de Build

```bash
# Verificar que la imagen fue construida con las optimizaciones correctas
cosign verify-attestation \
  --type custom \
  --certificate-identity-regexp="https://github.com/iagenerativa/SARAi_v2/.*" \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  ghcr.io/iagenerativa/sarai_v2:v2.6.0 | \
  jq '.payload | @base64d | fromjson | .predicate'

# Salida esperada:
# {
#   "platform": "linux/amd64,linux/arm64",
#   "cpu_flags": "-DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_BLAS=ON",
#   "blas": "OpenBLAS",
#   "builder": "GitHub Actions",
#   "timestamp": "2025-10-27T..."
# }
```

## 🚀 Verificación Completa (Script)

Crea un script `verify_sarai.sh` para automatizar todas las verificaciones:

```bash
#!/bin/bash
set -e

IMAGE="ghcr.io/iagenerativa/sarai_v2:v2.6.0"
CERT_IDENTITY="https://github.com/iagenerativa/SARAi_v2/.*"
OIDC_ISSUER="https://token.actions.githubusercontent.com"

echo "🔍 Verificando SARAi v2.6.0..."
echo ""

# 1. Verificar firma
echo "1️⃣ Verificando firma criptográfica..."
cosign verify \
  --certificate-identity-regexp="$CERT_IDENTITY" \
  --certificate-oidc-issuer="$OIDC_ISSUER" \
  "$IMAGE" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "   ✅ Firma válida"
else
    echo "   ❌ Firma inválida - NO USAR"
    exit 1
fi

# 2. Verificar SBOM
echo "2️⃣ Verificando SBOM..."
cosign verify-attestation \
  --type spdxjson \
  --certificate-identity-regexp="$CERT_IDENTITY" \
  --certificate-oidc-issuer="$OIDC_ISSUER" \
  "$IMAGE" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "   ✅ SBOM válido"
else
    echo "   ⚠️ SBOM no verificable"
fi

# 3. Verificar entorno
echo "3️⃣ Verificando entorno de build..."
BUILD_ENV=$(cosign verify-attestation \
  --type custom \
  --certificate-identity-regexp="$CERT_IDENTITY" \
  --certificate-oidc-issuer="$OIDC_ISSUER" \
  "$IMAGE" 2>/dev/null | jq -r '.payload | @base64d | fromjson | .predicate.cpu_flags')

if echo "$BUILD_ENV" | grep -q "AVX2"; then
    echo "   ✅ Optimizaciones AVX2 presentes"
else
    echo "   ⚠️ Sin optimizaciones AVX2"
fi

echo ""
echo "✅ Todas las verificaciones pasaron"
echo "🚀 Imagen segura para usar"
```

Ejecutar:
```bash
chmod +x verify_sarai.sh
./verify_sarai.sh
```

## 📊 Inspección del SBOM

### Ver todas las dependencias

```bash
# Descargar SBOM desde GitHub Release
wget https://github.com/iagenerativa/SARAi_v2/releases/download/v2.6.0/sbom.spdx.json

# Listar todas las dependencias
jq '.packages[] | {name, version, license}' sbom.spdx.json | less
```

### Buscar vulnerabilidades conocidas

```bash
# Instalar Grype (escáner de vulnerabilidades)
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

# Escanear SBOM
grype sbom:sbom.spdx.json

# Escanear directamente la imagen (alternativa)
grype ghcr.io/iagenerativa/sarai_v2:v2.6.0
```

## 🔐 Verificación Offline

Si no tienes conexión a internet, puedes verificar usando el SBOM descargado:

```bash
# 1. Descargar SBOM del release
wget https://github.com/iagenerativa/SARAi_v2/releases/download/v2.6.0/sbom.spdx.json

# 2. Verificar integridad con SHA-256 (publicado en el release)
sha256sum sbom.spdx.json
# Comparar con el hash en GitHub Release notes

# 3. Inspeccionar dependencias críticas
jq '.packages[] | select(.name | contains("llama") or contains("torch"))' sbom.spdx.json
```

## 📋 Checklist de Verificación

Antes de usar una release de SARAi en producción:

- [ ] Verificar firma Cosign (MUST)
- [ ] Verificar SBOM attestation (SHOULD)
- [ ] Revisar entorno de build (NICE TO HAVE)
- [ ] Escanear vulnerabilidades con Grype (RECOMMENDED)
- [ ] Comprobar que la versión de tag coincide con la documentación
- [ ] Revisar el changelog en GitHub Release

## ⚠️ Qué hacer si la verificación falla

**Si `cosign verify` falla**:
1. ❌ NO usar la imagen
2. 🔍 Verificar que usaste el comando correcto (typos en la URL)
3. 📧 Reportar en GitHub Issues con el output completo
4. 🔄 Intentar con una versión anterior conocida-buena

**Si `verify-attestation` falla pero `verify` pasa**:
- ⚠️ La imagen está firmada pero el SBOM podría estar corrupto
- Proceder con precaución
- Revisar manualmente el SBOM descargado del release

## 🔗 Referencias

- **Sigstore Cosign**: https://docs.sigstore.dev/cosign/overview/
- **SBOM (SPDX)**: https://spdx.dev/
- **SBOM (CycloneDX)**: https://cyclonedx.org/
- **Grype Scanner**: https://github.com/anchore/grype
- **Syft SBOM Generator**: https://github.com/anchore/syft

## 📝 Notas de Implementación

### ¿Por qué keyless OIDC?

SARAi usa **Cosign keyless** (OIDC) en lugar de claves privadas porque:
- ✅ No hay secretos que rotar o filtrar
- ✅ La firma está atada al workflow de GitHub Actions
- ✅ Transparencia total (logs públicos en Rekor)
- ✅ Más fácil de auditar

### ¿Qué se firma exactamente?

1. **Imagen Docker**: El digest SHA-256 completo
2. **SBOM**: Attestation SPDX JSON firmado
3. **Build Environment**: Metadata del entorno de construcción

### Niveles de confianza

| Nivel | Verificación | Confianza | Uso |
|-------|--------------|-----------|-----|
| 🔴 Ninguna | No verificado | 0% | ❌ NUNCA |
| 🟡 Básica | Solo `cosign verify` | 80% | Desarrollo local |
| 🟢 Completa | Firma + SBOM + Entorno | 95% | Producción |
| 🔵 Paranoid | Completa + Grype scan | 99% | Infraestructura crítica |

## 🎓 Tutorial Paso a Paso

### Escenario 1: Desplegar en producción

```bash
# 1. Verificar release
./verify_sarai.sh

# 2. Pull de la imagen verificada
docker pull ghcr.io/iagenerativa/sarai_v2:v2.6.0

# 3. Ejecutar con docker-compose
docker-compose up -d

# 4. Verificar health
curl http://localhost:8080/health
```

### Escenario 2: Auditoría de seguridad

```bash
# 1. Descargar todos los artifacts del release
wget https://github.com/iagenerativa/SARAi_v2/releases/download/v2.6.0/sbom.spdx.json
wget https://github.com/iagenerativa/SARAi_v2/releases/download/v2.6.0/sbom.cyclonedx.json
wget https://github.com/iagenerativa/SARAi_v2/releases/download/v2.6.0/build_env.json

# 2. Escanear vulnerabilidades
grype sbom:sbom.spdx.json > vulnerabilities.txt

# 3. Revisar dependencias críticas
jq '.packages[] | select(.licenses[] | contains("GPL"))' sbom.spdx.json

# 4. Verificar build environment
jq . build_env.json
```

### Escenario 3: Build reproducible

```bash
# 1. Clonar el repo en el mismo commit
git clone https://github.com/iagenerativa/SARAi_v2.git
cd SARAi_v2
git checkout v2.6.0

# 2. Build local
docker build -t sarai_local:v2.6.0 .

# 3. Comparar layers (no será idéntico byte-a-byte pero estructura similar)
dive sarai_local:v2.6.0
dive ghcr.io/iagenerativa/sarai_v2:v2.6.0
```

---

**Mantra de Verificación v2.6**: 
> _"Confía, pero verifica. Y luego verifica de nuevo con Cosign."_
