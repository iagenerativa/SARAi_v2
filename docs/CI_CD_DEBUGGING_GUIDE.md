# CI/CD Debugging Guide - SARAi Release Workflow

## 🎯 Propósito

El workflow de release (`release.yml`) ahora incluye **logging detallado con 16 milestones** para:
1. **Debug preciso**: Saber EXACTAMENTE en qué milestone falla
2. **No repetir trabajo**: Artifacts guardados permiten skip de etapas exitosas
3. **Trazabilidad completa**: Timeline con timestamps desde el inicio

---

## 📊 Los 16 Milestones

| Milestone | Nombre | Descripción | Artifact | Crítico |
|-----------|--------|-------------|----------|---------|
| **M0** | Init | Deployment initialization | No | ✅ |
| **M1** | Checkout | Clone del repositorio | No | ✅ |
| **M2** | Buildx | Docker Buildx setup | No | ✅ |
| **M3** | GHCR Login | Autenticación GHCR | No | ✅ |
| **M4** | Repo Extract | Nombre del repo (lowercase) | No | ✅ |
| **M5** | Docker Build | Build multi-arch + push | ✅ Sí (7d) | ✅ |
| **M6** | Cosign Install | Instalación de Cosign | No | ✅ |
| **M7** | Syft Install | Instalación de Syft | No | ✅ |
| **M8** | SBOM Generate | Generación de SBOM | ✅ Sí (90d) | ✅ |
| **M9** | Image Sign | Firma con Cosign OIDC | No | ✅ |
| **M10** | SBOM Attest | Attestation del SBOM | No | ✅ |
| **M11** | CHANGELOG | Extracción release notes | No | ✅ |
| **M12** | GitHub Release | Creación de release | No | ✅ |
| **M13** | Python Setup | Setup Python 3.11 | No | ⚠️ |
| **M14** | Grafana Publish | Dashboard Grafana | No | ⚠️ |
| **M15** | Signature Verify | Verificación de firma | No | ✅ |
| **M16** | Final Summary | Timeline completo | ✅ Sí (90d) | ✅ |

**Leyenda**:
- ✅ **Crítico**: Fallo bloquea el deployment
- ⚠️ **Opcional**: `continue-on-error: true`, no bloquea

---

## 🔍 Cómo Debuggear un Fallo

### Paso 1: Revisar GITHUB_STEP_SUMMARY

Cuando un workflow falla, GitHub Actions muestra un **summary** en la parte superior de los logs:

```
🚀 SARAi Deployment Log - v2.6.2

Started: 2025-10-28 02:00:00 UTC
Trigger: Tag push v2.6.2
Runner: ubuntu-latest

📋 Deployment Milestones

✅ M0: Deployment initialized at 02:00:00
✅ M1: Checkout - 02:00:05 UTC
✅ M2: Docker Buildx - 02:00:10 UTC
✅ M3: GHCR Login - 02:00:15 UTC
✅ M4: Repo Extract - 02:00:17 UTC
❌ M5: Docker Build - FAILED at 02:05:23 UTC  <-- AQUÍ FALLÓ
```

**Interpretación**:
- El fallo ocurrió en **M5: Docker Build**
- Tiempo transcurrido: ~5 minutos desde inicio
- Etapas M0-M4 completadas exitosamente

### Paso 2: Buscar el Error Específico

Expande el step "🏗️ MILESTONE 5: Build & Push Docker Image" en los logs de GitHub Actions.

**Errores comunes**:

#### Error 1: Dockerfile no encontrado
```
ERROR: failed to solve: failed to read dockerfile
```
**Solución**: Verificar que `Dockerfile` existe en la raíz del repo.

#### Error 2: Build multi-arch falla
```
ERROR: failed to solve: executor failed running [/bin/sh -c ...]
```
**Solución**: Revisar comandos en Dockerfile que pueden no funcionar en ARM64.

#### Error 3: Push a GHCR falla
```
ERROR: failed to push: unexpected status: 403 Forbidden
```
**Solución**: Verificar permisos `packages: write` en el workflow.

### Paso 3: Descargar Milestone Artifacts

Si el workflow llegó hasta M5 o M8, los **artifacts fueron guardados**:

1. Ve a la pestaña "Actions" → Selecciona el workflow fallido
2. Baja hasta "Artifacts"
3. Descarga:
   - `docker-build-milestone` (si M5 completó)
   - `sbom-milestone` (si M8 completó)
   - `deployment-milestones-v2.6.2` (siempre disponible)

**Contenido de `deployment-milestones-v2.6.2`**:
```
.milestones/
├── M0_init.txt         # deployment_started=1730077200
├── M1_checkout.txt     # checkout_complete=1730077205
├── M2_buildx.txt       # buildx_setup=1730077210
├── M3_ghcr.txt         # ghcr_login=1730077215
├── M4_repo.txt         # extract_repo=1730077217
└── M5_docker.txt       # docker_build=1730077523
                        # digest=sha256:abc123...
```

### Paso 4: Analizar Timeline

El artifact `deployment-milestones-v2.6.2` te permite:

1. **Calcular duración exacta** de cada milestone:
   ```bash
   M1_TIME=$(cat M1_checkout.txt | grep checkout_complete | cut -d'=' -f2)
   M0_TIME=$(cat M0_init.txt | grep deployment_started | cut -d'=' -f2)
   DURATION=$((M1_TIME - M0_TIME))
   echo "M1 tardó: ${DURATION}s"
   ```

2. **Identificar cuellos de botella**:
   - M5 (Docker Build) normalmente tarda 3-8 minutos
   - Si tarda >15 min → problema de build o red

3. **Comparar con runs anteriores**:
   - Descarga milestones de un run exitoso
   - Compara timestamps para ver qué cambió

---

## 🔄 Cómo Re-ejecutar Después de Fix

### Opción 1: Re-trigger con Nuevo Tag

Si corregiste el error (ej: Dockerfile):

```bash
# Commitear fix
git add Dockerfile
git commit -m "fix(docker): Corregir comando que falla en ARM64"
git push origin master

# Crear nuevo tag
git tag v2.6.3
git push origin v2.6.3
```

**Ventaja**: Workflow arranca desde M0 con el fix aplicado.

### Opción 2: Re-run del Mismo Tag (experimental)

Si el error fue temporal (ej: red):

```bash
# Borrar tag remoto
git push origin :refs/tags/v2.6.2

# Re-crear y pushear
git tag -f v2.6.2
git push origin v2.6.2
```

**Desventaja**: GitHub puede cachear el tag anterior.

### Opción 3: Re-run Manual en GitHub

1. Ve a Actions → Selecciona el workflow fallido
2. Click "Re-run failed jobs"

**Limitación**: No aplica fixes, solo reintenta.

---

## 📈 Interpretar el Final Summary

Cuando el workflow completa exitosamente, verás:

```markdown
## 🎉 Deployment Complete!

Total Duration: 12m 34s
Completed: 2025-10-28 02:12:34 UTC

### 📦 Published Artifacts

- Docker Image: ghcr.io/iagenerativa/sarai_v2:v2.6.2
- Digest: sha256:abc123def456...
- Platforms: linux/amd64, linux/arm64
- SBOM: Attached to release (SPDX + CycloneDX)
- Signature: Cosign keyless OIDC ✅
- Release: https://github.com/iagenerativa/SARAi_v2/releases/tag/v2.6.2

### 🏁 Milestone Timeline

- M0_init: +0s
- M1_checkout: +5s
- M2_buildx: +10s
- M3_ghcr: +15s
- M4_repo: +17s
- M5_docker: +323s  <-- Build tardó 5m 23s
- M6_cosign: +330s
- M7_syft: +335s
- M8_sbom: +365s
- M9_sign: +380s
- M10_attest: +395s
- M11_changelog: +400s
- M12_release: +420s
- M14_grafana: +450s (skipped si no hay secrets)
- M15_verify: +465s
- M16_summary: +470s
```

**Insights**:
- Total: 12m 34s (754s)
- M5 (Docker Build) fue el más largo: 5m 23s (normal para multi-arch)
- M8-M10 (SBOM + firma) tomó ~30s (eficiente)

---

## 🛠️ Troubleshooting Específico

### M5: Docker Build Falla

**Síntoma**: Build falla con error de plataforma ARM64.

**Diagnóstico**:
```bash
# Test local del build multi-arch
docker buildx build --platform linux/amd64,linux/arm64 -t sarai:test .
```

**Solución**: Añadir `--load` solo para amd64 en test local:
```bash
docker buildx build --platform linux/amd64 --load -t sarai:test .
docker run --rm sarai:test python -c "import sys; print(sys.version)"
```

### M8: SBOM Genera 0 Packages

**Síntoma**: `package_count=0` en `.milestones/M8_sbom.txt`.

**Diagnóstico**: Syft no detecta dependencias correctamente.

**Solución**:
1. Verificar que `requirements.txt` o `pyproject.toml` existe
2. Añadir metadata al Dockerfile:
   ```dockerfile
   LABEL org.opencontainers.image.source="https://github.com/iagenerativa/SARAi_v2"
   ```

### M9: Cosign Sign Falla con OIDC

**Síntoma**: `error: signing []: getting signer: ...OIDC...`

**Diagnóstico**: Permisos insuficientes.

**Solución**: Verificar en `release.yml`:
```yaml
permissions:
  id-token: write  # CRÍTICO para Cosign OIDC
```

### M14: Grafana Skipped

**Síntoma**: `grafana_skip=true` en milestone.

**Diagnóstico**: Secrets no configurados (esperado en repos públicos).

**Solución (opcional)**:
1. Crear Grafana Cloud account
2. Generar API key
3. Añadir secrets en GitHub:
   - Settings → Secrets → Actions
   - `GRAFANA_API_KEY`: tu API key
   - `GRAFANA_URL`: `https://your-org.grafana.net`

**No crítico**: El deployment completa sin Grafana.

---

## 📊 Metrics y Benchmarks

**Tiempos esperados** (runner ubuntu-latest):

| Milestone | Tiempo Esperado | Si Excede |
|-----------|-----------------|-----------|
| M0-M4 | <30s | Red lenta o runner ocupado |
| M5 (Docker Build) | 3-8 min | >15 min → revisar Dockerfile |
| M6-M7 (Installs) | <10s | >30s → cache de apt lento |
| M8 (SBOM) | 30-60s | >2 min → imagen muy grande |
| M9-M10 (Sign) | 10-20s | >1 min → OIDC lento |
| M11-M12 (Release) | 5-15s | >30s → muchos artifacts |
| M14 (Grafana) | 5-10s | Skipped si no hay secrets |
| M15 (Verify) | 5-10s | >30s → red lenta |

**Total esperado**: 8-15 minutos para deployment completo.

---

## 🎯 Checklist Pre-Deployment

Antes de hacer `git tag vX.Y.Z && git push origin vX.Y.Z`:

- [ ] `Dockerfile` existe y funciona localmente
- [ ] `docker build -t sarai:test .` pasa sin errores
- [ ] `CHANGELOG.md` tiene sección para `vX.Y.Z`
- [ ] Secrets configurados (opcional):
  - [ ] `GRAFANA_API_KEY`
  - [ ] `GRAFANA_URL`
- [ ] Permisos del workflow OK:
  - [ ] `contents: write`
  - [ ] `packages: write`
  - [ ] `id-token: write`
  - [ ] `attestations: write`

---

## � Errores de Infraestructura de GitHub Actions

### Error: "No space left on device"

**Síntoma**:
```
System.IO.IOException: No space left on device : '/home/runner/actions-runner/cached/_diag/Worker_*.log'
```

**Causa**: El runner de GitHub Actions asignado se quedó sin espacio en disco. **NO es culpa del código**.

**Diagnóstico**:
1. El error aparece en cualquier milestone (M0, M1, M5, etc.)
2. Ocurre ANTES de ejecutar tu código (durante init del runner)
3. Archivo afectado: `/home/runner/actions-runner/cached/_diag/...`

**Soluciones** (en orden de prioridad):

1. **Re-run del workflow** (GitHub asignará nuevo runner con espacio):
   - Ir a: `https://github.com/USER/REPO/actions`
   - Click en el workflow fallido
   - Botón: **"Re-run jobs"** (esquina superior derecha)
   - ✅ Esto resuelve el 95% de los casos

2. **Limpiar artifacts antiguos del repositorio**:
   ```bash
   # Manualmente en GitHub:
   # Settings → Actions → Artifacts
   # Borrar artifacts antiguos (>30 días)
   
   # O vía GitHub CLI:
   gh api repos/{owner}/{repo}/actions/artifacts --paginate \
     | jq -r '.artifacts[] | select(.expired == false) | .id' \
     | xargs -I {} gh api --method DELETE repos/{owner}/{repo}/actions/artifacts/{}
   ```

3. **Si persiste** (raro):
   - Esperar 30-60 minutos (GitHub limpia runners automáticamente)
   - Cambiar runner: `runs-on: ubuntu-22.04` → `runs-on: ubuntu-20.04`
   - Contactar GitHub Support si ocurre >3 veces consecutivas

**NO hagas**:
- ❌ Modificar el código (no es un bug tuyo)
- ❌ Reducir el tamaño del Dockerfile (el error ocurre antes del build)
- ❌ Cambiar `.milestones` logging (no afecta al espacio del runner)

**Casos reales**:
- Tag `v2.6.3` (28 Oct 2025): Error en init del runner
  - **Solución**: Re-run jobs
  - **Resultado**: Workflow completó exitosamente en segundo intento

---

## �📚 Referencias

- **Workflow File**: `.github/workflows/release.yml`
- **Milestones Dir**: `.milestones/` (en artifacts)
- **SBOM Format**: SPDX 2.3 + CycloneDX 1.4
- **Cosign Docs**: https://docs.sigstore.dev/cosign/overview/
- **Syft Docs**: https://github.com/anchore/syft
- **GitHub Actions Status**: https://www.githubstatus.com/

---

**Última actualización**: 28 de octubre de 2025  
**Versión del workflow**: v2.6.3 con 16 milestones + "No space" troubleshooting
