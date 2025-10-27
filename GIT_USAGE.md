# SARAi v2 - Guía de Uso del Repositorio Git

**Repositorio**: SARAi_v2  
**Estado actual**: v2.10-m2-closed ✅  
**Fecha**: 2025-10-27

---

## 📦 Estado del Repositorio

```bash
# Ver commit actual
git log --oneline --decorate -1

# Salida esperada:
# e668322 (HEAD -> master, tag: v2.10-m2-closed) 🎉 M2 COMPLETADO...
```

**Archivos trackeados**: 85  
**Último commit**: M2 COMPLETADO (2025-10-27)  
**Tag actual**: v2.10-m2-closed

---

## 🏷️ Tags y Versionado

### Convención de Tags

```
v{MAJOR}.{MINOR}-{MILESTONE}-{STATUS}

Ejemplos:
  v2.10-m2-closed      → Milestone 2 completado
  v2.11-m3-wip         → Milestone 3 en progreso
  v3.0-stable          → Release estable v3.0
```

### Ver Tags

```bash
# Listar todos los tags
git tag -l

# Ver detalles de un tag
git show v2.10-m2-closed

# Ver tags con anotaciones
git tag -l -n9
```

---

## 📝 Workflow de Desarrollo

### 1. Trabajo en Nueva Feature

```bash
# Crear rama para feature
git checkout -b feature/m2.6-devsecops

# Hacer cambios...
git add scripts/publish_grafana.py
git commit -m "feat(m2.6): Añadir script de publicación Grafana

- Script para publicar dashboard a Grafana Cloud
- Autenticación vía API key
- Dashboard ID: 21902
"

# Volver a master
git checkout master

# Merge (fast-forward si es posible)
git merge feature/m2.6-devsecops
```

### 2. Cerrar un Milestone

```bash
# Consolidar cambios
git add -A
git commit -m "🎉 M2.6 COMPLETADO: DevSecOps + Release Firmado

Componentes:
  ✅ Firma Cosign OIDC
  ✅ SBOM automático (Syft)
  ✅ Grafana dashboard publicado
  ✅ CI/CD GitHub Actions

KPIs: 4/4 CUMPLIDOS
Tests: 30/30 PASSED
"

# Crear tag
git tag -a v2.11-m2.6-closed -m "SARAi v2.11 - M2.6 Completado

DevSecOps: Firma Cosign + SBOM + Grafana + CI/CD
Tests: 30/30 PASSED
"
```

### 3. Ver Historial

```bash
# Log compacto
git log --oneline --graph --decorate --all

# Log con stats
git log --stat --max-count=5

# Buscar commits
git log --grep="M2.5" --oneline
```

---

## 🔍 Comandos Útiles

### Estado del Repo

```bash
# Status compacto
git status --short

# Ver diferencias no staged
git diff

# Ver diferencias staged
git diff --cached
```

### Inspección de Archivos

```bash
# Ver archivos en un commit
git ls-tree --name-only v2.10-m2-closed

# Ver cambios en un archivo
git log -p -- core/trm_classifier.py

# Ver quién modificó cada línea
git blame core/embeddings.py
```

### Restauración

```bash
# Descartar cambios no staged
git restore core/embeddings.py

# Descartar cambios staged
git restore --staged core/embeddings.py

# Volver a un commit anterior (CUIDADO)
git reset --hard v2.10-m2-closed
```

---

## 📊 Estadísticas del Proyecto

### Líneas de Código

```bash
# Por tipo de archivo
git ls-files | xargs wc -l | sort -rn | head -20

# Solo Python
git ls-files '*.py' | xargs wc -l | tail -1
```

### Contribuciones

```bash
# Commits por autor
git shortlog -sn

# Actividad por día
git log --date=short --format="%ad" | sort | uniq -c
```

### Cambios en Milestone

```bash
# Archivos modificados en M2
git diff --name-status v2.9..v2.10-m2-closed

# Estadísticas de cambios
git diff --stat v2.9..v2.10-m2-closed
```

---

## 🚀 Workflows Avanzados

### Cherry-pick de Feature

```bash
# Aplicar commit específico de otra rama
git cherry-pick abc1234

# Cherry-pick sin commit automático
git cherry-pick -n abc1234
```

### Stash para Cambios Temporales

```bash
# Guardar cambios sin commit
git stash save "WIP: Implementando M3.1"

# Listar stashes
git stash list

# Recuperar stash
git stash pop
```

### Bisect para Debugging

```bash
# Encontrar commit que introdujo un bug
git bisect start
git bisect bad                  # Commit actual tiene bug
git bisect good v2.10-m2-closed # Commit antiguo funciona

# Git irá dividiendo el rango
# Para cada commit, prueba y ejecuta:
git bisect good  # o git bisect bad

# Al final:
git bisect reset
```

---

## 🔐 Backup y Seguridad

### Crear Bundle (Backup Offline)

```bash
# Crear backup del repo completo
git bundle create sarai_v2_backup_$(date +%Y%m%d).bundle --all

# Verificar bundle
git bundle verify sarai_v2_backup_20251027.bundle

# Restaurar desde bundle
git clone sarai_v2_backup_20251027.bundle sarai_v2_restored
```

### Verificar Integridad

```bash
# Verificar objetos del repo
git fsck --full

# Optimizar repo (cleanup)
git gc --aggressive --prune=now
```

---

## 📋 Checklist para Nuevos Milestones

- [ ] Crear rama feature si es desarrollo largo
- [ ] Commits frecuentes con mensajes descriptivos
- [ ] Tests pasando antes de commit
- [ ] Actualizar docs/M{X}_COMPLETION_REPORT.md
- [ ] Consolidar con `git add -A && git commit`
- [ ] Crear tag anotado: `git tag -a v2.X-mX-closed`
- [ ] Actualizar TODO list
- [ ] Verificar: `git log --oneline -5`

---

## 🎯 Ejemplos de Mensajes de Commit

### Formato Estándar

```
<tipo>(<scope>): <descripción corta>

<cuerpo opcional con detalles>

<footer opcional: referencias, breaking changes>
```

### Tipos Comunes

- `feat`: Nueva feature
- `fix`: Bug fix
- `docs`: Documentación
- `test`: Tests
- `refactor`: Refactorización sin cambio funcional
- `perf`: Mejora de rendimiento
- `chore`: Tareas de mantenimiento

### Ejemplos Reales

```bash
# Feature nueva
git commit -m "feat(rag): Añadir auditoría HMAC para búsquedas web

- Log de queries con timestamp
- Firma SHA-256 por línea
- Verificación de integridad
"

# Bug fix
git commit -m "fix(embeddings): Eliminar BitsAndBytes que causaba NaN

BitsAndBytes incompatible con CPU, causaba NaN en embeddings.
Solución: Cargar modelo sin cuantización con dtype=torch.float32.

Closes #42
"

# Documentación
git commit -m "docs(m2): Añadir reporte de consolidación completo

- M1.2: Migración a embeddings semánticos
- M2.5: RAG + SearXNG
- KPIs finales: 7/7 cumplidos
"
```

---

## 🔗 Referencias

- **Guía Git oficial**: https://git-scm.com/doc
- **Git Flow**: https://nvie.com/posts/a-successful-git-branching-model/
- **Conventional Commits**: https://www.conventionalcommits.org/

---

**Última actualización**: 2025-10-27  
**Estado del repo**: v2.10-m2-closed ✅
