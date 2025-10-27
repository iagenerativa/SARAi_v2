# Commit Message Template para SARAi v2.11

## Formato Recomendado

```
<tipo>(<scope>): <descripción corta>

<descripción detallada opcional>

<footer opcional: referencias, breaking changes, etc.>
```

---

## Tipos de Commit

- **feat**: Nueva funcionalidad
- **fix**: Corrección de bug
- **docs**: Solo cambios en documentación
- **style**: Formato, espacios, etc. (sin cambio de código)
- **refactor**: Refactorización sin cambiar funcionalidad
- **perf**: Mejora de rendimiento
- **test**: Añadir o corregir tests
- **build**: Cambios en build system o dependencias
- **ci**: Cambios en CI/CD
- **chore**: Mantenimiento (actualización de deps, etc.)
- **revert**: Revertir commit anterior

---

## Scopes Comunes

- **audio**: Audio router, Omni pipeline, voz
- **rag**: RAG agent, web cache, búsqueda
- **skills**: Home Ops, Network Diag
- **core**: TRM, MCP, Graph, ModelPool
- **docker**: Dockerfiles, compose, hardening
- **docs**: CHANGELOG, ARCHITECTURE, README
- **config**: sarai.yaml, .env
- **tests**: Tests unitarios, integración, benchmarks

---

## Ejemplos de Commits

### Nueva Funcionalidad
```
feat(audio): implementar audio router con fallback Sentinel

- Añadir LanguageDetector (Whisper-tiny + fasttext)
- Implementar route_audio() con lógica de routing
- Garantizar 0% crash rate con fallback a omni-es

Closes #42
```

### Corrección de Bug
```
fix(rag): corregir timeout en SearXNG búsqueda

Aumentar timeout de 5s a 10s para evitar falsos positivos
de fallback Sentinel en conexiones lentas.

Fixes #38
```

### Documentación
```
docs(roadmap): añadir planning completo v2.11

- 5 Milestones con fechas y criterios
- Testing strategy completa
- Troubleshooting guide
- Cronograma detallado 3 semanas

Total: ~450 líneas
```

### Refactorización
```
refactor(core): extraer lógica de routing a función helper

Mejora legibilidad del graph.py sin cambiar funcionalidad.
```

### Tests
```
test(audio): añadir tests para audio_router fallback

- test_sentinel_fallback: audio corrupto
- test_language_detection: precisión >95%
- test_routing_logic: omni vs nllb

Cobertura: 90%
```

### Docker
```
build(docker): aplicar hardening kernel-level

- security_opt: no-new-privileges
- cap_drop: ALL
- read_only: true
- tmpfs: /tmp

Superficie de ataque: -99%
```

### Breaking Changes
```
feat(config): cambiar AUDIO_ENGINE a enum estricto

BREAKING CHANGE: Valores válidos ahora son solo:
  - omni3b
  - nllb
  - lfm2
  - disabled

Valores antiguos como "omni" ya no funcionan.

Migration: Editar .env y usar "omni3b" en vez de "omni"
```

---

## Buenas Prácticas

1. **Usa imperativo**: "añadir" no "añadido", "fix" no "fixed"
2. **Primera línea < 72 caracteres**
3. **Cuerpo opcional explica QUÉ y POR QUÉ, no CÓMO**
4. **Referencias issues con #numero**
5. **Breaking changes en footer**
6. **Un commit = un cambio lógico** (no mezclar tipos)

---

## Ejemplos de Commits BAD ❌

```
# Demasiado genérico
fix: arreglos varios

# No especifica scope
feat: añadir algo

# Mezcla tipos
feat+fix: nueva feature y corrección de bug

# Pasado en vez de imperativo
fixed: corregido el bug de audio
```

---

## Configuración Git

```bash
# Usar este template por defecto
git config commit.template .github/COMMIT_TEMPLATE.md

# Abrir editor para commits
git config core.editor "nano"  # o vim, code --wait, etc.
```

---

## Flujo de Trabajo Recomendado

1. **Hacer cambios** en una sola área (audio, docs, etc.)
2. **Revisar cambios**: `git diff`
3. **Stagear selectivamente**: `git add -p` (parcial)
4. **Commit con template**: `git commit` (abre editor)
5. **Escribir mensaje** siguiendo formato
6. **Push**: `git push origin main` (o feature branch)

---

**Ejemplo de sesión completa**:

```bash
# Implementar audio router
git add agents/audio_router.py
git commit -m "feat(audio): implementar audio router con fallback Sentinel

- LanguageDetector con Whisper-tiny + fasttext
- route_audio() con lógica de routing omni/nllb/lfm2
- Garantizar 0% crash rate (fallback a omni-es)

KPIs:
- Latencia LID: <50ms
- Precisión: >95%
- Crash rate: 0%

Closes #42"

# Actualizar documentación
git add CHANGELOG.md ARCHITECTURE.md
git commit -m "docs(changelog): documentar 3 refinamientos v2.11

- Ajuste A: Audio Router
- Ajuste B: AUDIO_ENGINE flag
- Ajuste C: Docker Hardening

Total: ~260 líneas"

# Push todo
git push origin main
```

---

**Versión**: 1.0  
**Última actualización**: 2025-10-27  
