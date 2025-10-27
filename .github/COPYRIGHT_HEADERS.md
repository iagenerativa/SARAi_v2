# Copyright Headers para SARAi v2.11

Este documento contiene los headers de copyright que deben añadirse a los archivos fuente del proyecto.

---

## Header para Archivos Python (.py)

```python
#!/usr/bin/env python3
"""
[Nombre del archivo] - [Descripción breve]

Copyright (c) 2025 Noel
Licensed under CC BY-NC-SA 4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/

Este archivo es parte de SARAi v2.11 "Omni-Sentinel".
No se permite uso comercial sin permiso del autor.
"""
```

---

## Header para Archivos YAML/JSON

```yaml
# SARAi v2.11 "Omni-Sentinel"
# Copyright (c) 2025 Noel
# Licensed under CC BY-NC-SA 4.0
# https://creativecommons.org/licenses/by-nc-sa/4.0/
```

---

## Header para Dockerfiles

```dockerfile
# SARAi v2.11 "Omni-Sentinel"
# Copyright (c) 2025 Noel
# Licensed under CC BY-NC-SA 4.0
# https://creativecommons.org/licenses/by-nc-sa/4.0/
# No commercial use without permission.
```

---

## Header para Scripts Shell (.sh)

```bash
#!/bin/bash
# SARAi v2.11 "Omni-Sentinel"
# Copyright (c) 2025 Noel
# Licensed under CC BY-NC-SA 4.0
# https://creativecommons.org/licenses/by-nc-sa/4.0/
```

---

## Header para Archivos Markdown (Documentación)

```markdown
<!--
SARAi v2.11 "Omni-Sentinel"
Copyright (c) 2025 Noel
Licensed under CC BY-NC-SA 4.0

Este documento es parte del proyecto SARAi.
Atribución requerida al compartir o modificar.
-->
```

---

## Aplicación de Headers

### Archivos Críticos que DEBEN tener header:

1. **Core del sistema**:
   - `core/trm_classifier.py`
   - `core/mcp.py`
   - `core/graph.py`
   - `core/model_pool.py`

2. **Agentes**:
   - `agents/audio_router.py` ✅ (añadir)
   - `agents/omni_pipeline.py`
   - `agents/rag_agent.py`
   - `agents/expert_agent.py`
   - `agents/tiny_agent.py`

3. **Skills**:
   - `skills/home_ops.py`
   - `skills/network_diag.py` (cuando se implemente)

4. **Configuración**:
   - `config/sarai.yaml`
   - `docker-compose.override.yml`
   - `Dockerfile.omni`

5. **Scripts**:
   - `scripts/download_gguf_models.py`
   - `scripts/train_trm_mini.py`
   - `scripts/online_tune.py`

### Archivos Opcionales:

- Tests (pueden tener header simplificado)
- Archivos generados automáticamente
- Archivos de configuración de terceros

---

## Script de Aplicación Automática

```bash
#!/bin/bash
# Script para añadir headers de copyright a archivos sin header

HEADER_PY='#!/usr/bin/env python3
"""
Copyright (c) 2025 Noel
Licensed under CC BY-NC-SA 4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/

Parte de SARAi v2.11 "Omni-Sentinel"
"""

'

# Encuentra archivos .py sin header de copyright
find . -name "*.py" -type f ! -path "./venv/*" ! -path "./.venv/*" | while read file; do
    if ! grep -q "Copyright (c) 2025 Noel" "$file"; then
        echo "Añadiendo header a: $file"
        # Crear archivo temporal con header
        echo "$HEADER_PY" | cat - "$file" > temp && mv temp "$file"
    fi
done

echo "Headers añadidos a archivos Python."
```

---

## Verificación de Cumplimiento

```bash
# Verificar que todos los archivos .py tienen header de copyright
find . -name "*.py" -type f ! -path "./venv/*" | while read file; do
    if ! grep -q "Copyright (c) 2025 Noel" "$file"; then
        echo "❌ Falta header: $file"
    fi
done

# Si no hay output, todos los archivos tienen header ✅
```

---

**Nota**: Los headers deben añadirse manualmente a archivos existentes durante el proceso de revisión de código. Nuevos archivos deben crearse con el header correspondiente desde el inicio.
