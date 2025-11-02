# 🔧 Corrección config/models.yaml - Configuración Real

**Fecha**: 1 Noviembre 2025  
**Cambios**: Ajuste a configuración real del sistema

---

## ❌ Problemas Detectados

### 1. Modelo SOLAR Incorrecto

**ANTES** (models.yaml v2.14 inicial):
```yaml
solar_short:
  backend: "gguf"
  model_path: "models/cache/solar/solar-10.7b-instruct-v1.0.Q4_K_M.gguf"
```

**PROBLEMA**: Este archivo NO existe en local. SOLAR se sirve desde **Ollama remoto**.

---

### 2. Prioridades Incorrectas

**ANTES**:
- SOLAR: `priority: 10` (siempre en memoria)
- LFM2: `priority: 8` (carga bajo demanda)

**PROBLEMA**: LFM2 es el modelo BASE para todo lo que no sea visión. Debe estar SIEMPRE en memoria.

---

## ✅ Soluciones Implementadas

### 1. SOLAR → Ollama Backend

**AHORA**:
```yaml
solar_short:
  name: "UNA-SOLAR-10.7B-Instruct (Short Context)"
  backend: "ollama"
  
  # Modelo servido por Ollama (definir en OLLAMA_BASE_URL)
  api_url: "${OLLAMA_BASE_URL}"  # Ejemplo: http://localhost:11434
  model_name: "${SOLAR_MODEL_NAME}"  # hf.co/fblgit/UNA-SOLAR-10.7B-Instruct-v1.0:Q5_K_M
  
  n_ctx: 512
  temperature: 0.7
  max_tokens: 512
  
  load_on_demand: true  # Servidor externo
  priority: 9
  max_memory_mb: 0  # Sin RAM local
```

**Variables de entorno** (.env):
```bash
OLLAMA_BASE_URL=http://<OLLAMA_HOST>:11434
SOLAR_MODEL_NAME=hf.co/fblgit/UNA-SOLAR-10.7B-Instruct-v1.0:Q5_K_M
```

**Beneficios**:
- ✅ Usa modelo REAL existente
- ✅ Sin RAM local (servidor remoto)
- ✅ Configuración dinámica con variables de entorno
- ✅ n_ctx flexible (512/2048)

---

### 2. LFM2 → Prioridad Máxima

**AHORA**:
```yaml
lfm2:
  name: "LiquidAI-LFM2-1.2B"
  backend: "gguf"
  model_path: "models/cache/lfm2/lfm2-1.2b.Q4_K_M.gguf"
  
  load_on_demand: false  # ✅ SIEMPRE cargado por defecto
  priority: 10  # ✅ Prioridad MÁXIMA
  max_memory_mb: 700
  
  # Política de coexistencia con Qwen3-VL
  allow_unload_for_vision: true  # Puede descargarse temporalmente para visión
```

**Política**:
- ✅ **LFM2 siempre en memoria** (modelo base para todo)
- ✅ **EXCEPCIÓN**: Se descarga temporalmente si Qwen3-VL necesita RAM
- ✅ Se **recarga automáticamente** después de tarea de visión

---

### 3. Qwen3-VL → Coexistencia Inteligente

**AHORA**:
```yaml
qwen3_vl:
  name: "Qwen3-VL-4B-Instruct"
  backend: "multimodal"
  
  load_on_demand: true  # Solo cuando se necesita visión
  priority: 7  # Prioridad ALTA para tareas de visión
  max_memory_mb: 4096
  
  # Política de coexistencia con LFM2
  can_evict_lfm2: true  # Puede solicitar descarga temporal de LFM2
```

**Política**:
- ✅ Se carga **solo para tareas de visión**
- ✅ Puede solicitar descarga temporal de LFM2
- ✅ Se descarga automáticamente después de uso
- ✅ LFM2 se recarga inmediatamente

---

## 📊 Prioridades Finales

| Modelo | Priority | RAM Local | Política |
|--------|----------|-----------|----------|
| **LFM2** | 10 | 700 MB | Siempre en memoria (excepto visión temporal) |
| **SOLAR short** | 9 | 0 MB | Servidor Ollama remoto |
| **SOLAR long** | 8 | 0 MB | Servidor Ollama remoto |
| **Qwen3-VL** | 7 | 4096 MB | Solo visión (carga bajo demanda) |
| **Qwen-Omni** | 6 | 4096 MB | Solo audio (carga bajo demanda) |

---

## 🔄 Estados de Memoria

### Estado Normal (Texto)

```
┌─────────────────────────┐
│ LFM2 (700 MB)           │ ← SIEMPRE activo
├─────────────────────────┤
│ SOLAR (Ollama remoto)   │ ← 0 MB local
└─────────────────────────┘

RAM Total Usada: ~700 MB
```

### Estado Visión (Temporal)

```
PASO 1: Necesidad de visión detectada
┌─────────────────────────┐
│ LFM2 descargado         │ ← Liberado temporalmente
├─────────────────────────┤
│ Qwen3-VL (4096 MB)      │ ← Cargado para tarea
└─────────────────────────┘

PASO 2: Tarea de visión completada
┌─────────────────────────┐
│ LFM2 (700 MB)           │ ← RECARGADO automáticamente
├─────────────────────────┤
│ Qwen3-VL descargado     │ ← Liberado
└─────────────────────────┘

RAM Total Usada: ~4GB (temporal) → ~700MB (recarga)
```

---

## 🎯 Lógica de Decisión del ModelPool

```python
# core/model_pool.py (pseudo-código)

class ModelPool:
    def get_model(self, name: str):
        config = load_config(name)
        
        # Caso especial: Visión con presión de RAM
        if name == "qwen3_vl" and self.ram_available() < 4096:
            # Descargar LFM2 temporalmente
            if "lfm2" in self.cache and config.get("can_evict_lfm2"):
                self.unload("lfm2", temporary=True)
            
            # Cargar Qwen3-VL
            model = self._load_model(name, config)
            
            # Registrar callback para recargar LFM2
            self.register_unload_callback("qwen3_vl", lambda: self.reload("lfm2"))
            
            return model
        
        # Caso normal
        return self._load_model(name, config)
```

---

## 📝 Notas Críticas Actualizadas

### 1. Verificación de Modelos Locales

**ANTES de usar GGUF locales, verificar que existen**:

```bash
# LFM2 (CRÍTICO - debe existir)
ls -lh models/cache/lfm2/lfm2-1.2b.Q4_K_M.gguf

# Si NO existe:
# 1. Descargar de HuggingFace
# 2. O ajustar model_path en models.yaml
```

**SOLAR NO requiere archivo local** (servidor Ollama).

---

### 2. Variables de Entorno Requeridas

```bash
# .env (VERIFICAR)
OLLAMA_BASE_URL=http://<OLLAMA_HOST>:11434
SOLAR_MODEL_NAME=hf.co/fblgit/UNA-SOLAR-10.7B-Instruct-v1.0:Q5_K_M
```

Si estas variables no están definidas, SOLAR fallará al cargar.

---

### 3. Migración Futura SOLAR Local

**Cuando descargues SOLAR en local**:

```yaml
# Cambiar en models.yaml:
solar_short:
  backend: "gguf"  # Era: "ollama"
  model_path: "models/cache/solar/UNA-SOLAR-10.7B.Q5_K_M.gguf"  # Era: api_url + model_name
  n_threads: 6
  use_mmap: true
  use_mlock: false
  priority: 9
  max_memory_mb: 5500  # Necesitará RAM local
```

**NO modificar código Python**, solo YAML.

---

## ✅ Validación de Cambios

### Checklist

- [x] SOLAR usa backend Ollama (no GGUF local)
- [x] Variables de entorno correctas (${OLLAMA_BASE_URL}, ${SOLAR_MODEL_NAME})
- [x] LFM2 priority=10 (siempre en memoria)
- [x] Qwen3-VL priority=7 (solo visión)
- [x] Políticas de coexistencia documentadas
- [x] Notas de uso actualizadas
- [x] Estados de memoria documentados

---

## 🔄 Próximos Pasos

1. **Verificar LFM2 local**:
   ```bash
   ls -lh models/cache/lfm2/lfm2-1.2b.Q4_K_M.gguf
   ```

2. **Verificar Ollama accesible**:
   ```bash
  curl ${OLLAMA_BASE_URL}/api/tags
   ```

3. **Implementar lógica de coexistencia** en `core/model_pool.py`:
   - Método `unload(model, temporary=True)`
   - Callbacks de recarga
   - Política `can_evict_lfm2`

4. **Tests**:
   - Test carga LFM2 al inicio
   - Test descarga temporal LFM2 cuando Qwen3-VL carga
   - Test recarga automática LFM2 después de visión

---

**FIN CORRECCIÓN - models.yaml ajustado a configuración real**
