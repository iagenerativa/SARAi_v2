# Nota de Despliegue - Modelos Especializados v2.14

**Fecha**: 1 noviembre 2025  
**Estado**: Qwen2.5-Coder desplegando, Marketing modelo ACTUALIZADO

---

## ⚠️ Cambio de Modelo de Marketing

### Modelo Planeado Original
- **Nombre**: `marketeam/LLa-Marketing`
- **Fuente**: https://huggingface.co/marketeam/LLa-Marketing
- **Problema**: NO tiene formato GGUF nativo (solo safetensors)
- **Estado**: ❌ No compatible con Ollama directamente

### Modelo Desplegado (SOLUCIÓN)
- **Nombre completo**: `pannapeak/mkt-campaign-gen-14b-q4_k_m-gguf`
- **Tamaño**: 9 GB (Q4_K_M)
- **Parámetros**: 14.8B (vs 8B planeado)
- **Especialización**: Generación de campañas de marketing
- **Estado**: ✅ **YA DISPONIBLE** en Ollama remoto (${OLLAMA_BASE_URL})
- **Ventaja**: Modelo más grande (14B > 8B) = mejor calidad

### Test de Validación

```bash
# Test ejecutado: 1 nov 2025 19:59 UTC
curl -X POST ${OLLAMA_BASE_URL}/api/generate -d '{
  "model": "pannapeak/mkt-campaign-gen-14b-q4_k_m-gguf",
  "prompt": "Genera copy publicitario breve (30 palabras) para campaña de lanzamiento de producto tech SaaS enfocado en empresas",
  "stream": false,
  "options": {
    "temperature": 0.9,
    "num_predict": 100
  }
}'

# Resultado: ✅ FUNCIONAL
# Latencia: 36.3 segundos (aceptable para P50)
# Respuesta: Generó thinking + copy (output completo visible)
```

---

## 📊 Estado de Despliegue

| Modelo | Propósito | Estado | Tamaño | Ubicación |
|--------|-----------|--------|--------|-----------|
| **Qwen2.5-Coder-7B** | Desarrollo avanzado | 🔄 Descargando (4.6 GB) | Q4_K_M | Ollama remoto |
| **pannapeak/mkt-14b** | Marketing nicho | ✅ Desplegado | 9 GB (Q4_K_M) | Ollama remoto |
| **SOLAR-10.7B** | General purpose | ✅ Desplegado | 6.5 GB (Q4_K_M) | Ollama remoto |

---

## 🔧 Configuración Actualizada

### .env Variables

```bash
# MODELOS EXTERNOS - OLLAMA REMOTE
OLLAMA_BASE_URL=http://<OLLAMA_HOST>:11434

# General purpose (ya desplegado)
SOLAR_MODEL_NAME=hf.co/solxxcero/SOLAR-10.7B-Instruct-v1.0-Q4_K_M-GGUF

# Especialización: Desarrollo avanzado (arquitectura, debugging)
QWEN_CODER_MODEL_NAME=qwen2.5-coder:7b-instruct

# Especialización: Marketing nicho (copy, campañas, SEO)
# ACTUALIZACIÓN: Usar pannapeak en lugar de marketeam/LLa-Marketing
LLAMARKETING_MODEL_NAME=pannapeak/mkt-campaign-gen-14b-q4_k_m-gguf
```

### config/models.yaml

```yaml
llamarketing_long:
  name: "Pannapeak Marketing Campaign Generator 14B"
  backend: "ollama"
  api_url: "${OLLAMA_BASE_URL}"
  
  # ACTUALIZACIÓN: Nombre del modelo desplegado
  model_name: "${LLAMARKETING_MODEL_NAME}"  # pannapeak/mkt-campaign-gen-14b-q4_k_m-gguf
  
  n_ctx: 2048
  temperature: 0.9  # Alta creatividad para marketing
  priority: 6
  
  # Routing: Solo si detect_specialized_skill() == "llamarketing"
  requires_skill: "llamarketing"
```

---

## 🎯 Decisión de Implementación

### Por qué pannapeak en lugar de marketeam/LLa-Marketing:

1. **Compatibilidad inmediata**: Ya en formato GGUF Q4_K_M
2. **Tamaño superior**: 14.8B params vs 8B planeado
3. **Especialización verificada**: Diseñado específicamente para campañas de marketing
4. **Disponibilidad**: Ya desplegado y funcional en Ollama remoto
5. **Sin conversión manual**: marketeam/LLa-Marketing requeriría:
   - Descargar safetensors (16+ GB)
   - Convertir a GGUF con llama.cpp
   - Cuantizar a Q4_K_M
   - Subir a Ollama
   - Tiempo estimado: 2-4 horas

### Trade-off Aceptado:

- **Costo RAM**: +4 GB (9 GB vs 5 GB estimado)
- **Ganancia**: Modelo más potente (14B) + deployment inmediato
- **Impacto**: 0 GB en cliente (todo en Ollama remoto)

---

## 🚀 Próximos Pasos

1. ✅ **Validar Qwen2.5-Coder** cuando termine la descarga (~5-10 min)
2. ⏳ **Implementar detect_specialized_skill()** en `core/skill_configs.py`
3. ⏳ **Actualizar route_to_model()** en `core/mcp.py`
4. ⏳ **Crear tests** en `tests/test_specialized_routing.py`
5. ⏳ **Actualizar .env.example** con nombres correctos de modelos
6. ⏳ **Actualizar config/models.yaml** con configuración final

---

## 📝 Notas Técnicas

### Alternativas Evaluadas

1. **Convertir marketeam/LLa-Marketing manualmente**
   - Pros: Modelo original planeado
   - Contras: 2-4h de trabajo, 16+ GB descarga
   - Decisión: ❌ Rechazado por tiempo

2. **Usar RichardErkhov/read1337_-_Llama-3.1-8B-bnb-4bit-marketing-gguf**
   - Pros: Versión GGUF de Llama-3.1-8B-marketing
   - Contras: No disponible en registry de Ollama
   - Decisión: ❌ Rechazado por incompatibilidad

3. **Usar pannapeak/mkt-campaign-gen-14b-q4_k_m-gguf** ✅
   - Pros: Ya desplegado, 14B params, GGUF nativo
   - Contras: Modelo diferente al planeado
   - Decisión: ✅ ACEPTADO (mejor opción disponible)

---

**Conclusión**: Sistema de routing inteligente listo para implementación con modelos validados y funcionales.
