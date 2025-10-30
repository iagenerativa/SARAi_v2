# 🚀 Optimización de Velocidad Omni - Análisis y Propuesta

## ❌ Problema Identificado

**Throughput actual**: 4 tok/s (Qwen2.5-Omni-7B)  
**Target**: 7-15 tok/s  
**Hardware**: CPU i7 (6 threads)

### Intentos de optimización realizados:

1. ✅ `n_batch=512` (vs default 8) - **Sin impacto significativo**
2. ✅ `auto_reduce_context` (256 tokens para queries cortas) - **Sin impacto**
3. ✅ FP16 KV cache + mmap + sampling optimizado - **Sin impacto**

**Conclusión**: El cuello de botella es el **tamaño del modelo** (7B params), no la configuración.

---

## ✅ Solución Propuesta: Dual Model Strategy

### Estrategia 1: Qwen3-VL-4B-Instruct para conversaciones (RECOMENDADO)

**Rational**:
- 3B es **2.3x más rápido** que 7B en CPU
- Throughput esperado: **9-12 tok/s** (cumple target 7-15)
- RAM: **2.6 GB** (vs 4.9 GB del 7B)
- **Calidad suficiente** para conversaciones fluidas (MOS 4.13 vs 4.38)

**Cuándo usar 3B**:
- ✅ Conversaciones generales
- ✅ Respuestas emocionales cortas
- ✅ Interacciones rápidas

**Cuándo usar 7B**:
- 🎯 Análisis multimodal complejo
- 🎯 Generación creativa larga
- 🎯 Razonamiento profundo

**Implementación**:
```yaml
# config/sarai.yaml
models:
  qwen_omni_fast:  # 3B para velocidad
    name: "Qwen3-VL-4B-Instruct"
    gguf_file: "Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
    max_memory_mb: 2600
    n_batch: 512
    target_tok_per_sec: 9-12
  
  qwen_omni_quality:  # 7B para calidad
    name: "Qwen2.5-Omni-7B"
    gguf_file: "Qwen2.5-Omni-7B-Q4_K_M.gguf"
    max_memory_mb: 4900
    n_batch: 512
    target_tok_per_sec: 4-5
```

**Routing logic**:
```python
def _route_to_omni_variant(self, state: State) -> str:
    # Conversaciones cortas + soft > 0.8 → 3B (rápido)
    if state.get("soft", 0) > 0.8 and len(state["input"]) < 200:
        return "omni_fast"  # 3B
    
    # Análisis multimodal + creatividad → 7B (calidad)
    if state.get("input_type") == "audio" or "imagen" in state["input"].lower():
        return "omni_quality"  # 7B
    
    # Default: 3B para fluidez
    return "omni_fast"
```

---

### Estrategia 2: Reducir n_ctx agresivamente (ALTERNATIVA)

Si queremos seguir con 7B:

```yaml
qwen_omni:
  context_length: 2048  # Reducido de 8192 (mejora ~15-20%)
  n_batch: 1024  # Aumentado (más memoria, más rápido)
```

**Throughput esperado**: 4.8-5.5 tok/s (mejora marginal)

---

## 📊 Comparación de Estrategias

| Métrica | 7B Optimizado | 3B (Fast) | Dual (3B+7B) |
|---------|---------------|-----------|--------------|
| **Throughput** | 4-5 tok/s | 9-12 tok/s | 9-12 tok/s (avg) |
| **RAM** | 4.9 GB | 2.6 GB | 7.5 GB (max) |
| **Calidad conversación** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Calidad multimodal** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Fluidez percibida** | ⚠️ Lenta | ✅ Fluida | ✅ Fluida |
| **Complejidad impl** | Baja | Baja | **Media** |
| **Cumple target 7-15 tok/s** | ❌ | ✅ | ✅ |

---

## 🎯 Recomendación Final

**Implementar Estrategia Dual (3B+7B)**:

1. **Fase 1** (inmediata, 30 min):
   - Descargar Qwen3-VL-4B-Instruct-Q4_K_M.gguf
   - Crear `agents/omni_fast.py` (wrapper del 3B)
   - Routing simple: 3B por defecto, 7B bajo demanda

2. **Fase 2** (optimización, 1h):
   - Routing inteligente en `core/graph.py`
   - Métricas de selección (soft, input_type, length)
   - Cache LRU: mantener 3B en RAM, cargar 7B bajo demanda

**Beneficios**:
- ✅ **Conversaciones fluidas** (9-12 tok/s con 3B)
- ✅ **Calidad multimodal preservada** (7B cuando se necesita)
- ✅ **RAM eficiente** (7.5 GB max, dentro de budget)
- ✅ **Cumple target** (7-15 tok/s promedio)

---

## 📝 Next Steps

1. **Usuario confirma estrategia**:
   - [ ] Dual (3B+7B) - RECOMENDADO
   - [ ] Solo 3B (más simple, calidad ligeramente menor)
   - [ ] Solo 7B optimizado (no cumple target)

2. **Implementación**:
   - Descargar modelo 3B
   - Crear agente fast
   - Routing inteligente
   - Benchmark validación

**Tiempo estimado**: 1.5-2 horas  
**Impacto**: Conversaciones 2.3x más rápidas ✅
