# Análisis de Memoria: SOLAR + LFM2 + Omni Simultáneos

## 🎯 Pregunta

¿Podemos tener **SOLAR + LFM2 + Omni todos en memoria** ahora que SOLAR puede usar servidor Ollama remoto?

## 🆕 Actualización (29 Oct 2025)

**GGUF de Qwen2.5-Omni disponible**: `unsloth/Qwen2.5-Omni-7B-GGUF` (Q4_K_M, 4.68 GB).

**DECISIÓN ESTRATÉGICA**: Upgrade de 3B a **7B** por función crítica multimodal en AGI.

- **Opción 1 (Solo texto)**: GGUF Q4_K_M = 4.9 GB (compatible con llama-cpp-python)
- **Opción 2 (Multimodal)**: Transformers 4bit = 5.1 GB (audio+imagen+video+TTS)
- **Decisión SARAi v2.16**: Arquitectura híbrida (GGUF para texto, Transformers bajo demanda para multimodal)

Los cálculos a continuación asumen **7B GGUF Q4_K_M** (4.9 GB) para multimodal estratégico.

**Justificación**: Con SOLAR offloaded (11.6 GB liberados), podemos priorizar calidad multimodal (+10-18% performance en todas las modalidades).

---

## 📊 Análisis de Consumo de RAM

### Escenario 1: SOLAR GGUF Nativo (Anterior)

| Componente | RAM (GB) | Notas |
|------------|----------|-------|
| **SOLAR-10.7B GGUF** | 11.8 GB | GGUF Q4_K_M local, n_ctx=2048 |
| **LFM2-1.2B** | 0.7 GB | Tiny tier, n_ctx=2048 |
| **Qwen-Omni-7B** | 4.9 GB | **7B Q4_K_M (Unsloth)**, solo texto |
| **EmbeddingGemma** | 0.15 GB | Siempre en memoria |
| **TRM-Router + Mini** | 0.05 GB | Clasificadores |
| **Sistema + Python** | 1.5 GB | OS + runtime |
| **TOTAL** | **19.1 GB** | ❌ **EXCEDE 16GB** (OOM garantizado) |

**Conclusión Anterior**: Solo podíamos cargar **2 de 3 LLMs** simultáneos (Expert + Tiny O Expert + Omni).

---

### Escenario 2: SOLAR Ollama HTTP (NUEVO v2.16)

| Componente | RAM (GB) | Notas |
|------------|----------|-------|
| **SOLAR Cliente HTTP** | 0.2 GB | Solo cliente requests + metadata |
| **LFM2-1.2B** | 0.7 GB | Tiny tier, n_ctx=2048 |
| **Qwen-Omni-7B** | 4.9 GB | **7B Q4_K_M (Unsloth)**, solo texto |
| **EmbeddingGemma** | 0.15 GB | Siempre en memoria |
| **TRM-Router + Mini** | 0.05 GB | Clasificadores |
| **Sistema + Python** | 1.5 GB | OS + runtime |
| **TOTAL** | **7.5 GB** | ✅ **SOLO 47% de 16GB** |

**Conclusión Nueva v2.16**: ✅ **SÍ, podemos tener los 3 LLMs "disponibles"** con **8.5 GB libres** (53% de RAM libre).

**Nota**: Si Omni usa Transformers (multimodal completo), RAM = 5.1 GB → Total 7.7 GB (52% libre).

---

## 🚀 Beneficios de la Estrategia Híbrida

### 1. RAM Liberada

```
ANTES (GGUF nativo):    16.3 GB → OOM
AHORA (Ollama HTTP):     4.7 GB → 70% libre
AHORRO:                 11.6 GB (-71%)
```

### 2. Capacidad Simultánea

**Antes**:
- ✅ SOLAR + LFM2 = 12.5 GB (OK)
- ❌ SOLAR + Omni = 13.9 GB (borderline)
- ❌ SOLAR + LFM2 + Omni = 16.3 GB (OOM)

**Ahora**:
- ✅ SOLAR (HTTP) + LFM2 + Omni = 4.7 GB (sobra RAM)
- ✅ Podríamos añadir más modelos pequeños si fuera necesario

### 3. Latencia Adicional Aceptable

| Métrica | GGUF Local | Ollama HTTP (LAN) | Δ |
|---------|------------|-------------------|---|
| **Primera inferencia** | ~30ms | ~80ms | +50ms |
| **Tokens/s** | 2.11-2.20 | 2.0-2.5 | ~5% más lento |
| **Prompt eval (512 tok)** | ~15s | ~16s | +1s |

**Conclusión**: +50ms de latencia es **ACEPTABLE** para ganar 11.6 GB de RAM.

---

## 🏗️ Arquitectura Propuesta

### Model Pool Optimizado

```python
# core/model_pool.py (pseudo-código)

class ModelPool:
    """
    Pool híbrido: HTTP para modelos grandes, GGUF para pequeños
    """
    
    def __init__(self):
        # Modelos SIEMPRE en memoria (pequeños)
        self.embedding = load_embedding_gemma()    # 150 MB
        self.trm_router = load_trm_router()        # 50 MB
        
        # Modelos cargados bajo demanda
        self.cache = {
            "solar": None,      # HTTP (200 MB cliente)
            "lfm2": None,       # GGUF (700 MB)
            "omni": None        # GGUF (2.1 GB)
        }
    
    def get(self, model_name: str):
        """Carga modelo si no está en cache"""
        if model_name == "solar":
            # Usar cliente HTTP (carga instantánea)
            if self.cache["solar"] is None:
                from agents.solar_ollama import SolarOllama
                self.cache["solar"] = SolarOllama()
            return self.cache["solar"]
        
        elif model_name == "lfm2":
            # GGUF local (700 MB)
            if self.cache["lfm2"] is None:
                self.cache["lfm2"] = load_lfm2_gguf()
            return self.cache["lfm2"]
        
        elif model_name == "omni":
            # GGUF local (2.1 GB)
            if self.cache["omni"] is None:
                self.cache["omni"] = load_omni_gguf()
            return self.cache["omni"]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Retorna uso de RAM por modelo"""
        return {
            "embedding": 0.15,
            "trm_router": 0.05,
            "solar": 0.2 if self.cache["solar"] else 0,
            "lfm2": 0.7 if self.cache["lfm2"] else 0,
            "omni": 2.1 if self.cache["omni"] else 0,
            "system": 1.5,
            "total": sum(...)
        }
```

### Política de Carga

**NUNCA descargar modelos** (nueva estrategia):

```python
# Antes (v2.15): Descarga si RAM > 12GB
if psutil.virtual_memory().available / (1024**3) < 4.0:
    self.unload("omni")  # Liberar RAM

# Ahora (v2.16): TODOS en memoria simultáneamente
# Total: 4.7 GB < 12 GB límite → nunca necesitamos descargar
```

---

## 🎭 Casos de Uso Mejorados

### Caso 1: Multimodal + Razonamiento

**Antes**: No podíamos tener Omni + SOLAR juntos (13.9 GB)

```python
# Usuario envía imagen + pregunta técnica
user_input = {
    "image": "foto.jpg",
    "text": "Explica qué sale en esta gráfica de red"
}

# Necesitábamos descargar Omni después de procesar imagen
omni = model_pool.get("omni")
description = omni.process_image(user_input["image"])
model_pool.unload("omni")  # ❌ Descarga para liberar RAM

solar = model_pool.get("solar")  # ❌ Carga GGUF (11.8 GB)
response = solar.generate(f"Contexto: {description}\n\n{user_input['text']}")
```

**Ahora**: Ambos disponibles simultáneamente

```python
# ✅ Omni y SOLAR disponibles al mismo tiempo
omni = model_pool.get("omni")      # Ya en memoria (2.1 GB)
solar = model_pool.get("solar")    # HTTP instantáneo (0.2 GB)

description = omni.process_image(user_input["image"])
response = solar.generate(f"Contexto: {description}\n\n{user_input['text']}")
# Total RAM: 2.1 + 0.2 = 2.3 GB → ¡11.3 GB libres!
```

### Caso 2: Pipeline Híbrido (Soft → Hard)

**Antes**: LFM2 → SOLAR requería descarga intermedia

```python
# Modulación de tono
lfm2 = model_pool.get("lfm2")
tone = lfm2.generate("Reformula con empatía: ...")
model_pool.unload("lfm2")  # ❌ Descarga

solar = model_pool.get("solar")  # ❌ Carga GGUF
response = solar.generate(f"Con este tono: {tone}, responde: ...")
```

**Ahora**: Pipeline sin descarga

```python
# ✅ LFM2 y SOLAR disponibles simultáneamente
lfm2 = model_pool.get("lfm2")    # 0.7 GB
solar = model_pool.get("solar")  # 0.2 GB HTTP

tone = lfm2.generate("Reformula con empatía: ...")
response = solar.generate(f"Con este tono: {tone}, responde: ...")
# Total: 0.9 GB → ¡Sin descarga!
```

### Caso 3: Voz + RAG + Expert

**Antes**: Imposible (Omni 2.1 GB + SOLAR 11.8 GB + LFM2 0.7 GB = 14.6 GB)

**Ahora**: Perfectamente viable

```python
# ✅ Los 3 modelos disponibles
omni = model_pool.get("omni")    # Voz (2.1 GB)
solar = model_pool.get("solar")  # Expert (0.2 GB HTTP)
lfm2 = model_pool.get("lfm2")    # Modulación (0.7 GB)

# Pipeline completo
audio_input = user_records_voice()
text = omni.transcribe(audio_input)           # Omni STT
facts = solar.generate(f"Datos sobre: {text}")  # Expert
response = lfm2.modulate(facts, tone="friendly")  # Modulación

# Total: 3.0 GB → ¡13 GB libres para búsqueda web!
```

---

## ⚠️ Trade-offs y Consideraciones

### 1. Dependencia de Red

| Aspecto | GGUF Local | Ollama HTTP |
|---------|------------|-------------|
| **Red requerida** | No | Sí (LAN) |
| **Falla si...** | Disco lleno | Servidor caído |
| **Latencia** | Constante | Variable (red) |
| **Ancho de banda** | 0 MB/s | ~20 KB/s (tokens) |

**Mitigación**: Fallback automático a GGUF si Ollama no disponible

```python
def get_solar_with_fallback():
    try:
        return SolarOllama()  # Intento 1: HTTP
    except ConnectionError:
        return SolarNative()  # Intento 2: GGUF local
```

### 2. Latencia de Red

**LAN (subred privada)**:
- Ping: ~1ms
- Overhead HTTP: ~50ms por request
- Streaming: Token-por-token sin lag perceptible
- **Conclusión**: ✅ Aceptable

**WAN (Internet)**:
- Ping: 20-100ms
- Overhead HTTP: 100-300ms por request
- **Conclusión**: ⚠️ Solo para development, no producción

### 3. Escalabilidad del Servidor

**Servidor Ollama (<OLLAMA_HOST>)**:
- Capacidad: ~2-3 requests simultáneas (depende de RAM)
- Uso actual: Solo desarrollo de SARAi
- **Conclusión**: ✅ OK para 1-2 desarrolladores

**Si crece el equipo**:
- Opción A: Servidor más grande
- Opción B: Volver a GGUF local en producción
- Opción C: Ollama en cluster (múltiples servidores)

---

## 🎯 Recomendación Final

### SÍ, podemos tener SOLAR + LFM2 + Omni simultáneos ✅

**Configuración recomendada**:

```yaml
# config/sarai.yaml

models:
  solar:
    backend: "ollama_http"  # Nuevo
    fallback: "gguf_native" # Si HTTP falla
    
  lfm2:
    backend: "gguf_native"  # Pequeño, mejor local
    
  omni:
    backend: "gguf_native"  # Multimodal, mejor local

memory:
  max_ram_gb: 12
  max_concurrent_llms: 3  # NUEVO (antes era 2)
  strategy: "hybrid_http_gguf"
```

**Uso de RAM esperado**:
- **Mínimo**: 2.5 GB (solo sistema + embeddings)
- **Típico**: 4.7 GB (todos los modelos cargados)
- **Máximo**: 6.0 GB (con skills + cache adicional)
- **Margen**: 10 GB libres (62% de RAM disponible)

### Ventajas

1. ✅ **Latencia reducida**: Sin descarga/carga intermedia
2. ✅ **Pipelines complejos**: Multimodal + Expert + Soft
3. ✅ **RAG + Voz**: Espacio para búsqueda web + audio
4. ✅ **Desarrollo rápido**: No esperar cargas de 11.8 GB

### Desventajas (mitigables)

1. ⚠️ Depende de red LAN (fallback a GGUF)
2. ⚠️ +50ms latencia HTTP (aceptable)
3. ⚠️ Servidor único (escalar si crece equipo)

---

## 🚀 Implementación Sugerida

### 1. Actualizar Model Pool

```python
# core/model_pool.py

MAX_CONCURRENT_LLMS = 3  # Cambiar de 2 a 3
```

### 2. Configurar Estrategia Híbrida

```python
# core/model_pool.py

def get_solar():
    """Estrategia híbrida con fallback"""
    prefer_http = os.getenv("SOLAR_PREFER_HTTP", "true").lower() == "true"
    
    if prefer_http:
        try:
            return SolarOllama()
        except ConnectionError:
            logger.warning("Ollama no disponible, usando GGUF local")
            return SolarNative()
    else:
        return SolarNative()
```

### 3. Añadir a `.env`

```bash
# Preferir HTTP en desarrollo
SOLAR_PREFER_HTTP=true

# Permitir 3 LLMs simultáneos
MAX_CONCURRENT_LLMS=3
```

### 4. Validar con Benchmark

```python
# scripts/test_concurrent_models.py

def test_all_models_loaded():
    pool = ModelPool()
    
    # Cargar los 3
    solar = pool.get("solar")
    lfm2 = pool.get("lfm2")
    omni = pool.get("omni")
    
    # Verificar RAM
    usage = pool.get_memory_usage()
    assert usage["total"] < 12.0, "Excede límite de RAM"
    
    # Probar inferencia simultánea
    assert solar.generate("test") is not None
    assert lfm2.generate("test") is not None
    assert omni.process_image(dummy_image) is not None
    
    print(f"✅ RAM usada: {usage['total']:.1f} GB < 12 GB")
```

---

## 📊 Comparativa Final

| Métrica | Antes (GGUF) | Ahora (HTTP) | Mejora |
|---------|--------------|--------------|--------|
| **Modelos simultáneos** | 2 | 3 | +50% |
| **RAM total** | 12.5 GB | 4.7 GB | -62% |
| **RAM libre** | 3.5 GB | 11.3 GB | +223% |
| **Latencia SOLAR** | 30ms | 80ms | +50ms |
| **Carga inicial** | 60s | <1s | -98% |
| **Pipelines complejos** | ❌ | ✅ | Habilitado |

**Conclusión**: El trade-off de **+50ms latencia** por **+11.6 GB RAM libre** es **EXCELENTE**.

---

**Respuesta corta**: 

# ✅ SÍ, con Ollama HTTP podemos tener SOLAR + LFM2 + Omni-3B simultáneos

- **RAM usada**: 4.7 GB (vs 16.3 GB antes)
- **RAM libre**: 11.3 GB (70% disponible)
- **Latencia extra**: +50ms (aceptable)
- **Beneficio**: Pipelines multimodales complejos sin descargas
