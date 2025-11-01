# 🚀 Unified Model Wrapper: Arquitectura Universal para SARAi

**Versión**: v2.14 REDISEÑO FINAL  
**Fecha**: 31 Octubre 2025  
**Propósito**: Capa de abstracción modular que permita evolucionar SARAi sin limitaciones de hardware actuales

---

## 🎯 Visión del Usuario

> "las capacidades multimodales no sólo están en la parte de QWEN3, quiero que el wrapper 
> potencie **todas las capacidades que tiene SARAi** y que me permita ofrecerle **nuevas capacidades**, 
> aprovechando la potencia de **langchain** para poder hacer nuevas implementaciones de modelos 
> **sin tener que hacer un gran desarrollo** y así preparamos a SARAi para poder **evolucionar en el futuro**, 
> cuando tenga menos limitaciones de hardware."

---

## 🏗️ Principios Arquitectónicos

### 1. **Abstracción Universal**
- Un wrapper para TODOS los modelos (text, audio, vision, multimodal)
- Interfaz unificada independiente del backend (GGUF, Transformers, API, gRPC)

### 2. **LangChain Native**
- Todo orquestado con StateGraph (no código imperativo)
- Nuevos modelos = nuevos nodos (plug & play)
- Chain composition para pipelines complejos

### 3. **Evolución Sin Refactorización**
- Agregar modelo nuevo: solo crear `ModelConfig` + nodo
- Backend flexible: CPU → GPU → Cloud sin cambiar lógica
- Hardware upgrades = solo cambiar config YAML

### 4. **Modularidad Phoenix**
- Cada modelo es un "skill especializado"
- Skills NO son contenedores, son configuraciones de modelo
- Composición > Duplicación

---

## 📐 Arquitectura: Unified Model Wrapper

### Componente 1: `core/unified_model_wrapper.py`

```python
"""
Unified Model Wrapper: Capa de abstracción para TODOS los modelos SARAi

Filosofía:
- Interfaz única para text, audio, vision, multimodal
- Backend abstraído (GGUF, Transformers, Ollama, gRPC, OpenAI API)
- LangChain native (Runnable interface)
- Configuración YAML > hard-coded logic

Casos de uso:
1. Text generation (SOLAR, LFM2)
2. Vision analysis (Qwen3-VL)
3. Audio processing (Vosk, MeloTTS)
4. Multimodal (Qwen2.5-Omni)
5. Future: Any model (Gemini, Claude, Llama-4, etc.)
"""

from typing import Dict, Any, Optional, List, Union, Literal
from pathlib import Path
import yaml
from dataclasses import dataclass
from abc import ABC, abstractmethod

# LangChain imports
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings


@dataclass
class ModelConfig:
    """
    Configuración universal de modelo
    
    Cubre TODOS los tipos: text, audio, vision, multimodal
    """
    name: str
    type: Literal["text", "audio", "vision", "multimodal", "embedding"]
    backend: Literal["gguf", "transformers", "ollama", "grpc", "openai_api"]
    
    # Backend-specific
    model_path: Optional[str] = None  # GGUF file path
    repo_id: Optional[str] = None  # HuggingFace repo
    api_url: Optional[str] = None  # gRPC/REST endpoint
    api_key: Optional[str] = None  # API authentication
    
    # Runtime config
    n_ctx: int = 2048
    temperature: float = 0.8
    max_tokens: int = 256
    
    # Multimodal config
    supports_images: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    
    # Loading strategy
    load_on_demand: bool = True  # False = preload
    ttl_seconds: int = 60  # Auto-unload timeout
    priority: int = 5  # 1-10 (10 = highest priority)


class UnifiedModelWrapper(Runnable, ABC):
    """
    Wrapper base compatible con LangChain Runnable
    
    Ventajas:
    - Compatible con LCEL (LangChain Expression Language)
    - Composable con | operator
    - Streaming nativo
    - Async support
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self._load_lock = None  # Thread-safe loading
    
    @abstractmethod
    def _load_model(self):
        """Carga modelo según backend"""
        pass
    
    @abstractmethod
    def _invoke_sync(self, input: Any) -> Any:
        """Invocación síncrona"""
        pass
    
    def invoke(
        self,
        input: Union[str, Dict, List[BaseMessage]],
        config: Optional[RunnableConfig] = None
    ) -> Any:
        """
        LangChain Runnable interface
        
        Acepta:
        - str: prompt directo
        - Dict: {"text": "...", "image": "...", "audio": "..."}
        - List[BaseMessage]: chat history
        """
        if self.model is None and self.config.load_on_demand:
            self._load_model()
        
        return self._invoke_sync(input)
    
    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None):
        """Async invocation"""
        # TODO: Implementar versión async real
        return self.invoke(input, config)
    
    def unload(self):
        """Libera modelo de memoria"""
        if self.model is not None:
            del self.model
            self.model = None
            
            import gc
            gc.collect()


# ============================================================================
# Backend Implementations
# ============================================================================

class GGUFModelWrapper(UnifiedModelWrapper):
    """Wrapper para modelos GGUF (llama-cpp-python)"""
    
    def _load_model(self):
        from llama_cpp import Llama
        
        self.model = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_threads=6,
            use_mmap=True,
            use_mlock=False,
            verbose=False
        )
    
    def _invoke_sync(self, input: Any) -> str:
        if isinstance(input, str):
            prompt = input
        elif isinstance(input, dict):
            prompt = input.get("text", "")
        elif isinstance(input, list):  # Chat messages
            prompt = self._messages_to_prompt(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
        
        response = self.model(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return response["choices"][0]["text"]
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convierte chat messages a prompt plano"""
        parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"Assistant: {msg.content}")
        
        parts.append("Assistant:")
        return "\n\n".join(parts)


class TransformersModelWrapper(UnifiedModelWrapper):
    """Wrapper para modelos Transformers (GPU 4-bit)"""
    
    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.repo_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.repo_id)
    
    def _invoke_sync(self, input: Any) -> str:
        if isinstance(input, str):
            prompt = input
        else:
            prompt = str(input)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class OllamaModelWrapper(UnifiedModelWrapper):
    """Wrapper para modelos Ollama (API local)"""
    
    def _load_model(self):
        # Ollama no requiere carga explícita
        # El servidor Ollama gestiona modelos
        pass
    
    def _invoke_sync(self, input: Any) -> str:
        import requests
        
        if isinstance(input, str):
            prompt = input
        else:
            prompt = str(input)
        
        response = requests.post(
            f"{self.config.api_url}/api/generate",
            json={
                "model": self.config.name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
        )
        
        return response.json()["response"]


class MultimodalModelWrapper(UnifiedModelWrapper):
    """
    Wrapper para modelos multimodales (Qwen3-VL, Qwen2.5-Omni)
    
    Soporta:
    - Text + Image
    - Text + Audio
    - Text + Video (frames)
    """
    
    def _load_model(self):
        # Similar a TransformersModelWrapper pero con soporte multimodal
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.repo_id,
            load_in_4bit=True,
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(self.config.repo_id)
    
    def _invoke_sync(self, input: Dict) -> str:
        """
        Input esperado:
        {
            "text": "Describe esta imagen",
            "image": "/path/to/image.jpg",  # Optional
            "audio": "/path/to/audio.wav",  # Optional
            "video": "/path/to/video.mp4"   # Optional (frames)
        }
        """
        from PIL import Image
        
        text = input.get("text", "")
        image_path = input.get("image")
        
        if image_path:
            image = Image.open(image_path)
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            ).to(self.model.device)
        else:
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens
        )
        
        return self.processor.decode(outputs[0], skip_special_tokens=True)


class OpenAIAPIWrapper(UnifiedModelWrapper):
    """
    Wrapper para APIs compatibles OpenAI
    
    Permite usar:
    - OpenAI GPT-4, GPT-4V
    - Anthropic Claude (via compatibility layer)
    - Google Gemini (via compatibility layer)
    - Local LLMs con OpenAI API (llama.cpp server, vLLM, etc.)
    """
    
    def _load_model(self):
        # API no requiere carga
        pass
    
    def _invoke_sync(self, input: Any) -> str:
        import openai
        
        openai.api_key = self.config.api_key
        openai.api_base = self.config.api_url or "https://api.openai.com/v1"
        
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        elif isinstance(input, list):  # LangChain messages
            messages = [
                {"role": "user" if isinstance(msg, HumanMessage) else "assistant", 
                 "content": msg.content}
                for msg in input
            ]
        else:
            messages = [{"role": "user", "content": str(input)}]
        
        response = openai.ChatCompletion.create(
            model=self.config.name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content


# ============================================================================
# Model Registry & Factory
# ============================================================================

class ModelRegistry:
    """
    Registry centralizado de modelos
    
    Carga configuración desde YAML:
    - config/models.yaml: modelos disponibles
    - Permite agregar modelos sin tocar código
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        
        self.wrappers: Dict[str, UnifiedModelWrapper] = {}
    
    def get_model(self, name: str) -> UnifiedModelWrapper:
        """
        Obtiene modelo por nombre lógico
        
        Ejemplos:
        - "solar_short" → SOLAR GGUF n_ctx=512
        - "qwen3_vl" → Qwen3-VL multimodal
        - "gpt4_vision" → OpenAI GPT-4V (si API key configurada)
        """
        if name in self.wrappers:
            return self.wrappers[name]
        
        # Cargar config
        model_config_dict = self.configs.get(name)
        if not model_config_dict:
            raise ValueError(f"Modelo '{name}' no encontrado en config/models.yaml")
        
        config = ModelConfig(**model_config_dict)
        
        # Factory: Crear wrapper según backend
        wrapper = self._create_wrapper(config)
        
        # Cache si load_on_demand=False
        if not config.load_on_demand:
            wrapper._load_model()
        
        self.wrappers[name] = wrapper
        return wrapper
    
    def _create_wrapper(self, config: ModelConfig) -> UnifiedModelWrapper:
        """Factory de wrappers según backend"""
        
        if config.backend == "gguf":
            return GGUFModelWrapper(config)
        
        elif config.backend == "transformers":
            if config.type == "multimodal":
                return MultimodalModelWrapper(config)
            else:
                return TransformersModelWrapper(config)
        
        elif config.backend == "ollama":
            return OllamaModelWrapper(config)
        
        elif config.backend == "openai_api":
            return OpenAIAPIWrapper(config)
        
        else:
            raise ValueError(f"Backend no soportado: {config.backend}")
    
    def list_models(self) -> List[str]:
        """Lista todos los modelos disponibles"""
        return list(self.configs.keys())
    
    def unload_all(self):
        """Libera todos los modelos cargados"""
        for wrapper in self.wrappers.values():
            wrapper.unload()
        
        self.wrappers.clear()


# ============================================================================
# Singleton Factory
# ============================================================================

_model_registry: Optional[ModelRegistry] = None

def get_model_registry() -> ModelRegistry:
    """Singleton global de registry"""
    global _model_registry
    
    if _model_registry is None:
        _model_registry = ModelRegistry()
    
    return _model_registry


def get_model(name: str) -> UnifiedModelWrapper:
    """Shortcut para obtener modelo"""
    return get_model_registry().get_model(name)
```

---

## 📋 Configuración YAML: `config/models.yaml`

```yaml
# ============================================================================
# Text Models (GGUF CPU-optimized)
# ============================================================================

solar_short:
  name: "SOLAR-10.7B-Instruct"
  type: "text"
  backend: "gguf"
  model_path: "models/gguf/solar-10.7b-Q4_K_M.gguf"
  n_ctx: 512
  temperature: 0.7
  max_tokens: 256
  load_on_demand: false  # Siempre en memoria (core model)
  priority: 10

solar_long:
  name: "SOLAR-10.7B-Instruct"
  type: "text"
  backend: "gguf"
  model_path: "models/gguf/solar-10.7b-Q4_K_M.gguf"
  n_ctx: 2048
  temperature: 0.7
  max_tokens: 512
  load_on_demand: true
  ttl_seconds: 60
  priority: 9

lfm2:
  name: "LFM2-1.2B"
  type: "text"
  backend: "gguf"
  model_path: "models/gguf/lfm2-1.2b-Q4_K_M.gguf"
  n_ctx: 2048
  temperature: 0.9
  max_tokens: 256
  load_on_demand: false
  priority: 10

# ============================================================================
# Multimodal Models
# ============================================================================

qwen3_vl:
  name: "Qwen3-VL-4B"
  type: "multimodal"
  backend: "transformers"
  repo_id: "Qwen/Qwen3-VL-4B"
  n_ctx: 2048
  temperature: 0.8
  max_tokens: 300
  supports_images: true
  supports_video: true
  load_on_demand: true
  ttl_seconds: 120  # Más tiempo para multimodal
  priority: 7

qwen_omni:
  name: "Qwen2.5-Omni-7B"
  type: "multimodal"
  backend: "transformers"
  repo_id: "Qwen/Qwen2.5-Omni-7B"
  n_ctx: 2048
  temperature: 0.8
  max_tokens: 256
  supports_audio: true
  supports_images: true
  load_on_demand: true
  ttl_seconds: 90
  priority: 6

# ============================================================================
# Future: Cloud Models (cuando hardware mejore)
# ============================================================================

gpt4_vision:
  name: "gpt-4-vision-preview"
  type: "multimodal"
  backend: "openai_api"
  api_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"  # Leer de env var
  temperature: 0.7
  max_tokens: 500
  supports_images: true
  load_on_demand: true
  priority: 5

claude_opus:
  name: "claude-3-opus-20240229"
  type: "text"
  backend: "openai_api"  # Via compatibility layer
  api_url: "https://api.anthropic.com/v1"  # Requires adapter
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.8
  max_tokens: 1024
  load_on_demand: true
  priority: 4

# ============================================================================
# Local Alternatives (Ollama)
# ============================================================================

ollama_llama3:
  name: "llama3:8b"
  type: "text"
  backend: "ollama"
  api_url: "http://localhost:11434"
  temperature: 0.8
  max_tokens: 512
  load_on_demand: true
  priority: 3
```

---

## 🔗 Integración LangChain: Composición de Pipelines

```python
# core/langchain_pipelines.py
"""
Pipelines SARAi usando LangChain Expression Language (LCEL)

Ventajas:
- Composición declarativa con | operator
- Streaming automático
- Retry/fallback integrados
- Async nativo
"""

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.unified_model_wrapper import get_model


# ============================================================================
# Pipeline 1: Text Generation Simple
# ============================================================================

def create_text_pipeline(model_name: str = "solar_short"):
    """
    Pipeline básico: prompt → model → output
    
    Uso:
        pipeline = create_text_pipeline()
        response = pipeline.invoke("¿Qué es Python?")
    """
    model = get_model(model_name)
    
    return model | StrOutputParser()


# ============================================================================
# Pipeline 2: Vision Analysis
# ============================================================================

def create_vision_pipeline():
    """
    Pipeline multimodal: image + text → Qwen3-VL → analysis
    
    Uso:
        pipeline = create_vision_pipeline()
        result = pipeline.invoke({
            "text": "¿Qué se muestra en esta imagen?",
            "image": "/path/to/image.jpg"
        })
    """
    qwen3_vl = get_model("qwen3_vl")
    
    return qwen3_vl | StrOutputParser()


# ============================================================================
# Pipeline 3: Hybrid (Text + Vision Fallback)
# ============================================================================

def create_hybrid_pipeline_with_fallback():
    """
    Pipeline con fallback: Intenta vision, si falla usa text-only
    
    Uso:
        pipeline = create_hybrid_pipeline_with_fallback()
        result = pipeline.invoke({
            "text": "Explica React",
            "image": None  # Si None, usa fallback
        })
    """
    from langchain_core.runnables import RunnableBranch
    
    qwen3_vl = get_model("qwen3_vl")
    solar = get_model("solar_long")
    
    # Routing condicional
    def has_image(input_dict):
        return input_dict.get("image") is not None
    
    pipeline = RunnableBranch(
        (has_image, qwen3_vl),  # Si hay imagen → vision
        solar  # Else → text-only
    ) | StrOutputParser()
    
    return pipeline


# ============================================================================
# Pipeline 4: Video Conference (Multi-step)
# ============================================================================

def create_video_conference_pipeline():
    """
    Pipeline complejo:
    1. Frames → Qwen3-VL → visual_context
    2. Audio → STT → transcript
    3. (visual_context + transcript) → SOLAR → summary
    
    Uso:
        pipeline = create_video_conference_pipeline()
        summary = pipeline.invoke({
            "frames": [frame1, frame2, ...],
            "audio": audio_bytes
        })
    """
    from langchain_core.runnables import RunnableParallel
    
    qwen3_vl = get_model("qwen3_vl")
    solar = get_model("solar_long")
    
    # Step 1: Procesamiento paralelo de frames y audio
    parallel_processing = RunnableParallel(
        visual_context=RunnableLambda(lambda x: process_frames_with_qwen(x["frames"], qwen3_vl)),
        transcript=RunnableLambda(lambda x: transcribe_audio(x["audio"]))
    )
    
    # Step 2: Síntesis con SOLAR
    synthesis_prompt = ChatPromptTemplate.from_template("""
    Genera resumen ejecutivo de esta reunión:
    
    Contexto visual: {visual_context}
    Transcripción: {transcript}
    
    Estructura:
    1. Tema principal
    2. Decisiones clave
    3. Action items
    """)
    
    pipeline = (
        parallel_processing 
        | synthesis_prompt 
        | solar 
        | StrOutputParser()
    )
    
    return pipeline


def process_frames_with_qwen(frames, model):
    """Helper: Analiza frames con Qwen3-VL"""
    analyses = []
    for frame in frames[:3]:  # Solo 3 frames clave
        result = model.invoke({
            "text": "Describe esta escena brevemente",
            "image": frame
        })
        analyses.append(result)
    
    return " | ".join(analyses)


def transcribe_audio(audio_bytes):
    """Helper: STT con Vosk"""
    # TODO: Implementar con Vosk
    return "[Transcripción placeholder]"
```

---

## ✅ Beneficios de esta Arquitectura

| Aspecto | Antes | Con Unified Wrapper |
|---------|-------|---------------------|
| **Agregar modelo nuevo** | Refactorizar `model_pool.py` + agents | Solo agregar entry en `models.yaml` |
| **Cambiar backend** | Modificar código Python | Solo cambiar `backend:` en YAML |
| **LangChain integration** | Manual, imperativo | Nativo (Runnable interface) |
| **Composición pipelines** | Hard-coded | LCEL (declarativo con \|) |
| **Fallback logic** | Try-except manual | `RunnableBranch` integrado |
| **Async support** | Thread hacks | `ainvoke()` nativo |
| **Streaming** | Complejo | Automático (LangChain) |
| **Future models** | Cambios masivos | Config-driven (0 código) |

---

## 🚀 Evolutibilidad: Casos de Uso Futuros

### Caso 1: Agregar GPT-4 Vision (cuando tengas API key)

**Código necesario**: 0 líneas  
**Configuración**: 6 líneas YAML

```yaml
# config/models.yaml
gpt4_vision:
  name: "gpt-4-vision-preview"
  backend: "openai_api"
  api_key: "${OPENAI_API_KEY}"
  supports_images: true
```

Uso inmediato:
```python
gpt4 = get_model("gpt4_vision")
response = gpt4.invoke({"text": "Analiza", "image": "img.jpg"})
```

### Caso 2: Migrar a GPU cuando mejore hardware

**Cambio en config/models.yaml**:
```yaml
# ANTES (CPU)
solar_short:
  backend: "gguf"
  model_path: "models/gguf/solar.gguf"

# DESPUÉS (GPU)
solar_short:
  backend: "transformers"
  repo_id: "upstage/SOLAR-10.7B-Instruct-v1.0"
```

**Código Python**: SIN CAMBIOS (interfaz unificada)

### Caso 3: Video conferencia con Gemini Pro Vision

```yaml
# Agregar en models.yaml
gemini_pro_vision:
  name: "gemini-pro-vision"
  backend: "openai_api"  # Via compatibility adapter
  api_url: "https://generativelanguage.googleapis.com/v1"
  api_key: "${GOOGLE_API_KEY}"
  supports_images: true
```

```python
# Pipeline sin cambios, solo swap de modelo
pipeline = create_video_conference_pipeline()
# Internamente usa gemini_pro_vision en lugar de qwen3_vl
```

---

## 📊 Plan de Implementación

### Fase 1: Core Wrapper (4h)
- [ ] `core/unified_model_wrapper.py` (500 LOC)
  * Clase base `UnifiedModelWrapper`
  * Implementar `GGUFModelWrapper`
  * Implementar `MultimodalModelWrapper`
  * Implementar `ModelRegistry`
  
- [ ] `config/models.yaml` (100 LOC)
  * Definir modelos actuales (SOLAR, LFM2, Qwen3-VL)
  
- [ ] Tests `tests/test_unified_wrapper.py` (200 LOC)
  * Test registry carga modelos
  * Test GGUF wrapper funciona
  * Test multimodal wrapper (mock images)

### Fase 2: LangChain Pipelines (3h)
- [ ] `core/langchain_pipelines.py` (300 LOC)
  * `create_text_pipeline()`
  * `create_vision_pipeline()`
  * `create_hybrid_pipeline_with_fallback()`
  * `create_video_conference_pipeline()`

- [ ] Tests `tests/test_pipelines.py` (150 LOC)

### Fase 3: Integración Graph (2h)
- [ ] Refactorizar `core/graph.py` para usar pipelines
- [ ] Nodos LangChain en lugar de funciones imperativas
- [ ] Migrar `model_pool` a `ModelRegistry`

### Fase 4: Documentación (1h)
- [ ] README: "Agregar nuevos modelos"
- [ ] Ejemplos de pipelines LCEL

**Total**: ~10 horas

---

**Mantra v2.14 Universal**:  
_"Un wrapper para gobernarlos a todos. Configuración > Código.  
LangChain orquesta, YAML define, el futuro es plug & play."_
