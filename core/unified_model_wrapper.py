"""
Unified Model Wrapper - Abstracción Universal para Modelos de SARAi v2.14

Este módulo implementa una capa de abstracción que permite usar cualquier modelo
(GGUF, Transformers, Multimodal, APIs) con una interfaz unificada basada en LangChain.

Filosofía:
    "SARAi no debe conocer sus modelos. Solo debe invocar capacidades.
    YAML define, LangChain orquesta, el wrapper abstrae.
    Cuando el hardware mejore, solo cambiamos configuración, nunca código."

Características:
    - LangChain Runnable interface (LCEL compatible)
    - Backend-agnostic (GGUF, Transformers, Ollama, OpenAI API)
    - Config-driven (models.yaml)
    - Async nativo
    - Streaming support
    - Fallback automático

Backends soportados:
    - GGUF: llama-cpp-python (CPU optimizado)
    - Transformers: HuggingFace con 4-bit quantization (GPU)
    - Multimodal: Qwen3-VL (vision)
    - Ollama: API local
    - OpenAI API: GPT-4, Claude, Gemini (cloud)

Autor: SARAi v2.14
Fecha: 1 Noviembre 2025
"""

import os
import gc
import time
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, Iterator, Protocol
from pathlib import Path
import yaml

# LangChain imports
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# Type hints
InputType = Union[str, Dict[str, Any], List[BaseMessage]]
OutputType = Union[str, Dict[str, Any]]

logger = logging.getLogger(__name__)


# ============================================================================
# BASE CLASS - Unified Model Wrapper
# ============================================================================

class UnifiedModelWrapper(Runnable):
    """
    Clase base abstracta para TODOS los modelos de SARAi.
    
    Implementa la interfaz LangChain Runnable para compatibilidad completa
    con LCEL (LangChain Expression Language).
    
    Atributos:
        name (str): Nombre del modelo (registry key)
        model_type (str): Tipo de modelo (text, audio, vision, multimodal)
        backend (str): Backend usado (gguf, transformers, etc.)
        config (dict): Configuración completa del modelo
        model (Any): Instancia del modelo cargado
        is_loaded (bool): Estado de carga del modelo
        last_access (float): Timestamp del último acceso (para TTL)
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Inicializa el wrapper (sin cargar modelo aún).
        
        Args:
            name: Nombre del modelo (ej: "solar_short")
            config: Configuración del modelo desde models.yaml
        """
        self.name = name
        self.model_type = config.get("type", "text")
        self.backend = config["backend"]
        self.config = config
        self.model = None
        self.is_loaded = False
        self.last_access = 0.0
        
        logger.info(f"Initialized wrapper for {name} (backend: {self.backend})")
    
    # ------------------------------------------------------------------------
    # LangChain Runnable Interface
    # ------------------------------------------------------------------------
    
    def invoke(self, input: InputType, config: Optional[Dict] = None) -> OutputType:
        """
        Ejecuta el modelo de forma síncrona (interfaz LangChain).
        
        Args:
            input: Texto, dict con multimodal data, o mensajes LangChain
            config: Configuración opcional de runtime
            
        Returns:
            Respuesta del modelo (str o dict)
        """
        self._ensure_loaded()
        self.last_access = time.time()
        
        try:
            return self._invoke_sync(input, config)
        except Exception as e:
            logger.error(f"Error invoking {self.name}: {e}")
            raise
    
    async def ainvoke(self, input: InputType, config: Optional[Dict] = None) -> OutputType:
        """
        Ejecuta el modelo de forma asíncrona (interfaz LangChain).
        
        Por defecto llama a invoke() en thread pool.
        Backends específicos pueden override para async nativo.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, input, config)
    
    def stream(self, input: InputType, config: Optional[Dict] = None) -> Iterator[str]:
        """
        Stream de tokens (si el backend lo soporta).
        
        Por defecto retorna respuesta completa.
        Backends específicos pueden override.
        """
        response = self.invoke(input, config)
        yield response
    
    # ------------------------------------------------------------------------
    # Model Lifecycle Management
    # ------------------------------------------------------------------------
    
    def _ensure_loaded(self) -> None:
        """Carga el modelo si no está en memoria."""
        if not self.is_loaded:
            logger.info(f"Loading model {self.name}...")
            self.model = self._load_model()
            self.is_loaded = True
            logger.info(f"Model {self.name} loaded successfully")
    
    def unload(self) -> None:
        """
        Descarga el modelo de memoria explícitamente.
        
        Útil para gestión manual de RAM.
        """
        if self.is_loaded:
            logger.info(f"Unloading model {self.name}...")
            del self.model
            self.model = None
            self.is_loaded = False
            gc.collect()
            logger.info(f"Model {self.name} unloaded")
    
    # ------------------------------------------------------------------------
    # Abstract Methods (implementados por backends)
    # ------------------------------------------------------------------------
    
    @abstractmethod
    def _load_model(self) -> Any:
        """
        Carga el modelo específico del backend.
        
        Returns:
            Instancia del modelo cargado
        """
        pass
    
    @abstractmethod
    def _invoke_sync(self, input: InputType, config: Optional[Dict] = None) -> OutputType:
        """
        Invocación específica del backend.
        
        Args:
            input: Input procesado
            config: Config de runtime
            
        Returns:
            Respuesta del modelo
        """
        pass


# ============================================================================
# BACKEND 1: GGUF Model Wrapper (llama-cpp-python)
# ============================================================================

class GGUFModelWrapper(UnifiedModelWrapper):
    """
    Wrapper para modelos GGUF usando llama-cpp-python.
    
    Optimizado para CPU con cuantización Q4_K_M.
    Usado por: SOLAR, LFM2
    
    Config esperada:
        model_path: Ruta al archivo .gguf
        n_ctx: Longitud de contexto
        n_threads: Threads de CPU
        temperature: (opcional) Default 0.7
    """
    
    def _load_model(self) -> Any:
        """Carga modelo GGUF con llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
        
        model_path = self.config["model_path"]
        
        # Validar que el archivo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found: {model_path}")
        
        # Configuración de carga
        n_ctx = self.config.get("n_ctx", 2048)
        n_threads = self.config.get("n_threads", os.cpu_count() - 2)
        use_mmap = self.config.get("use_mmap", True)
        use_mlock = self.config.get("use_mlock", False)
        
        logger.info(f"Loading GGUF model from {model_path}")
        logger.info(f"Config: n_ctx={n_ctx}, n_threads={n_threads}")
        
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            verbose=False
        )
    
    def _invoke_sync(self, input: InputType, config: Optional[Dict] = None) -> str:
        """Genera texto con GGUF."""
        # Convertir input a string
        if isinstance(input, list):
            # LangChain messages
            prompt = "\n".join([msg.content for msg in input if hasattr(msg, 'content')])
        elif isinstance(input, dict):
            prompt = input.get("text", str(input))
        else:
            prompt = str(input)
        
        # Parámetros de generación
        temperature = config.get("temperature") if config else None
        temperature = temperature or self.config.get("temperature", 0.7)
        
        max_tokens = config.get("max_tokens") if config else None
        max_tokens = max_tokens or self.config.get("max_tokens", 512)
        
        # Generar
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "Human:", "User:"],  # Stop sequences comunes
            echo=False
        )
        
        return response["choices"][0]["text"].strip()


# ============================================================================
# BACKEND 2: Transformers Model Wrapper (HuggingFace)
# ============================================================================

class TransformersModelWrapper(UnifiedModelWrapper):
    """
    Wrapper para modelos HuggingFace con cuantización 4-bit.
    
    Usado cuando hay GPU disponible.
    Futuro: SOLAR, LFM2 en GPU
    
    Config esperada:
        repo_id: HuggingFace repo (ej: "upstage/SOLAR-10.7B-Instruct-v1.0")
        load_in_4bit: (opcional) Default True
        device_map: (opcional) Default "auto"
    """
    
    def _load_model(self) -> Any:
        """Carga modelo con Transformers + 4-bit quantization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers"
            )
        
        repo_id = self.config["repo_id"]
        load_in_4bit = self.config.get("load_in_4bit", True)
        device_map = self.config.get("device_map", "auto")
        
        logger.info(f"Loading Transformers model: {repo_id}")
        logger.info(f"Quantization: 4-bit={load_in_4bit}, device_map={device_map}")
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
        # Cargar modelo
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
            trust_remote_code=True
        )
        
        return model
    
    def _invoke_sync(self, input: InputType, config: Optional[Dict] = None) -> str:
        """Genera texto con Transformers."""
        # Convertir input a string
        if isinstance(input, list):
            prompt = "\n".join([msg.content for msg in input if hasattr(msg, 'content')])
        elif isinstance(input, dict):
            prompt = input.get("text", str(input))
        else:
            prompt = str(input)
        
        # Tokenizar
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Mover a device del modelo
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Parámetros de generación
        temperature = config.get("temperature") if config else None
        temperature = temperature or self.config.get("temperature", 0.7)
        
        max_tokens = config.get("max_tokens") if config else None
        max_tokens = max_tokens or self.config.get("max_tokens", 512)
        
        # Generar
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decodificar
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remover prompt del output
        response = response[len(prompt):].strip()
        
        return response


# ============================================================================
# BACKEND 3: Multimodal Model Wrapper (Vision + Audio)
# ============================================================================

class MultimodalModelWrapper(UnifiedModelWrapper):
    """
    Wrapper para modelos multimodales (Qwen3-VL).
    
    Soporta:
        - Texto + Imágenes
        - Texto + Audio
        - Texto + Video (frames)
    
    Config esperada:
        repo_id: HuggingFace repo
        supports_images: bool
        supports_audio: bool
        supports_video: bool
    """
    
    def _load_model(self) -> Any:
        """Carga modelo multimodal."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers"
            )
        
        repo_id = self.config["repo_id"]
        
        logger.info(f"Loading Multimodal model: {repo_id}")
        
        # Cargar processor (tokenizer + image processor)
        self.processor = AutoProcessor.from_pretrained(repo_id)
        
        # Cargar modelo
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model
    
    def _invoke_sync(self, input: InputType, config: Optional[Dict] = None) -> str:
        """
        Genera respuesta multimodal.
        
        Input esperado (dict):
            {
                "text": str,
                "image": str | Path | List[str],  # Opcional
                "audio": bytes | str,  # Opcional
                "video": List[str]  # Opcional (frames)
            }
        """
        if not isinstance(input, dict):
            # Fallback a texto simple
            input = {"text": str(input)}
        
        text = input.get("text", "")
        
        # Procesar imagen si presente
        if "image" in input and self.config.get("supports_images"):
            return self._process_with_image(text, input["image"], config)
        
        # Procesar audio si presente
        elif "audio" in input and self.config.get("supports_audio"):
            return self._process_with_audio(text, input["audio"], config)
        
        # Procesar video si presente
        elif "video" in input and self.config.get("supports_video"):
            return self._process_with_video(text, input["video"], config)
        
        # Solo texto
        else:
            return self._process_text_only(text, config)
    
    def _process_with_image(self, text: str, image_path: Union[str, List[str]], 
                           config: Optional[Dict] = None) -> str:
        """Procesa texto + imagen(es)."""
        from PIL import Image
        
        # Cargar imagen(es)
        if isinstance(image_path, list):
            images = [Image.open(img) for img in image_path]
        else:
            images = [Image.open(image_path)]
        
        # Procesar con processor multimodal
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt"
        )
        
        # Mover a device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generar
        max_tokens = config.get("max_tokens", 512) if config else 512
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens
        )
        
        # Decodificar
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        response = response[len(text):].strip()
        
        return response
    
    def _process_with_audio(self, text: str, audio_data: Union[bytes, str],
                           config: Optional[Dict] = None) -> str:
    """Procesa texto + audio (STT + análisis)."""
    # Placeholder: soporte de audio pendiente de implementación
        logger.warning("Audio processing not fully implemented yet")
        return self._process_text_only(text, config)
    
    def _process_with_video(self, text: str, frames: List[str],
                           config: Optional[Dict] = None) -> str:
        """Procesa texto + video (frames)."""
        # Video = secuencia de imágenes
        return self._process_with_image(text, frames, config)
    
    def _process_text_only(self, text: str, config: Optional[Dict] = None) -> str:
        """Fallback a texto simple."""
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        max_tokens = config.get("max_tokens", 512) if config else 512
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return response[len(text):].strip()


# ============================================================================
# BACKEND 4: Ollama Model Wrapper (API Local)
# ============================================================================

class OllamaModelWrapper(UnifiedModelWrapper):
    """
    Wrapper para Ollama API local.
    
    Permite usar modelos locales vía API REST sin cargar en memoria Python.
    Útil para Llama3 70B, Mixtral, etc.
    
    Config esperada:
        api_url: URL de Ollama (ej: "http://localhost:11434")
        model_name: Nombre del modelo en Ollama (ej: "llama3:70b")
    """
    
    def _load_model(self) -> None:
        """
        Ollama no requiere carga en memoria Python.
        Solo verificamos que el servidor esté disponible y resolvemos
        variables de entorno críticas (api_url, model_name).
        """
        import requests
        import re

        def resolve_env(value: Optional[str], *, default: Optional[str] = None, label: str = "value") -> str:
            """Resuelve ${VAR} usando variables de entorno."""
            if not value:
                return default if default is not None else ""

            pattern = re.compile(r"\$\{([^}]+)\}")

            def replace(match: re.Match) -> str:
                env_var = match.group(1)
                env_value = os.getenv(env_var)
                if env_value is None:
                    logger.warning(
                        "Environment variable %s referenced in %s but not set",
                        env_var,
                        label,
                    )
                    return match.group(0)
                return env_value

            resolved = pattern.sub(replace, value)

            if "${" in resolved:
                # Sustituir por default si queda sin resolver
                if default is not None:
                    logger.warning(
                        "Environment variable unresolved in %s: %s. Using default %s",
                        label,
                        resolved,
                        default,
                    )
                    return default
            return resolved

        # Resolver API URL y asegurar formato correcto (sin hardcodear IPs)
        env_api_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_URL")
        raw_api_url = self.config.get("api_url") or env_api_url or "http://localhost:11434"
        api_url = resolve_env(
            raw_api_url,
            default=env_api_url or "http://localhost:11434",
            label="api_url",
        )
        api_url = api_url.rstrip("/") if api_url else "http://localhost:11434"

        try:
            response = requests.get(f"{api_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Ollama server available at {api_url}")
        except Exception as e:
            raise ConnectionError(f"Ollama server not available: {e}")

        tags_payload = response.json() if response.content else {}
        available_models = [
            model.get("name")
            for model in tags_payload.get("models", [])
            if isinstance(model, dict) and model.get("name")
        ]

        # Resolver model_name con fallback al primer modelo disponible
        raw_model_name = self.config.get("model_name")
        default_model = available_models[0] if available_models else ""
        model_name = resolve_env(raw_model_name, default=default_model, label="model_name")

        if not model_name:
            raise ValueError(
                "No Ollama model available. Configure 'model_name' or ensure the server has at least one model."
            )

        if available_models and model_name not in available_models:
            logger.warning(
                "Requested Ollama model '%s' not found. Available: %s. Using '%s' instead.",
                model_name,
                ", ".join(available_models),
                available_models[0],
            )
            model_name = available_models[0]

        # Guardar valores resueltos para reutilizar en invoke
        self._resolved_api_url = api_url
        self._resolved_model_name = model_name
        self._available_ollama_models = available_models

        return None  # No model object
    
    def _invoke_sync(self, input: InputType, config: Optional[Dict] = None) -> str:
        """Genera texto vía Ollama API."""
        import requests
        
        # Convertir input a string
        if isinstance(input, list):
            prompt = "\n".join([msg.content for msg in input if hasattr(msg, 'content')])
        elif isinstance(input, dict):
            prompt = input.get("text", str(input))
        else:
            prompt = str(input)
        
        # Usar valores resueltos (de _load_model); si no existen, forzar carga
        if not hasattr(self, "_resolved_api_url") or not hasattr(self, "_resolved_model_name"):
            # Carga perezosa segura
            self._ensure_loaded()

        env_api_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_URL")
        api_url = getattr(
            self,
            "_resolved_api_url",
            self.config.get("api_url") or env_api_url or "http://localhost:11434",
        )
        model_name = getattr(self, "_resolved_model_name", self.config.get("model_name"))
        temperature = config.get("temperature", 0.7) if config else 0.7
        
        # Request
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        response = requests.post(
            f"{api_url}/api/generate",
            json=payload,
            timeout=300  # 5 min timeout
        )
        response.raise_for_status()
        
        return response.json()["response"]


# ============================================================================
# BACKEND 5: OpenAI API Wrapper (GPT-4, Claude, Gemini)
# ============================================================================

class OpenAIAPIWrapper(UnifiedModelWrapper):
    """
    Wrapper para APIs compatibles con OpenAI (cloud).
    
    Soporta:
        - OpenAI (GPT-4, GPT-4V)
        - Anthropic Claude (vía proxy OpenAI-compatible)
        - Google Gemini (vía proxy OpenAI-compatible)
    
    Config esperada:
        api_key: API key (o ${ENV_VAR})
        api_url: Base URL (ej: "https://api.openai.com/v1")
        model_name: Nombre del modelo (ej: "gpt-4-vision-preview")
    """
    
    def _load_model(self) -> None:
        """
        APIs no requieren carga.
        Solo validamos API key.
        """
        api_key = self.config.get("api_key")
        
        # Resolver variable de entorno si es ${VAR}
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var)
            
            if not api_key:
                raise ValueError(f"API key environment variable not set: {env_var}")
        
        self.api_key = api_key
        
        if not self.api_key:
            logger.warning(f"No API key configured for {self.name}")
        
        return None  # No model object
    
    def _invoke_sync(self, input: InputType, config: Optional[Dict] = None) -> str:
        """Genera texto vía OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install openai"
            )
        
        # Configurar cliente
        api_url = self.config.get("api_url", "https://api.openai.com/v1")
        model_name = self.config["model_name"]
        
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=api_url
        )
        
        # Convertir input a messages
        if isinstance(input, list):
            messages = [
                {"role": "user" if isinstance(msg, HumanMessage) else "assistant", 
                 "content": msg.content}
                for msg in input
            ]
        elif isinstance(input, dict):
            # Multimodal (imagen)
            if "image" in input:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input["text"]},
                        {"type": "image_url", "image_url": {"url": input["image"]}}
                    ]
                }]
            else:
                messages = [{"role": "user", "content": input.get("text", str(input))}]
        else:
            messages = [{"role": "user", "content": str(input)}]
        
        # Parámetros
        temperature = config.get("temperature", 0.7) if config else 0.7
        max_tokens = config.get("max_tokens", 1024) if config else 1024
        
        # Request
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content


# ============================================================================
# BACKEND 6: Embedding Model Wrapper (EmbeddingGemma-300M)
# ============================================================================

class EmbeddingModelWrapper(UnifiedModelWrapper):
    """
    Wrapper para modelos de embeddings (vector representations).
    
    Soporta:
        - EmbeddingGemma-300M (Google, 768-dim)
        - Otros modelos de embeddings HuggingFace
    
    CRÍTICO: Este wrapper retorna VECTORES (np.ndarray), no texto.
    
    Config esperada:
        repo_id: HuggingFace repo (ej: "google/embeddinggemma-300m-qat-q4_0-unquantized")
        embedding_dim: Dimensión del vector (ej: 768)
        quantization: "4bit" | "8bit" | "fp16"
        device: "cpu" | "cuda"
        cache_dir: Directorio de cache
    
    API:
        invoke(text: str) -> np.ndarray  # Retorna vector 1D
        invoke(texts: List[str]) -> List[np.ndarray]  # Batch processing
    """
    
    def _load_model(self) -> Any:
        """
        Carga modelo de embeddings desde HuggingFace.
        
        Usa SentenceTransformers para simplicidad y compatibilidad.
        """
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        repo_id = self.config.get("repo_id") or self.config.get("source")
        cache_dir = self.config.get("cache_dir", "models/cache/embeddings")
        device_str = self.config.get("device", "cpu")

        if not repo_id:
            raise ValueError(f"Embedding model {self.name} missing 'repo_id' or 'source'")

        device = torch.device(device_str)
        dtype = torch.float16 if device.type != "cpu" else torch.float32

        logger.info(f"🔄 Loading embedding model: {repo_id} on {device_str}")

        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            cache_dir=cache_dir
        )

        model = AutoModel.from_pretrained(
            repo_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map="cpu" if device.type == "cpu" else "auto",
            low_cpu_mem_usage=True
        )

        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._device = device
        self._dtype = dtype
        self._normalize = self.config.get("normalize", True)
        self._max_length = self.config.get("max_length", 512)

        expected_dim = self.config.get("embedding_dim", 768)
        self._embedding_dim = expected_dim

        # Probar embedding rápido para validar dimensión
        with torch.no_grad():
            sample_inputs = tokenizer(
                ["dummy"],
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt"
            )
            sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}
            outputs = model(**sample_inputs)
            sample_embedding = outputs.last_hidden_state.mean(dim=1)
            sample_dim = sample_embedding.shape[-1]

        if sample_dim != expected_dim:
            logger.warning(
                "Embedding dimension mismatch: expected %s, got %s",
                expected_dim,
                sample_dim,
            )
            self._embedding_dim = sample_dim

        logger.info(
            "✅ Embedding model loaded: %s-dim vectors (normalize=%s)",
            self._embedding_dim,
            self._normalize,
        )

        return model
    
    def _invoke_sync(self, input: InputType, config: Optional[Dict] = None) -> Any:
        """
        Genera embeddings para texto(s).
        
        Args:
            input: str o List[str]
            config: Configuración opcional (no usado en embeddings)
        
        Returns:
            np.ndarray (1D si input es str)
            List[np.ndarray] (si input es List[str])
        """
        import numpy as np
        import torch
        
        # Normalizar input
        if isinstance(input, str):
            texts = [input]
            single_input = True
        elif isinstance(input, list):
            # Convertir LangChain messages a strings
            if input and hasattr(input[0], 'content'):
                texts = [msg.content for msg in input]
            else:
                texts = [str(item) for item in input]
            single_input = False
        else:
            texts = [str(input)]
            single_input = True
        
        # Generar embeddings
        tokenizer = self._tokenizer
        device = self._device
        max_length = self._max_length

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings_tensor = outputs.last_hidden_state.mean(dim=1)

            if self._normalize:
                embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)

        embeddings = embeddings_tensor.cpu().numpy()
        
        # Retornar según input
        if single_input:
            return embeddings[0]  # 1D array
        else:
            return list(embeddings)  # List of 1D arrays
    
    def get_embedding(self, text: str) -> 'np.ndarray':
        """
        Método de conveniencia para obtener embedding de un texto.
        
        Alias de invoke() para claridad semántica.
        """
        return self.invoke(text)
    
    def batch_encode(self, texts: List[str]) -> 'np.ndarray':
        """
        Procesa batch de textos y retorna matriz 2D.
        
        Returns:
            np.ndarray con shape (len(texts), embedding_dim)
        """
        import numpy as np
        embeddings = self.invoke(texts)
        return np.array(embeddings)


# ============================================================================
# MODEL REGISTRY - Factory Pattern + YAML Config
# ============================================================================

class ModelRegistry:
    """
    Registry centralizado de modelos.
    
    Carga configuraciones desde config/models.yaml y crea wrappers
    según el backend especificado.
    
    Factory pattern: ModelRegistry.get_model("solar_short") → GGUFModelWrapper
    
    Características:
        - Singleton pattern (una instancia global)
        - Lazy loading (modelos se cargan bajo demanda)
        - Cache automático (misma instancia si ya cargado)
        - Config-driven (YAML > código)
    """
    
    _instance = None
    _models: Dict[str, Any] = {}  # Cambio: Any en lugar de UnifiedModelWrapper
    _config: Optional[Dict] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load_config(cls, config_path: str = "config/models.yaml") -> None:
        """
        Carga configuración de modelos desde YAML.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Models config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            cls._config = yaml.safe_load(f)
        
        logger.info(f"Loaded {len(cls._config)} model configurations")
    
    @classmethod
    def get_model(cls, name: str) -> UnifiedModelWrapper:
        """
        Obtiene o crea wrapper para el modelo especificado.
        
        Args:
            name: Nombre del modelo (key en YAML)
            
        Returns:
            Wrapper del modelo (ya instanciado)
            
        Raises:
            ValueError: Si el modelo no existe en config
        """
        # Lazy load config si no está cargado
        if cls._config is None:
            cls.load_config()
        
        # Retornar del cache si ya existe
        if name in cls._models:
            logger.debug(f"Model {name} retrieved from cache")
            return cls._models[name]
        
        # Validar que existe en config
        if name not in cls._config:
            available = ", ".join(cls._config.keys())
            raise ValueError(
                f"Model '{name}' not found in config. "
                f"Available models: {available}"
            )
        
        # Crear wrapper según backend
        config = cls._config[name]
        backend = config["backend"]
        
        logger.info(f"Creating wrapper for {name} (backend: {backend})")
        
        if backend == "gguf":
            wrapper = GGUFModelWrapper(name, config)
        elif backend == "transformers":
            wrapper = TransformersModelWrapper(name, config)
        elif backend == "multimodal":
            wrapper = MultimodalModelWrapper(name, config)
        elif backend == "ollama":
            wrapper = OllamaModelWrapper(name, config)
        elif backend == "openai_api":
            wrapper = OpenAIAPIWrapper(name, config)
        elif backend == "embedding":  # NEW v2.14
            wrapper = EmbeddingModelWrapper(name, config)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        
        # Cachear
        cls._models[name] = wrapper
        
        return wrapper
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        Lista todos los modelos disponibles en config.
        
        Returns:
            Lista de nombres de modelos
        """
        if cls._config is None:
            cls.load_config()
        
        return list(cls._config.keys())
    
    @classmethod
    def unload_model(cls, name: str) -> None:
        """
        Descarga un modelo específico de memoria.
        
        Args:
            name: Nombre del modelo
        """
        if name in cls._models:
            cls._models[name].unload()
            del cls._models[name]
            logger.info(f"Model {name} unloaded from registry")
    
    @classmethod
    def unload_all(cls) -> None:
        """Descarga TODOS los modelos de memoria."""
        for name in list(cls._models.keys()):
            cls.unload_model(name)
        
        logger.info("All models unloaded from registry")
    
    @classmethod
    def get_loaded_models(cls) -> List[str]:
        """
        Retorna lista de modelos actualmente cargados.
        
        Returns:
            Lista de nombres de modelos cargados
        """
        return [
            name for name, wrapper in cls._models.items()
            if wrapper.is_loaded
        ]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_model(name: str) -> UnifiedModelWrapper:
    """
    Función de conveniencia para obtener modelo.
    
    Alias de ModelRegistry.get_model()
    
    Args:
        name: Nombre del modelo
        
    Returns:
        Wrapper del modelo
        
    Example:
        >>> solar = get_model("solar_short")
        >>> response = solar.invoke("¿Qué es Python?")
    """
    return ModelRegistry.get_model(name)


def list_available_models() -> List[str]:
    """
    Lista modelos disponibles.
    
    Alias de ModelRegistry.list_models()
    """
    return ModelRegistry.list_models()


# ============================================================================
# MODULE-LEVEL INITIALIZATION
# ============================================================================

# Intentar cargar config automáticamente al importar
try:
    ModelRegistry.load_config()
except FileNotFoundError:
    logger.warning(
        "config/models.yaml not found. "
        "Models will not be available until config is loaded."
    )


if __name__ == "__main__":
    # Demo básico
    print("🎯 Unified Model Wrapper v2.14")
    print("\nModelos disponibles:")
    
    try:
        for model_name in list_available_models():
            print(f"  - {model_name}")
    except:
        print("  (No config loaded)")
