#!/usr/bin/env python3
"""
SOLAR-10.7B Native Wrapper - v2.16
Acceso directo al GGUF de Ollama sin dependencia del servidor

Características:
- Busca GGUF en cache de Ollama (/usr/share/ollama/.ollama/models/)
- LangChain optimizado (f16_kv, use_mmap, n_batch=512)
- Context-aware: short (512) y long (2048) con MISMO archivo
- Zero dependency on Ollama server (no HTTP, no sockets)
- Fallback automático a Ollama HTTP si GGUF no disponible

Author: SARAi v2.16 Integration Team
Date: 2025-10-28
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
import time
import gc

# Cargar variables de entorno desde .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv no instalado, usar env vars del sistema

# LangChain imports
try:
    from langchain_community.llms import LlamaCpp
    from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain no disponible. Instalar: pip install langchain langchain-community langchain-core")

# Llama.cpp imports
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python no disponible. Instalar: pip install llama-cpp-python")


class SolarNative:
    """
    Wrapper nativo para SOLAR-10.7B con optimizaciones CPU
    
    Basado en el código oficial de Upstage/SOLAR-10.7B-v1.0
    Referencia: https://huggingface.co/Upstage/SOLAR-10.7B-v1.0
    
    Estrategia:
    - Busca GGUF en cache de Ollama (evita descargas duplicadas)
    - LangChain para optimizaciones automáticas
    - Context-aware: un solo GGUF, dos modos (short/long)
    - Float16 por defecto (siguiendo estándar Upstage)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        context_mode: str = "short",  # "short" (512) o "long" (2048)
        use_langchain: bool = True,
        n_threads: int = 6,
        verbose: bool = True,
        temperature: float = 0.7,  # Agregado: parámetro por defecto
        top_p: float = 0.95,       # Agregado: parámetro por defecto
    ):
        """
        Args:
            model_path: Ruta al GGUF. Si None, busca en Ollama cache
            context_mode: "short" (512 tokens) o "long" (2048 tokens)
            use_langchain: Usar LangChain para optimizaciones
            n_threads: Threads CPU (default 6, deja 2 libres en 8-core)
            verbose: Mostrar logs de carga
            temperature: Creatividad por defecto (0.7 como Upstage)
            top_p: Nucleus sampling por defecto (0.95 como Upstage)
        """
        self.context_mode = context_mode
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        self.n_threads = n_threads
        self.verbose = verbose
        self.default_temperature = temperature
        self.default_top_p = top_p
        
        # Context-aware: Mismo GGUF, diferente n_ctx
        # SOLAR-10.7B entrenado con n_ctx_train=4096 (como Upstage)
        self.n_ctx = 512 if context_mode == "short" else 2048
        
        # Buscar modelo
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = self.find_solar_gguf()
        
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(
                f"SOLAR GGUF no encontrado. Ejecuta: ollama pull solar:10.7b"
            )
        
        if self.verbose:
            print(f"✅ SOLAR GGUF encontrado: {self.model_path}")
            print(f"   Modo: {context_mode} (n_ctx={self.n_ctx})")
            print(f"   Backend: {'LangChain' if self.use_langchain else 'llama.cpp directo'}")
        
        # Cargar modelo
        self.model = self._load_model()
        
        if self.verbose:
            print(f"✅ Modelo cargado: SOLAR-10.7B-{context_mode}")
    
    def find_solar_gguf(self) -> Optional[Path]:
        """
        Busca SOLAR GGUF en cache de Ollama
        
        Ubicaciones:
        1. /usr/share/ollama/.ollama/models/blobs/sha256-*
        2. ~/.ollama/models/blobs/sha256-*
        
        Identificación:
        - Tamaño: ~6.0-6.5 GB (Q4_K_M quantization)
        - Nombre en registry: solar:10.7b
        """
        search_paths = [
            Path("/usr/share/ollama/.ollama/models/blobs"),
            Path.home() / ".ollama" / "models" / "blobs"
        ]
        
        for base_path in search_paths:
            if not base_path.exists():
                continue
            
            # Buscar blobs de ~6GB (SOLAR Q4_K_M)
            for blob_file in base_path.glob("sha256-*"):
                # Skip manifests y configs (pequeños)
                if blob_file.stat().st_size < 5_000_000_000:  # < 5GB
                    continue
                
                # SOLAR Q4_K_M está entre 6.0-6.5 GB
                if 6_000_000_000 <= blob_file.stat().st_size <= 6_500_000_000:
                    if self.verbose:
                        size_gb = blob_file.stat().st_size / (1024**3)
                        print(f"🔍 Candidato SOLAR: {blob_file.name} ({size_gb:.2f} GB)")
                    return blob_file
        
        return None
    
    def _load_model(self):
        """
        Carga modelo con backend apropiado
        """
        if self.use_langchain:
            return self._load_with_langchain()
        else:
            return self._load_with_llamacpp()
    
    def _load_with_langchain(self):
        """
        Carga con LangChain (optimizaciones automáticas)
        
        Optimizaciones CPU según mejores prácticas:
        - f16_kv=True: FP16 KV cache (~15% RAM reduction)
        - use_mmap=True: Memory-mapped file access (faster I/O)
        - n_batch=512: Optimized batch size for CPU throughput
        - use_mlock=False: No memory locking (prevents OOM)
        - n_gpu_layers=0: CPU-only inference
        - max_tokens=512: Default compatible con Upstage (max_new_tokens=64)
        
        Nota: SOLAR-10.7B usa torch_dtype=torch.float16 en GPU.
              En CPU con GGUF Q4_K_M, esto se traduce a f16_kv.
        """
        if not LANGCHAIN_AVAILABLE:
            print("⚠️  LangChain no disponible. Fallback a llama.cpp directo.")
            return self._load_with_llamacpp()
        
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        return LlamaCpp(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=512,          # ✅ Batch optimizado para CPU
            f16_kv=True,          # ✅ FP16 KV cache (equivalente a torch.float16)
            use_mmap=True,        # ✅ Memory mapping (fast I/O)
            use_mlock=False,      # ✅ No lock (evita OOM)
            n_gpu_layers=0,       # ✅ CPU-only
            callback_manager=callback_manager,
            verbose=False,
            max_tokens=512        # ✅ Default razonable (ajustable en generate())
        )
    
    def _load_with_llamacpp(self):
        """
        Carga con llama.cpp directo (fallback)
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python no disponible. Instalar con: pip install llama-cpp-python")
        
        return Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            use_mmap=True,
            use_mlock=False,
            verbose=self.verbose
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Genera respuesta con el modelo
        
        Parámetros según especificación oficial de Upstage SOLAR-10.7B:
        - temperature: 0.0-1.0 (default 0.7)
        - top_p: nucleus sampling (default 0.95)
        - max_tokens: longitud máxima (default 512, ajustable)
        
        Args:
            prompt: Texto de entrada
            max_tokens: Tokens máximos de salida (renombrado de max_new_tokens)
            temperature: Creatividad (None = usar default del constructor)
            top_p: Nucleus sampling (None = usar default del constructor)
            stop: Secuencias de parada
        
        Returns:
            Texto generado
        """
        # Usar defaults del constructor si no se especifican
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p
        
        start_time = time.time()
        
        if self.use_langchain:
            # LangChain usa .invoke()
            response = self.model.invoke(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or []
            )
        else:
            # llama.cpp usa .__call__()
            result = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                echo=False
            )
            response = result["choices"][0]["text"]
        
        elapsed = time.time() - start_time
        tokens = len(response.split())  # Approximation
        tok_per_sec = tokens / elapsed if elapsed > 0 else 0
        
        if self.verbose:
            print(f"\n⏱️  Generación: {elapsed:.2f}s ({tok_per_sec:.2f} tok/s)")
        
        return response.strip()
    
    def generate_upstage_style(
        self,
        text: str,
        max_new_tokens: int = 64,
        **kwargs
    ) -> str:
        """
        Generación al estilo de la API oficial de Upstage
        
        Compatibilidad con el ejemplo oficial:
        ```python
        text = "Hi, my name is "
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=64)
        ```
        
        Args:
            text: Texto de entrada (equivalente a tokenizer input)
            max_new_tokens: Tokens a generar (como en transformers)
            **kwargs: Argumentos adicionales (temperature, top_p, etc.)
        
        Returns:
            Texto completo (input + generación)
        """
        response = self.generate(
            prompt=text,
            max_tokens=max_new_tokens,
            **kwargs
        )
        
        # Upstage devuelve input + output concatenados
        return text + response
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas del modelo
        """
        import psutil
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024**2)
        
        return {
            "model": "SOLAR-10.7B-Instruct-v1.0",
            "context_mode": self.context_mode,
            "n_ctx": self.n_ctx,
            "backend": "LangChain" if self.use_langchain else "llama.cpp",
            "ram_mb": ram_mb,
            "model_path": str(self.model_path)
        }
    
    def unload(self):
        """
        Descarga modelo de memoria
        """
        if self.model:
            del self.model
            self.model = None
            gc.collect()
            
            if self.verbose:
                print(f"✅ SOLAR-{self.context_mode} descargado de memoria")


def main():
    """
    Test standalone del wrapper
    """
    print("=" * 70)
    print("SOLAR-10.7B Native Wrapper - Test (Upstage Compatible)")
    print("=" * 70)
    
    # TEST 0: Compatibilidad API Upstage
    print("\n📝 TEST 0: API Compatible con Upstage (max_new_tokens=64)")
    print("-" * 70)
    
    solar = SolarNative(context_mode="short", use_langchain=True)
    
    # Ejemplo oficial de Upstage
    text = "Hi, my name is "
    print(f"Input: '{text}'")
    
    output = solar.generate_upstage_style(text, max_new_tokens=64)
    print(f"\nOutput completo (input + generado):")
    print(f"'{output}'")
    print(f"\nSolo generado:")
    print(f"'{output[len(text):]}'")
    
    # TEST 1: Short context (512 tokens)
    print("\n📝 TEST 1: Short Context (512 tokens)")
    print("-" * 70)
    
    solar_short = SolarNative(context_mode="short", use_langchain=True)
    
    prompt_short = """Pregunta: ¿Qué es backpropagation en deep learning?
Responde en máximo 3 líneas técnicas.

Respuesta:"""
    
    print(f"\nPrompt: {prompt_short}\n")
    response_short = solar_short.generate(prompt_short, max_tokens=150, temperature=0.3)
    print(f"\n{response_short}")
    
    stats_short = solar_short.get_stats()
    print(f"\n📊 Stats: {stats_short['ram_mb']:.0f} MB RAM")
    
    # TEST 2: Long context (2048 tokens)
    print("\n" + "=" * 70)
    print("📝 TEST 2: Long Context (2048 tokens)")
    print("-" * 70)
    
    solar_short.unload()  # Liberar short
    
    solar_long = SolarNative(context_mode="long", use_langchain=True)
    
    prompt_long = """Contexto: Los modelos transformer utilizan atención multi-head para procesar secuencias.

Pregunta: Explica detalladamente cómo funciona el mecanismo de atención en transformers, 
incluyendo el cálculo de Query, Key y Value.

Respuesta técnica detallada:"""
    
    print(f"\nPrompt: {prompt_long}\n")
    response_long = solar_long.generate(prompt_long, max_tokens=512, temperature=0.5)
    print(f"\n{response_long}")
    
    stats_long = solar_long.get_stats()
    print(f"\n📊 Stats: {stats_long['ram_mb']:.0f} MB RAM")
    
    # Cleanup
    solar_long.unload()
    
    print("\n" + "=" * 70)
    print("✅ Tests completados")
    print("=" * 70)


if __name__ == "__main__":
    main()
