"""
ModelPool v2.4: Cache LRU/TTL con sistema de fallback tolerante a fallos
Gestión automática de memoria + resiliencia para SARAi v2.4
"""

import os
import time
import gc
from typing import Dict, Any, Optional
from collections import OrderedDict
from pathlib import Path
import yaml
from huggingface_hub import hf_hub_download


class ModelPool:
    """
    Cache inteligente para modelos LLM con:
    - LRU (Least Recently Used): Descarga el modelo menos usado
    - TTL (Time To Live): Auto-descarga tras N segundos sin uso
    - Backend abstraído: GGUF (CPU) o 4-bit (GPU) según config
    - Prefetch cache: Modelos precargados por el Prefetcher
    - GGUF Context-Aware: expert_short y expert_long usan el mismo archivo
    """
    
    def __init__(self, config_path: str = "config/sarai.yaml"):
        """
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cache: OrderedDict = OrderedDict()  # {model_name: model_object}
        self.cache_prefetch: Dict[str, Any] = {}  # {model_name: prefetched_model} NEW v2.3
        self.timestamps: Dict[str, float] = {}   # {model_name: last_access_time}
        
        # Configuración de runtime
        runtime_cfg = self.config.get('runtime', {})
        self.backend = runtime_cfg.get('backend', 'cpu')
        self.max_models = runtime_cfg.get('max_concurrent_llms', 2)
        
        # Configuración de memoria
        memory_cfg = self.config.get('memory', {})
        self.ttl = memory_cfg.get('model_ttl_seconds', 45)  # Aumentado para prefetch
        self.max_ram_gb = memory_cfg.get('max_ram_gb', 12)
        
        print(f"[ModelPool v2.3] Inicializado - Backend: {self.backend}, "
              f"Max modelos: {self.max_models}, TTL: {self.ttl}s")
    
    def get(self, logical_name: str) -> Any:
        """
        Obtiene modelo del cache o lo carga si no existe
        logical_name puede ser: 'expert_short', 'expert_long', 'tiny', 'qwen_omni'
        
        Args:
            logical_name: Nombre lógico del modelo (incluye variantes de contexto)
        
        Returns:
            Objeto modelo cargado (Llama para CPU, tuple(model,tokenizer) para GPU)
        """
        # Limpiar modelos expirados
        self._cleanup_expired()
        
        # Si existe en cache, mover al final (LRU)
        if logical_name in self.cache:
            self.cache.move_to_end(logical_name)
            self.timestamps[logical_name] = time.time()
            print(f"[ModelPool] Cache hit: {logical_name}")
            return self.cache[logical_name]
        
        # NEW v2.3: Comprobar cache de prefetch
        if logical_name in self.cache_prefetch:
            print(f"✅ Prefetch HIT: {logical_name} ya estaba cargado")
            self.cache[logical_name] = self.cache_prefetch.pop(logical_name)
            self.timestamps[logical_name] = time.time()
            return self.cache[logical_name]
        
        # Si cache lleno, eliminar el menos usado
        if len(self.cache) >= self.max_models:
            self._evict_lru()
        
        # NEW v2.4: Sistema de fallback tolerante a fallos
        # Intenta cargar el modelo solicitado, con degradación gradual
        model = self._load_with_fallback(logical_name, prefetch=False)
        
        if model is None:
            raise RuntimeError(f"❌ Todos los fallbacks fallaron para {logical_name}")
        
        self.cache[logical_name] = model
        self.timestamps[logical_name] = time.time()
        
        print(f"[ModelPool] {logical_name} cargado. Cache: {list(self.cache.keys())}")
        return model
    
    def _load_with_fallback(self, logical_name: str, prefetch: bool = False):
        """
        NEW v2.4: Sistema de fallback en cascada
        Garantiza que siempre se obtiene un modelo, degradando calidad si es necesario
        
        Cascada de fallback:
        1. expert_long → expert_short → tiny
        2. expert_short → tiny
        3. tiny → (sin fallback, es el último recurso)
        
        Args:
            logical_name: Modelo solicitado
            prefetch: Si es precarga o carga normal
        
        Returns:
            Modelo cargado o None si todos los fallbacks fallan
        """
        # Definir cadena de fallback
        fallback_chain = {
            "expert_long": ["expert_short", "tiny"],
            "expert_short": ["tiny"],
            "tiny": [],  # Sin fallback, es el mínimo
            "qwen_omni": []  # Multimodal sin fallback
        }
        
        # Intentar cargar el modelo solicitado
        try:
            print(f"[ModelPool] Cargando {logical_name} (backend: {self.backend})...")
            model = self._load_with_backend(logical_name, prefetch)
            print(f"✅ {logical_name} cargado exitosamente")
            return model
        
        except Exception as e:
            print(f"⚠️ Error cargando {logical_name}: {e}")
            
            # Intentar fallbacks en orden
            fallbacks = fallback_chain.get(logical_name, [])
            
            for fallback_name in fallbacks:
                try:
                    print(f"🔄 Intentando fallback: {fallback_name}")
                    model = self._load_with_backend(fallback_name, prefetch)
                    print(f"✅ Fallback exitoso: {fallback_name}")
                    
                    # Registrar métrica de fallback (para Prometheus)
                    self._record_fallback(logical_name, fallback_name)
                    
                    return model
                
                except Exception as fallback_error:
                    print(f"⚠️ Fallback {fallback_name} también falló: {fallback_error}")
                    continue
            
            # Si todos los fallbacks fallan
            print(f"❌ Todos los fallbacks agotados para {logical_name}")
            return None
    
    def _record_fallback(self, requested: str, used: str):
        """
        Registra evento de fallback para métricas
        
        Args:
            requested: Modelo solicitado
            used: Modelo usado como fallback
        """
        # Guardar en archivo de métricas (leído por /metrics endpoint)
        metrics_file = Path("state/model_fallbacks.log")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_file, "a") as f:
            import json
            from datetime import datetime
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "requested": requested,
                "used": used
            }) + "\n")
    
    def prefetch_model(self, logical_name: str):
        """
        NEW v2.3: Precarga modelo en segundo plano (llamado por Prefetcher)
        
        Args:
            logical_name: Nombre lógico del modelo a precargar
        """
        if logical_name in self.cache or logical_name in self.cache_prefetch:
            return  # Ya está cargado
        
        try:
            print(f"🔄 Prefetching {logical_name}...")
            model = self._load_with_backend(logical_name, prefetch=True)
            self.cache_prefetch[logical_name] = model
            print(f"✅ Prefetch completo: {logical_name}")
        except Exception as e:
            print(f"⚠️ Prefetch fallido para {logical_name}: {e}")
    
    def release(self, logical_name: str):
        """
        Libera modelo explícitamente (útil para multimodal)
        
        Args:
            logical_name: Nombre del modelo a liberar
        """
        if logical_name in self.cache:
            del self.cache[logical_name]
            del self.timestamps[logical_name]
            gc.collect()
            print(f"[ModelPool] {logical_name} liberado manualmente")
    
    def _load_with_backend(self, logical_name: str, prefetch: bool = False) -> Any:
        """
        Carga modelo según backend configurado
        NEW v2.3: Soporta GGUF Context-Aware (expert_short vs expert_long)
        
        CRÍTICO: 
        - CPU: usa llama-cpp-python + GGUF (10x más rápido)
        - GPU: usa transformers + 4-bit quantization
        
        Args:
            logical_name: Nombre lógico (expert_short, expert_long, tiny, qwen_omni)
            prefetch: Si True, carga con mínimos recursos (1 thread)
        
        Returns:
            Modelo cargado
        """
        # Mapeo de nombres lógicos a configuración base
        if logical_name.startswith("expert"):
            model_cfg_key = "expert"
            context_length = 512 if logical_name == "expert_short" else 2048
        elif logical_name == "tiny":
            model_cfg_key = "tiny"
            context_length = self.config['models']['tiny'].get('context_length', 2048)
        elif logical_name == "qwen_omni":
            model_cfg_key = "qwen_omni"
            context_length = self.config['models']['qwen_omni'].get('context_length', 2048)
        else:
            # Fallback: usar logical_name directamente
            model_cfg_key = logical_name
            context_length = None
        
        model_cfg = self.config['models'].get(model_cfg_key)
        if not model_cfg:
            raise ValueError(f"Modelo '{logical_name}' no encontrado en config")
        
        if self.backend == "cpu":
            return self._load_gguf(model_cfg, context_length, prefetch)
        elif self.backend == "gpu":
            return self._load_gpu_4bit(model_cfg)
        else:
            raise ValueError(f"Backend '{self.backend}' no soportado")
    
    def _load_gguf(self, model_cfg: Dict[str, Any], context_length: Optional[int] = None, 
                   prefetch: bool = False) -> Any:
        """
        Carga modelo GGUF con llama-cpp-python (CPU)
        NEW v2.3: context_length puede sobrescribir el default (Context-Aware)
        
        Args:
            model_cfg: Configuración del modelo desde YAML
            context_length: Override de n_ctx (para expert_short/expert_long)
            prefetch: Si True, usa solo 1 thread para no saturar CPU
        
        Returns:
            Objeto Llama
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python no instalado. Ejecuta: "
                "pip install llama-cpp-python"
            )
        
        # Descargar archivo GGUF desde HuggingFace
        gguf_file = model_cfg.get('gguf_file')
        if not gguf_file:
            raise ValueError(f"Falta 'gguf_file' en config para {model_cfg.get('name', 'unknown')}")
        
        model_path = hf_hub_download(
            repo_id=model_cfg['repo_id'],
            filename=gguf_file,
            cache_dir=model_cfg.get('cache_dir', './models/cache')
        )
        
        # Determinar n_ctx (context_length override o config default)
        n_ctx = context_length if context_length is not None else model_cfg.get('context_length', 2048)
        
        # Determinar n_threads (1 para prefetch, full para carga normal)
        runtime_cfg = self.config.get('runtime', {})
        if prefetch:
            n_threads = 1  # Mínimo impacto durante prefetch
        else:
            n_threads = runtime_cfg.get('n_threads', max(1, os.cpu_count() - 2))
        
        # Configuración de memoria
        memory_cfg = self.config.get('memory', {})
        use_mmap = memory_cfg.get('use_mmap', True)
        use_mlock = memory_cfg.get('use_mlock', False)
        
        # Cargar con llama-cpp
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            verbose=False
        )
    
    def _load_gpu_4bit(self, model_cfg: Dict[str, Any]) -> tuple:
        """
        Carga modelo 4-bit con transformers (GPU)
        
        Args:
            model_cfg: Configuración del modelo desde YAML
        
        Returns:
            Tuple (model, tokenizer)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers o bitsandbytes no instalados. Ejecuta: "
                "pip install transformers bitsandbytes accelerate"
            )
        
        # Configuración 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg['repo_id'],
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=model_cfg.get('cache_dir')
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg['repo_id'],
            cache_dir=model_cfg.get('cache_dir')
        )
        
        return (model, tokenizer)
    
    def _cleanup_expired(self):
        """
        Elimina modelos no usados en más de TTL segundos
        """
        now = time.time()
        to_remove = [
            name for name, timestamp in self.timestamps.items()
            if now - timestamp > self.ttl
        ]
        
        for name in to_remove:
            print(f"[ModelPool] TTL expirado para {name}, descargando...")
            del self.cache[name]
            del self.timestamps[name]
        
        if to_remove:
            gc.collect()
    
    def _evict_lru(self):
        """
        Elimina el modelo menos recientemente usado (LRU)
        """
        if not self.cache:
            return
        
        # OrderedDict mantiene orden de inserción/acceso
        lru_name = next(iter(self.cache))
        print(f"[ModelPool] Cache lleno, eliminando LRU: {lru_name}")
        
        del self.cache[lru_name]
        del self.timestamps[lru_name]
        gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del pool
        
        Returns:
            Dict con métricas del cache
        """
        now = time.time()
        return {
            "backend": self.backend,
            "models_loaded": len(self.cache),
            "max_capacity": self.max_models,
            "models_in_cache": list(self.cache.keys()),
            "time_since_last_access": {
                name: round(now - ts, 2)
                for name, ts in self.timestamps.items()
            }
        }


# Singleton global (inicializado en main.py)
_global_pool: Optional[ModelPool] = None


def get_model_pool(config_path: str = "config/sarai.yaml") -> ModelPool:
    """
    Obtiene instancia singleton del ModelPool
    
    Args:
        config_path: Ruta al archivo de configuración
    
    Returns:
        Instancia de ModelPool
    """
    global _global_pool
    if _global_pool is None:
        _global_pool = ModelPool(config_path)
    return _global_pool
