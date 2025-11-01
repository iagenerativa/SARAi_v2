"""
ModelPool v2.4: Cache LRU/TTL con sistema de fallback tolerante a fallos
Gestión automática de memoria + resiliencia para SARAi v2.4

NEW v2.16 (Risk #5): Timeout dinámico basado en n_ctx
"""

import os
import time
import gc
from typing import Dict, Any, Optional
from collections import OrderedDict
from pathlib import Path
import yaml
from huggingface_hub import hf_hub_download


def _calculate_timeout(n_ctx: int) -> int:
    """
    NEW v2.16 (Risk #5): Calcula timeout adaptativo según contexto
    
    Fórmula: timeout = 10s + (n_ctx / 1024) * 10s
    
    Tabla de referencia:
    - n_ctx=512:  10 + (512/1024)*10  = 15s
    - n_ctx=1024: 10 + (1024/1024)*10 = 20s
    - n_ctx=2048: 10 + (2048/1024)*10 = 30s
    - n_ctx=4096: 10 + (4096/1024)*10 = 50s
    - n_ctx=8192: 10 + (8192/1024)*10 = 90s → max(90, 60) = 60s
    
    Args:
        n_ctx: Tamaño del contexto (context window)
    
    Returns:
        Timeout en segundos (máximo 60s)
    """
    base_timeout = 10  # Segundos base para contextos pequeños
    scaling_factor = 10  # Segundos adicionales por cada 1024 tokens
    
    timeout = base_timeout + (n_ctx / 1024) * scaling_factor
    
    # Límite superior: 60s (contextos muy grandes no deberían bloquear indefinidamente)
    return min(int(timeout), 60)


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
        
        # NEW v2.12: Cache separado para skills MoE
        self.skills_cache: OrderedDict = OrderedDict()  # {skill_name: model_object}
        self.skills_timestamps: Dict[str, float] = {}   # {skill_name: last_access_time}
        
        # Configuración de runtime
        runtime_cfg = self.config.get('runtime', {})
        self.backend = runtime_cfg.get('backend', 'cpu')
        self.max_models = runtime_cfg.get('max_concurrent_llms', 2)
        
        # NEW v2.12: Máximo de skills simultáneos (3 skills + expert/tiny base)
        self.max_skills = runtime_cfg.get('max_concurrent_skills', 3)
        
        # Configuración de memoria
        memory_cfg = self.config.get('memory', {})
        self.ttl = memory_cfg.get('model_ttl_seconds', 45)  # Aumentado para prefetch
        self.max_ram_gb = memory_cfg.get('max_ram_gb', 12)
        
        print(f"[ModelPool v2.12] Inicializado - Backend: {self.backend}, "
              f"Max modelos: {self.max_models}, Max skills: {self.max_skills}, TTL: {self.ttl}s")
    
    def get(self, logical_name: str) -> Any:
        """
        Obtiene modelo del cache o lo carga si no existe
        logical_name puede ser: 'expert_short', 'expert_long', 'tiny', 'qwen_omni', 'qwen3_vl_4b'
        
        NEW v2.12: Qwen3-VL-4B para análisis de imagen/video
        - Se carga bajo demanda cuando input_type in ["image", "video"]
        - TTL: 60s (auto-descarga rápida para liberar RAM)
        - Política: Se descarga si RAM libre < 4GB
        
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
            "tiny": [],
            "qwen_omni": [],  # Audio/multimodal sin fallback
            "qwen3_vl_4b": []  # NEW v2.12: Visión sin fallback
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
    
    def get_skill(self, skill_name: str) -> Any:
        """
        NEW v2.12: Obtiene modelo de skill MoE del cache o lo carga bajo demanda
        
        Skills disponibles (según config/sarai.yaml → models.skills):
        - programming: Desarrollo de software (Python, JS, etc.)
        - diagnosis: Diagnóstico de sistemas
        - finance: Análisis financiero
        - logic: Razonamiento lógico
        - creative: Generación creativa
        - reasoning: Razonamiento complejo
        
        Gestión de memoria:
        - Máximo 3 skills simultáneos en RAM (self.max_skills)
        - LRU: descarga el skill menos usado cuando se alcanza límite
        - TTL: auto-descarga tras 45s sin uso
        - Impacto RAM: ~800MB por skill (GGUF IQ4_NL, n_ctx=1024)
        
        Args:
            skill_name: Nombre del skill a cargar
        
        Returns:
            Objeto Llama del skill cargado
        
        Raises:
            ValueError: Si skill_name no existe en configuración
            RuntimeError: Si falla la carga y no hay fallback
        
        Example:
            ```python
            pool = get_model_pool()
            prog_skill = pool.get_skill("programming")
            response = prog_skill.create_completion(
                "Escribe una función Python para ordenar lista"
            )
            ```
        """
        # Limpiar skills expirados
        self._cleanup_expired_skills()
        
        # Si existe en cache, mover al final (LRU) y actualizar timestamp
        if skill_name in self.skills_cache:
            self.skills_cache.move_to_end(skill_name)
            self.skills_timestamps[skill_name] = time.time()
            print(f"[ModelPool] Skill cache hit: {skill_name}")
            return self.skills_cache[skill_name]
        
        # Si cache de skills lleno, eliminar el menos usado
        if len(self.skills_cache) >= self.max_skills:
            self._evict_lru_skill()
        
        # Cargar skill desde configuración
        skills_config = self.config.get('models', {}).get('skills', {})
        
        if skill_name not in skills_config:
            available = list(skills_config.keys())
            raise ValueError(
                f"Skill '{skill_name}' no encontrado en config. "
                f"Skills disponibles: {available}"
            )
        
        skill_cfg = skills_config[skill_name]
        
        # Cargar skill (siempre GGUF en CPU por ahora)
        try:
            print(f"[ModelPool] Cargando skill '{skill_name}'...")
            skill_model = self._load_gguf(
                model_cfg=skill_cfg,
                context_length=skill_cfg.get('context_length', 1024),
                prefetch=False
            )
            
            # Guardar en cache
            self.skills_cache[skill_name] = skill_model
            self.skills_timestamps[skill_name] = time.time()
            
            print(f"✅ Skill '{skill_name}' cargado. Skills activos: {list(self.skills_cache.keys())}")
            return skill_model
        
        except Exception as e:
            print(f"❌ Error cargando skill '{skill_name}': {e}")
            raise RuntimeError(f"No se pudo cargar skill '{skill_name}': {e}")
    
    def release_skill(self, skill_name: str):
        """
        NEW v2.12: Libera skill explícitamente
        
        Args:
            skill_name: Nombre del skill a liberar
        """
        if skill_name in self.skills_cache:
            del self.skills_cache[skill_name]
            del self.skills_timestamps[skill_name]
            gc.collect()
            print(f"[ModelPool] Skill '{skill_name}' liberado manualmente")
    
    def _cleanup_expired_skills(self):
        """
        NEW v2.12: Elimina skills no usados en más de TTL segundos
        """
        now = time.time()
        to_remove = [
            name for name, timestamp in self.skills_timestamps.items()
            if now - timestamp > self.ttl
        ]
        
        for name in to_remove:
            print(f"[ModelPool] TTL expirado para skill '{name}', descargando...")
            del self.skills_cache[name]
            del self.skills_timestamps[name]
        
        if to_remove:
            gc.collect()
    
    def _evict_lru_skill(self):
        """
        NEW v2.12: Elimina el skill menos recientemente usado (LRU)
        """
        if not self.skills_cache:
            return
        
        # OrderedDict mantiene orden de inserción/acceso
        lru_name = next(iter(self.skills_cache))
        print(f"[ModelPool] Cache de skills lleno, eliminando LRU: {lru_name}")
        
        del self.skills_cache[lru_name]
        del self.skills_timestamps[lru_name]
        gc.collect()

    
    def _load_with_backend(self, logical_name: str, prefetch: bool = False) -> Any:
        """
        Carga modelo según backend configurado
        NEW v2.3: Soporta GGUF Context-Aware (expert_short vs expert_long)
        
        CRÍTICO: 
        - CPU: usa llama-cpp-python + GGUF (10x más rápido)
        - GPU: usa transformers + 4-bit quantization
        
        Args:
            logical_name: Nombre lógico (expert_short, expert_long, tiny, qwen_omni, qwen3_vl_4b)
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
        elif logical_name == "qwen3_vl_4b":
            # NEW v2.12: Modelo de visión (imagen/video)
            model_cfg_key = "qwen3_vl_4b"
            context_length = self.config['models']['qwen3_vl_4b'].get('context_length', 1024)
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
        
        # NEW v2.16 (Risk #5): Calcular timeout dinámico basado en n_ctx
        request_timeout = _calculate_timeout(n_ctx)
        
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
        
        print(f"[ModelPool] Cargando GGUF con n_ctx={n_ctx}, timeout={request_timeout}s")
        
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
            },
            # NEW v2.12: Stats de skills
            "skills_loaded": len(self.skills_cache),
            "max_skills_capacity": self.max_skills,
            "skills_in_cache": list(self.skills_cache.keys()),
            "skills_time_since_last_access": {
                name: round(now - ts, 2)
                for name, ts in self.skills_timestamps.items()
            }
        }
    
    def get_skill_client(self, skill_name: str):
        """
        NEW v2.16: Obtiene cliente gRPC para skill containerizado
        
        PHOENIX INTEGRATION:
        - Connection pooling: reutiliza clientes gRPC
        - Health checks: verifica que container esté activo
        - Auto-reconnect: re-crea cliente si conexión falla
        
        Args:
            skill_name: Nombre del skill ("draft", "image", "lora-trainer", etc.)
        
        Returns:
            Cliente gRPC del skill (tipo específico según skill)
        
        Raises:
            RuntimeError: Si skill no está disponible
        
        Example:
            ```python
            pool = get_model_pool()
            draft_client = pool.get_skill_client("draft")
            response = draft_client.Generate(request, timeout=10.0)
            ```
        """
        # Importar gRPC stubs (lazy import)
        try:
            from skills import skills_pb2_grpc
            import grpc
        except ImportError:
            raise ImportError(
                "gRPC no instalado. Ejecuta: pip install grpcio grpcio-tools"
            )
        
        # Cache de clientes gRPC (para reutilizar conexiones)
        if not hasattr(self, '_skill_clients'):
            self._skill_clients: Dict[str, Any] = {}
        
        # Si el cliente ya existe en cache, retornarlo
        if skill_name in self._skill_clients:
            # Verificar que la conexión sigue activa
            client = self._skill_clients[skill_name]
            try:
                # Health check simple (timeout corto)
                # TODO: Implementar método Health() en proto si no existe
                return client
            except:
                # Conexión muerta, eliminar y recrear
                print(f"[ModelPool] Conexión muerta para skill '{skill_name}', recreando...")
                del self._skill_clients[skill_name]
        
        # Mapeo de skill_name a puerto gRPC (según docker-compose)
        skill_ports = {
            "draft": 50051,      # skill_draft container
            "image": 50052,      # skill_image container
            "lora-trainer": 50053,  # skill_lora_trainer container
            "sql": 50054,        # skill_sql container (existente)
            "home_ops": 50055    # skill_home_ops container (existente)
        }
        
        if skill_name not in skill_ports:
            raise ValueError(
                f"Skill '{skill_name}' no reconocido. "
                f"Skills disponibles: {list(skill_ports.keys())}"
            )
        
        # Construir dirección del skill
        # En producción: "skill_<name>:50051" (DNS de Docker)
        # En desarrollo: "localhost:<port>" (port forwarding)
        use_docker_dns = os.getenv("USE_DOCKER_DNS", "true").lower() == "true"
        
        if use_docker_dns:
            address = f"skill_{skill_name}:{skill_ports[skill_name]}"
        else:
            address = f"localhost:{skill_ports[skill_name]}"
        
        # Crear canal gRPC
        print(f"[ModelPool] Conectando a skill '{skill_name}' en {address}...")
        
        channel = grpc.insecure_channel(
            address,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 30000),  # 30s keepalive
                ('grpc.keepalive_timeout_ms', 10000),  # 10s timeout
            ]
        )
        
        # Crear stub según el skill
        if skill_name in ["draft", "image", "lora-trainer"]:
            # Skills genéricos usan SkillsService
            client = skills_pb2_grpc.SkillsServiceStub(channel)
        else:
            # Skills específicos (sql, home_ops) tienen sus propios stubs
            # TODO: Manejar caso por caso si tienen protos diferentes
            client = skills_pb2_grpc.SkillsServiceStub(channel)
        
        # Guardar en cache
        self._skill_clients[skill_name] = client
        
        print(f"✅ Cliente gRPC para '{skill_name}' creado")
        
        return client
    
    def close_skill_clients(self):
        """
        NEW v2.16: Cierra todos los clientes gRPC
        Llamar al finalizar la aplicación
        """
        if hasattr(self, '_skill_clients'):
            for skill_name in list(self._skill_clients.keys()):
                print(f"[ModelPool] Cerrando cliente gRPC: {skill_name}")
                # gRPC channels se cierran automáticamente al liberar referencia
                del self._skill_clients[skill_name]
            
            self._skill_clients.clear()
            print("✅ Todos los clientes gRPC cerrados")


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
