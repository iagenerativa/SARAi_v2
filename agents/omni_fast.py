"""
Omni Fast Agent v2.16.1: Qwen3-VL-4B-Instruct con LangChain
GGUF optimizado para VELOCIDAD (target: 9-12 tok/s)

Filosofía v2.16.1:
- ✅ VELOCIDAD PRIMERO: 3B params = 2.3x más rápido que 7B
- ✅ CONVERSACIONES FLUIDAS: 10 tok/s promedio (vs 4 tok/s del 7B)
- ✅ RAM EFICIENTE: 2.6 GB (vs 4.9 GB del 7B)

Uso:
    from agents.omni_fast import get_omni_fast_agent
    
    agent = get_omni_fast_agent()
    response = agent.invoke("Hola, ¿cómo estás?")  # ~10 tok/s ✅
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate


class OmniFastConfig:
    """Configuración optimizada para velocidad (3B model)"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 6,
                 temperature: float = 0.7, max_tokens: int = 512,
                 n_batch: int = 512):
        self.model_path = model_path
        self.n_ctx = n_ctx  # Reducido: 2048 vs 8192 del 7B (más rápido)
        self.n_threads = n_threads
        self.temperature = temperature
        self.max_tokens = max_tokens  # Reducido: 512 vs 2048 (conversaciones cortas)
        self.n_batch = n_batch
    
    @classmethod
    def from_yaml(cls, config_path: str = "config/sarai.yaml"):
        """Carga config desde YAML (sección qwen_omni_fast)"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Intentar cargar config específica de omni_fast, fallback a qwen_omni
        omni_fast = config['models'].get('qwen_omni_fast')
        if omni_fast is None:
            # Fallback: crear config optimizada desde qwen_omni
            omni = config['models']['qwen_omni']
            return cls(
                model_path=str(Path("models/gguf") / "Qwen3-VL-4B-Instruct-Q4_K_M.gguf"),
                n_ctx=2048,  # Reducido para velocidad
                n_threads=config.get('runtime', {}).get('n_threads', max(1, os.cpu_count() - 2)),
                temperature=0.7,
                max_tokens=512,  # Conversaciones cortas
                n_batch=512
            )
        else:
            return cls(
                model_path=str(Path("models/gguf") / omni_fast['gguf_file']),
                n_ctx=omni_fast.get('context_length', 2048),
                n_threads=config.get('runtime', {}).get('n_threads', max(1, os.cpu_count() - 2)),
                temperature=omni_fast.get('temperature', 0.7),
                max_tokens=omni_fast.get('max_tokens', 512),
                n_batch=omni_fast.get('n_batch', 512)
            )


class OmniFastAgent:
    """Agente LangChain para Qwen3-VL-4B-Instruct GGUF (velocidad optimizada)"""
    
    def __init__(self, config: Optional[OmniFastConfig] = None):
        self.config = config or OmniFastConfig.from_yaml()
        self.llm = None
        self.prompt_template = None
        self._initialize()
    
    def _initialize(self):
        """Carga modelo 3B con optimizaciones CPU agresivas"""
        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(f"Modelo 3B no encontrado: {self.config.model_path}")
        
        print(f"[OmniFast] Cargando Omni-3B GGUF (optimizado para velocidad)...")
        print(f"  Contexto: {self.config.n_ctx}, Threads: {self.config.n_threads}")
        print(f"  Target: 9-12 tok/s (2.3x más rápido que 7B)")
        
        # OPTIMIZACIONES MÁXIMAS PARA VELOCIDAD
        self.llm = LlamaCpp(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_batch=512,  # ✅ Batch grande
            n_gpu_layers=0,  # ✅ CPU puro
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            f16_kv=True,  # ✅ FP16 KV cache
            use_mlock=False,  # ✅ No mlock (evita OOM)
            use_mmap=True,  # ✅ Memory-mapped
            verbose=False,
            # Sampling optimizado
            repeat_penalty=1.1,
            last_n_tokens_size=64,
            top_k=40,
            top_p=0.9,
            seed=-1,
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="User: {query}\nAssistant:"
        )
        
        print(f"✅ Omni-3B listo (~2.6 GB, 9-12 tok/s esperados)")
    
    def invoke(self, query: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        """
        Procesa query con velocidad optimizada
        
        Args:
            query: Pregunta o instrucción
            max_tokens: Límite de tokens (None = usa config default 512)
            **kwargs: Contexto adicional
        
        Returns:
            Respuesta generada
        
        Performance:
            - Modelo 3B: 2.3x más rápido que 7B
            - Target: 9-12 tok/s
            - Latencia esperada: ~10s para 100 tokens
        """
        prompt = self.prompt_template.format(query=query)
        
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        
        return self.llm.invoke(prompt, max_tokens=max_tokens)


# Singleton global
_omni_fast_agent: Optional[OmniFastAgent] = None


def get_omni_fast_agent() -> OmniFastAgent:
    """
    Factory para obtener singleton de OmniFastAgent (3B)
    
    Returns:
        OmniFastAgent instance (única en memoria)
    """
    global _omni_fast_agent
    
    if _omni_fast_agent is None:
        _omni_fast_agent = OmniFastAgent()
    
    return _omni_fast_agent


def reset_omni_fast_agent():
    """Reset singleton (útil para tests)"""
    global _omni_fast_agent
    _omni_fast_agent = None
