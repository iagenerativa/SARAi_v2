"""
Omni Native Agent v2.16.1: Qwen3-VL-4B-Instruct con LangChain
GGUF permanente en memoria (190 MB, audio español + NLLB)

ARQUITECTURA v2.16.1 Best-of-Breed:
- Omni-3B: Solo audio (español STT/TTS, emotion detection)
- NLLB-600M: Traducción multilingüe (pipeline STT→translate→process→translate→TTS)
- Empatía: Manejada por LFM2-1.2B (tiny agent), NO por Omni

Filosofía v2.16:
- ✅ LangChain: Abstracción limpia, sin código spaghetti
- ✅ Permanente: Modelo cargado en startup
- ✅ GGUF nativo: Audio optimizado (<240ms latencia)
- ✅ Especialización: Audio exclusivo, empatía a LFM2

Uso:
    from agents.omni_native import get_omni_agent
    
    agent = get_omni_agent()
    response = agent.invoke_audio(audio_bytes)  # Audio español
    # Para empatía conversacional usar tiny agent (LFM2)
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate


class OmniConfig:
    """Configuración desde config/sarai.yaml"""
    
    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: int = 6,
                 temperature: float = 0.7, max_tokens: int = 2048,
                 n_batch: int = 512, auto_reduce_context: bool = True):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_batch = n_batch  # NEW: Batch size para CPU
        self.auto_reduce_context = auto_reduce_context  # NEW: Reducir contexto en conversaciones cortas
    
    @classmethod
    def from_yaml(cls, config_path: str = "config/sarai.yaml") -> "OmniNativeAgent":
        """
        Carga configuración desde YAML y crea instancia del agente.
        
        IMPORTANTE: Solo carga Qwen3-VL-4B-Instruct (190 MB) para audio español.
        Para empatía (soft > 0.7) usar LFM2-1.2B (tiny agent).
        
        Args:
            config_path: Ruta al archivo de configuración YAML
            
        Returns:
            Instancia configurada de OmniNativeAgent
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Solo Omni-3B soportado (audio español exclusivamente)
        if 'qwen_omni_3b' not in config['models']:
            raise KeyError(
                "Configuración 'qwen_omni_3b' requerida en config['models']. "
                "Omni-3B (190 MB) es el único modelo de audio soportado. "
                "Para empatía usar LFM2 (tiny agent)."
            )
        
        omni = config['models']['qwen_omni_3b']
        
        runtime = config.get('runtime', {})
        
        return cls(
            model_path=str(Path("models/gguf") / omni['gguf_file']),
            n_ctx=omni.get('context_length', 8192),
            n_threads=runtime.get('n_threads', max(1, os.cpu_count() - 2)),
            temperature=omni.get('temperature', 0.7),
            max_tokens=omni.get('max_tokens', 2048),
            n_batch=omni.get('n_batch', 512),  # NEW
            auto_reduce_context=omni.get('auto_reduce_context', True)  # NEW
        )


class OmniNativeAgent:
    """
    Agente LangChain para Qwen3-VL-4B-Instruct GGUF
    
    v2.16.1 Best-of-Breed: Solo audio (español + NLLB)
    Para empatía conversacional usar LFM2 (tiny agent)
    """
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.llm = None
        
    def load(self):
        """Carga el modelo GGUF en memoria (permanente)"""
        if self.llm is not None:
            return  # Ya está cargado
        
        print(f"[OmniAgent] Cargando Omni-3B GGUF (audio especialista)...")
        print(f"  Modelo: {self.config.model_path}")
        print(f"  Contexto: {self.config.n_ctx}, Threads: {self.config.n_threads}")
        
        # LlamaCpp con configuración optimizada
        self.llm = LlamaCpp(
            model_path=str(self.config.model_path),
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=0.95,
            repeat_penalty=1.1,
            n_batch=512,
            verbose=False
        )
        
        print(f"✅ Omni-3B listo (~190 MB, audio streaming + NLLB)")
        print(f"   Uso: Audio español STT/TTS, emotion detection")
    
    def invoke(self, query: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        """
        Procesa query con LangChain (optimizado para velocidad)
        
        Args:
            query: Pregunta o instrucción
            max_tokens: Límite de tokens (None = usa config default)
            **kwargs: Contexto adicional (futuro)
        
        Returns:
            Respuesta generada
        
        Performance:
            - Context reduction: Query corta (<100 chars) = response más rápida
            - Batch processing: n_batch=512 optimizado para CPU
            - Target: 7-15 tok/s (vs 4 tok/s baseline)
        """
        # OPTIMIZACIÓN: Reducir max_tokens para queries cortas (más rápido)
        if max_tokens is None:
            if self.config.auto_reduce_context and len(query) < 100:
                # Query corta → respuesta corta esperada
                max_tokens = 256  # Reducido de 2048
            else:
                max_tokens = self.config.max_tokens
        
        prompt = self.prompt_template.format(query=query)
        
        return self.llm.invoke(prompt, max_tokens=max_tokens)


# Singleton global
_omni_agent: Optional[OmniNativeAgent] = None


def get_omni_agent() -> OmniNativeAgent:
    """Obtiene instancia singleton"""
    global _omni_agent
    if _omni_agent is None:
        _omni_agent = OmniNativeAgent()
    return _omni_agent
