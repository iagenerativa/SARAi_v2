"""
Capa 1: I/O Asíncrono Full-Duplex
Procesamiento de audio de entrada y salida con hilos independientes

Componentes:
    - Canal IN: Audio → VAD → STT → BERT → LoRA Router
    - Canal OUT: Decisión → [TRM/LLM/NLLB] → TTS → Playback
    - Orchestrator: Coordina ambos canales
"""

from .vosk_streaming import VoskSTTStreaming, VoskStreamingSession
from .lora_router import LoRARouter
from .audio_emotion_lite import EmotionAudioLite, EmotionResult
from .input_thread import InputThread
from .output_thread import OutputThread
from .orchestrator import Layer1Orchestrator

__all__ = [
    'VoskSTTStreaming',
    'VoskStreamingSession',
    'LoRARouter',
    'EmotionAudioLite',
    'EmotionResult',
    'InputThread',
    'OutputThread',
    'Layer1Orchestrator'
]
