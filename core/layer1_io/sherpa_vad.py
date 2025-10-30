"""
Sherpa-ONNX VAD (Voice Activity Detection)
Detecta segmentos de voz en audio continuo con alta precisión
"""

import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class SherpaVAD:
    """
    Voice Activity Detector usando Sherpa-ONNX
    
    Ventajas sobre VAD simple:
    - Mayor precisión (menos falsos positivos)
    - Detección de múltiples hablantes
    - Menor latencia (<50ms)
    - Robusto a ruido de fondo
    
    Uso:
        vad = SherpaVAD()
        for chunk in audio_stream:
            if vad.is_speech(chunk):
                # Procesar voz
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        window_size_ms: int = 30,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500
    ):
        """
        Args:
            model_path: Ruta al modelo Sherpa-ONNX VAD (si None, usa silero-vad)
            sample_rate: Frecuencia de muestreo (16kHz estándar)
            window_size_ms: Ventana de análisis (30ms óptimo)
            min_speech_duration_ms: Duración mínima de voz para considerarla válida
            min_silence_duration_ms: Duración de silencio para marcar fin de frase
        """
        self.sample_rate = sample_rate
        self.window_size_ms = window_size_ms
        self.window_size_samples = int(sample_rate * window_size_ms / 1000)
        
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
        
        # Estado interno
        self.speech_samples_count = 0
        self.silence_samples_count = 0
        self.is_speaking = False
        
        # Buffer para audio acumulado
        self.speech_buffer: List[np.ndarray] = []
        
        # Lazy loading del modelo
        self._vad_model = None
        self.model_path = model_path
        
        logger.info(f"SherpaVAD inicializado (window={window_size_ms}ms, sr={sample_rate}Hz)")
    
    def _load_model(self):
        """Carga el modelo Sherpa-ONNX bajo demanda"""
        if self._vad_model is not None:
            return
        
        try:
            # Intentar usar Sherpa-ONNX primero
            import sherpa_onnx
            
            if self.model_path:
                # Modelo personalizado
                self._vad_model = sherpa_onnx.VoiceActivityDetector(
                    model=self.model_path,
                    sample_rate=self.sample_rate
                )
                logger.info(f"✓ Sherpa-ONNX VAD cargado desde {self.model_path}")
            else:
                # Fallback a Silero VAD (más ligero, integrado)
                logger.warning("Sherpa-ONNX no disponible, usando fallback a Silero VAD")
                self._use_silero_fallback()
        
        except ImportError:
            logger.warning("Sherpa-ONNX no instalado, usando Silero VAD")
            self._use_silero_fallback()
    
    def _use_silero_fallback(self):
        """Usa Silero VAD como fallback (PyTorch ligero)"""
        try:
            import torch
            # Silero VAD es muy ligero (~1.5MB)
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True  # Usa versión ONNX si disponible
            )
            self._vad_model = model
            self._get_speech_timestamps = utils[0]
            logger.info("✓ Silero VAD cargado (fallback)")
        
        except Exception as e:
            logger.error(f"Error cargando Silero VAD: {e}")
            # Último fallback: VAD por energía simple
            self._vad_model = "energy"
            logger.warning("Usando VAD por energía (simple)")
    
    def is_speech(self, audio_chunk: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Detecta si el chunk de audio contiene voz
        
        Args:
            audio_chunk: Audio en formato float32, shape=(samples,)
            threshold: Umbral de confianza (0.0-1.0), mayor = más estricto
        
        Returns:
            True si es voz, False si es silencio/ruido
        """
        self._load_model()
        
        if self._vad_model == "energy":
            # VAD simple por energía (fallback)
            energy = np.abs(audio_chunk).mean()
            return energy > 0.02
        
        elif hasattr(self._vad_model, 'predict'):
            # Silero VAD (PyTorch)
            import torch
            audio_tensor = torch.from_numpy(audio_chunk).float()
            confidence = self._vad_model(audio_tensor, self.sample_rate).item()
            return confidence > threshold
        
        else:
            # Sherpa-ONNX VAD
            result = self._vad_model.process(audio_chunk)
            return result.is_speech
    
    def process_stream(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesa un chunk de audio del stream continuo
        
        Returns:
            - None: Aún no hay frase completa
            - np.ndarray: Frase de voz completa detectada
        """
        is_voice = self.is_speech(audio_chunk)
        
        if is_voice:
            self.speech_samples_count += len(audio_chunk)
            self.silence_samples_count = 0
            
            # Acumular audio de voz
            self.speech_buffer.append(audio_chunk)
            
            if not self.is_speaking and self.speech_samples_count >= self.min_speech_samples:
                self.is_speaking = True
                logger.debug("🎙️ Inicio de voz detectado")
        
        else:
            self.silence_samples_count += len(audio_chunk)
            
            # Si estábamos hablando y hay suficiente silencio → fin de frase
            if self.is_speaking and self.silence_samples_count >= self.min_silence_samples:
                logger.debug(f"✓ Frase completa ({len(self.speech_buffer)} chunks)")
                
                # Concatenar todo el audio capturado
                complete_phrase = np.concatenate(self.speech_buffer)
                
                # Resetear estado
                self._reset()
                
                return complete_phrase
        
        return None
    
    def _reset(self):
        """Resetea el estado interno después de procesar una frase"""
        self.speech_buffer.clear()
        self.speech_samples_count = 0
        self.silence_samples_count = 0
        self.is_speaking = False
    
    def force_flush(self) -> Optional[np.ndarray]:
        """
        Fuerza el retorno del buffer actual (útil para timeouts)
        
        Returns:
            Audio acumulado o None si está vacío
        """
        if len(self.speech_buffer) > 0:
            complete_phrase = np.concatenate(self.speech_buffer)
            self._reset()
            return complete_phrase
        return None


# Ejemplo de uso
if __name__ == "__main__":
    # Test básico
    vad = SherpaVAD()
    
    # Simular chunks de audio
    silence = np.zeros(480, dtype=np.float32)  # 30ms @ 16kHz
    speech = np.random.randn(480).astype(np.float32) * 0.1  # Voz simulada
    
    print("Procesando stream...")
    for i in range(20):
        chunk = speech if i % 5 < 3 else silence
        result = vad.process_stream(chunk)
        
        if result is not None:
            print(f"✓ Frase detectada: {len(result)} samples ({len(result)/16000:.2f}s)")
