"""
Vosk STT (Speech-to-Text)
Wrapper optimizado del reconocedor Vosk para español
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
import logging
from vosk import Model, KaldiRecognizer

logger = logging.getLogger(__name__)


class VoskSTT:
    """
    Speech-to-Text usando Vosk
    
    Características:
    - 100% offline
    - Latencia baja (~100-200ms)
    - Soporte streaming
    - Modelos pequeños (~38MB español)
    
    Uso:
        stt = VoskSTT()
        text = stt.transcribe(audio_array)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 16000
    ):
        """
        Args:
            model_path: Ruta al modelo Vosk (default: vosk-model-small-es-0.42)
            sample_rate: Frecuencia de muestreo (16kHz estándar)
        """
        self.sample_rate = sample_rate
        
        # Determinar ruta del modelo
        if model_path is None:
            base_path = Path(__file__).parent.parent.parent
            model_path = base_path / "models/vosk/vosk-model-small-es-0.42"
        
        self.model_path = Path(model_path)
        
        # Lazy loading
        self._model = None
        self._recognizer = None
        
        logger.info(f"VoskSTT inicializado (modelo: {self.model_path.name})")
    
    def _load_model(self):
        """Carga el modelo Vosk bajo demanda"""
        if self._model is not None:
            return
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo Vosk no encontrado: {self.model_path}\n"
                f"Descárgalo con:\n"
                f"  cd models/vosk\n"
                f"  wget https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip\n"
                f"  unzip vosk-model-small-es-0.42.zip"
            )
        
        self._model = Model(str(self.model_path))
        logger.info(f"✓ Modelo Vosk cargado desde {self.model_path}")
    
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio a texto
        
        Args:
            audio: Audio en formato float32, shape=(samples,)
        
        Returns:
            Texto transcrito (string vacío si no detecta nada)
        """
        self._load_model()
        
        # Crear recognizer para este audio
        recognizer = KaldiRecognizer(self._model, self.sample_rate)
        
        # Convertir a int16 para Vosk
        audio_int16 = (audio * 32768).astype(np.int16)
        
        # Procesar todo el audio
        recognizer.AcceptWaveform(audio_int16.tobytes())
        
        # Obtener resultado final
        result = json.loads(recognizer.FinalResult())
        
        return result.get('text', '').strip()
    
    def transcribe_streaming(self, audio_chunk: np.ndarray) -> dict:
        """
        Transcripción streaming (chunk por chunk)
        
        Returns:
            dict con:
                - 'partial': Transcripción parcial (mientras habla)
                - 'final': Transcripción final (cuando termina)
        """
        self._load_model()
        
        # Crear recognizer si no existe (para streaming persistente)
        if self._recognizer is None:
            self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
        
        # Convertir a int16
        audio_int16 = (audio_chunk * 32768).astype(np.int16)
        
        # Procesar chunk
        if self._recognizer.AcceptWaveform(audio_int16.tobytes()):
            # Chunk completó una frase
            result = json.loads(self._recognizer.Result())
            return {
                'partial': '',
                'final': result.get('text', '').strip()
            }
        else:
            # Aún procesando
            partial = json.loads(self._recognizer.PartialResult())
            return {
                'partial': partial.get('partial', '').strip(),
                'final': ''
            }
    
    def reset_streaming(self):
        """Resetea el recognizer streaming (útil para nueva frase)"""
        self._recognizer = None


# Ejemplo de uso
if __name__ == "__main__":
    stt = VoskSTT()
    
    # Test con audio simulado
    import wave
    print("Cargando modelo...")
    
    # Simular audio de voz (en producción vendría de VAD)
    audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1  # 2s de "audio"
    
    print("Transcribiendo...")
    text = stt.transcribe(audio)
    print(f"Resultado: '{text}'")
