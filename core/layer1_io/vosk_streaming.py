"""
SARAi v2.17 - Vosk STT Streaming
Speech-to-Text en modo streaming (chunks de 100ms)
"""

import json
import numpy as np
from vosk import Model, KaldiRecognizer
from pathlib import Path
from typing import Optional, Dict
import time


class VoskSTTStreaming:
    """
    Vosk STT en modo streaming para procesamiento continuo
    Genera transcripciones parciales cada 100ms
    """
    
    def __init__(self, model_path: str = "models/vosk/vosk-model-small-es-0.42"):
        """
        Args:
            model_path: Ruta al modelo Vosk
        """
        self.base_path = Path(__file__).parent.parent.parent
        self.model_path = self.base_path / model_path
        
        print(f"🎤 Cargando Vosk STT desde {self.model_path}...")
        start = time.perf_counter()
        
        self.model = Model(str(self.model_path))
        self.recognizer = None
        self.sample_rate = 16000
        
        load_time = (time.perf_counter() - start) * 1000
        print(f"✓ Vosk STT cargado en {load_time:.0f}ms")
    
    def create_recognizer(self) -> KaldiRecognizer:
        """
        Crea un nuevo recognizer para una sesión de streaming
        Cada sesión de audio necesita su propio recognizer
        """
        return KaldiRecognizer(self.model, self.sample_rate)
    
    def process_chunk(self, recognizer: KaldiRecognizer, audio_chunk: np.ndarray) -> Dict:
        """
        Procesa un chunk de audio (100ms típicamente)
        
        Args:
            recognizer: KaldiRecognizer activo
            audio_chunk: Audio en float32 [-1, 1]
        
        Returns:
            {
                "partial": str,      # Transcripción parcial (mientras habla)
                "final": str,        # Transcripción final (si detecta fin)
                "is_final": bool     # True si es resultado final
            }
        """
        # Convertir float32 a int16
        audio_int16 = (audio_chunk * 32768).astype(np.int16)
        
        # Procesar chunk
        if recognizer.AcceptWaveform(audio_int16.tobytes()):
            # Resultado final (detectó fin de frase)
            result = json.loads(recognizer.Result())
            return {
                "partial": "",
                "final": result.get("text", ""),
                "is_final": True
            }
        else:
            # Resultado parcial (aún hablando)
            partial_result = json.loads(recognizer.PartialResult())
            return {
                "partial": partial_result.get("partial", ""),
                "final": "",
                "is_final": False
            }
    
    def transcribe_complete(self, audio: np.ndarray) -> str:
        """
        Transcripción completa (modo no-streaming, para compatibilidad)
        
        Args:
            audio: Audio completo en float32 [-1, 1]
        
        Returns:
            Texto transcrito
        """
        recognizer = self.create_recognizer()
        
        # Convertir a int16
        audio_int16 = (audio * 32768).astype(np.int16)
        
        # Procesar todo el audio
        recognizer.AcceptWaveform(audio_int16.tobytes())
        
        # Obtener resultado final
        result = json.loads(recognizer.FinalResult())
        return result.get("text", "")
    
    def reset_recognizer(self, recognizer: KaldiRecognizer):
        """
        Resetea el recognizer para nueva sesión
        (En Vosk, se recomienda crear uno nuevo)
        """
        # Vosk no tiene reset, retornar nuevo recognizer
        return self.create_recognizer()


class VoskStreamingSession:
    """
    Sesión de streaming Vosk con buffer inteligente
    Acumula texto parcial y detecta fin de frase
    """
    
    def __init__(self, vosk_stt: VoskSTTStreaming):
        self.vosk = vosk_stt
        self.recognizer = vosk_stt.create_recognizer()
        self.partial_text = ""
        self.complete_sentences = []
        self.last_partial_time = time.time()
        self.silence_threshold = 1.5  # 1500ms sin cambios = fin de frase (más tolerante)
    
    def feed_audio(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Alimenta chunk de audio y retorna frase completa si detecta fin
        
        Args:
            audio_chunk: Chunk de audio (100ms típicamente)
        
        Returns:
            Frase completa si detectó fin, None si aún procesando
        """
        result = self.vosk.process_chunk(self.recognizer, audio_chunk)
        
        if result["is_final"] and result["final"]:
            # Frase completa detectada por Vosk
            complete = result["final"]
            self.complete_sentences.append(complete)
            self.partial_text = ""
            return complete
        
        elif result["partial"]:
            # Actualizar texto parcial
            if result["partial"] != self.partial_text:
                self.partial_text = result["partial"]
                self.last_partial_time = time.time()
            
            # Detectar fin por silencio (timeout)
            elif time.time() - self.last_partial_time > self.silence_threshold:
                if self.partial_text:
                    # Timeout alcanzado, considerar completo
                    complete = self.partial_text
                    self.complete_sentences.append(complete)
                    self.partial_text = ""
                    
                    # Resetear recognizer para nueva frase
                    self.recognizer = self.vosk.reset_recognizer(self.recognizer)
                    return complete
        
        return None
    
    def get_partial_text(self) -> str:
        """Retorna texto parcial actual (para preview)"""
        return self.partial_text
    
    def get_complete_sentences(self) -> list:
        """Retorna todas las frases completas acumuladas"""
        return self.complete_sentences.copy()
    
    def reset(self):
        """Resetea la sesión (nueva conversación)"""
        self.recognizer = self.vosk.reset_recognizer(self.recognizer)
        self.partial_text = ""
        self.complete_sentences = []
        self.last_partial_time = time.time()


# ============ Test de Streaming ============
if __name__ == "__main__":
    import sounddevice as sd
    
    print("=== Test Vosk STT Streaming ===\n")
    
    # Inicializar Vosk
    vosk = VoskSTTStreaming()
    session = VoskStreamingSession(vosk)
    
    # Configurar captura de audio
    chunk_duration = 0.1  # 100ms
    chunk_samples = int(vosk.sample_rate * chunk_duration)
    
    print(f"🎤 Habla naturalmente (Ctrl+C para salir)\n")
    print("Texto parcial aparecerá mientras hablas...")
    print("Frases completas se mostrarán al detectar fin\n")
    print("-" * 60)
    
    def audio_callback(indata, frames, time_info, status):
        """Callback para procesamiento streaming"""
        audio_chunk = indata[:, 0]  # Mono
        
        # Procesar chunk
        complete_sentence = session.feed_audio(audio_chunk)
        
        if complete_sentence:
            print(f"\n✅ Frase completa: \"{complete_sentence}\"")
            print("-" * 60)
        else:
            partial = session.get_partial_text()
            if partial:
                print(f"\r⏳ Parcial: {partial}...", end='', flush=True)
    
    try:
        with sd.InputStream(
            samplerate=vosk.sample_rate,
            channels=1,
            blocksize=chunk_samples,
            callback=audio_callback
        ):
            print("🔴 Streaming activo...")
            while True:
                sd.sleep(100)
    
    except KeyboardInterrupt:
        print("\n\n✓ Streaming detenido")
        print(f"\nTotal frases capturadas: {len(session.get_complete_sentences())}")
        for i, sentence in enumerate(session.get_complete_sentences(), 1):
            print(f"  {i}. {sentence}")
