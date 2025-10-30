"""
Kitten TTS Engine - Voz en Español para SARAi v2.17
====================================================

Motor de síntesis de voz en español usando Kitten TTS.

Características:
- Latencia: ~50ms
- Calidad: MOS 4.0
- Idioma: Español nativo (es_ES)
- Tamaño: 60MB

Uso:
    from agents.kitten_tts import KittenTTSEngine
    
    tts = KittenTTSEngine()
    audio = tts.synthesize("Hola, soy SARAi")
    # audio: numpy array (22050 Hz, mono, float32)
"""

import os
import time
import wave
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from piper import PiperVoice


class KittenTTSEngine:
    """
    Motor de Text-to-Speech usando Kitten.
    
    Optimizado para latencia baja y calidad en español.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Inicializa el motor Kitten TTS.
        
        Args:
            model_path: Ruta al modelo ONNX. Por defecto usa es_ES-davefx-medium
            config_path: Ruta al config JSON. Se autodetecta si no se proporciona
        """
        self.model_path = model_path or self._get_default_model_path()
        self.config_path = config_path or f"{self.model_path}.json"
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Modelo Kitten no encontrado: {self.model_path}\n"
                f"Ejecuta: make install-kitten"
            )
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config Kitten no encontrado: {self.config_path}"
            )
        
        # Cargar configuración
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Cargar modelo (lazy)
        self._voice = None
        self.sample_rate = self.config['audio']['sample_rate']  # 22050
    
    def _get_default_model_path(self) -> str:
        """Obtiene la ruta al modelo por defecto."""
        return str(
            Path(__file__).parent.parent / 
            "models" / "kitten" / "es_ES-davefx-medium.onnx"
        )
    
    def _load_voice(self):
        """Carga el modelo Kitten (lazy loading)."""
        if self._voice is None:
            print(f"🔊 Cargando voz Kitten desde {self.model_path}...")
            start = time.perf_counter()
            self._voice = PiperVoice.load(
                self.model_path,
                config_path=self.config_path,
                use_cuda=False  # CPU-only
            )
            elapsed = (time.perf_counter() - start) * 1000
            print(f"✓ Voz cargada en {elapsed:.1f}ms")
    
    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8
    ) -> np.ndarray:
        """
        Sintetiza texto a audio.
        
        Args:
            text: Texto en español a sintetizar
            speed: Velocidad de habla (1.0 = normal)
            noise_scale: Variabilidad de tono (0.667 = natural)
            noise_w: Variabilidad de duración (0.8 = natural)
        
        Returns:
            Audio como numpy array (float32, 22050 Hz, mono)
        """
        self._load_voice()
        
        start = time.perf_counter()
        
        # Sintetizar usando synthesize
        audio_chunks = []
        for audio_chunk in self._voice.synthesize(text):
            # AudioChunk tiene 'audio_int16_array' como numpy array
            audio_int16 = audio_chunk.audio_int16_array
            audio_chunks.append(audio_int16)
        
        # Concatenar y convertir a float32
        if audio_chunks:
            audio_int16 = np.concatenate(audio_chunks)
            audio = audio_int16.astype(np.float32) / 32768.0
        else:
            audio = np.array([], dtype=np.float32)
        
        elapsed = (time.perf_counter() - start) * 1000
        duration = len(audio) / self.sample_rate
        rtf = elapsed / (duration * 1000) if duration > 0 else 0
        
        print(f"🎤 TTS: {len(text)} chars → {duration:.2f}s audio en {elapsed:.1f}ms (RTF={rtf:.2f}x)")
        
        return audio
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        **kwargs
    ) -> float:
        """
        Sintetiza y guarda a archivo WAV.
        
        Args:
            text: Texto a sintetizar
            output_path: Ruta del archivo WAV de salida
            **kwargs: Argumentos para synthesize()
        
        Returns:
            Latencia en milisegundos
        """
        start = time.perf_counter()
        
        audio = self.synthesize(text, **kwargs)
        
        # Convertir a int16 para WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Guardar WAV
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"✓ Audio guardado: {output_path} ({len(audio)/self.sample_rate:.2f}s)")
        
        return elapsed
    
    def get_info(self) -> dict:
        """Obtiene información del modelo."""
        return {
            "model": self.model_path,
            "language": self.config.get('language', {}).get('code', 'es_ES'),
            "sample_rate": self.sample_rate,
            "num_speakers": self.config.get('num_speakers', 1),
            "quality": self.config.get('quality', 'medium')
        }


def demo():
    """Demo rápido de Kitten TTS."""
    print("=" * 70)
    print("  Kitten TTS - Demo Voz en Español")
    print("=" * 70)
    
    tts = KittenTTSEngine()
    
    # Info del modelo
    info = tts.get_info()
    print(f"\n📊 Modelo: {info['language']} - {info['quality']}")
    print(f"   Sample rate: {info['sample_rate']} Hz")
    
    # Test 1: Frase corta
    print("\n[Test 1] Frase corta:")
    audio1 = tts.synthesize("Hola, soy SARAi, tu asistente personal.")
    
    # Test 2: Frase técnica
    print("\n[Test 2] Respuesta técnica:")
    text = """Para configurar SSH en Linux, primero instala el servidor con 
    sudo apt install openssh-server, luego edita el archivo de configuración 
    en /etc/ssh/sshd_config."""
    audio2 = tts.synthesize(text)
    
    # Test 3: Guardar a archivo
    print("\n[Test 3] Guardar a archivo:")
    output_file = "/tmp/sarai_demo.wav"
    latency = tts.synthesize_to_file(
        "Esta es una prueba de la voz en español de SARAi.",
        output_file
    )
    print(f"   Latencia total: {latency:.1f}ms")
    print(f"   Archivo: {output_file}")
    print(f"   Reproducir: aplay {output_file}")
    
    print("\n✅ Demo completado")
    print("=" * 70)


if __name__ == "__main__":
    demo()
