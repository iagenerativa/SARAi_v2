"""
MeloTTS Engine for SARAi v2.17
High-quality Spanish TTS with natural expressiveness
"""

import numpy as np
import time
import os


class MeloTTSEngine:
    """Wrapper for MeloTTS - Spanish voice"""
    
    def __init__(self, language='ES', speaker='ES', device='cpu', speed=1.0, preload=False):
        """
        Args:
            language: Idioma del modelo (ES para español)
            speaker: ID del speaker (ES para español neutro)
            device: 'cpu' o 'cuda'
            speed: Velocidad de síntesis (1.0 = normal, 1.3 = más rápido)
            preload: Si True, carga el modelo inmediatamente
        """
        self.language = language
        self.speaker = speaker
        self.device = device
        self.speed = speed
        self.model = None
        self.speaker_id = None
        self.sample_rate = 44100  # MeloTTS usa 44.1kHz
        self.audio_cache = {}  # Cache de audio para respuestas comunes
        
        if preload:
            self._lazy_load()
    
    def _lazy_load(self):
        """Carga el modelo solo cuando se necesita"""
        if self.model is not None:
            return
        
        print(f"Loading MeloTTS ({self.language})...")
        start = time.perf_counter()
        
        try:
            from melo.api import TTS
            
            # Cargar modelo
            self.model = TTS(language=self.language, device=self.device)
            self.speaker_id = self.model.hps.data.spk2id[self.speaker]
            
            load_time = (time.perf_counter() - start) * 1000
            print(f"MeloTTS loaded in {load_time:.0f}ms")
            print(f"  Available speakers: {list(self.model.hps.data.spk2id.keys())}")
            print(f"  Selected speaker: {self.speaker} (ID: {self.speaker_id})")
            
        except Exception as e:
            print(f"❌ Error cargando MeloTTS: {e}")
            print("\nIntenta instalar dependencias:")
            print("  pip3 install git+https://github.com/myshell-ai/MeloTTS.git")
            print("  pip3 install mecab-python3 unidic-lite")
            raise
    
    def synthesize(self, text):
        """
        Sintetiza texto a audio con caché
        
        Args:
            text: Texto a sintetizar
            
        Returns:
            numpy array con audio (float32, sample_rate=44100)
        """
        self._lazy_load()
        
        if not text or not text.strip():
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)
        
        # Normalizar texto para caché
        text_normalized = text.strip().lower()
        
        # Verificar caché
        if text_normalized in self.audio_cache:
            return self.audio_cache[text_normalized].copy()
        
        try:
            import tempfile
            import soundfile as sf
            
            # Generar audio temporal
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Sintetizar
            self.model.tts_to_file(
                text,
                self.speaker_id,
                tmp_path,
                speed=self.speed
            )
            
            # Leer audio generado
            audio, sr = sf.read(tmp_path)
            
            # Eliminar archivo temporal
            os.unlink(tmp_path)
            
            # Asegurar formato correcto
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalizar si es necesario
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.abs(audio).max()
            
            # Convertir a mono si es estéreo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Guardar en caché si es texto corto (< 50 caracteres)
            if len(text_normalized) < 50:
                self.audio_cache[text_normalized] = audio.copy()
            
            return audio
            
        except Exception as e:
            print(f"TTS error: {e}")
            return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
    
    def get_sample_rate(self):
        """Retorna sample rate del modelo"""
        return self.sample_rate


# Función de utilidad para testing
def test_melo():
    """Prueba rápida de MeloTTS"""
    import soundfile as sf
    
    print("🧪 Test de MeloTTS\n")
    
    engine = MeloTTSEngine(language='ES', speaker='ES', speed=1.0)
    
    text = "Hola, soy SARAi. Esta es mi voz española nativa con MeloTTS."
    
    print(f"Texto: \"{text}\"\n")
    print("Generando audio...")
    
    start = time.perf_counter()
    audio = engine.synthesize(text)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"✅ Audio generado en {elapsed:.0f}ms")
    print(f"   Shape: {audio.shape}")
    print(f"   Sample rate: {engine.sample_rate} Hz")
    print(f"   Duración: {len(audio)/engine.sample_rate:.2f}s")
    
    output_file = "/tmp/test_melo_sarai.wav"
    sf.write(output_file, audio, engine.sample_rate)
    print(f"\n🔊 Guardado en: {output_file}")
    print(f"   Escucha: aplay {output_file}")
    
    return audio


if __name__ == "__main__":
    test_melo()
