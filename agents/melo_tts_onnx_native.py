"""
Wrapper para melotts-onnx (versión nativa ONNX optimizada).

Esta implementación usa modelos ONNX pre-exportados que son 2-3x más
rápidos que la versión PyTorch en CPU.

Beneficios:
- ONNX Runtime con instrucciones AVX/AVX2/AVX512
- Modelos pre-cuantizados y optimizados
- Sin overhead de PyTorch
- Latencia típica: 300-500ms (vs 1000-1500ms PyTorch)
"""

import os
import time
import tempfile
import numpy as np
import soundfile as sf
import logging
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class MeloTTSONNXEngine:
    """
    Engine TTS basado en melotts-onnx (ONNX Runtime nativo).
    
    API compatible con MeloTTSEngine pero con mejor rendimiento.
    """
    
    def __init__(
        self,
        language: str = "ES",
        speaker: str = "ES",
        device: str = "cpu",
        speed: float = 1.0,
        preload: bool = False
    ):
        """
        Args:
            language: Código de idioma (ES, EN, FR, ZH, JP, KR)
            speaker: ID del speaker (por defecto igual al idioma)
            device: 'cpu' (ONNX Runtime solo soporta CPU por ahora)
            speed: Factor de velocidad (1.0 = normal, 1.3 = 30% más rápido)
            preload: Si True, carga modelos inmediatamente
        """
        self.language = language.upper()
        self.speaker = speaker.upper()
        self.speed = speed
        
        self.model = None
        self._load_time = None
        
        # Cache de audio para respuestas cortas
        self.audio_cache: Dict[str, np.ndarray] = {}
        
        # Mapeo de códigos de idioma
        self._lang_map = {
            "ES": "ES",  # Español
            "EN": "EN",  # Inglés
            "FR": "FR",  # Francés
            "ZH": "ZH",  # Chino
            "JP": "JP",  # Japonés
            "KR": "KR",  # Coreano
        }
        
        if preload:
            self._lazy_load()
    
    def _lazy_load(self):
        """Carga el modelo ONNX solo cuando se necesita."""
        if self.model is not None:
            return
        
        start = time.perf_counter()
        print(f"Loading MeloTTS-ONNX ({self.language})...")
        
        try:
            from melo_onnx import MeloTTS_ONNX
        except ImportError:
            raise RuntimeError(
                "melotts-onnx no instalado. "
                "Ejecuta: pip3 install melotts-onnx"
            )
        
        # Crear directorio de modelos si no existe
        model_dir = Path("models/melo_onnx")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar modelo ONNX
        lang_code = self._lang_map.get(self.language, "ES")
        
        self.model = MeloTTS_ONNX(
            language=lang_code,
            device="cpu",
            model_path=str(model_dir)
        )
        
        self._load_time = (time.perf_counter() - start) * 1000
        
        print(f"MeloTTS-ONNX loaded in {self._load_time:.0f}ms")
        print(f"  Language: {lang_code}")
        print(f"  Speed: {self.speed}x")
        print(f"  Backend: ONNX Runtime")
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Genera audio a partir de texto usando ONNX Runtime.
        
        Args:
            text: Texto a sintetizar
            
        Returns:
            Audio como array numpy float32, sample rate 44100 Hz
        """
        # Comprobar caché primero
        cache_key = text.lower().strip()
        if cache_key in self.audio_cache:
            return self.audio_cache[cache_key].copy()
        
        # Lazy load del modelo
        self._lazy_load()
        
        # Crear archivo temporal para el audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            # Generar audio con ONNX
            # La API de melo_onnx genera directamente a archivo
            self.model.synthesize(
                text=text,
                output_path=temp_path,
                speaker_id=self.speaker,
                speed=self.speed
            )
            
            # Leer audio generado
            audio, sr = sf.read(temp_path)
            
            # Verificar sample rate
            if sr != 44100:
                # Resamplear si es necesario
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
            
            # Convertir a float32 si es necesario
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalizar amplitud
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val * 0.95
            
            # Guardar en caché si es texto corto
            if len(text) < 50:
                self.audio_cache[cache_key] = audio.copy()
            
            return audio
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def get_sample_rate(self) -> int:
        """Retorna la frecuencia de muestreo del audio."""
        return 44100
    
    def clear_cache(self):
        """Limpia el caché de audio."""
        self.audio_cache.clear()
    
    def get_stats(self) -> dict:
        """Retorna estadísticas del engine."""
        return {
            "model_loaded": self.model is not None,
            "load_time_ms": self._load_time,
            "cache_size": len(self.audio_cache),
            "language": self.language,
            "speaker": self.speaker,
            "speed": self.speed,
            "backend": "ONNX Runtime"
        }


def benchmark_onnx_native():
    """
    Benchmark de melotts-onnx nativo.
    """
    print("=" * 70)
    print("🚀 Benchmark: MeloTTS-ONNX (Nativo)")
    print("=" * 70 + "\n")
    
    test_phrases = [
        "Hola",
        "Buenos días",
        "¿Cómo estás?",
        "El sistema está funcionando correctamente",
        "Procesando tu solicitud",
        "Hola",  # Repetida para test de caché
    ]
    
    # Crear engine
    print("📥 Cargando MeloTTS-ONNX...\n")
    engine = MeloTTSONNXEngine(language="ES", speed=1.3, preload=True)
    
    print("\n📊 Síntesis de frases:\n")
    
    times = []
    for i, phrase in enumerate(test_phrases, 1):
        start = time.perf_counter()
        audio = engine.synthesize(phrase)
        elapsed = (time.perf_counter() - start) * 1000
        
        duration = len(audio) / 44100
        rtf = elapsed / 1000 / duration if duration > 0 else 0
        
        cached = " (CACHE)" if phrase == "Hola" and i == 6 else ""
        
        print(f"[{i}] '{phrase}'{cached}")
        print(f"    Latencia: {elapsed:6.0f}ms | Audio: {duration:.2f}s | RTF: {rtf:.2f}x")
        
        if not cached:  # No contar cache en promedios
            times.append(elapsed)
    
    # Estadísticas
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print("\n" + "=" * 70)
    print("📈 RESULTADOS\n")
    print(f"  Promedio:  {avg_time:6.0f}ms")
    print(f"  Mínimo:    {min_time:6.0f}ms")
    print(f"  Máximo:    {max_time:6.0f}ms")
    print(f"\n  Backend:   ONNX Runtime")
    print(f"  Speed:     1.3x (30% más rápido)")
    print(f"  Cache:     {engine.get_stats()['cache_size']} entradas")
    print("=" * 70)


if __name__ == "__main__":
    # Ejecutar benchmark
    benchmark_onnx_native()
