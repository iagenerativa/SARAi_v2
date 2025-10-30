"""
Kitten TTS Engine for SARAi v2.17
Ultra-lightweight TTS using KittenML
"""

import numpy as np
import time
import re


class KittenTTSEngine:
    """Wrapper for Kitten TTS"""
    
    # Diccionario de correcciones fonéticas para forzar pronunciación española
    SPANISH_PHONETIC_FIXES = {
        # Palabras comunes mal pronunciadas
        r'\bdetalles\b': 'de-ta-yes',
        r'\bdetalle\b': 'de-ta-ye',
        r'\bmás\b': 'mas',
        r'\bestás\b': 'es-tas',
        r'\bestá\b': 'es-ta',
        r'\bpuedes\b': 'pue-des',
        r'\bpuede\b': 'pue-de',
        r'\bquieres\b': 'quie-res',
        r'\bquiere\b': 'quie-re',
        r'\bdatos\b': 'da-tos',
        r'\bdato\b': 'da-to',
        r'\barchivos\b': 'ar-chi-vos',
        r'\barchivo\b': 'ar-chi-vo',
        r'\bcarpetas\b': 'car-pe-tas',
        r'\bcarpeta\b': 'car-pe-ta',
        # Tecnicismos anglicismos
        r'\bserver\b': 'ser-vi-dor',
        r'\berror\b': 'e-rror',
        r'\binternet\b': 'in-ter-net',
        r'\bemail\b': 'i-meil',
        r'\bweb\b': 'güeb',
        r'\bonline\b': 'en línea',
        r'\boffline\b': 'desconectado',
        r'\bpassword\b': 'contraseña',
        r'\bclick\b': 'clic',
        r'\bdownload\b': 'descargar',
        r'\bupload\b': 'subir',
        r'\bfile\b': 'archivo',
        r'\bfiles\b': 'archivos',
        r'\bfolder\b': 'carpeta',
        r'\bfolders\b': 'carpetas',
        # Tecnicismos
        r'\bAPI\b': 'a-pe-i',
        r'\bUSB\b': 'u-ese-be',
        r'\bRAM\b': 'ram',
        r'\bCPU\b': 'ce-pe-u',
        r'\bGPU\b': 'ge-pe-u',
        r'\bSSD\b': 'e-se-e-se-de',
        r'\bHDD\b': 'a-che-de-de',
        # Verbos comunes
        r'\bpuedo\b': 'pue-do',
        r'\btengo\b': 'ten-go',
        r'\btienes\b': 'tie-nes',
        r'\bhay\b': 'ai',
        r'\bes\b': 'es',
        r'\bson\b': 'son',
    }
    
    def __init__(self, model_path=None, voice="expr-voice-4-f", speed=1.2, fix_spanish=True):
        """
        Args:
            model_path: No usado (compatibilidad), modelo se descarga automáticamente
            voice: Voz a usar (expr-voice-4-f = española neutral)
            speed: Multiplicador de velocidad (1.2 = 20% más rápido)
            fix_spanish: Si True, aplica correcciones fonéticas para español
        """
        self.voice = voice
        self.speed = speed
        self.fix_spanish = fix_spanish
        self.model = None
        self.sample_rate = 24000
    
    def _lazy_load(self):
        if self.model is not None:
            return
        
        print(f"Loading Kitten TTS (voice: {self.voice})...")
        start = time.perf_counter()
        
        from kittentts import KittenTTS
        self.model = KittenTTS("KittenML/kitten-tts-nano-0.2")
        
        load_time = (time.perf_counter() - start) * 1000
        print(f"Kitten TTS loaded in {load_time:.0f}ms")
    
    def _normalize_spanish_text(self, text):
        """Normaliza texto para mejorar pronunciación española"""
        if not self.fix_spanish:
            return text
        
        # Aplicar correcciones fonéticas
        normalized = text
        for pattern, replacement in self.SPANISH_PHONETIC_FIXES.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def synthesize(self, text):
        self._lazy_load()
        
        if not text or not text.strip():
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)
        
        try:
            # Normalizar texto para pronunciación española
            normalized_text = self._normalize_spanish_text(text)
            
            # Generar audio con voz configurada
            audio = self.model.generate(normalized_text, voice=self.voice)
            
            # Acelerar audio si speed > 1.0
            if self.speed != 1.0:
                from scipy.signal import resample
                target_length = int(len(audio) / self.speed)
                audio = resample(audio, target_length)
            
            # Normalizar a float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.abs(audio).max()
            
            return audio
        except Exception as e:
            print(f"TTS error: {e}")
            return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
    
    def get_sample_rate(self):
        return self.sample_rate
