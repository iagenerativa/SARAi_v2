#!/usr/bin/env python3
"""
agents/audio_router.py - Audio Router con Fallback Sentinel v2.11

Copyright (c) 2025 Noel
Licensed under CC BY-NC-SA 4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/

Este archivo es parte de SARAi v2.11 "Omni-Sentinel".
No se permite uso comercial sin permiso del autor.

---

Audio Router con Detección de Idioma (LID) y Fallback Sentinel

Decide qué pipeline de IA (Omni-3B o NLLB) debe procesar el audio,
con un fallback seguro a Omni-Español si falla la detección de idioma.

Filosofía v2.11 "Home Sentinel":
    "El sistema nunca falla, se degrada elegantemente."

Pipeline:
    Audio → Whisper-tiny (LID) → fasttext (detección idioma)
          ├─► Idioma en OMNI_LANGS (es, en) → Omni-3B (alta empatía)
          ├─► Idioma en NLLB_LANGS (fr, de, ja) → NLLB (traducción)
          └─► Fallo o idioma desconocido → SENTINEL FALLBACK (omni-es)

KPIs:
    - Latencia LID: <50ms (Whisper-tiny + fasttext)
    - Precisión LID: >95% (idiomas conocidos)
    - Fallback rate: <5% (solo idiomas desconocidos)

Autor: SARAi v2.11 "Home Sentinel"
"""

import os
import logging
from typing import Tuple, Optional
from pathlib import Path

import numpy as np

# Whisper-tiny para transcripción rápida (LID)
try:
    import whisper
except ImportError:
    whisper = None

# fasttext para detección de idioma
try:
    import fasttext
except ImportError:
    fasttext = None

# Core SARAi
from core.audit import is_safe_mode

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

AUDIO_ENGINE = os.getenv("AUDIO_ENGINE", "omni3b")

# Idiomas soportados nativamente por Omni-3B (baja latencia, alta empatía)
OMNI_LANGS = ["es", "en"]

# Idiomas que requieren traducción vía NLLB
NLLB_LANGS_RAW = os.getenv("LANGUAGES", "es,en,fr,de,ja")
NLLB_LANGS = [lang.strip() for lang in NLLB_LANGS_RAW.split(",")]

# Modelos para LID
WHISPER_MODEL = "tiny"  # 39M params, ~150MB, latencia ~20ms
LID_MODEL_PATH = "models/lid.176.ftz"  # fasttext pre-entrenado

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioConfig:
    """Configuración de audio router desde .env"""
    
    def __init__(self):
        # Leer env vars en tiempo de ejecución (no usar globales)
        self.engine = os.getenv("AUDIO_ENGINE", "omni3b")
        
        # Parse languages
        languages_str = os.getenv("LANGUAGES", "es,en,fr,de,ja")
        self.languages = [lang.strip() for lang in languages_str.split(",")]
        
        # Omni langs son siempre es, en
        self.omni_langs = ["es", "en"]
        
        # NLLB langs son los que NO están en omni_langs
        self.nllb_langs = [l for l in self.languages if l not in self.omni_langs]


def get_audio_config() -> AudioConfig:
    """Factory para obtener configuración de audio"""
    return AudioConfig()


# ============================================================================
# DETECCIÓN DE IDIOMA (Language Identification)
# ============================================================================
class LanguageDetector:
    """
    Detector de idioma ligero con Whisper-tiny + fasttext
    
    Pipeline:
    1. Whisper-tiny transcribe el audio (STT rápido)
    2. fasttext detecta el idioma del texto
    3. Retorna código ISO 639-1 (es, en, fr, etc.)
    
    Fallback: Si falla cualquier paso, retorna "es" (español)
    """
    
    def __init__(self):
        self.whisper_model = None
        self.lid_model = None
        self.load_models()
    
    def load_models(self):
        """Carga modelos de forma lazy (solo si existen)"""
        global whisper, fasttext
        
        # Whisper-tiny
        if whisper is not None:
            try:
                self.whisper_model = whisper.load_model(WHISPER_MODEL)
                logger.info(f"✅ Whisper-tiny cargado ({WHISPER_MODEL})")
            except Exception as e:
                logger.warning(f"⚠️  No se pudo cargar Whisper: {e}")
        else:
            logger.warning("⚠️  whisper no instalado (pip install openai-whisper)")
        
        # fasttext LID
        if fasttext is not None and os.path.exists(LID_MODEL_PATH):
            try:
                self.lid_model = fasttext.load_model(LID_MODEL_PATH)
                logger.info(f"✅ fasttext LID cargado: {LID_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"⚠️  No se pudo cargar fasttext: {e}")
        else:
            logger.warning(f"⚠️  fasttext LID no encontrado: {LID_MODEL_PATH}")
    
    def detect(self, audio_bytes: bytes) -> str:
        """
        Detecta idioma del audio
        
        Args:
            audio_bytes: Audio raw (WAV, 16kHz recomendado)
        
        Returns:
            Código ISO 639-1 (es, en, fr, etc.)
            Si falla: "es" (español, fallback Sentinel)
        """
        try:
            # 1. Transcribir con Whisper-tiny
            if self.whisper_model is None:
                raise ValueError("Whisper no disponible")
            
            # Convertir bytes a array numpy (asume WAV 16kHz mono)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            result = self.whisper_model.transcribe(
                audio_array,
                fp16=False,  # CPU mode
                language=None  # Auto-detect
            )
            text = result["text"]
            
            logger.info(f"📝 Transcripción Whisper: {text[:100]}...")
            
            # 2. Detectar idioma con fasttext
            if self.lid_model is None:
                raise ValueError("fasttext LID no disponible")
            
            # fasttext espera texto sin newlines
            text_clean = text.replace("\n", " ").strip()
            
            predictions = self.lid_model.predict(text_clean, k=1)
            lang_label = predictions[0][0]  # '__label__es'
            confidence = predictions[1][0]
            
            # Limpiar etiqueta
            lang = lang_label.replace("__label__", "")
            
            logger.info(f"🌍 Idioma detectado: {lang} (confianza: {confidence:.2f})")
            
            # 3. Validar que esté en whitelist
            if lang not in NLLB_LANGS:
                logger.warning(f"⚠️  Idioma {lang} no en whitelist, usando 'es'")
                return "es"
            
            return lang
        
        except Exception as e:
            # SENTINEL FALLBACK: Cualquier fallo → español
            logger.error(f"❌ Error en detección de idioma: {e}")
            logger.info("🛡️  SENTINEL FALLBACK: Asumiendo español (es)")
            return "es"


# ============================================================================
# ROUTER DE AUDIO
# ============================================================================

# Singleton para reutilizar modelos
_detector = None

def get_language_detector() -> LanguageDetector:
    """Factory para detector de idioma (singleton)"""
    global _detector
    if _detector is None:
        _detector = LanguageDetector()
    return _detector


def route_audio(audio_bytes: bytes) -> Tuple[str, bytes, Optional[str]]:
    """
    Decide qué pipeline de voz usar según el idioma detectado
    
    Args:
        audio_bytes: Audio raw
    
    Returns:
        Tupla (engine, audio_bytes, lang_code)
        - engine: "omni" | "nllb" | "lfm2"
        - audio_bytes: Audio original (sin modificar)
        - lang_code: Código ISO 639-1 (None si engine="omni")
    
    Lógica de enrutamiento:
        1. Safe Mode activo → fallback a "lfm2" (solo texto)
        2. AUDIO_ENGINE="disabled" → "lfm2"
        3. AUDIO_ENGINE="lfm2" → "lfm2"
        4. Detectar idioma:
           - Idioma en OMNI_LANGS (es, en) → "omni"
           - Idioma en NLLB_LANGS (fr, de, ja, etc.) → "nllb"
           - Fallo o desconocido → SENTINEL FALLBACK ("omni", "es")
    """
    # 1. Safe Mode check
    if is_safe_mode():
        logger.warning("🚨 Safe Mode activo - Voz bloqueada, usando LFM2")
        return ("lfm2", audio_bytes, None)
    
    # 2. Flag AUDIO_ENGINE (leer en runtime, no global)
    audio_engine = os.getenv("AUDIO_ENGINE", "omni3b")
    
    if audio_engine == "disabled":
        logger.info("🔇 AUDIO_ENGINE=disabled, usando LFM2")
        return ("lfm2", audio_bytes, None)
    
    if audio_engine == "lfm2":
        logger.info("📝 AUDIO_ENGINE=lfm2, modo solo texto")
        return ("lfm2", audio_bytes, None)
    
    # 3. Detección de idioma
    detector = get_language_detector()
    
    try:
        lang = detector.detect(audio_bytes)
    except Exception as e:
        # SENTINEL FALLBACK: Si falla LID → Omni-es
        logger.error(f"❌ Error en LID: {e}. Fallback a Omni-es.")
        return ("omni", audio_bytes, "es")
    
    # 4. Enrutamiento por idioma
    # Leer NLLB_LANGS de env
    languages_str = os.getenv("LANGUAGES", "es,en,fr,de,ja")
    nllb_langs = [l.strip() for l in languages_str.split(",")]
    
    if lang in OMNI_LANGS:
        # Idioma nativo de Omni-3B
        logger.info(f"✅ Idioma '{lang}' soportado nativamente por Omni-3B")
        return ("omni", audio_bytes, None)
    
    elif lang in nllb_langs and audio_engine == "nllb":
        # Idioma requiere traducción
        logger.info(f"🌐 Idioma '{lang}' requiere traducción NLLB")
        return ("nllb", audio_bytes, lang)
    
    else:
        # SENTINEL FALLBACK: Idioma desconocido o NLLB no habilitado
        logger.warning(
            f"⚠️  Idioma '{lang}' no soportado o AUDIO_ENGINE != 'nllb'. "
            "Usando Omni-3B en español."
        )
        return ("omni", audio_bytes, "es")


# ============================================================================
# ESTADÍSTICAS Y MÉTRICAS
# ============================================================================

# Métricas globales del router
_router_stats = {
    "total_requests": 0,
    "fallback_count": 0,
    "lid_failures": 0,
    "languages_detected": {}
}


def get_router_stats() -> dict:
    """Retorna estadísticas del router para monitoreo"""
    stats = _router_stats.copy()
    
    # Calcular fallback rate
    if stats["total_requests"] > 0:
        stats["fallback_rate"] = stats["fallback_count"] / stats["total_requests"]
    else:
        stats["fallback_rate"] = 0.0
    
    return stats


# ============================================================================
# CLI (para testing)
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python -m agents.audio_router <audio.wav>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Leer audio
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    # Router
    engine, _, lang = route_audio(audio_bytes)
    
    print(f"\n✅ Resultado del routing:")
    print(f"   Engine: {engine}")
    print(f"   Idioma: {lang if lang else 'Auto-detectado (Omni nativo)'}")
    print(f"   Audio size: {len(audio_bytes)} bytes")
