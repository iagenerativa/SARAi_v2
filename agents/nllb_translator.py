"""
NLLB-200 Translation Pipeline for SARAi v2.11

Implements multi-language support for non-native languages (fr, de, ja, pt, it, ru)
using the NLLB-200 (No Language Left Behind) model.

Pipeline:
    Audio → Whisper STT → NLLB(src→es) → LFM2 generation → NLLB(es→src) → TTS

Author: SARAi Team
License: CC-BY-NC-SA 4.0
"""

import os
import logging
from typing import Optional, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLLBTranslator:
    """
    NLLB-200 translator for multi-language support.
    
    Supports bidirectional translation between Spanish and:
    - French (fr)
    - German (de)
    - Japanese (ja)
    - Portuguese (pt)
    - Italian (it)
    - Russian (ru)
    
    Uses facebook/nllb-200-distilled-600M with 4-bit quantization for CPU efficiency.
    """
    
    # Language code mapping (ISO 639-1 → NLLB codes)
    LANG_MAP = {
        "es": "spa_Latn",  # Spanish
        "fr": "fra_Latn",  # French
        "de": "deu_Latn",  # German
        "ja": "jpn_Jpan",  # Japanese
        "pt": "por_Latn",  # Portuguese
        "it": "ita_Latn",  # Italian
        "ru": "rus_Cyrl",  # Russian
        "en": "eng_Latn",  # English (fallback)
    }
    
    SUPPORTED_LANGS = ["fr", "de", "ja", "pt", "it", "ru"]
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        """
        Initialize NLLB translator with quantized model.
        
        Args:
            model_name: HuggingFace model identifier
        
        RAM Usage: ~1.2GB with 4-bit quantization
        """
        self.model_name = model_name
        self.device = "cpu"  # SARAi runs on CPU only
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load NLLB model with 4-bit quantization (CPU compatible)"""
        try:
            logger.info(f"Loading NLLB model: {self.model_name}")
            
            # Tokenizer (lightweight, always loaded)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="models/cache/nllb_tokenizer"
            )
            
            # Model with 4-bit quantization
            # Note: For CPU, we use load_in_8bit=True (4-bit not supported on CPU)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,  # CPU compatible
                device_map="cpu",
                cache_dir="models/cache/nllb_model"
            )
            
            logger.info("✅ NLLB model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NLLB model: {e}")
            raise
    
    def translate(
        self, 
        text: str, 
        src_lang: str, 
        tgt_lang: str,
        max_length: int = 512
    ) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Input text to translate
            src_lang: Source language code (ISO 639-1, e.g., "fr")
            tgt_lang: Target language code (ISO 639-1, e.g., "es")
            max_length: Maximum output tokens
        
        Returns:
            Translated text
        
        Raises:
            ValueError: If language not supported
        """
        # Validate languages
        if src_lang not in self.LANG_MAP:
            raise ValueError(f"Source language '{src_lang}' not supported. "
                           f"Supported: {list(self.LANG_MAP.keys())}")
        
        if tgt_lang not in self.LANG_MAP:
            raise ValueError(f"Target language '{tgt_lang}' not supported. "
                           f"Supported: {list(self.LANG_MAP.keys())}")
        
        # Convert to NLLB codes
        src_nllb = self.LANG_MAP[src_lang]
        tgt_nllb = self.LANG_MAP[tgt_lang]
        
        # Tokenize
        self.tokenizer.src_lang = src_nllb
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length
        )
        
        # Translate
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_nllb],
                max_length=max_length,
                num_beams=4,  # Beam search for quality
                early_stopping=True
            )
        
        # Decode
        translated_text = self.tokenizer.batch_decode(
            translated_tokens, 
            skip_special_tokens=True
        )[0]
        
        return translated_text
    
    def __del__(self):
        """Cleanup: release model from memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("NLLB model unloaded")


class NLLBPipeline:
    """
    Full translation pipeline integrating STT → Translation → LLM → Translation → TTS.
    
    This is the high-level interface used by audio_router.py when processing
    non-native languages (fr, de, ja, pt, it, ru).
    """
    
    def __init__(self, translator: Optional[NLLBTranslator] = None):
        """
        Initialize pipeline.
        
        Args:
            translator: Optional pre-initialized NLLBTranslator instance
        """
        self.translator = translator or NLLBTranslator()
    
    def process_audio(
        self,
        audio_bytes: bytes,
        detected_lang: str,
        whisper_stt_func,  # Function: audio_bytes → text
        llm_generate_func,  # Function: text → response
        tts_func  # Function: (text, lang) → audio_bytes
    ) -> Dict:
        """
        Process audio through full NLLB pipeline.
        
        Args:
            audio_bytes: Input audio (WAV format)
            detected_lang: Detected language (fr, de, ja, pt, it, ru)
            whisper_stt_func: STT function (audio → text)
            llm_generate_func: LLM generation function (text → response)
            tts_func: TTS function (text, lang → audio)
        
        Returns:
            {
                "text_input": str,  # Original text in source language
                "text_input_es": str,  # Translated to Spanish
                "response_es": str,  # LLM response in Spanish
                "response_target": str,  # Response translated back to source
                "audio_output": bytes,  # TTS audio in source language
                "latency_ms": float  # Total pipeline latency
            }
        """
        import time
        start_time = time.time()
        
        # Validate language
        if detected_lang not in self.translator.SUPPORTED_LANGS:
            raise ValueError(f"Language '{detected_lang}' not supported by NLLB pipeline")
        
        logger.info(f"🌍 NLLB Pipeline started for language: {detected_lang}")
        
        # STEP 1: STT (Whisper) - transcribe in source language
        text_input = whisper_stt_func(audio_bytes)
        logger.info(f"STT ({detected_lang}): {text_input[:100]}...")
        
        # STEP 2: Translate source → Spanish
        text_input_es = self.translator.translate(
            text_input, 
            src_lang=detected_lang, 
            tgt_lang="es"
        )
        logger.info(f"Translation ({detected_lang}→es): {text_input_es[:100]}...")
        
        # STEP 3: LLM generation in Spanish
        response_es = llm_generate_func(text_input_es)
        logger.info(f"LLM (es): {response_es[:100]}...")
        
        # STEP 4: Translate response Spanish → source language
        response_target = self.translator.translate(
            response_es,
            src_lang="es",
            tgt_lang=detected_lang
        )
        logger.info(f"Translation (es→{detected_lang}): {response_target[:100]}...")
        
        # STEP 5: TTS in source language
        audio_output = tts_func(response_target, detected_lang)
        
        # Timing
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"✅ NLLB Pipeline completed in {latency_ms:.0f}ms")
        
        return {
            "text_input": text_input,
            "text_input_es": text_input_es,
            "response_es": response_es,
            "response_target": response_target,
            "audio_output": audio_output,
            "latency_ms": latency_ms
        }


# Singleton for global access
_translator_instance: Optional[NLLBTranslator] = None

def get_nllb_translator() -> NLLBTranslator:
    """
    Factory function to get singleton NLLB translator.
    
    Ensures only one model instance is loaded in memory.
    """
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = NLLBTranslator()
    return _translator_instance


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Convenience function for quick translation.
    
    Args:
        text: Text to translate
        src_lang: Source language (ISO 639-1)
        tgt_lang: Target language (ISO 639-1)
    
    Returns:
        Translated text
    
    Example:
        >>> translate_text("Bonjour le monde", "fr", "es")
        "Hola mundo"
    """
    translator = get_nllb_translator()
    return translator.translate(text, src_lang, tgt_lang)


if __name__ == "__main__":
    """Test standalone translation"""
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python -m agents.nllb_translator <text> <src_lang> <tgt_lang>")
        print("Example: python -m agents.nllb_translator 'Bonjour' fr es")
        sys.exit(1)
    
    text, src, tgt = sys.argv[1], sys.argv[2], sys.argv[3]
    
    try:
        result = translate_text(text, src, tgt)
        print(f"\nOriginal ({src}): {text}")
        print(f"Translated ({tgt}): {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
