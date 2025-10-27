"""
Test Suite for NLLB-200 Translation Pipeline (M3.1 Fase 3)

Tests bidirectional translation for 6 languages:
- French (fr)
- German (de)
- Japanese (ja)
- Portuguese (pt)
- Italian (it)
- Russian (ru)

Author: SARAi Team
License: CC-BY-NC-SA 4.0
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.nllb_translator import (
    NLLBTranslator,
    NLLBPipeline,
    get_nllb_translator,
    translate_text
)


# ============================================================================
# Test Data: Sample translations for validation
# ============================================================================

SAMPLE_TRANSLATIONS = {
    # French → Spanish
    ("Bonjour, comment allez-vous?", "fr", "es"): "Hola, ¿cómo está?",
    ("Je suis heureux de vous rencontrer", "fr", "es"): "Estoy feliz de conocerte",
    
    # German → Spanish
    ("Guten Tag, wie geht es Ihnen?", "de", "es"): "Buen día, ¿cómo está?",
    ("Ich liebe Musik", "de", "es"): "Amo la música",
    
    # Japanese → Spanish
    ("こんにちは", "ja", "es"): "Hola",
    ("ありがとうございます", "ja", "es"): "Muchas gracias",
    
    # Portuguese → Spanish
    ("Olá, tudo bem?", "pt", "es"): "Hola, ¿todo bien?",
    ("Obrigado", "pt", "es"): "Gracias",
    
    # Italian → Spanish
    ("Ciao, come stai?", "it", "es"): "Hola, ¿cómo estás?",
    ("Grazie mille", "it", "es"): "Muchas gracias",
    
    # Russian → Spanish
    ("Привет", "ru", "es"): "Hola",
    ("Спасибо", "ru", "es"): "Gracias",
}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer"""
    tokenizer = Mock()
    tokenizer.src_lang = "spa_Latn"
    tokenizer.lang_code_to_id = {
        "spa_Latn": 0,
        "fra_Latn": 1,
        "deu_Latn": 2,
        "jpn_Jpan": 3,
        "por_Latn": 4,
        "ita_Latn": 5,
        "rus_Cyrl": 6,
        "eng_Latn": 7,
    }
    
    # Mock tokenize output
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
    }
    
    # Mock decode output
    tokenizer.batch_decode.return_value = ["Mocked translation"]
    
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock NLLB model"""
    model = Mock()
    
    # Mock generate output (tensor of token IDs)
    model.generate.return_value = torch.tensor([[10, 20, 30, 40, 50]])
    
    return model


@pytest.fixture
def mock_translator(mock_tokenizer, mock_model):
    """Mock NLLBTranslator with mocked dependencies"""
    with patch('agents.nllb_translator.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('agents.nllb_translator.AutoModelForSeq2SeqLM.from_pretrained', return_value=mock_model):
        
        translator = NLLBTranslator()
        return translator


# ============================================================================
# Test: NLLBTranslator Initialization
# ============================================================================

class TestNLLBTranslatorInit:
    """Test translator initialization and model loading"""
    
    def test_language_map_complete(self):
        """Verify all supported languages are mapped"""
        assert "es" in NLLBTranslator.LANG_MAP
        assert "fr" in NLLBTranslator.LANG_MAP
        assert "de" in NLLBTranslator.LANG_MAP
        assert "ja" in NLLBTranslator.LANG_MAP
        assert "pt" in NLLBTranslator.LANG_MAP
        assert "it" in NLLBTranslator.LANG_MAP
        assert "ru" in NLLBTranslator.LANG_MAP
    
    def test_supported_langs_list(self):
        """Verify SUPPORTED_LANGS excludes Spanish and English"""
        assert "es" not in NLLBTranslator.SUPPORTED_LANGS
        assert "en" not in NLLBTranslator.SUPPORTED_LANGS
        assert len(NLLBTranslator.SUPPORTED_LANGS) == 6
    
    @patch('agents.nllb_translator.AutoTokenizer.from_pretrained')
    @patch('agents.nllb_translator.AutoModelForSeq2SeqLM.from_pretrained')
    def test_model_loading(self, mock_model, mock_tokenizer):
        """Test that model is loaded with correct parameters"""
        translator = NLLBTranslator()
        
        # Verify tokenizer loaded
        mock_tokenizer.assert_called_once()
        assert "cache_dir" in mock_tokenizer.call_args.kwargs
        
        # Verify model loaded with 8-bit quantization (CPU)
        mock_model.assert_called_once()
        assert mock_model.call_args.kwargs["load_in_8bit"] is True
        assert mock_model.call_args.kwargs["device_map"] == "cpu"


# ============================================================================
# Test: Translation Function
# ============================================================================

class TestTranslation:
    """Test core translation functionality"""
    
    def test_translate_french_to_spanish(self, mock_translator):
        """Test French → Spanish translation"""
        result = mock_translator.translate("Bonjour", "fr", "es")
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify model.generate was called
        mock_translator.model.generate.assert_called_once()
    
    def test_translate_with_max_length(self, mock_translator):
        """Test translation with custom max_length"""
        result = mock_translator.translate(
            "Long text...", 
            "de", 
            "es", 
            max_length=256
        )
        
        # Verify max_length passed to generate
        call_kwargs = mock_translator.model.generate.call_args.kwargs
        assert call_kwargs["max_length"] == 256
    
    def test_translate_unsupported_source_language(self, mock_translator):
        """Test that unsupported source language raises ValueError"""
        with pytest.raises(ValueError, match="Source language.*not supported"):
            mock_translator.translate("Hello", "zh", "es")
    
    def test_translate_unsupported_target_language(self, mock_translator):
        """Test that unsupported target language raises ValueError"""
        with pytest.raises(ValueError, match="Target language.*not supported"):
            mock_translator.translate("Hola", "es", "zh")
    
    def test_translate_bidirectional(self, mock_translator):
        """Test bidirectional translation (es ↔ fr)"""
        # es → fr
        result_fr = mock_translator.translate("Hola", "es", "fr")
        assert isinstance(result_fr, str)
        
        # fr → es
        result_es = mock_translator.translate("Bonjour", "fr", "es")
        assert isinstance(result_es, str)
    
    def test_translate_all_supported_languages(self, mock_translator):
        """Test translation for all 6 supported languages"""
        for lang in NLLBTranslator.SUPPORTED_LANGS:
            # lang → es
            result = mock_translator.translate("Test", lang, "es")
            assert isinstance(result, str)
            
            # es → lang
            result = mock_translator.translate("Prueba", "es", lang)
            assert isinstance(result, str)


# ============================================================================
# Test: NLLBPipeline
# ============================================================================

class TestNLLBPipeline:
    """Test full translation pipeline"""
    
    @pytest.fixture
    def mock_stt_func(self):
        """Mock STT function"""
        return Mock(return_value="Bonjour, comment ça va?")
    
    @pytest.fixture
    def mock_llm_func(self):
        """Mock LLM generation function"""
        return Mock(return_value="Estoy bien, gracias por preguntar.")
    
    @pytest.fixture
    def mock_tts_func(self):
        """Mock TTS function"""
        return Mock(return_value=b"AUDIO_DATA_TTS")
    
    def test_pipeline_initialization(self, mock_translator):
        """Test pipeline can be initialized with translator"""
        pipeline = NLLBPipeline(translator=mock_translator)
        assert pipeline.translator is not None
    
    def test_pipeline_initialization_default(self):
        """Test pipeline creates translator if not provided"""
        with patch('agents.nllb_translator.NLLBTranslator') as MockTranslator:
            pipeline = NLLBPipeline()
            MockTranslator.assert_called_once()
    
    def test_pipeline_process_audio_french(
        self, 
        mock_translator, 
        mock_stt_func, 
        mock_llm_func, 
        mock_tts_func
    ):
        """Test full pipeline for French audio"""
        # Mock translator.translate to return realistic values
        def mock_translate(text, src_lang, tgt_lang):
            if src_lang == "fr" and tgt_lang == "es":
                return "Hola, ¿cómo estás?"  # fr → es
            elif src_lang == "es" and tgt_lang == "fr":
                return "Je vais bien, merci de demander."  # es → fr
            return text
        
        mock_translator.translate = Mock(side_effect=mock_translate)
        
        pipeline = NLLBPipeline(translator=mock_translator)
        
        result = pipeline.process_audio(
            audio_bytes=b"AUDIO_DATA_FR",
            detected_lang="fr",
            whisper_stt_func=mock_stt_func,
            llm_generate_func=mock_llm_func,
            tts_func=mock_tts_func
        )
        
        # Verify all steps executed
        mock_stt_func.assert_called_once_with(b"AUDIO_DATA_FR")
        assert mock_translator.translate.call_count == 2  # fr→es, es→fr
        mock_llm_func.assert_called_once()
        mock_tts_func.assert_called_once()
        
        # Verify result structure
        assert "text_input" in result
        assert "text_input_es" in result
        assert "response_es" in result
        assert "response_target" in result
        assert "audio_output" in result
        assert "latency_ms" in result
        
        # Verify latency is reasonable (should be <5000ms in real scenario)
        assert result["latency_ms"] > 0
        assert result["latency_ms"] < 10000  # Mock should be fast
    
    def test_pipeline_unsupported_language(
        self, 
        mock_translator, 
        mock_stt_func, 
        mock_llm_func, 
        mock_tts_func
    ):
        """Test pipeline rejects unsupported language"""
        pipeline = NLLBPipeline(translator=mock_translator)
        
        with pytest.raises(ValueError, match="not supported by NLLB pipeline"):
            pipeline.process_audio(
                audio_bytes=b"AUDIO_DATA",
                detected_lang="zh",  # Chinese not supported
                whisper_stt_func=mock_stt_func,
                llm_generate_func=mock_llm_func,
                tts_func=mock_tts_func
            )
    
    def test_pipeline_latency_tracking(
        self, 
        mock_translator, 
        mock_stt_func, 
        mock_llm_func, 
        mock_tts_func
    ):
        """Test pipeline tracks latency correctly"""
        import time
        
        # Make functions sleep to simulate latency
        def slow_stt(audio):
            time.sleep(0.1)
            return "Text"
        
        def slow_llm(text):
            time.sleep(0.1)
            return "Response"
        
        def slow_tts(text, lang):
            time.sleep(0.1)
            return b"AUDIO"
        
        mock_translator.translate = Mock(return_value="Translated")
        
        pipeline = NLLBPipeline(translator=mock_translator)
        
        result = pipeline.process_audio(
            audio_bytes=b"AUDIO",
            detected_lang="fr",
            whisper_stt_func=slow_stt,
            llm_generate_func=slow_llm,
            tts_func=slow_tts
        )
        
        # Total sleep = 300ms, should be reflected in latency
        assert result["latency_ms"] >= 300


# ============================================================================
# Test: Singleton Pattern
# ============================================================================

class TestSingleton:
    """Test singleton factory function"""
    
    @patch('agents.nllb_translator.NLLBTranslator')
    def test_get_nllb_translator_singleton(self, MockTranslator):
        """Test that get_nllb_translator returns same instance"""
        # Reset singleton
        import agents.nllb_translator
        agents.nllb_translator._translator_instance = None
        
        # First call creates instance
        translator1 = get_nllb_translator()
        MockTranslator.assert_called_once()
        
        # Second call reuses instance
        translator2 = get_nllb_translator()
        assert MockTranslator.call_count == 1  # Not called again
        assert translator1 is translator2


# ============================================================================
# Test: Convenience Function
# ============================================================================

class TestConvenienceFunction:
    """Test translate_text convenience function"""
    
    @patch('agents.nllb_translator.get_nllb_translator')
    def test_translate_text_uses_singleton(self, mock_get_translator):
        """Test that translate_text uses singleton translator"""
        mock_translator = Mock()
        mock_translator.translate.return_value = "Translated"
        mock_get_translator.return_value = mock_translator
        
        result = translate_text("Bonjour", "fr", "es")
        
        mock_get_translator.assert_called_once()
        mock_translator.translate.assert_called_once_with("Bonjour", "fr", "es")
        assert result == "Translated"


# ============================================================================
# Test: Integration (End-to-End simulation)
# ============================================================================

class TestIntegration:
    """Integration tests simulating real usage"""
    
    def test_full_voice_interaction_french(self, mock_translator):
        """Simulate full voice interaction in French"""
        # User speaks French
        audio_input = b"FRENCH_AUDIO_WAV"
        
        # STT transcribes
        stt_result = "Quel temps fait-il aujourd'hui?"
        
        # NLLB translates to Spanish
        mock_translator.translate = Mock(side_effect=[
            "¿Qué tiempo hace hoy?",  # fr → es
            "Il fait beau aujourd'hui."  # es → fr
        ])
        
        # LLM responds in Spanish
        llm_response = "Hace buen tiempo hoy."
        
        # NLLB translates back to French (already mocked above)
        
        # TTS generates French audio
        tts_output = b"FRENCH_AUDIO_RESPONSE"
        
        # Execute pipeline
        pipeline = NLLBPipeline(translator=mock_translator)
        result = pipeline.process_audio(
            audio_bytes=audio_input,
            detected_lang="fr",
            whisper_stt_func=lambda x: stt_result,
            llm_generate_func=lambda x: llm_response,
            tts_func=lambda text, lang: tts_output
        )
        
        # Verify complete flow
        assert result["text_input"] == stt_result
        assert result["text_input_es"] == "¿Qué tiempo hace hoy?"
        assert result["response_es"] == llm_response
        assert result["response_target"] == "Il fait beau aujourd'hui."
        assert result["audio_output"] == tts_output
    
    def test_multiple_languages_in_sequence(self, mock_translator):
        """Test handling multiple languages in sequence"""
        languages = ["fr", "de", "ja", "pt", "it", "ru"]
        
        mock_translator.translate = Mock(return_value="Respuesta traducida")
        
        pipeline = NLLBPipeline(translator=mock_translator)
        
        for lang in languages:
            result = pipeline.process_audio(
                audio_bytes=b"AUDIO",
                detected_lang=lang,
                whisper_stt_func=lambda x: f"Text in {lang}",
                llm_generate_func=lambda x: "Respuesta",
                tts_func=lambda text, lang: b"AUDIO_RESPONSE"
            )
            
            assert result["latency_ms"] > 0
            assert result["audio_output"] == b"AUDIO_RESPONSE"


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_text_translation(self, mock_translator):
        """Test translating empty string"""
        result = mock_translator.translate("", "fr", "es")
        # Should not crash, return empty or minimal output
        assert isinstance(result, str)
    
    def test_very_long_text(self, mock_translator):
        """Test translation with very long text (truncation)"""
        long_text = "Bonjour " * 1000  # Very long
        result = mock_translator.translate(long_text, "fr", "es", max_length=128)
        
        # Verify max_length is respected
        call_kwargs = mock_translator.model.generate.call_args.kwargs
        assert call_kwargs["max_length"] == 128
    
    def test_special_characters(self, mock_translator):
        """Test translation with special characters"""
        text_with_special = "Hola! ¿Cómo estás? 😊"
        result = mock_translator.translate(text_with_special, "es", "fr")
        assert isinstance(result, str)


# ============================================================================
# Test: Performance Benchmarks
# ============================================================================

class TestPerformance:
    """Performance and latency tests"""
    
    def test_translation_latency_target(self, mock_translator):
        """Test that translation meets <2s latency target (simulated)"""
        import time
        
        start = time.time()
        mock_translator.translate("Bonjour le monde", "fr", "es")
        latency = (time.time() - start) * 1000
        
        # Mock should be fast (<50ms)
        assert latency < 100
    
    def test_pipeline_latency_target(self, mock_translator):
        """Test that full pipeline meets <2s target (simulated)"""
        mock_translator.translate = Mock(return_value="Translated")
        
        pipeline = NLLBPipeline(translator=mock_translator)
        
        result = pipeline.process_audio(
            audio_bytes=b"AUDIO",
            detected_lang="fr",
            whisper_stt_func=lambda x: "Text",
            llm_generate_func=lambda x: "Response",
            tts_func=lambda text, lang: b"AUDIO"
        )
        
        # Mock pipeline should be <500ms
        assert result["latency_ms"] < 500


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
