"""
tests/test_audio_router.py - Tests para Audio Router v2.11

Valida el comportamiento del router con diferentes escenarios:
- Detección correcta de idiomas
- Fallback cuando falla LID
- Safe mode handling
- Configuración desde .env
"""

import pytest
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from agents.audio_router import (
    route_audio,
    get_audio_config,
    get_language_detector,
    LanguageDetector,
    AudioConfig,
    OMNI_LANGS,
    NLLB_LANGS
)


@pytest.fixture
def mock_safe_mode_off():
    """Mock para desactivar safe mode durante tests"""
    with patch('agents.audio_router.is_safe_mode', return_value=False):
        yield


@pytest.fixture
def mock_safe_mode_on():
    """Mock para activar safe mode"""
    with patch('agents.audio_router.is_safe_mode', return_value=True):
        yield


@pytest.fixture
def sample_audio_bytes():
    """Audio de prueba (bytes ficticios)"""
    return b"FAKE_AUDIO_DATA_FOR_TESTING"


@pytest.fixture
def mock_language_detector():
    """Mock del LanguageDetector"""
    detector = Mock(spec=LanguageDetector)
    detector.detect = Mock()
    return detector


class TestAudioConfig:
    """Tests para la configuración del motor de audio"""
    
    def test_default_config(self):
        """Verifica configuración por defecto"""
        with patch.dict(os.environ, {}, clear=True):
            config = get_audio_config()
            
            assert config.engine == "omni3b"
            assert "es" in config.languages
            assert "en" in config.languages
            assert config.omni_langs == OMNI_LANGS
    
    def test_custom_config_from_env(self):
        """Verifica lectura de .env"""
        with patch.dict(os.environ, {
            'AUDIO_ENGINE': 'nllb',
            'LANGUAGES': 'es,en,fr,de'
        }):
            config = get_audio_config()
            
            assert config.engine == "nllb"
            assert config.languages == ["es", "en", "fr", "de"]
            assert "fr" in config.nllb_langs
    
    def test_disabled_engine(self):
        """Verifica engine deshabilitado"""
        with patch.dict(os.environ, {'AUDIO_ENGINE': 'disabled'}):
            config = get_audio_config()
            assert config.engine == "disabled"


class TestLanguageDetector:
    """Tests para el detector de idioma"""
    
    @patch('agents.audio_router.whisper')
    @patch('agents.audio_router.fasttext')
    def test_detect_spanish(self, mock_ft, mock_whisper, sample_audio_bytes):
        """Verifica detección de español"""
        # Mock Whisper transcription
        mock_whisper_model = Mock()
        mock_whisper_model.transcribe.return_value = {"text": "Hola mundo"}
        mock_whisper.load_model.return_value = mock_whisper_model
        
        # Mock fasttext prediction
        mock_ft_model = Mock()
        mock_ft_model.predict.return_value = (["__label__spa"], [0.95])
        mock_ft.load_model.return_value = mock_ft_model
        
        detector = LanguageDetector()
        lang = detector.detect(sample_audio_bytes)
        
        assert lang == "es"
    
    @patch('agents.audio_router.whisper')
    @patch('agents.audio_router.fasttext')
    @patch('agents.audio_router.np.frombuffer')
    @patch('agents.audio_router.os.path.exists', return_value=True)
    def test_detect_english(self, mock_exists, mock_frombuffer, mock_ft, mock_whisper, sample_audio_bytes):
        """Verifica detección de inglés"""
        # Mock np.frombuffer para evitar error con bytes fake
        mock_audio_array = np.zeros(1000, dtype=np.float32)
        mock_frombuffer.return_value = mock_audio_array
        
        mock_whisper_model = Mock()
        mock_whisper_model.transcribe.return_value = {"text": "Hello world"}
        mock_whisper.load_model.return_value = mock_whisper_model
        
        mock_ft_model = Mock()
        mock_ft_model.predict.return_value = (["__label__eng"], [0.98])
        mock_ft.load_model.return_value = mock_ft_model
        
        detector = LanguageDetector()
        lang = detector.detect(sample_audio_bytes)
        
        assert lang == "en"
    
    @patch('agents.audio_router.whisper')
    def test_detect_empty_transcription(self, mock_whisper, sample_audio_bytes):
        """Verifica fallback cuando transcripción está vacía"""
        mock_whisper_model = Mock()
        mock_whisper_model.transcribe.return_value = {"text": ""}
        mock_whisper.load_model.return_value = mock_whisper_model
        
        detector = LanguageDetector()
        lang = detector.detect(sample_audio_bytes)
    
    @pytest.mark.skip(reason="Requiere micrófono real - ejecutar manualmente con pytest -k real_microphone")
    def test_detect_with_real_microphone(self):
        """
        Test INTERACTIVO con micrófono real
        
        Uso:
            pytest tests/test_audio_router.py::TestLanguageDetector::test_detect_with_real_microphone -s
        
        O sin skip:
            pytest tests/test_audio_router.py -k real_microphone -s --no-skip
        """
        try:
            import pyaudio
            import wave
            import tempfile
        except ImportError:
            pytest.skip("pyaudio no instalado. Instalar con: pip install pyaudio")
        
        print("\n" + "="*60)
        print("🎤 TEST INTERACTIVO: Detección de idioma con micrófono")
        print("="*60)
        
        # Configuración de grabación
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5
        
        print(f"\n📋 Configuración:")
        print(f"   - Duración: {RECORD_SECONDS} segundos")
        print(f"   - Sample rate: {RATE} Hz")
        print(f"   - Channels: {CHANNELS}")
        
        input("\n▶️  Presiona ENTER para comenzar a grabar...")
        
        # Inicializar PyAudio
        p = pyaudio.PyAudio()
        
        print("\n🔴 GRABANDO... (habla en cualquier idioma)")
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            # Indicador de progreso
            if i % 10 == 0:
                print("█", end="", flush=True)
        
        print(" ✅ Grabación completada\n")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wf = wave.open(temp_wav.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Leer como bytes
            with open(temp_wav.name, 'rb') as f:
                audio_bytes = f.read()
        
        # DETECCIÓN con LanguageDetector REAL (sin mocks)
        print("🔍 Detectando idioma...")
        detector = LanguageDetector()
        
        try:
            detected_lang = detector.detect(audio_bytes)
            
            print("\n" + "="*60)
            print(f"✅ RESULTADO: {detected_lang.upper()}")
            print("="*60)
            
            # Mapeo de códigos a nombres
            lang_names = {
                "es": "Español",
                "en": "English",
                "fr": "Français",
                "de": "Deutsch",
                "ja": "日本語",
                "pt": "Português",
                "it": "Italiano",
                "ru": "Русский",
                "zh": "中文",
                "ar": "العربية",
                "hi": "हिन्दी",
                "ko": "한국어"
            }
            
            lang_name = lang_names.get(detected_lang, "Desconocido")
            print(f"\n🌍 Idioma detectado: {lang_name} ({detected_lang})")
            
            # Verificación manual
            print("\n❓ ¿Es correcto? (y/n): ", end="")
            user_confirm = input().strip().lower()
            
            if user_confirm == 'y':
                print("✅ Test PASSED - Detección correcta")
                assert True
            else:
                print("❌ Test FAILED - Detección incorrecta")
                pytest.fail(f"Usuario indicó que '{detected_lang}' es incorrecto")
        
        except Exception as e:
            print(f"\n❌ ERROR en detección: {e}")
            pytest.fail(f"Detección falló: {e}")
        
        # Debe asumir español por defecto
        assert lang == "es"

class TestAudioRouter:
    """Tests para la lógica de enrutamiento"""
    
    def test_route_to_omni_spanish(self, mock_safe_mode_off, sample_audio_bytes):
        """Verifica routing a Omni para español"""
        with patch('agents.audio_router.get_language_detector') as mock_get_detector:
            mock_detector = Mock()
            mock_detector.detect.return_value = "es"
            mock_get_detector.return_value = mock_detector
            
            engine, audio, lang = route_audio(sample_audio_bytes)
            
            assert engine == "omni"
            assert audio == sample_audio_bytes
            assert lang is None  # Omni no necesita lang target
    
    def test_route_to_omni_english(self, mock_safe_mode_off, sample_audio_bytes):
        """Verifica routing a Omni para inglés"""
        with patch('agents.audio_router.get_language_detector') as mock_get_detector:
            mock_detector = Mock()
            mock_detector.detect.return_value = "en"
            mock_get_detector.return_value = mock_detector
            
            engine, audio, lang = route_audio(sample_audio_bytes)
            
            assert engine == "omni"
            assert audio == sample_audio_bytes
            assert lang is None
    
    def test_route_to_nllb_french(self, mock_safe_mode_off, sample_audio_bytes):
        """Verifica routing a NLLB para francés"""
        with patch('agents.audio_router.get_language_detector') as mock_get_detector, \
             patch.dict(os.environ, {'AUDIO_ENGINE': 'nllb', 'LANGUAGES': 'es,en,fr'}):
            
            mock_detector = Mock()
            mock_detector.detect.return_value = "fr"
            mock_get_detector.return_value = mock_detector
            
            engine, audio, lang = route_audio(sample_audio_bytes)
            
            assert engine == "nllb"
            assert audio == sample_audio_bytes
            assert lang == "fr"
    
    def test_sentinel_fallback_on_lid_failure(self, mock_safe_mode_off, sample_audio_bytes):
        """Verifica Sentinel fallback cuando falla LID"""
        with patch('agents.audio_router.get_language_detector') as mock_get_detector:
            mock_detector = Mock()
            mock_detector.detect.side_effect = Exception("LID failed")
            mock_get_detector.return_value = mock_detector
            
            engine, audio, lang = route_audio(sample_audio_bytes)
            
            # SENTINEL: Debe fallar gracefully a omni-es
            assert engine == "omni"
            assert audio == sample_audio_bytes
            assert lang == "es"
    
    def test_sentinel_fallback_on_unknown_language(self, mock_safe_mode_off, sample_audio_bytes):
        """Verifica Sentinel fallback para idioma desconocido"""
        with patch('agents.audio_router.get_language_detector') as mock_get_detector:
            mock_detector = Mock()
            mock_detector.detect.return_value = "zh"  # Chino no soportado
            mock_get_detector.return_value = mock_detector
            
            engine, audio, lang = route_audio(sample_audio_bytes)
            
            # SENTINEL: Idioma no soportado → omni-es
            assert engine == "omni"
            assert audio == sample_audio_bytes
            assert lang == "es"
    
    def test_safe_mode_forces_lfm2(self, mock_safe_mode_on, sample_audio_bytes):
        """Verifica que Safe Mode fuerza fallback a LFM2"""
        # No necesita mock de detector porque safe mode debe cortocircuitar
        engine, audio, lang = route_audio(sample_audio_bytes)
        
        assert engine == "lfm2"
        assert audio == sample_audio_bytes
        assert lang is None
    
    def test_disabled_engine_forces_lfm2(self, mock_safe_mode_off, sample_audio_bytes):
        """Verifica que AUDIO_ENGINE=disabled fuerza LFM2"""
        with patch.dict(os.environ, {'AUDIO_ENGINE': 'disabled'}):
            engine, audio, lang = route_audio(sample_audio_bytes)
            
            assert engine == "lfm2"
            assert audio == sample_audio_bytes
            assert lang is None
    
    def test_audio_bytes_never_modified(self, mock_safe_mode_off, sample_audio_bytes):
        """Verifica que audio_bytes NUNCA se modifica (pasa sin cambios)"""
        with patch('agents.audio_router.get_language_detector') as mock_get_detector:
            mock_detector = Mock()
            mock_detector.detect.return_value = "es"
            mock_get_detector.return_value = mock_detector
            
            _, returned_audio, _ = route_audio(sample_audio_bytes)
            
            # CRÍTICO: Audio debe pasar sin modificar
            assert returned_audio is sample_audio_bytes
            assert id(returned_audio) == id(sample_audio_bytes)


class TestIntegration:
    """Tests de integración del pipeline completo"""
    
    @pytest.mark.skipif(
        not os.path.exists("models/cache/lid.176.bin"),
        reason="Modelo fasttext no descargado"
    )
    def test_real_audio_routing(self):
        """Test con audio real (si está disponible)"""
        # TODO: Implementar con archivo de audio real
        pytest.skip("Requiere archivo de audio de prueba")
    
    def test_fallback_rate_metrics(self):
        """Verifica que fallback_rate se trackea correctamente"""
        from agents.audio_router import get_router_stats
        
        stats = get_router_stats()
        
        assert "fallback_rate" in stats
        assert "languages_detected" in stats
        assert "total_requests" in stats
        assert "lid_failures" in stats


# Configuración de pytest
def pytest_configure(config):
    """Configuración global de pytest"""
    config.addinivalue_line(
        "markers", "integration: marca tests de integración (lentos)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
