"""
tests/test_omni_pipeline.py - Tests para Omni Pipeline v2.11

Valida el comportamiento del motor de voz Qwen2.5-Omni-3B:
- Carga del modelo ONNX
- STT + detección emocional
- TTS empático
- Integración con Safe Mode
- Auditoría HMAC
- API REST endpoints
"""

import pytest
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

# Mock de onnxruntime antes de importar omni_pipeline
sys_modules_backup = {}
if 'onnxruntime' not in sys.modules:
    sys.modules['onnxruntime'] = MagicMock()

from agents.omni_pipeline import (
    OmniSentinelEngine,
    AudioAuditLogger,
    process_audio_stream,
    SENTINEL_AUDIO_RESPONSES,
    TARGET_LATENCY_MS,
    SAMPLE_RATE
)


@pytest.fixture
def mock_onnx_session():
    """Mock de ONNX InferenceSession"""
    session = Mock()
    
    # Mock de inputs/outputs del modelo
    mock_input = Mock()
    mock_input.name = "audio"
    session.get_inputs.return_value = [mock_input]
    
    mock_output_text = Mock()
    mock_output_text.name = "text"
    mock_output_emo = Mock()
    mock_output_emo.name = "emotion_vec"
    mock_output_z = Mock()
    mock_output_z.name = "z_embedding"
    
    session.get_outputs.return_value = [mock_output_text, mock_output_emo, mock_output_z]
    
    # Mock del forward pass (STT + Emoción)
    def mock_run(outputs, inputs):
        audio_input = inputs["audio"]
        
        # Simular outputs del modelo
        text = "Hola, ¿cómo estás?"
        emotion_vec = np.random.rand(15).astype(np.float32)
        emotion_vec[1] = 0.9  # "happy" dominante
        z_embed = np.random.rand(768).astype(np.float32)
        
        return [text, emotion_vec, z_embed]
    
    session.run = Mock(side_effect=mock_run)
    
    return session


@pytest.fixture
def sample_audio_22k():
    """Audio de prueba a 22050 Hz"""
    duration = 2.0  # segundos
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    
    # Señal sinusoidal simple (440 Hz)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    return audio


@pytest.fixture
def mock_safe_mode_off():
    """Mock para desactivar safe mode"""
    with patch('agents.omni_pipeline.is_safe_mode', return_value=False):
        yield


@pytest.fixture
def mock_safe_mode_on():
    """Mock para activar safe mode"""
    with patch('agents.omni_pipeline.is_safe_mode', return_value=True):
        yield


@pytest.fixture
def temp_model_path(tmp_path):
    """Path temporal para modelo fake"""
    model_file = tmp_path / "fake_omni_model.onnx"
    model_file.write_bytes(b"FAKE_ONNX_MODEL_DATA")
    return str(model_file)


class TestOmniSentinelEngine:
    """Tests para el motor principal Omni-3B"""
    
    @patch('agents.omni_pipeline.ort.InferenceSession')
    def test_load_model_success(self, mock_onnx, temp_model_path, mock_onnx_session):
        """Verifica carga exitosa del modelo"""
        mock_onnx.return_value = mock_onnx_session
        
        engine = OmniSentinelEngine(temp_model_path)
        
        assert engine.session is not None
        assert engine.model_path == temp_model_path
        mock_onnx.assert_called_once()
    
    @patch('agents.omni_pipeline.ort.InferenceSession')
    def test_load_model_not_found(self, mock_onnx):
        """Verifica error cuando modelo no existe"""
        with pytest.raises(FileNotFoundError):
            engine = OmniSentinelEngine("nonexistent_model.onnx")
    
    @patch('agents.omni_pipeline.ort.InferenceSession')
    def test_stt_with_emotion(self, mock_onnx, temp_model_path, mock_onnx_session, sample_audio_22k):
        """Verifica STT + detección emocional"""
        mock_onnx.return_value = mock_onnx_session
        
        engine = OmniSentinelEngine(temp_model_path)
        result = engine.stt_with_emotion(sample_audio_22k)
        
        # Verificar estructura del resultado
        assert "text" in result
        assert "emotion" in result
        assert "emotion_vector" in result
        assert "embedding_z" in result
        assert "latency_ms" in result
        
        # Verificar tipos
        assert isinstance(result["text"], str)
        assert isinstance(result["emotion"], str)
        assert isinstance(result["emotion_vector"], np.ndarray)
        assert isinstance(result["embedding_z"], np.ndarray)
        assert isinstance(result["latency_ms"], float)
        
        # Verificar dimensiones
        assert result["emotion_vector"].shape == (15,)
        assert result["embedding_z"].shape == (768,)
        
        # Verificar latencia razonable
        assert result["latency_ms"] < TARGET_LATENCY_MS
    
    @patch('agents.omni_pipeline.ort.InferenceSession')
    def test_stt_normalizes_audio(self, mock_onnx, temp_model_path, mock_onnx_session):
        """Verifica que el audio se normaliza correctamente"""
        mock_onnx.return_value = mock_onnx_session
        
        # Audio con valores fuera de rango [-1, 1]
        audio_unnorm = np.array([0, 5.0, 10.0, -5.0, -10.0], dtype=np.float32)
        
        engine = OmniSentinelEngine(temp_model_path)
        
        # Capturar la llamada a session.run para verificar normalización
        original_run = engine.session.run
        captured_input = None
        
        def capture_run(outputs, inputs):
            nonlocal captured_input
            captured_input = inputs["audio"]
            return original_run(outputs, inputs)
        
        engine.session.run = Mock(side_effect=capture_run)
        engine.stt_with_emotion(audio_unnorm)
        
        # Verificar que el audio fue normalizado
        assert captured_input is not None
        assert np.abs(captured_input).max() <= 1.0
    
    @patch('agents.omni_pipeline.ort.InferenceSession')
    def test_emotion_detection_categories(self, mock_onnx, temp_model_path, mock_onnx_session, sample_audio_22k):
        """Verifica que las emociones detectadas están en las categorías válidas"""
        mock_onnx.return_value = mock_onnx_session
        
        engine = OmniSentinelEngine(temp_model_path)
        result = engine.stt_with_emotion(sample_audio_22k)
        
        valid_emotions = [
            "neutral", "happy", "sad", "angry", "frustrated",
            "surprised", "fearful", "disgusted", "calm", "excited",
            "bored", "confused", "determined", "hopeful", "worried"
        ]
        
        assert result["emotion"] in valid_emotions
    
    @patch('agents.omni_pipeline.ort.InferenceSession')
    def test_tts_empathic_basic(self, mock_onnx, temp_model_path, mock_onnx_session):
        """Verifica TTS empático básico"""
        mock_onnx.return_value = mock_onnx_session
        
        # Mock adicional para TTS (backward pass)
        def mock_run_tts(outputs, inputs):
            # Si es texto input → generar audio
            if "text" in inputs:
                # Generar audio sintético de 1 segundo
                audio_out = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
                return [audio_out]
            else:
                # STT path
                text = "Hola, ¿cómo estás?"
                emotion_vec = np.random.rand(15).astype(np.float32)
                z_embed = np.random.rand(768).astype(np.float32)
                return [text, emotion_vec, z_embed]
        
        mock_onnx_session.run = Mock(side_effect=mock_run_tts)
        
        engine = OmniSentinelEngine(temp_model_path)
        audio_out, latency_ms = engine.tts_empathic("Hola, ¿cómo estás?", "calm")
        
        assert isinstance(audio_out, np.ndarray)
        assert audio_out.shape[0] > 0  # Audio no vacío
        assert isinstance(latency_ms, float)
        assert latency_ms < TARGET_LATENCY_MS


class TestAudioAuditLogger:
    """Tests para auditoría HMAC de voz"""
    
    def test_log_interaction_creates_files(self, tmp_path):
        """Verifica que se crean archivos de log + HMAC"""
        log_dir = tmp_path / "audio_logs"
        log_dir.mkdir()
        
        secret = b"test-secret-key"
        logger = AudioAuditLogger(log_dir, secret)
        
        # Datos de prueba
        audio_hash = "abc123"
        stt_result = {
            "text": "Hola",
            "emotion": "happy",
            "latency_ms": 150.0
        }
        llm_response = "¿En qué puedo ayudarte?"
        tts_latency = 100.0
        context = "test"
        
        logger.log_interaction(audio_hash, stt_result, llm_response, tts_latency, context)
        
        # Verificar que se crearon archivos
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        
        log_file = log_dir / f"voice_interactions_{today}.jsonl"
        hmac_file = log_dir / f"voice_interactions_{today}.jsonl.hmac"
        
        assert log_file.exists()
        assert hmac_file.exists()
    
    def test_hmac_signature_valid(self, tmp_path):
        """Verifica que la firma HMAC es válida"""
        log_dir = tmp_path / "audio_logs"
        log_dir.mkdir()
        
        secret = b"test-secret-key"
        logger = AudioAuditLogger(log_dir, secret)
        
        # Log de interacción
        audio_hash = "test123"
        stt_result = {"text": "Test", "emotion": "neutral", "latency_ms": 100.0}
        logger.log_interaction(audio_hash, stt_result, "Response", 50.0, "test")
        
        # Leer log + HMAC
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        
        with open(log_dir / f"voice_interactions_{today}.jsonl") as f:
            log_line = f.read().strip()
        
        with open(log_dir / f"voice_interactions_{today}.jsonl.hmac") as f:
            stored_hmac = f.read().strip()
        
        # Verificar HMAC
        import hmac
        import hashlib
        computed_hmac = hmac.new(secret, log_line.encode(), hashlib.sha256).hexdigest()
        
        assert computed_hmac == stored_hmac


class TestProcessAudioStream:
    """Tests para la función principal de procesamiento"""
    
    @patch('agents.omni_pipeline.OmniSentinelEngine')
    def test_process_audio_stream_basic(self, mock_engine_class, mock_safe_mode_off, sample_audio_22k):
        """Verifica procesamiento básico de audio"""
        # Mock del engine
        mock_engine = Mock()
        mock_engine.stt_with_emotion.return_value = {
            "text": "Hola",
            "emotion": "happy",
            "emotion_vector": np.random.rand(15),
            "embedding_z": np.random.rand(768),
            "latency_ms": 100.0
        }
        mock_engine.tts_empathic.return_value = (sample_audio_22k, 50.0)
        mock_engine_class.return_value = mock_engine
        
        result = process_audio_stream(
            sample_audio_22k.tobytes(), 
            context="test",
            engine_instance=mock_engine
        )
        
        assert "text" in result
        assert "audio" in result
        assert "latency_ms" in result
    
    @patch('agents.omni_pipeline.OmniSentinelEngine')
    def test_process_audio_stream_safe_mode(self, mock_engine_class, mock_safe_mode_on, sample_audio_22k):
        """Verifica que Safe Mode retorna sentinel response"""
        result = process_audio_stream(sample_audio_22k.tobytes(), context="test")
        
        assert result["text"] == SENTINEL_AUDIO_RESPONSES["safe_mode"]["text"]
        assert result["sentinel_triggered"] is True
        
        # Engine NO debe ser llamado en safe mode
        mock_engine_class.assert_not_called()


class TestAPIEndpoints:
    """Tests para los endpoints de la API REST"""
    
    @pytest.fixture
    def client(self):
        """Cliente de Flask para testing"""
        from agents.omni_pipeline import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health_endpoint(self, client):
        """Verifica endpoint /health"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
    
    @patch('agents.omni_pipeline.is_safe_mode', return_value=True)
    def test_voice_gateway_safe_mode(self, mock_safe, client):
        """Verifica que /voice-gateway respeta Safe Mode"""
        response = client.post('/voice-gateway')
        
        assert response.status_code == 200
        assert b"modo seguro" in response.data  # Audio sentinel


# Configuración de pytest
def pytest_configure(config):
    """Configuración global"""
    config.addinivalue_line(
        "markers", "slow: marca tests lentos (requieren modelo real)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
