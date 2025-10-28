"""
tests/test_emotion_model.py

Suite de tests para wrapper de modelos de detección emocional

Testing:
- EmotionPrediction dataclass
- EmotionModelWrapper (lazy loading, predict, batch)
- Helper functions (factory, mapping)
- Edge cases y fallbacks

Author: SARAi v2.11
Date: 2025-10-28
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Importar módulo a testear
from agents.emotion_model import (
    EmotionPrediction,
    EmotionModelWrapper,
    create_emotion_model,
    map_emotion_to_category,
    TRANSFORMERS_AVAILABLE
)


# ============================================
# SKIP SI TRANSFORMERS NO DISPONIBLE
# ============================================

pytestmark = pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE,
    reason="transformers no disponible (pip install transformers)"
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def sample_audio():
    """Audio sintético de 1 segundo"""
    return np.random.randn(16000).astype(np.float32)


@pytest.fixture
def mock_model():
    """Mock del modelo Wav2Vec2"""
    model = Mock()
    model.config.id2label = {
        0: "angry",
        1: "happy",
        2: "sad",
        3: "neutral"
    }
    model.eval = Mock()
    model.to = Mock(return_value=model)
    return model


@pytest.fixture
def mock_processor():
    """Mock del processor"""
    processor = Mock()
    processor.return_value = {
        "input_values": np.random.randn(1, 16000).astype(np.float32)
    }
    return processor


# ============================================
# TESTS: EmotionPrediction
# ============================================

def test_emotion_prediction_creation():
    """EmotionPrediction se crea correctamente"""
    prediction = EmotionPrediction(
        emotion="happy",
        confidence=0.85,
        all_scores={"happy": 0.85, "sad": 0.10, "angry": 0.05}
    )
    
    assert prediction.emotion == "happy"
    assert prediction.confidence == 0.85
    assert len(prediction.all_scores) == 3
    assert prediction.raw_logits is None


def test_emotion_prediction_repr():
    """EmotionPrediction tiene repr legible"""
    prediction = EmotionPrediction(
        emotion="sad",
        confidence=0.72,
        all_scores={}
    )
    
    repr_str = repr(prediction)
    assert "sad" in repr_str
    assert "0.72" in repr_str


# ============================================
# TESTS: EmotionModelWrapper (Mocked)
# ============================================

@patch('agents.emotion_model.Wav2Vec2ForSequenceClassification')
@patch('agents.emotion_model.Wav2Vec2Processor')
def test_wrapper_initialization(mock_processor_cls, mock_model_cls):
    """Wrapper se inicializa correctamente"""
    wrapper = EmotionModelWrapper(
        model_name="wav2vec2-emotion-en",
        device="cpu"
    )
    
    assert wrapper.device == "cpu"
    assert wrapper._model is None  # Lazy loading
    assert wrapper._processor is None


@patch('agents.emotion_model.Wav2Vec2ForSequenceClassification')
@patch('agents.emotion_model.Wav2Vec2Processor')
def test_wrapper_auto_device(mock_processor_cls, mock_model_cls):
    """Wrapper auto-detecta device"""
    wrapper = EmotionModelWrapper(model_name="test")
    
    # Debe ser cpu o cuda según disponibilidad
    assert wrapper.device in ["cpu", "cuda"]


@patch('agents.emotion_model.Wav2Vec2ForSequenceClassification')
@patch('agents.emotion_model.Wav2Vec2Processor')
def test_load_model(mock_processor_cls, mock_model_cls, mock_model, mock_processor):
    """load_model() carga modelo correctamente"""
    # Configurar mocks
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor_cls.from_pretrained.return_value = mock_processor
    
    wrapper = EmotionModelWrapper(model_name="test", device="cpu")
    wrapper.load_model()
    
    # Verificar que se llamó from_pretrained
    assert mock_model_cls.from_pretrained.called
    assert mock_processor_cls.from_pretrained.called
    
    # Verificar que modelo se movió a cpu
    mock_model.to.assert_called_with("cpu")
    mock_model.eval.assert_called_once()


@patch('agents.emotion_model.Wav2Vec2ForSequenceClassification')
@patch('agents.emotion_model.Wav2Vec2Processor')
@patch('agents.emotion_model.torch')
def test_predict_basic(mock_torch, mock_processor_cls, mock_model_cls, 
                       mock_model, mock_processor, sample_audio):
    """predict() retorna EmotionPrediction válido"""
    # Configurar mocks
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor_cls.from_pretrained.return_value = mock_processor
    
    # Mock processor output (debe retornar tensors con .to())
    mock_tensor = MagicMock()
    mock_tensor.to.return_value = mock_tensor
    mock_processor.return_value = {
        "input_values": mock_tensor
    }
    
    # Mock logits y softmax
    mock_logits = MagicMock()
    mock_logits.cpu.return_value.numpy.return_value = np.array([[2.0, 5.0, 1.0, 0.5]])
    mock_model.return_value.logits = mock_logits
    
    # Mock softmax (happy tiene score más alto)
    mock_softmax = MagicMock()
    mock_softmax.cpu.return_value.numpy.return_value = np.array([[0.1, 0.7, 0.15, 0.05]])
    mock_torch.nn.functional.softmax.return_value = mock_softmax
    
    wrapper = EmotionModelWrapper(model_name="test", device="cpu")
    prediction = wrapper.predict(sample_audio)
    
    assert isinstance(prediction, EmotionPrediction)
    assert prediction.emotion == "happy"  # Max score
    assert 0.0 <= prediction.confidence <= 1.0


def test_predict_empty_audio_fails():
    """predict() falla con audio vacío"""
    wrapper = EmotionModelWrapper(model_name="test", device="cpu")
    
    with pytest.raises(ValueError, match="Audio vacío"):
        wrapper.predict(np.array([]))


@patch('agents.emotion_model.Wav2Vec2ForSequenceClassification')
@patch('agents.emotion_model.Wav2Vec2Processor')
def test_get_emotion_labels(mock_processor_cls, mock_model_cls, mock_model, mock_processor):
    """get_emotion_labels() retorna lista correcta"""
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor_cls.from_pretrained.return_value = mock_processor
    
    wrapper = EmotionModelWrapper(model_name="test", device="cpu")
    labels = wrapper.get_emotion_labels()
    
    assert isinstance(labels, list)
    assert len(labels) == 4
    assert "happy" in labels
    assert "sad" in labels


# ============================================
# TESTS: Helper Functions
# ============================================

def test_create_emotion_model_factory():
    """create_emotion_model() factory funciona"""
    wrapper = create_emotion_model("wav2vec2-emotion-en", device="cpu")
    
    assert isinstance(wrapper, EmotionModelWrapper)
    assert wrapper.device == "cpu"


def test_map_emotion_to_category_angry():
    """map_emotion_to_category() mapea variantes de angry"""
    assert map_emotion_to_category("ang") == "angry"
    assert map_emotion_to_category("anger") == "angry"
    assert map_emotion_to_category("ANGRY") == "angry"
    assert map_emotion_to_category("enojado") == "angry"


def test_map_emotion_to_category_happy():
    """map_emotion_to_category() mapea variantes de happy"""
    assert map_emotion_to_category("hap") == "happy"
    assert map_emotion_to_category("happiness") == "happy"
    assert map_emotion_to_category("feliz") == "happy"
    assert map_emotion_to_category("joy") == "happy"


def test_map_emotion_to_category_sad():
    """map_emotion_to_category() mapea variantes de sad"""
    assert map_emotion_to_category("sad") == "sad"
    assert map_emotion_to_category("sadness") == "sad"
    assert map_emotion_to_category("triste") == "sad"


def test_map_emotion_to_category_unknown():
    """map_emotion_to_category() retorna neutral para desconocidos"""
    assert map_emotion_to_category("unknown_emotion") == "neutral"
    assert map_emotion_to_category("xyz") == "neutral"
    assert map_emotion_to_category("") == "neutral"


def test_map_emotion_to_category_case_insensitive():
    """map_emotion_to_category() es case-insensitive"""
    assert map_emotion_to_category("HAPPY") == "happy"
    assert map_emotion_to_category("Happy") == "happy"
    assert map_emotion_to_category("hApPy") == "happy"


# ============================================
# TESTS: Edge Cases
# ============================================

@patch('agents.emotion_model.Wav2Vec2ForSequenceClassification')
@patch('agents.emotion_model.Wav2Vec2Processor')
def test_supported_models_dict(mock_processor_cls, mock_model_cls):
    """SUPPORTED_MODELS contiene modelos conocidos"""
    wrapper = EmotionModelWrapper(model_name="wav2vec2-emotion-en")
    
    assert "wav2vec2-emotion-en" in EmotionModelWrapper.SUPPORTED_MODELS
    assert "hubert-emotion" in EmotionModelWrapper.SUPPORTED_MODELS


@patch('agents.emotion_model.Wav2Vec2ForSequenceClassification')
@patch('agents.emotion_model.Wav2Vec2Processor')
def test_custom_model_id(mock_processor_cls, mock_model_cls, mock_model, mock_processor):
    """Wrapper acepta model IDs personalizados"""
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor_cls.from_pretrained.return_value = mock_processor
    
    custom_id = "user/custom-emotion-model"
    wrapper = EmotionModelWrapper(model_name=custom_id, device="cpu")
    
    assert wrapper.model_id == custom_id


# ============================================
# BENCHMARK (opcional, -k bench)
# ============================================

@pytest.mark.slow
@patch('agents.emotion_model.Wav2Vec2ForSequenceClassification')
@patch('agents.emotion_model.Wav2Vec2Processor')
@patch('agents.emotion_model.torch')
def test_benchmark_method(mock_torch, mock_processor_cls, mock_model_cls,
                          mock_model, mock_processor):
    """benchmark() retorna métricas válidas"""
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_processor_cls.from_pretrained.return_value = mock_processor
    
    # Mock predicción rápida
    mock_logits = MagicMock()
    mock_logits.cpu.return_value.numpy.return_value = np.array([[1, 2, 3, 4]])
    mock_model.return_value.logits = mock_logits
    
    mock_softmax = MagicMock()
    mock_softmax.cpu.return_value.numpy.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
    mock_torch.nn.functional.softmax.return_value = mock_softmax
    
    mock_processor.return_value = {"input_values": np.random.randn(1, 16000)}
    
    wrapper = EmotionModelWrapper(model_name="test", device="cpu")
    stats = wrapper.benchmark(audio_duration_s=1.0, num_iterations=3)
    
    assert "mean_time_ms" in stats
    assert "std_time_ms" in stats
    assert "throughput_samples_s" in stats
    assert stats["device"] == "cpu"


# ============================================
# INTEGRATION TEST (REAL MODEL - SKIPPED)
# ============================================

@pytest.mark.skip(reason="Requiere download real del modelo (~300MB)")
def test_real_model_integration():
    """
    Test de integración con modelo real (solo manual)
    
    Para ejecutar:
        pytest tests/test_emotion_model.py::test_real_model_integration -v -s
    """
    wrapper = create_emotion_model("wav2vec2-emotion-en", device="cpu")
    
    # Audio sintético feliz (tono alto)
    audio = np.sin(2 * np.pi * 600 * np.linspace(0, 1, 16000)).astype(np.float32)
    
    prediction = wrapper.predict(audio)
    
    print(f"\n🎭 Predicción: {prediction}")
    print(f"📊 Scores: {prediction.all_scores}")
    
    assert prediction.emotion in wrapper.get_emotion_labels()
    assert 0.0 <= prediction.confidence <= 1.0
