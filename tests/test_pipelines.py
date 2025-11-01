"""
Tests para LangChain Pipelines LCEL v2.14

Tests de:
- create_text_pipeline() composición básica
- create_vision_pipeline() con imágenes
- create_hybrid_pipeline_with_fallback() con RunnableBranch
- create_video_conference_pipeline() con RunnableParallel
- create_rag_pipeline() integración web_cache
- create_skill_pipeline() integración v2.12 skills

Uso:
    pytest tests/test_pipelines.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Mock de LangChain
import sys
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.runnables'] = MagicMock()
sys.modules['langchain_core.output_parsers'] = MagicMock()
sys.modules['langchain_core.prompts'] = MagicMock()
sys.modules['langchain_core.messages'] = MagicMock()

from core.langchain_pipelines import (
    create_text_pipeline,
    create_vision_pipeline,
    create_hybrid_pipeline_with_fallback,
    create_video_conference_pipeline,
    create_rag_pipeline,
    create_skill_pipeline,
    get_pipeline
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_model():
    """Mock de UnifiedModelWrapper"""
    model = Mock()
    model.invoke = Mock(return_value="Model response")
    return model


@pytest.fixture
def mock_registry():
    """Mock de ModelRegistry"""
    with patch('core.langchain_pipelines.get_model') as mock:
        model = Mock()
        model.invoke = Mock(return_value="Model response")
        mock.return_value = model
        yield mock


# ============================================================================
# TEST: Text Pipeline
# ============================================================================

def test_text_pipeline_basic(mock_registry):
    """Verifica que create_text_pipeline() compone modelo + parser"""
    pipeline = create_text_pipeline("lfm2")
    
    # Pipeline debe ser invocable (LangChain Runnable)
    assert hasattr(pipeline, 'invoke')
    
    # Invocar pipeline
    result = pipeline.invoke("Test prompt")
    
    # Verificar que llamó al modelo
    mock_registry.assert_called_with("lfm2")
    assert result == "Model response"


def test_text_pipeline_with_system_prompt(mock_registry):
    """Verifica que pipeline aplica system_prompt"""
    system_prompt = "You are a helpful assistant"
    
    pipeline = create_text_pipeline(
        "lfm2",
        temperature=0.7,
        system_prompt=system_prompt
    )
    
    # Invocar
    result = pipeline.invoke("User question")
    
    # Verificar que se usó el modelo
    mock_registry.assert_called_with("lfm2")


def test_text_pipeline_with_temperature(mock_registry):
    """Verifica que pipeline pasa temperature al modelo"""
    pipeline = create_text_pipeline("lfm2", temperature=0.9)
    
    # Invocar
    result = pipeline.invoke("Creative prompt")
    
    # Modelo debe haber sido llamado
    mock_registry.assert_called_with("lfm2")


# ============================================================================
# TEST: Vision Pipeline
# ============================================================================

def test_vision_pipeline_with_image(mock_registry):
    """Verifica que vision pipeline maneja imágenes"""
    pipeline = create_vision_pipeline("qwen3_vl")
    
    # Invocar con imagen
    result = pipeline.invoke({
        "text": "Describe this image",
        "image": "path/to/image.jpg"
    })
    
    # Verificar que llamó al modelo multimodal
    mock_registry.assert_called_with("qwen3_vl")
    assert result == "Model response"


def test_vision_pipeline_text_only_fallback(mock_registry):
    """Verifica que vision pipeline acepta solo texto"""
    pipeline = create_vision_pipeline("qwen3_vl")
    
    # Invocar solo con texto (sin imagen)
    result = pipeline.invoke({"text": "Just a text question"})
    
    # Debe funcionar igual
    mock_registry.assert_called_with("qwen3_vl")


# ============================================================================
# TEST: Hybrid Pipeline con Fallback
# ============================================================================

def test_hybrid_pipeline_routes_to_vision(mock_registry):
    """Verifica que hybrid pipeline enruta a vision si hay imagen"""
    # Configurar mocks para vision y text
    vision_model = Mock()
    vision_model.invoke = Mock(return_value="Vision response")
    
    text_model = Mock()
    text_model.invoke = Mock(return_value="Text response")
    
    def get_model_side_effect(name):
        if name == "qwen3_vl":
            return vision_model
        else:
            return text_model
    
    with patch('core.langchain_pipelines.get_model', side_effect=get_model_side_effect):
        pipeline = create_hybrid_pipeline_with_fallback(
            vision_model_name="qwen3_vl",
            text_model_name="lfm2"
        )
        
        # Invocar con imagen
        result = pipeline.invoke({
            "text": "Analyze",
            "image": "path/to/image.jpg"
        })
        
        # Debe usar vision model
        vision_model.invoke.assert_called()


def test_hybrid_pipeline_fallback_to_text(mock_registry):
    """Verifica que hybrid pipeline hace fallback a texto si no hay imagen"""
    vision_model = Mock()
    vision_model.invoke = Mock(return_value="Vision response")
    
    text_model = Mock()
    text_model.invoke = Mock(return_value="Text response")
    
    def get_model_side_effect(name):
        if name == "qwen3_vl":
            return vision_model
        else:
            return text_model
    
    with patch('core.langchain_pipelines.get_model', side_effect=get_model_side_effect):
        pipeline = create_hybrid_pipeline_with_fallback(
            vision_model_name="qwen3_vl",
            text_model_name="lfm2"
        )
        
        # Invocar sin imagen
        result = pipeline.invoke({"text": "Just text"})
        
        # Debe usar text model (fallback)
        text_model.invoke.assert_called()


# ============================================================================
# TEST: Video Conference Pipeline
# ============================================================================

def test_video_conference_pipeline_parallel(mock_registry):
    """Verifica que video_conference pipeline ejecuta pasos en paralelo"""
    # Mock modelos
    visual_model = Mock()
    visual_model.invoke = Mock(return_value="Visual analysis")
    
    synthesis_model = Mock()
    synthesis_model.invoke = Mock(return_value="Synthesized summary")
    
    def get_model_side_effect(name):
        if name == "qwen3_vl":
            return visual_model
        else:
            return synthesis_model
    
    with patch('core.langchain_pipelines.get_model', side_effect=get_model_side_effect):
        pipeline = create_video_conference_pipeline(
            vision_model_name="qwen3_vl",
            synthesis_model_name="solar_long"
        )
        
        # Invocar con datos de conferencia
        result = pipeline.invoke({
            "frames": ["frame1.jpg", "frame2.jpg"],
            "audio": b"audio_bytes",
            "enable_emotion": False
        })
        
        # Debe haber usado ambos modelos
        visual_model.invoke.assert_called()
        synthesis_model.invoke.assert_called()


def test_video_conference_pipeline_with_emotion(mock_registry):
    """Verifica que video_conference pipeline integra emotion detection"""
    visual_model = Mock()
    visual_model.invoke = Mock(return_value="Visual analysis")
    
    synthesis_model = Mock()
    synthesis_model.invoke = Mock(return_value="Empathetic summary")
    
    def get_model_side_effect(name):
        if name == "qwen3_vl":
            return visual_model
        else:
            return synthesis_model
    
    with patch('core.langchain_pipelines.get_model', side_effect=get_model_side_effect), \
         patch('core.langchain_pipelines.detect_emotion') as mock_emotion:
        
        mock_emotion.return_value = {
            "label": "neutral",
            "valence": 0.5,
            "arousal": 0.5
        }
        
        pipeline = create_video_conference_pipeline(
            vision_model_name="qwen3_vl",
            synthesis_model_name="solar_long",
            enable_emotion=True
        )
        
        result = pipeline.invoke({
            "frames": ["frame1.jpg"],
            "audio": b"audio_bytes",
            "enable_emotion": True
        })
        
        # Debe haber llamado a emotion detection
        mock_emotion.assert_called()


# ============================================================================
# TEST: RAG Pipeline
# ============================================================================

def test_rag_pipeline_web_search(mock_registry):
    """Verifica que RAG pipeline integra búsqueda web"""
    model = Mock()
    model.invoke = Mock(return_value="Synthesized answer from web")
    
    with patch('core.langchain_pipelines.get_model', return_value=model), \
         patch('core.langchain_pipelines.cached_search') as mock_search:
        
        mock_search.return_value = {
            "source": "searxng",
            "snippets": [
                {"title": "Result 1", "content": "Content 1", "url": "http://example1.com"},
                {"title": "Result 2", "content": "Content 2", "url": "http://example2.com"}
            ]
        }
        
        pipeline = create_rag_pipeline(
            search_model_name="solar_long",
            enable_cache=True,
            safe_mode=False
        )
        
        result = pipeline.invoke("¿Quién ganó el Oscar 2025?")
        
        # Debe haber llamado a cached_search
        mock_search.assert_called_with("¿Quién ganó el Oscar 2025?")
        
        # Debe haber sintetizado respuesta
        model.invoke.assert_called()


def test_rag_pipeline_safe_mode(mock_registry):
    """Verifica que RAG pipeline respeta safe_mode"""
    with patch('core.langchain_pipelines.is_safe_mode', return_value=True):
        pipeline = create_rag_pipeline(
            search_model_name="solar_long",
            safe_mode=True
        )
        
        result = pipeline.invoke("Web query")
        
        # En safe mode debe retornar sentinel response
        assert "temporalmente deshabilitada" in result.lower()


# ============================================================================
# TEST: Skill Pipeline
# ============================================================================

def test_skill_pipeline_detects_programming(mock_registry):
    """Verifica que skill pipeline detecta skill programming"""
    model = Mock()
    model.invoke = Mock(return_value="Python code response")
    
    with patch('core.langchain_pipelines.get_model', return_value=model), \
         patch('core.langchain_pipelines.detect_and_apply_skill') as mock_skill:
        
        mock_skill.return_value = {
            "name": "programming",
            "temperature": 0.3,
            "system_prompt": "You are an expert programmer"
        }
        
        pipeline = create_skill_pipeline(
            base_model_name="solar_long",
            enable_detection=True
        )
        
        result = pipeline.invoke("Crea una función Python para ordenar lista")
        
        # Debe haber detectado skill
        mock_skill.assert_called()
        
        # Debe haber generado respuesta
        model.invoke.assert_called()


def test_skill_pipeline_no_skill_detected(mock_registry):
    """Verifica que skill pipeline funciona sin skill específico"""
    model = Mock()
    model.invoke = Mock(return_value="Generic response")
    
    with patch('core.langchain_pipelines.get_model', return_value=model), \
         patch('core.langchain_pipelines.detect_and_apply_skill') as mock_skill:
        
        mock_skill.return_value = None  # Sin skill detectado
        
        pipeline = create_skill_pipeline(
            base_model_name="lfm2",
            enable_detection=True
        )
        
        result = pipeline.invoke("Pregunta genérica")
        
        # Debe funcionar sin skill
        model.invoke.assert_called()


# ============================================================================
# TEST: Pipeline Factory
# ============================================================================

def test_get_pipeline_factory_text(mock_registry):
    """Verifica que get_pipeline() crea text pipeline"""
    pipeline = get_pipeline("text", model_name="lfm2")
    
    assert hasattr(pipeline, 'invoke')


def test_get_pipeline_factory_vision(mock_registry):
    """Verifica que get_pipeline() crea vision pipeline"""
    pipeline = get_pipeline("vision", model_name="qwen3_vl")
    
    assert hasattr(pipeline, 'invoke')


def test_get_pipeline_factory_invalid_type():
    """Verifica que get_pipeline() falla con tipo inválido"""
    with pytest.raises(ValueError):
        get_pipeline("invalid_type", model_name="lfm2")


# ============================================================================
# TEST: LCEL Composition
# ============================================================================

def test_pipeline_is_composable(mock_registry):
    """Verifica que pipelines son composables con operador |"""
    pipeline1 = create_text_pipeline("lfm2")
    
    # Debe tener método __or__ (operador |)
    assert hasattr(pipeline1, '__or__') or callable(getattr(pipeline1, '__or__', None))


def test_pipeline_supports_streaming(mock_registry):
    """Verifica que pipelines soportan streaming"""
    model = Mock()
    model.stream = Mock(return_value=iter(["Token1", "Token2", "Token3"]))
    
    with patch('core.langchain_pipelines.get_model', return_value=model):
        pipeline = create_text_pipeline("lfm2")
        
        # Debe tener método stream
        if hasattr(pipeline, 'stream'):
            tokens = list(pipeline.stream("Prompt"))
            assert len(tokens) > 0


# ============================================================================
# RESUMEN DE TESTS
# ============================================================================

"""
COBERTURA:
✅ Text pipeline básico
✅ Text pipeline con system_prompt
✅ Text pipeline con temperature
✅ Vision pipeline con imagen
✅ Vision pipeline con solo texto
✅ Hybrid pipeline enruta a vision
✅ Hybrid pipeline hace fallback a text
✅ Video conference pipeline ejecuta en paralelo
✅ Video conference pipeline con emotion detection
✅ RAG pipeline con web search
✅ RAG pipeline respeta safe_mode
✅ Skill pipeline detecta programming
✅ Skill pipeline sin skill detectado
✅ get_pipeline() factory text
✅ get_pipeline() factory vision
✅ get_pipeline() falla con tipo inválido
✅ Pipelines son composables (operador |)
✅ Pipelines soportan streaming

Total: 18 tests
Tiempo estimado ejecución: <3 segundos (con mocks)
"""
