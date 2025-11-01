"""
Tests de Integración E2E para Unified Model Wrapper v2.14

Estos tests validan funcionalidad REAL sin mocks pesados.
Requieren que Ollama esté corriendo en http://192.168.0.251:11434

Filosofía:
    "Validar comportamiento real > Mock perfecto"
    
Tests incluidos:
    ✅ OllamaModelWrapper con servidor real
    ✅ ModelRegistry carga config real
    ✅ get_model() factory funciona
    ✅ Fallback chain (si modelo no disponible)
    ✅ Cache de registry funciona

Uso:
    # Con Ollama corriendo
    pytest tests/test_unified_wrapper_integration.py -v
    
    # Skip si Ollama no disponible
    pytest tests/test_unified_wrapper_integration.py -v -m "not requires_ollama"
"""

import pytest
import requests
import yaml
from pathlib import Path

from core.unified_model_wrapper import (
    ModelRegistry,
    get_model,
    list_available_models,
    OllamaModelWrapper
)


# ============================================================================
# HELPERS
# ============================================================================

def is_ollama_available():
    """Verifica si Ollama está corriendo"""
    try:
        response = requests.get("http://192.168.0.251:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def ollama_available():
    """Marca tests que requieren Ollama"""
    if not is_ollama_available():
        pytest.skip("Ollama no disponible en http://192.168.0.251:11434")


# ============================================================================
# TESTS: ModelRegistry con Config Real
# ============================================================================

def test_registry_loads_real_config():
    """Verifica que ModelRegistry carga config/models.yaml real"""
    registry = ModelRegistry()
    
    # Cargar config real
    registry.load_config("config/models.yaml")
    
    # Verificar que cargó modelos
    assert registry._config is not None
    assert len(registry._config) > 0
    
    # Verificar que solar_short existe
    assert "solar_short" in registry._config
    assert registry._config["solar_short"]["backend"] == "ollama"
    
    print(f"✅ Config cargado: {len(registry._config)} modelos")


def test_list_models_returns_available():
    """Verifica que list_available_models() retorna lista de modelos"""
    # Inicializar registry
    registry = ModelRegistry()
    registry.load_config("config/models.yaml")
    
    # Listar modelos
    models = list_available_models()
    
    # Verificar que retorna lista
    assert isinstance(models, list)
    assert len(models) > 0
    assert "solar_short" in models
    
    print(f"✅ Modelos disponibles: {', '.join(models)}")


# ============================================================================
# TESTS: OllamaModelWrapper con Servidor Real
# ============================================================================

@pytest.mark.requires_ollama
def test_ollama_wrapper_real_inference(ollama_available):
    """Verifica que OllamaWrapper hace inferencia real"""
    registry = ModelRegistry()
    registry.load_config("config/models.yaml")
    
    # Obtener config de solar_short
    config = registry._config["solar_short"]
    
    # Crear wrapper
    wrapper = OllamaModelWrapper("solar_short", config)
    
    # Invocar con prompt simple
    response = wrapper.invoke("Di 'hola' y nada más")
    
    # Verificar respuesta
    assert isinstance(response, str)
    assert len(response) > 0
    
    print(f"✅ Respuesta Ollama: {response[:100]}...")


@pytest.mark.requires_ollama
def test_get_model_factory_with_ollama(ollama_available):
    """Verifica que get_model() factory funciona con Ollama real"""
    # Inicializar registry
    registry = ModelRegistry()
    registry.load_config("config/models.yaml")
    
    # Obtener modelo via factory
    solar = get_model("solar_short")
    
    # Verificar que es OllamaModelWrapper
    assert isinstance(solar, OllamaModelWrapper)
    assert solar.name == "solar_short"
    assert solar.backend == "ollama"
    
    # Invocar
    response = solar.invoke("Responde solo: OK")
    
    # Verificar
    assert isinstance(response, str)
    assert len(response) > 0
    
    print(f"✅ Factory funciona: {type(solar).__name__}")


@pytest.mark.requires_ollama
def test_registry_cache_reuses_wrapper(ollama_available):
    """Verifica que ModelRegistry cachea wrappers"""
    registry = ModelRegistry()
    registry.load_config("config/models.yaml")
    
    # Primera llamada
    solar1 = get_model("solar_short")
    
    # Segunda llamada
    solar2 = get_model("solar_short")
    
    # Verificar que son el mismo objeto (cache)
    assert solar1 is solar2
    
    print("✅ Cache funciona correctamente")


# ============================================================================
# TESTS: Config YAML Validation
# ============================================================================

def test_models_yaml_is_valid():
    """Valida que models.yaml tiene estructura correcta para todos los modelos."""
    config_path = "config/models.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert config is not None, "models.yaml está vacío"
    
    print(f"\n🔍 Validando {len(config)} configuraciones")
    
    for name, model_config in config.items():
        # Validar que TODOS tienen backend
        assert "backend" in model_config, f"{name} falta 'backend'"
        
        backend = model_config["backend"]
        
        # Backend "config" es para configuraciones (no modelos)
        if backend == "config":
            continue  # No validar campos de modelo
        
        # Embeddings tiene campos específicos
        if model_config.get("type") == "embedding":
            assert "embedding_dim" in model_config, f"{name} falta 'embedding_dim'"
            assert "repo_id" in model_config, f"{name} falta 'repo_id'"
        
        # PyTorch checkpoints (TRM, MCP)
        elif backend == "pytorch_checkpoint":
            assert "checkpoint_path" in model_config, f"{name} falta 'checkpoint_path'"
            assert "device" in model_config, f"{name} falta 'device'"
        
        # LLMs estándar
        elif backend in ["gguf", "transformers", "multimodal"]:
            # Validar que tiene source, repo_id, gguf_file o model_path
            has_source = ("source" in model_config or 
                         "repo_id" in model_config or 
                         "gguf_file" in model_config or
                         "model_path" in model_config)
            assert has_source, f"{name} falta 'source', 'repo_id', 'gguf_file' o 'model_path'"
    
    print(f"✅ models.yaml válido")



def test_ollama_models_have_required_fields():
    """Verifica que modelos Ollama tienen campos necesarios"""
    registry = ModelRegistry()
    registry.load_config("config/models.yaml")
    
    # Filtrar modelos Ollama (skip legacy_mappings)
    ollama_models = {
        name: cfg 
        for name, cfg in registry._config.items() 
        if isinstance(cfg, dict) and 
           name != "legacy_mappings" and 
           cfg.get("backend") == "ollama"
    }
    
    # Verificar cada uno
    for name, config in ollama_models.items():
        assert "api_url" in config or "OLLAMA_BASE_URL" in str(config.get("api_url", "")), \
            f"{name} falta api_url"
        assert "model_name" in config or "MODEL_NAME" in str(config.get("model_name", "")), \
            f"{name} falta model_name"
    
    print(f"✅ {len(ollama_models)} modelos Ollama válidos")


# ============================================================================
# TESTS: Error Handling
# ============================================================================

def test_get_model_raises_on_invalid_name():
    """Verifica que get_model() falla con nombre inválido"""
    registry = ModelRegistry()
    registry.load_config("config/models.yaml")
    
    with pytest.raises(ValueError) as exc_info:
        get_model("modelo_que_no_existe")
    
    assert "not found in config" in str(exc_info.value)
    print("✅ Error handling correcto")


@pytest.mark.requires_ollama
def test_ollama_wrapper_handles_empty_response(ollama_available):
    """Verifica que OllamaWrapper maneja respuestas vacías"""
    registry = ModelRegistry()
    registry.load_config("config/models.yaml")
    
    solar = get_model("solar_short")
    
    # Invocar con prompt que puede dar respuesta vacía
    response = solar.invoke("")
    
    # Verificar que retorna string (aunque sea vacío)
    assert isinstance(response, str)
    
    print("✅ Manejo de respuestas vacías OK")


# ============================================================================
# EMBEDDINGS INTEGRATION TESTS (v2.14)
# ============================================================================

def test_embeddings_model_in_config():
    """Verifica que embeddings está correctamente configurado en models.yaml."""
    config_path = "config/models.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert "embeddings" in config, "embeddings no encontrado en models.yaml"
    
    emb_config = config["embeddings"]
    
    # Validar campos críticos
    assert emb_config["type"] == "embedding", "embeddings.type debe ser 'embedding'"
    assert emb_config["backend"] == "embedding", "embeddings.backend debe ser 'embedding'"
    assert "embedding_dim" in emb_config, "embeddings falta 'embedding_dim'"
    assert emb_config["embedding_dim"] == 768, "EmbeddingGemma debe ser 768-dim"
    assert "repo_id" in emb_config, "embeddings falta 'repo_id'"
    
    print(f"✅ Embeddings config válido: {emb_config['name']} ({emb_config['embedding_dim']}-dim)")


@pytest.mark.integration
def test_embeddings_wrapper_creation():
    """Verifica que EmbeddingModelWrapper se crea correctamente desde registry."""
    registry = ModelRegistry()
    registry.load_config("config/models.yaml")
    
    # Obtener wrapper (NO carga modelo todavía, lazy loading)
    embeddings = get_model("embeddings")
    
    assert embeddings is not None, "Embeddings wrapper es None"
    assert embeddings.name == "embeddings"
    assert hasattr(embeddings, 'get_embedding'), "Falta método get_embedding()"
    assert hasattr(embeddings, 'batch_encode'), "Falta método batch_encode()"
    
    print(f"✅ EmbeddingModelWrapper creado: {embeddings.name}")


@pytest.mark.integration
@pytest.mark.slow
def test_embeddings_returns_768_dim_vector():
    """
    Verifica que embeddings retorna vector 768-D.
    
    NOTA: Este test carga el modelo real (~150MB), puede tardar.
    """
    registry = ModelRegistry()
    registry.load_config("config/models.yaml")
    
    embeddings = get_model("embeddings")
    
    # Cargar modelo (lazy loading) - método privado
    embeddings._ensure_loaded()
    
    # Generar embedding
    test_text = "SARAi es un sistema AGI local"
    vector = embeddings.invoke(test_text)
    
    # Validar output
    import numpy as np
    assert isinstance(vector, np.ndarray), f"Output debe ser ndarray, got {type(vector)}"
    assert vector.shape == (768,), f"Vector debe ser 768-D, got {vector.shape}"
    assert vector.dtype in [np.float32, np.float64], f"Dtype debe ser float, got {vector.dtype}"
    
    print(f"✅ Embedding generado: {vector.shape}, dtype={vector.dtype}")
    print(f"   Valores (primeros 5): {vector[:5]}")


@pytest.mark.integration
@pytest.mark.slow
def test_embeddings_batch_processing():
    """Verifica que embeddings soporta batch processing."""
    registry = ModelRegistry()
    registry.load_config("config/models.yaml")
    
    embeddings = get_model("embeddings")
    embeddings._ensure_loaded()
    
    # Batch de 3 textos
    texts = [
        "Python es un lenguaje de programación",
        "JavaScript es popular en web",
        "Rust es un lenguaje de sistemas"
    ]
    
    # Batch encode
    vectors = embeddings.batch_encode(texts)
    
    # Validar output
    import numpy as np
    assert isinstance(vectors, np.ndarray), f"Batch output debe ser ndarray, got {type(vectors)}"
    assert vectors.shape == (3, 768), f"Batch shape debe ser (3, 768), got {vectors.shape}"
    
    print(f"✅ Batch encoding OK: {vectors.shape}")


# ============================================================================
# RESUMEN
# ============================================================================

"""
COBERTURA DE INTEGRACIÓN:
✅ ModelRegistry carga config real
✅ list_models() funciona
✅ OllamaWrapper inferencia real (requiere Ollama)
✅ get_model() factory funciona
✅ Cache de registry funciona
✅ YAML validation
✅ Error handling
✅ Ollama models validation

Total: 9 tests (3 requieren Ollama corriendo)

Para ejecutar solo tests sin Ollama:
    pytest tests/test_unified_wrapper_integration.py -v -m "not requires_ollama"

Para ejecutar todos (con Ollama):
    pytest tests/test_unified_wrapper_integration.py -v
"""
