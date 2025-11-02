"""
MCP Coverage Completer - Lleva core/mcp.py al 100%

Tests pragmáticos que cubren branches faltantes sin modificar arquitectura:
- Cache quantization con numpy arrays
- Skill detection edge cases
- Route_to_skills threshold filtering
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


@pytest.fixture
def dummy_embedder():
    """Embedder que retorna numpy arrays"""
    class DummyEmbed:
        def encode(self, text):
            return np.array([0.1] * 512)
    return DummyEmbed()


# =============================================================================
# CACHE QUANTIZE CON NUMPY (Lines 48-55)
# =============================================================================

def test_mcp_cache_quantize_with_numpy_arrays(dummy_embedder):
    """Cubre _quantize con numpy arrays correctos"""
    from core.mcp import MCPCache
    
    cache = MCPCache(embedder=dummy_embedder, ttl=60)
    
    # Crear embedding con numpy
    emb = np.array([0.5] * 512)
    key = cache._quantize(emb)
    
    # Verificar que se cuantiza correctamente
    assert isinstance(key, np.ndarray)
    assert key.dtype == np.uint8
    assert len(key) == 512


def test_mcp_cache_set_and_get_with_numpy(dummy_embedder):
    """Cubre set/get completo con embeddings numpy"""
    from core.mcp import MCPCache
    
    cache = MCPCache(embedder=dummy_embedder, ttl=60)
    
    # Set
    cache.set("test context", 0.7, 0.3)
    
    # Get
    result = cache.get("test context")
    
    assert result is not None
    alpha, beta = result
    assert alpha == pytest.approx(0.7)
    assert beta == pytest.approx(0.3)


# =============================================================================
# ROUTE_TO_SKILLS THRESHOLD (Lines 412-425)
# =============================================================================

def test_route_to_skills_filters_below_threshold():
    """Cubre filtrado de skills con score < 0.3"""
    from core.mcp import route_to_skills
    
    scores = {
        "hard": 0.5,
        "soft": 0.5,
        "programming": 0.8,  # Sobre threshold
        "financial": 0.2,    # Bajo threshold (filtrado)
        "creative": 0.1      # Bajo threshold (filtrado)
    }
    
    skills = route_to_skills(scores)
    
    # Solo programming debería pasar
    assert "programming" in skills
    assert "financial" not in skills
    assert "creative" not in skills


def test_route_to_skills_returns_top_k():
    """Cubre retorno de top-3 skills por score"""
    from core.mcp import route_to_skills
    
    scores = {
        "hard": 0.6,
        "soft": 0.4,
        "programming": 0.9,
        "financial": 0.8,
        "creative": 0.7,
        "diagnosis": 0.6,
        "reasoning": 0.5
    }
    
    skills = route_to_skills(scores)
    
    # Debe retornar top-3
    assert len(skills) <= 3
    assert "programming" in skills  # Highest
    assert "financial" in skills    # Second
    assert "creative" in skills     # Third


