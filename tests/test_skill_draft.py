#!/usr/bin/env python3
"""
Test de skill_draft - Verificación de filosofía Phoenix correcta

Valida que skill_draft:
1. Es una configuración de prompting sobre LFM2 (NO modelo separado)
2. Usa keywords/longtail para detección
3. Aplica parámetros optimizados (temperature, max_tokens)
4. NO añade RAM extra (reutiliza LFM2)
5. Latencia <500ms (sin overhead gRPC)

Autor: SARAi Dev Team
Fecha: 2 Noviembre 2025
Versión: v2.16 (corrección filosófica)
"""

import pytest
import time
from unittest.mock import MagicMock, patch


def test_draft_skill_exists():
    """Test 1: draft skill existe en ALL_SKILLS"""
    from core.skill_configs import ALL_SKILLS, get_skill
    
    assert "draft" in ALL_SKILLS, "draft skill NO encontrado en ALL_SKILLS"
    
    draft_skill = get_skill("draft")
    assert draft_skill is not None, "get_skill('draft') retorna None"
    assert draft_skill.name == "draft"
    assert draft_skill.preferred_model == "lfm2", "draft debe preferir LFM2 (tiny)"


def test_draft_skill_keywords():
    """Test 2: Keywords y longtail patterns de draft skill"""
    from core.skill_configs import get_skill
    
    draft_skill = get_skill("draft")
    
    # Verificar keywords
    expected_keywords = ["draft", "borrador", "iteración", "refinamiento", "inicial", "sketch"]
    for keyword in expected_keywords:
        assert keyword in draft_skill.keywords, f"Keyword '{keyword}' faltante"
    
    # Verificar parámetros optimizados
    assert draft_skill.temperature == 0.9, "Temperature debe ser alta (0.9) para creatividad"
    assert draft_skill.max_tokens == 150, "max_tokens debe ser 150 para drafts cortos"
    assert draft_skill.top_p == 0.95


def test_draft_skill_detection():
    """Test 3: Detección de draft skill por keywords"""
    from core.skill_configs import match_skill_by_keywords
    
    # Test queries que deben detectar draft
    draft_queries = [
        "genera un draft inicial",
        "necesito un borrador rápido",
        "primera iteración del texto",
        "sketch de la idea"
    ]
    
    for query in draft_queries:
        skill = match_skill_by_keywords(query)
        assert skill is not None, f"Query '{query}' no detectó ningún skill"
        assert skill.name == "draft", f"Query '{query}' detectó '{skill.name}' en vez de 'draft'"


def test_draft_skill_prompt_application():
    """Test 4: Aplicación de prompt especializado"""
    from core.skill_configs import get_skill
    
    draft_skill = get_skill("draft")
    user_query = "Explica la relatividad general"
    
    full_prompt = draft_skill.build_prompt(user_query)
    
    # Verificar que incluye system prompt
    assert "rapid draft generator" in full_prompt, "System prompt NO aplicado"
    assert "50-150 tokens" in full_prompt, "Guideline de longitud NO presente"
    assert user_query in full_prompt, "User query NO incluido en prompt"


def test_draft_skill_no_extra_ram():
    """Test 5: draft skill NO añade RAM (usa LFM2 ya cargado)"""
    from core.skill_configs import get_skill
    
    draft_skill = get_skill("draft")
    
    # Verificar que NO hay referencias a modelos adicionales
    assert draft_skill.preferred_model == "lfm2", "draft NO debe usar modelo separado"
    
    # Verificar que config NO tiene paths a modelos
    config_dict = draft_skill.to_dict()
    assert "model_path" not in config_dict, "draft NO debe tener model_path"
    assert "gguf_file" not in config_dict, "draft NO debe tener gguf_file"


def test_draft_skill_in_omni_loop():
    """Test 6: draft skill se integra correctamente en Omni-Loop"""
    from core.omni_loop import OmniLoop
    from unittest.mock import MagicMock, Mock
    import sys
    
    # Mock ModelPool module antes de importar
    mock_model_pool_module = Mock()
    mock_get_model_pool = MagicMock()
    mock_model_pool_module.get_model_pool = mock_get_model_pool
    sys.modules['core.model_pool'] = mock_model_pool_module
    
    try:
        # Mock LFM2 response
        mock_lfm2 = MagicMock()
        mock_lfm2.return_value = {
            "choices": [{"text": "Draft response sobre relatividad"}],
            "usage": {"completion_tokens": 50}
        }
        
        mock_pool = MagicMock()
        mock_pool.get.return_value = mock_lfm2
        mock_get_model_pool.return_value = mock_pool
        
        # Crear OmniLoop con draft skill
        loop = OmniLoop()
        
        # Ejecutar iteración (sin image ni previous_response)
        iteration = loop._run_iteration(
            prompt="Explica la relatividad general",
            image_path=None,
            iteration=1,
            previous_response=None
        )
        
        # Verificar que llamó a LFM2 (no servicios externos)
        assert mock_pool.get.called, "ModelPool.get() NO fue llamado"
        assert mock_pool.get.call_args[0][0] == "tiny", "Debe usar 'tiny' (LFM2)"
        
        # Verificar resultado
        assert iteration.response == "Draft response sobre relatividad"
        assert iteration.source == "draft_skill_lfm2", "Source debe indicar draft_skill_lfm2"
    
    finally:
        # Limpiar mock
        if 'core.model_pool' in sys.modules:
            del sys.modules['core.model_pool']


def test_draft_skill_latency_target():
    """Test 7: Latencia < 500ms (objetivo Phoenix)"""
    import time
    from core.omni_loop import OmniLoop
    from unittest.mock import MagicMock, Mock
    import sys
    
    # Mock ModelPool module
    mock_model_pool_module = Mock()
    mock_get_model_pool = MagicMock()
    mock_model_pool_module.get_model_pool = mock_get_model_pool
    sys.modules['core.model_pool'] = mock_model_pool_module
    
    try:
        # Mock LFM2 con latencia realista (300ms)
        def slow_generate(*args, **kwargs):
            time.sleep(0.3)  # Simular generación de 300ms
            return {
                "choices": [{"text": "Draft rápido"}],
                "usage": {"completion_tokens": 30}
            }
        
        mock_lfm2 = MagicMock(side_effect=slow_generate)
        
        mock_pool = MagicMock()
        mock_pool.get.return_value = mock_lfm2
        mock_get_model_pool.return_value = mock_pool
        
        loop = OmniLoop()
        
        start = time.perf_counter()
        iteration = loop._run_iteration(
            prompt="Draft inicial",
            image_path=None,
            iteration=1,
            previous_response=None
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Verificar latencia
        assert elapsed_ms < 500, f"Latency {elapsed_ms:.1f}ms > 500ms (overhead detectado)"
        assert iteration.latency_ms < 500
    
    finally:
        # Limpiar mock
        if 'core.model_pool' in sys.modules:
            del sys.modules['core.model_pool']


def test_draft_skill_vs_grpc_philosophy():
    """Test 8: Verificar que NO usa servicios externos (filosofía Phoenix)"""
    from core.omni_loop import OmniLoop
    import inspect
    
    loop = OmniLoop()
    
    # Verificar que _call_draft_skill NO usa servicios externos
    source = inspect.getsource(loop._call_draft_skill)
    
    # Strings que NO deben aparecer (indicadores de arquitectura incorrecta)
    forbidden_strings = [
        "grpc.channel",
        "skills_pb2",
        "draft_client",
        "get_skill_client"
    ]
    
    for forbidden in forbidden_strings:
        assert forbidden not in source, f"'{forbidden}' encontrado - viola filosofía Phoenix"
    
    # Strings que DEBEN aparecer (filosofía correcta)
    required_strings = [
        "detect_and_apply_skill",
        "get_model_pool",
        "tiny",  # LFM2
        "skill_config"
    ]
    
    for required in required_strings:
        assert required in source, f"'{required}' NO encontrado - implementación incorrecta"


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST SKILL_DRAFT - Filosofía Phoenix Correcta")
    print("="*70 + "\n")
    
    pytest.main([__file__, "-v", "-s"])

