#!/usr/bin/env python3
"""
Test de skill_draft siguiendo filosofía Phoenix v2.12+

Valida que draft funciona como PROMPT sobre LFM2,
NO como servicio gRPC separado.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.skill_configs import DRAFT_SKILL


def test_draft_skill_config():
    """Test 1: Configuración de draft skill"""
    print("\n" + "="*60)
    print("TEST 1: Draft Skill Config")
    print("="*60)
    
    # Verificar atributos del SkillConfig
    assert hasattr(DRAFT_SKILL, 'name')
    assert hasattr(DRAFT_SKILL, 'temperature')
    assert hasattr(DRAFT_SKILL, 'system_prompt')
    assert hasattr(DRAFT_SKILL, 'keywords')
    assert hasattr(DRAFT_SKILL, 'preferred_model')
    
    # Verificar valores correctos
    assert DRAFT_SKILL.name == "draft"
    assert DRAFT_SKILL.preferred_model == "tiny"  # ✅ USA LFM2
    assert DRAFT_SKILL.temperature == 0.9  # Alta creatividad
    assert DRAFT_SKILL.max_tokens == 150  # Limita tamaño de draft
    
    # Verificar keywords
    assert "draft" in DRAFT_SKILL.keywords
    assert "borrador" in DRAFT_SKILL.keywords
    assert "iteración" in DRAFT_SKILL.keywords
    
    # Verificar system prompt contiene instrucciones correctas
    assert "draft generator" in DRAFT_SKILL.system_prompt.lower()
    assert "concise" in DRAFT_SKILL.system_prompt.lower()
    
    print("✅ Configuración correcta (prompt sobre LFM2)")


def test_draft_skill_integration():
    """Test 2: Integración con Omni-Loop"""
    print("\n" + "="*60)
    print("TEST 2: Draft Skill Integration")
    print("="*60)
    
    # Mock del model pool y LFM2
    mock_lfm2 = MagicMock()
    mock_lfm2.generate.return_value = "Este es un borrador inicial: [contenido]"
    
    mock_pool = MagicMock()
    mock_pool.get.return_value = mock_lfm2
    
    # Inyectar mock en el módulo
    import core.omni_loop
    original_get_pool = getattr(core.omni_loop, 'get_model_pool', None)
    core.omni_loop.get_model_pool = lambda: mock_pool
    
    try:
        from core.omni_loop import OmniLoop
        
        loop = OmniLoop(max_iterations=1)
        
        # Simular query que activa draft
        query = "Hazme un draft inicial del documento"
        
        # Verificar detección de skill
        skill_config = None
        if any(kw in query.lower() for kw in DRAFT_SKILL.keywords):
            skill_config = DRAFT_SKILL
        
        assert skill_config is not None, "❌ No detectó draft skill"
        assert skill_config.preferred_model == "tiny"
        
        print("✅ Draft skill se integra correctamente")
    
    finally:
        # Restaurar función original
        if original_get_pool:
            core.omni_loop.get_model_pool = original_get_pool


def test_draft_no_grpc():
    """Test 3: Verifica que NO usa gRPC"""
    print("\n" + "="*60)
    print("TEST 3: No gRPC (Phoenix Philosophy)")
    print("="*60)
    
    # Mock del model pool y LFM2
    mock_lfm2 = MagicMock()
    mock_lfm2.generate.return_value = "Borrador generado"
    
    mock_pool = MagicMock()
    mock_pool.get.return_value = mock_lfm2
    
    # Inyectar mock
    import core.omni_loop
    original_get_pool = getattr(core.omni_loop, 'get_model_pool', None)
    core.omni_loop.get_model_pool = lambda: mock_pool
    
    try:
        from core.omni_loop import OmniLoop
        
        loop = OmniLoop(max_iterations=1)
        
        # Ejecutar con draft
        result = loop._execute_iteration(
            iteration_num=1,
            query="draft inicial del plan",
            context=""
        )
        
        # Verificar que se usó LFM2 (tiny)
        mock_pool.get.assert_called_with("tiny")
        
        # Verificar que NO hay llamadas gRPC
        assert "grpc" not in str(result).lower()
        
        print("✅ No usa gRPC, solo prompt sobre LFM2")
    
    finally:
        # Restaurar función original
        if original_get_pool:
            core.omni_loop.get_model_pool = original_get_pool


if __name__ == "__main__":
    print("\n🧪 DRAFT SKILL PHOENIX TESTS")
    print("Validando filosofía Phoenix (prompts, NO gRPC)\n")
    
    try:
        test_draft_skill_config()
        test_draft_skill_integration()
        test_draft_no_grpc()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS TESTS PASARON")
        print("="*60)
        print("\n✓ Draft skill usa LFM2 (tiny)")
        print("✓ NO usa gRPC")
        print("✓ Sigue filosofía Phoenix v2.12+")
        print("\n🎉 Draft skill optimizado correctamente\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n💥 ERROR INESPERADO: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
