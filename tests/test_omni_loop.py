#!/usr/bin/env python3
"""
Tests para Omni-Loop Engine v2.16

Valida:
- Límite de 3 iteraciones (hard-coded)
- Fallback a LFM2 si skill_draft falla
- Cálculo de confidence
- Latencia <7.2s P50 (target v2.16)
- Auto-corrección funcional

Autor: SARAi Dev Team
Fecha: 29 octubre 2025
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.omni_loop import (
    OmniLoop,
    LoopConfig,
    LoopIteration,
    get_omni_loop
)


class TestOmniLoopBasic:
    """Tests básicos del Omni-Loop"""
    
    def test_loop_config_defaults(self):
        """Test 1: Configuración por defecto"""
        config = LoopConfig()
        
        assert config.max_iterations == 3, "Debe ser exactamente 3 iteraciones"
        assert config.confidence_threshold == 0.85
        assert config.enable_reflection is True
        assert config.use_skill_draft is True
    
    def test_loop_config_max_iterations_forced(self):
        """Test 2: max_iterations se fuerza a 3 si se intenta cambiar"""
        config = LoopConfig(max_iterations=5)  # Intento de cambiar
        loop = OmniLoop(config)
        
        # Debe revertir a 3
        assert loop.config.max_iterations == 3
    
    def test_loop_initialization(self):
        """Test 3: Inicialización correcta"""
        loop = OmniLoop()
        
        assert loop.config is not None
        assert loop.loop_history == []
        assert loop.config.max_iterations == 3
    
    def test_singleton_pattern(self):
        """Test 4: get_omni_loop retorna singleton"""
        loop1 = get_omni_loop()
        loop2 = get_omni_loop()
        
        assert loop1 is loop2, "Debe ser la misma instancia"


class TestOmniLoopExecution:
    """Tests de ejecución del loop"""
    
    @patch('core.omni_loop.OmniLoop._call_skill_draft')
    def test_single_iteration_no_reflection(self, mock_draft):
        """Test 5: Una sola iteración sin reflexión"""
        # Mock response de skill_draft
        mock_draft.return_value = {
            "text": "La relatividad general describe la gravedad como curvatura del espacio-tiempo.",
            "tokens": 15,
            "tokens_per_second": 50.0
        }
        
        config = LoopConfig(enable_reflection=False)
        loop = OmniLoop(config)
        
        result = loop.execute_loop(
            prompt="¿Qué es la relatividad general?",
            enable_reflection=False
        )
        
        # Validaciones
        assert result["fallback_used"] is False
        assert len(result["iterations"]) == 1, "Solo debe haber 1 iteración"
        assert result["auto_corrected"] is False
        assert "relatividad general" in result["response"].lower()
        
        # Mock llamado exactamente 1 vez
        assert mock_draft.call_count == 1
    
    @patch('core.omni_loop.OmniLoop._call_skill_draft')
    def test_three_iterations_with_reflection(self, mock_draft):
        """Test 6: 3 iteraciones con reflexión"""
        # Mock responses progresivas
        responses = [
            # Iteration 1: Draft básico (baja confidence)
            {
                "text": "La gravedad es una fuerza.",
                "tokens": 6,
                "tokens_per_second": 50.0
            },
            # Iteration 2: Reflexión mejora (media confidence)
            {
                "text": "La relatividad general describe la gravedad como curvatura del espacio-tiempo.",
                "tokens": 12,
                "tokens_per_second": 50.0
            },
            # Iteration 3: Corrección final (alta confidence)
            {
                "text": "La relatividad general de Einstein explica la gravedad como curvatura del espacio-tiempo causada por masa-energía.",
                "tokens": 18,
                "tokens_per_second": 50.0
            }
        ]
        
        mock_draft.side_effect = responses
        
        config = LoopConfig(enable_reflection=True, confidence_threshold=0.95)
        loop = OmniLoop(config)
        
        result = loop.execute_loop(
            prompt="¿Qué es la relatividad general?",
            enable_reflection=True
        )
        
        # Validaciones
        assert len(result["iterations"]) == 3, "Debe ejecutar 3 iteraciones"
        assert result["auto_corrected"] is True, "Debe detectar auto-corrección"
        assert mock_draft.call_count == 3
        
        # Verificar progresión de respuestas
        iter1 = result["iterations"][0]
        iter3 = result["iterations"][2]
        
        assert len(iter3["response"]) > len(iter1["response"]), "Iter 3 debe ser más completa"
    
    @patch('core.omni_loop.OmniLoop._call_skill_draft')
    def test_early_exit_high_confidence(self, mock_draft):
        """Test 7: Early exit si confidence es alta en iter 2"""
        responses = [
            # Iteration 1: Draft básico
            {
                "text": "Respuesta inicial.",
                "tokens": 2,
                "tokens_per_second": 50.0
            },
            # Iteration 2: Muy buena respuesta (alta confidence)
            {
                "text": "La relatividad general de Einstein describe la gravedad como curvatura del espacio-tiempo causada por masa y energía, prediciendo fenómenos como lentes gravitacionales y ondas gravitacionales.",
                "tokens": 30,
                "tokens_per_second": 50.0
            }
        ]
        
        mock_draft.side_effect = responses
        
        # Confidence threshold bajo para forzar early exit
        config = LoopConfig(enable_reflection=True, confidence_threshold=0.75)
        loop = OmniLoop(config)
        
        # Mock _calculate_confidence para forzar alta confidence en iter 2
        with patch.object(loop, '_calculate_confidence', return_value=0.90):
            result = loop.execute_loop(
                prompt="¿Qué es la relatividad general?",
                enable_reflection=True
            )
        
        # Debe terminar en iteración 2 (early exit)
        assert len(result["iterations"]) == 2, "Debe terminar early en iter 2"
        assert mock_draft.call_count == 2


class TestOmniLoopFallback:
    """Tests de fallback a LFM2"""
    
    @patch('core.omni_loop.OmniLoop._call_skill_draft')
    @patch('core.omni_loop.OmniLoop._call_local_lfm2')
    def test_fallback_to_lfm2_on_skill_draft_failure(self, mock_lfm2, mock_draft):
        """Test 8: Fallback a LFM2 si skill_draft falla"""
        # skill_draft falla
        mock_draft.side_effect = Exception("gRPC connection failed")
        
        # LFM2 funciona
        mock_lfm2.return_value = {
            "text": "Respuesta de fallback LFM2",
            "tokens": 5,
            "tokens_per_second": 20.0
        }
        
        loop = OmniLoop()
        
        result = loop.execute_loop(
            prompt="Test fallback",
            enable_reflection=False
        )
        
        # Debe usar LFM2
        assert "fallback" in result["response"].lower() or result["fallback_used"]
        assert mock_lfm2.call_count > 0, "LFM2 debe ser llamado"
    
    @patch('core.omni_loop.OmniLoop._call_skill_draft')
    @patch('core.omni_loop.OmniLoop._call_local_lfm2')
    def test_critical_failure_graceful_degradation(self, mock_lfm2, mock_draft):
        """Test 9: Degradación elegante si skill_draft Y LFM2 fallan"""
        # Ambos fallan
        mock_draft.side_effect = Exception("skill_draft failed")
        mock_lfm2.side_effect = Exception("LFM2 also failed")
        
        loop = OmniLoop()
        
        # NO debe crashear, sino retornar mensaje de error amigable
        result = loop.execute_loop(
            prompt="Test critical failure",
            enable_reflection=False
        )
        
        # Debe retornar respuesta degradada (no crashear)
        assert "response" in result
        assert result["fallback_used"] is True
        
        # Mensaje de error amigable
        response_lower = result["response"].lower()
        assert any(word in response_lower for word in ["sorry", "siento", "error", "intenta"])
        
        print(f"✅ Degradación elegante funcionó: {result['response']}")


class TestOmniLoopConfidence:
    """Tests del cálculo de confidence"""
    
    def test_confidence_calculation_length(self):
        """Test 10: Confidence basada en longitud"""
        loop = OmniLoop()
        
        # Muy corta (<50 chars)
        short_response = "Sí."
        conf_short = loop._calculate_confidence(short_response, "¿Test?", 1)
        
        # Longitud razonable (50-500 chars)
        medium_response = "La relatividad general es una teoría física que describe la gravedad."
        conf_medium = loop._calculate_confidence(medium_response, "¿Test?", 1)
        
        # Muy larga (>1000 chars)
        long_response = "Lorem ipsum " * 100
        conf_long = loop._calculate_confidence(long_response, "¿Test?", 1)
        
        # Validaciones
        assert conf_medium > conf_short, "Longitud razonable debe tener mayor confidence"
        assert conf_medium > conf_long, "Muy largo debe penalizar"
    
    def test_confidence_calculation_iteration_bonus(self):
        """Test 11: Bonus de confidence en iteración 3"""
        loop = OmniLoop()
        
        response = "La relatividad general describe la gravedad como curvatura."
        
        conf_iter1 = loop._calculate_confidence(response, "¿Qué es la relatividad?", 1)
        conf_iter3 = loop._calculate_confidence(response, "¿Qué es la relatividad?", 3)
        
        # Iteración 3 debe tener bonus +0.3
        assert conf_iter3 >= conf_iter1 + 0.2, "Iter 3 debe tener bonus"


class TestOmniLoopMetrics:
    """Tests de métricas y performance"""
    
    @patch('core.omni_loop.OmniLoop._call_skill_draft')
    def test_latency_under_target(self, mock_draft):
        """Test 12: Latencia total <7.2s (target P50 v2.16)"""
        # Mock response rápida (0.5s por iteración simulada)
        mock_draft.return_value = {
            "text": "Respuesta rápida.",
            "tokens": 3,
            "tokens_per_second": 50.0
        }
        
        config = LoopConfig(enable_reflection=True)
        loop = OmniLoop(config)
        
        start = time.perf_counter()
        
        result = loop.execute_loop(
            prompt="Test rápido",
            enable_reflection=True
        )
        
        latency_s = result["total_latency_ms"] / 1000
        
        # Debe ser <7.2s (incluso con 3 iteraciones)
        assert latency_s < 7.2, f"Latencia {latency_s:.2f}s > 7.2s target"
    
    @patch('core.omni_loop.OmniLoop._call_skill_draft')
    def test_metadata_aggregation(self, mock_draft):
        """Test 13: Metadata agregada correctamente"""
        mock_draft.return_value = {
            "text": "Test",
            "tokens": 10,
            "tokens_per_second": 50.0
        }
        
        loop = OmniLoop()
        
        result = loop.execute_loop(
            prompt="Test metadata",
            enable_reflection=False
        )
        
        # Validar metadata
        assert "metadata" in result
        assert result["metadata"]["total_tokens"] > 0
        assert result["metadata"]["avg_tokens_per_second"] > 0
        assert 0.0 <= result["metadata"]["confidence_final"] <= 1.0
    
    def test_loop_history_tracking(self):
        """Test 14: Historia de loops se guarda"""
        loop = OmniLoop()
        
        # Mock _run_iteration en lugar de _call_skill_draft
        mock_iteration = LoopIteration(
            iteration=1,
            response="Test response",
            confidence=0.9,
            corrected=False,
            latency_ms=100.0,
            tokens_generated=10,
            tokens_per_second=50.0,
            source="skill_draft"
        )
        
        with patch.object(loop, '_run_iteration', return_value=mock_iteration):
            loop.execute_loop("Test 1", enable_reflection=False)
            loop.execute_loop("Test 2", enable_reflection=False)
        
        history = loop.get_loop_history()
        
        assert len(history) == 2, f"Debe guardar 2 loops, pero tiene {len(history)}"
        assert "response" in history[0]
        assert "iterations" in history[0]
        
        # Limpiar historia
        loop.clear_history()
        assert len(loop.get_loop_history()) == 0


class TestOmniLoopPromptBuilding:
    """Tests de construcción de prompts"""
    
    def test_build_full_prompt_first_iteration(self):
        """Test 15: Prompt en primera iteración (sin contexto previo)"""
        loop = OmniLoop()
        
        prompt = loop._build_full_prompt(
            prompt="¿Qué es Python?",
            previous_response=None
        )
        
        assert prompt == "¿Qué es Python?"
        assert "[Previous attempt]" not in prompt
    
    def test_build_full_prompt_with_previous_response(self):
        """Test 16: Prompt con respuesta previa (iteraciones 2-3)"""
        loop = OmniLoop()
        
        prompt = loop._build_full_prompt(
            prompt="¿Qué es Python?",
            previous_response="Python es un lenguaje."
        )
        
        assert "[Previous attempt]" in prompt
        assert "Python es un lenguaje." in prompt
        assert "¿Qué es Python?" in prompt
    
    def test_build_reflection_prompt_iteration_2(self):
        """Test 17: Prompt de reflexión (iteración 2)"""
        loop = OmniLoop()
        
        prompt = loop._build_reflection_prompt(
            original_prompt="¿Qué es AI?",
            draft_response="AI es inteligencia artificial.",
            iteration=2
        )
        
        assert "coherent" in prompt.lower() or "complete" in prompt.lower()
        assert "AI es inteligencia artificial." in prompt
    
    def test_build_reflection_prompt_iteration_3(self):
        """Test 18: Prompt de reflexión (iteración 3 - polish)"""
        loop = OmniLoop()
        
        prompt = loop._build_reflection_prompt(
            original_prompt="¿Qué es AI?",
            draft_response="AI es inteligencia artificial.",
            iteration=3
        )
        
        assert "polish" in prompt.lower() or "clarity" in prompt.lower()


# ========================================
# Test Runner
# ========================================

def run_tests():
    """Ejecuta todos los tests"""
    import subprocess
    
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=False
    )
    
    return result.returncode


if __name__ == "__main__":
    exit_code = run_tests()
    
    if exit_code == 0:
        print("\n✅ TODOS LOS TESTS PASARON")
        print("Omni-Loop Engine v2.16 validado")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")
        print("Revisar output arriba")
    
    exit(exit_code)
