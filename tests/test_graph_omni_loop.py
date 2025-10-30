#!/usr/bin/env python3
"""
Tests de Integración LangGraph con Omni-Loop v2.16

Valida:
- Routing correcto a omni_loop cuando hay imagen + texto
- Estado se propaga correctamente
- Métricas de iteraciones se guardan
- Fallback funciona si omni_loop falla

Autor: SARAi Dev Team
Fecha: 29 octubre 2025
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import SARAiOrchestrator, State


class TestLangGraphOmniLoopIntegration:
    """Tests de integración del Omni-Loop en LangGraph"""
    
    @patch('core.model_pool.get_model_pool')
    @patch('core.graph.get_omni_loop')
    @patch('core.graph.get_expert_agent')
    @patch('core.graph.get_tiny_agent')
    @patch('core.graph.get_omni_agent')
    @patch('core.graph.create_rag_node')
    def test_routing_to_omni_loop_with_image_and_text(
        self, 
        mock_rag,
        mock_omni_agent,
        mock_tiny,
        mock_expert,
        mock_omni_loop,
        mock_pool
    ):
        """Test 1: Routing directo del _route_to_agent cuando hay imagen + texto"""
        
        # Mock del omni_loop
        mock_loop_instance = Mock()
        mock_loop_instance.execute_loop.return_value = {
            "response": "Respuesta reflexiva sobre la imagen",
            "iterations": [
                {"iteration": 1, "response": "Draft 1", "confidence": 0.7},
                {"iteration": 2, "response": "Draft 2", "confidence": 0.85}
            ],
            "total_latency_ms": 1500.0,
            "auto_corrected": True,
            "fallback_used": False,
            "metadata": {
                "confidence_final": 0.85,
                "total_tokens": 50,
                "avg_tokens_per_second": 30.0,
                "num_iterations": 2
            }
        }
        mock_omni_loop.return_value = mock_loop_instance
        
        # Crear orchestrator con TRM simulado
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        # Probar directamente el routing (sin pasar por detect_input_type)
        # Simular que ya pasó por classify y mcp
        state = {
            "input": "¿Qué hay en esta imagen? Descríbelo en detalle.",
            "input_type": "text",  # Se establece como texto para que pase por routing normal
            "image_path": "/tmp/test_image.jpg",
            "audio_input": None,
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "video_path": None,
            "fps": None,
            "enable_reflection": True,
            "omni_loop_iterations": None,
            "auto_corrected": False,
            "hard": 0.6,
            "soft": 0.4,
            "web_query": 0.1,
            "alpha": 0.6,
            "beta": 0.4,
            "agent_used": "",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        # Testear el routing directamente
        route = orchestrator._route_to_agent(state)
        
        # Con imagen + texto significativo (>20 chars), debe ir a omni_loop
        assert route == "omni_loop", f"Expected omni_loop but got {route}"
    
    @patch('core.model_pool.get_model_pool')
    @patch('core.graph.get_omni_loop')
    @patch('core.graph.get_expert_agent')
    @patch('core.graph.get_tiny_agent')
    @patch('core.graph.get_omni_agent')
    @patch('core.graph.create_rag_node')
    def test_routing_to_vision_with_image_only(
        self,
        mock_rag,
        mock_omni_agent,
        mock_tiny,
        mock_expert,
        mock_omni_loop,
        mock_pool
    ):
        """Test 2: Routing a vision cuando hay solo imagen (sin texto significativo)"""
        
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        # Estado: imagen con texto corto (<20 chars)
        initial_state = {
            "input": "¿Qué es esto?",  # 13 chars
            "input_type": "text",
            "image_path": "/tmp/test_image.jpg",
            "audio_input": None,
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "video_path": None,
            "fps": None,
            "enable_reflection": True,
            "omni_loop_iterations": None,
            "auto_corrected": False,
            "hard": 0.0,
            "soft": 0.0,
            "web_query": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "agent_used": "",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        # Testear solo el routing (sin ejecutar el grafo completo)
        route = orchestrator._route_to_agent(initial_state)
        
        # Con texto corto, debe ir a vision specialist
        assert route == "vision", f"Expected vision but got {route}"
    
    @patch('core.model_pool.get_model_pool')
    @patch('core.graph.get_omni_loop')
    @patch('core.graph.get_expert_agent')
    @patch('core.graph.get_tiny_agent')
    @patch('core.graph.get_omni_agent')
    @patch('core.graph.create_rag_node')
    def test_omni_loop_disable_reflection_for_emotional_queries(
        self,
        mock_rag,
        mock_omni_agent,
        mock_tiny,
        mock_expert,
        mock_omni_loop,
        mock_pool
    ):
        """Test 3: Reflexión deshabilitada para queries emocionales (soft > 0.8)"""
        
        # Mock del omni_loop
        mock_loop_instance = Mock()
        mock_loop_instance.execute_loop.return_value = {
            "response": "Entiendo que estás triste",
            "iterations": [{"iteration": 1, "response": "Entiendo que estás triste", "confidence": 0.9}],
            "total_latency_ms": 500.0,
            "auto_corrected": False,
            "fallback_used": False,
            "metadata": {
                "confidence_final": 0.9,
                "total_tokens": 10,
                "avg_tokens_per_second": 20.0,
                "num_iterations": 1
            }
        }
        mock_omni_loop.return_value = mock_loop_instance
        
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        # Estado: query emocional (soft > 0.8)
        initial_state = {
            "input": "Me siento muy triste hoy",
            "input_type": "text",
            "image_path": "/tmp/sad_face.jpg",  # Imagen emocional
            "audio_input": None,
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "video_path": None,
            "fps": None,
            "enable_reflection": True,
            "omni_loop_iterations": None,
            "auto_corrected": False,
            "hard": 0.1,
            "soft": 0.85,  # Alta carga emocional
            "web_query": 0.0,
            "alpha": 0.2,
            "beta": 0.8,
            "agent_used": "",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        # Ejecutar solo el nodo omni_loop
        result = orchestrator._execute_omni_loop(initial_state)
        
        # Verificar que execute_loop fue llamado con enable_reflection=False
        call_args = mock_loop_instance.execute_loop.call_args
        assert call_args.kwargs["enable_reflection"] is False, "Reflection should be disabled for emotional queries"
    
    @patch('core.model_pool.get_model_pool')
    @patch('core.graph.get_omni_loop')
    @patch('core.graph.get_expert_agent')
    @patch('core.graph.get_tiny_agent')
    @patch('core.graph.get_omni_agent')
    @patch('core.graph.create_rag_node')
    def test_omni_loop_fallback_to_tiny(
        self,
        mock_rag,
        mock_omni_agent,
        mock_tiny_cls,
        mock_expert,
        mock_omni_loop,
        mock_pool
    ):
        """Test 4: Fallback a tiny agent si omni_loop falla"""
        
        # Mock del omni_loop que falla
        mock_loop_instance = Mock()
        mock_loop_instance.execute_loop.side_effect = Exception("Omni-Loop crashed")
        mock_omni_loop.return_value = mock_loop_instance
        
        # Mock del tiny agent (fallback)
        mock_tiny_instance = Mock()
        mock_tiny_instance.generate.return_value = "Respuesta de fallback del tiny agent"
        mock_tiny_cls.return_value = mock_tiny_instance
        
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        orchestrator.tiny_agent = mock_tiny_instance
        
        # Estado con imagen + texto
        initial_state = {
            "input": "Analiza esta imagen técnicamente",
            "input_type": "text",
            "image_path": "/tmp/technical.jpg",
            "audio_input": None,
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "video_path": None,
            "fps": None,
            "enable_reflection": True,
            "omni_loop_iterations": None,
            "auto_corrected": False,
            "hard": 0.6,
            "soft": 0.4,
            "web_query": 0.0,
            "alpha": 0.6,
            "beta": 0.4,
            "agent_used": "",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        # Ejecutar nodo omni_loop
        result = orchestrator._execute_omni_loop(initial_state)
        
        # Debe usar tiny como fallback
        assert result["agent_used"] == "tiny"
        assert "fallback" in result["response"].lower()
        assert result["auto_corrected"] is False
        
        # Verificar que tiny.generate fue llamado
        mock_tiny_instance.generate.assert_called_once()


class TestLangGraphRouting:
    """Tests del routing inteligente"""
    
    def test_routing_priority_rag_over_omni_loop(self):
        """Test 5: RAG tiene prioridad sobre omni_loop"""
        from core.graph import SARAiOrchestrator
        
        with patch('core.model_pool.get_model_pool'), \
             patch('core.graph.get_omni_loop'), \
             patch('core.graph.get_expert_agent'), \
             patch('core.graph.get_tiny_agent'), \
             patch('core.graph.get_omni_agent'), \
             patch('core.graph.create_rag_node'):
            
            orchestrator = SARAiOrchestrator(use_simulated_trm=True)
            
            state = {
                "input": "¿Quién ganó el Oscar 2025?",
                "image_path": "/tmp/oscars.jpg",  # Tiene imagen
                "web_query": 0.8,  # Pero web_query es alta
                "soft": 0.3
            }
            
            route = orchestrator._route_to_agent(state)
            
            # RAG debe tener prioridad
            assert route == "rag"
    
    def test_routing_multimodal_over_vision(self):
        """Test 6: Multimodal (imagen + texto) usa omni_loop, no vision"""
        from core.graph import SARAiOrchestrator
        
        with patch('core.model_pool.get_model_pool'), \
             patch('core.graph.get_omni_loop'), \
             patch('core.graph.get_expert_agent'), \
             patch('core.graph.get_tiny_agent'), \
             patch('core.graph.get_omni_agent'), \
             patch('core.graph.create_rag_node'):
            
            orchestrator = SARAiOrchestrator(use_simulated_trm=True)
            
            # Texto significativo (>20 chars) + imagen
            state = {
                "input": "Explica qué está pasando en esta imagen con mucho detalle",  # 60 chars
                "image_path": "/tmp/complex_scene.jpg",
                "web_query": 0.2,
                "soft": 0.3
            }
            
            route = orchestrator._route_to_agent(state)
            
            # Debe usar omni_loop (multimodal reflexivo)
            assert route == "omni_loop"


# ========================================
# Test Runner
# ========================================

def run_tests():
    """Ejecuta todos los tests"""
    import subprocess
    
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short", "-k", "TestLangGraphOmniLoopIntegration or TestLangGraphRouting"],
        capture_output=False
    )
    
    return result.returncode


if __name__ == "__main__":
    exit_code = run_tests()
    
    if exit_code == 0:
        print("\n✅ TODOS LOS TESTS DE INTEGRACIÓN PASARON")
        print("Omni-Loop integrado correctamente en LangGraph")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")
        print("Revisar output arriba")
    
    exit(exit_code)
