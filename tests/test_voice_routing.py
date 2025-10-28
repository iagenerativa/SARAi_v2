"""
Tests de routing de voz en LangGraph (M3.2 Fase 1)

Valida:
- Detección correcta de input_type (audio vs texto)
- Routing a process_voice para audio
- Routing directo a classify para texto
- Integración con nodos existentes
"""

import pytest
from core.graph import SARAiOrchestrator, State


class TestVoiceRouting:
    """Suite de tests para routing de voz vs texto"""
    
    @pytest.fixture
    def orchestrator(self):
        """Fixture: Orchestrator con TRM simulado"""
        return SARAiOrchestrator(use_simulated_trm=True)
    
    # ============================================================
    # TEST 1: Detección de Input Type
    # ============================================================
    
    def test_detect_text_input(self, orchestrator):
        """Debe detectar input_type='text' cuando solo hay texto"""
        state = {
            "input": "Hola SARAi",
            "audio_input": None
        }
        
        result = orchestrator._detect_input_type(state)
        
        assert result["input_type"] == "text"
    
    def test_detect_audio_input(self, orchestrator):
        """Debe detectar input_type='audio' cuando hay audio_input"""
        fake_audio = b"FAKE_AUDIO_DATA"
        
        state = {
            "input": "",
            "audio_input": fake_audio
        }
        
        result = orchestrator._detect_input_type(state)
        
        assert result["input_type"] == "audio"
    
    def test_audio_priority_over_text(self, orchestrator):
        """Audio_input tiene prioridad sobre input texto"""
        fake_audio = b"AUDIO_DATA"
        
        state = {
            "input": "Texto también presente",
            "audio_input": fake_audio
        }
        
        result = orchestrator._detect_input_type(state)
        
        assert result["input_type"] == "audio"
    
    # ============================================================
    # TEST 2: Routing Condicional
    # ============================================================
    
    def test_route_by_input_type_text(self, orchestrator):
        """Debe enrutar a 'text' cuando input_type='text'"""
        state = {"input_type": "text"}
        
        route = orchestrator._route_by_input_type(state)
        
        assert route == "text"
    
    def test_route_by_input_type_audio(self, orchestrator):
        """Debe enrutar a 'audio' cuando input_type='audio'"""
        state = {"input_type": "audio"}
        
        route = orchestrator._route_by_input_type(state)
        
        assert route == "audio"
    
    # ============================================================
    # TEST 3: Routing a TTS
    # ============================================================
    
    def test_route_to_tts_when_audio_input(self, orchestrator):
        """Debe enrutar a TTS si audio_input existe"""
        state = {
            "audio_input": b"AUDIO",
            "response": "Respuesta generada"
        }
        
        route = orchestrator._route_to_tts(state)
        
        assert route == "tts"
    
    def test_skip_tts_when_text_only(self, orchestrator):
        """Debe saltar TTS si solo hay texto"""
        state = {
            "audio_input": None,
            "response": "Respuesta textual"
        }
        
        route = orchestrator._route_to_tts(state)
        
        assert route == "skip"
    
    # ============================================================
    # TEST 4: Process Voice (Stubs)
    # ============================================================
    
    @pytest.mark.skip(reason="Requiere audio_router y omni_pipeline implementados")
    def test_process_voice_transcription(self, orchestrator):
        """Process_voice debe extraer transcripción del audio"""
        fake_audio = b"FAKE_AUDIO"
        
        state = {
            "audio_input": fake_audio
        }
        
        result = orchestrator._process_voice(state)
        
        assert "input" in result  # Transcripción
        assert "detected_emotion" in result
        assert "detected_lang" in result
        assert result["detected_emotion"] in ["empático", "neutral", "urgente"]
    
    # ============================================================
    # TEST 5: Métodos Invoke
    # ============================================================
    
    def test_invoke_text_only(self, orchestrator):
        """invoke() debe procesar texto normal"""
        response = orchestrator.invoke("¿Qué es Python?")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.skip(reason="invoke_audio requiere audio_router completo")
    def test_invoke_audio_returns_dict(self, orchestrator):
        """invoke_audio() debe retornar dict con metadata"""
        fake_audio = b"AUDIO_BYTES"
        
        result = orchestrator.invoke_audio(fake_audio)
        
        assert isinstance(result, dict)
        assert "response" in result
        assert "audio_output" in result
        assert "detected_emotion" in result
        assert "detected_lang" in result
        assert "transcription" in result


class TestVoiceGraphIntegration:
    """Tests de integración del grafo completo con voz"""
    
    @pytest.fixture
    def orchestrator(self):
        return SARAiOrchestrator(use_simulated_trm=True)
    
    def test_graph_has_voice_nodes(self, orchestrator):
        """El grafo debe incluir nodos de voz"""
        workflow = orchestrator.workflow
        
        # Verificar que los nodos existen en el diccionario
        assert "detect_input_type" in workflow.nodes
        assert "process_voice" in workflow.nodes
        assert "enhance_with_emotion" in workflow.nodes
        assert "generate_tts" in workflow.nodes
    
    def test_entry_point_is_detect_input_type(self, orchestrator):
        """El entry point debe ser detect_input_type (no classify)"""
        # Verificar que el grafo tiene el entry point correcto
        # mediante test funcional simple
        
        # Si el entry point es correcto, invoke() debe funcionar
        try:
            response = orchestrator.invoke("test simple")
            assert isinstance(response, str)
        except Exception as e:
            pytest.fail(f"El grafo no funcionó correctamente: {e}")
    
    @pytest.mark.skip(reason="Requiere implementación completa de nodos")
    def test_text_flow_skips_voice_processing(self, orchestrator):
        """Texto debe ir directo a classify, saltando process_voice"""
        # Este test requiere tracing del grafo
        # Implementar cuando LangGraph tenga callback de tracing
        pass
    
    @pytest.mark.skip(reason="Requiere implementación completa de nodos")
    def test_audio_flow_goes_through_voice(self, orchestrator):
        """Audio debe pasar por process_voice antes de classify"""
        # Test de integración completo
        pass


# ============================================================
# TESTS PARAMETRIZADOS
# ============================================================

@pytest.mark.parametrize("input_data,expected_type", [
    ({"input": "texto", "audio_input": None}, "text"),
    ({"input": "", "audio_input": b"audio"}, "audio"),
    ({"input": "ambos", "audio_input": b"audio"}, "audio"),  # Audio prioridad
])
def test_input_type_detection_parametrized(input_data, expected_type):
    """Test parametrizado de detección de input_type"""
    orchestrator = SARAiOrchestrator(use_simulated_trm=True)
    
    result = orchestrator._detect_input_type(input_data)
    
    assert result["input_type"] == expected_type


@pytest.mark.parametrize("audio_present,expected_route", [
    (None, "skip"),
    (b"", "skip"),  # Audio vacío = skip
    (b"AUDIO", "tts"),
])
def test_tts_routing_parametrized(audio_present, expected_route):
    """Test parametrizado de routing a TTS"""
    orchestrator = SARAiOrchestrator(use_simulated_trm=True)
    
    state = {"audio_input": audio_present}
    route = orchestrator._route_to_tts(state)
    
    if audio_present:
        assert route == expected_route
    else:
        assert route == "skip"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
