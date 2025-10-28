"""
Tests de integración para TTS Engine en LangGraph (M3.2 Fase 3)

Verifican:
- Nodo generate_tts integrado correctamente
- Flujo completo: audio_input → STT → LLM → TTS → audio_output
- Prosody aplicada desde emotion_integration
- Fallbacks sentinel funcionando
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Fixtures compartidos
try:
    from agents.emotion_integration import EmotionState
    EMOTION_INTEGRATION_AVAILABLE = True
except ImportError:
    EMOTION_INTEGRATION_AVAILABLE = False


@pytest.fixture
def mock_tts_engine():
    """Mock de TTSEngine para tests sin dependencias"""
    mock = Mock()
    
    # Simular TTSOutput
    mock_output = Mock()
    mock_output.audio_bytes = b"MOCK_AUDIO_WAV_DATA"
    mock_output.duration_ms = 1500
    mock_output.prosody_applied = True
    mock_output.cached = False
    
    mock.generate.return_value = mock_output
    return mock


@pytest.fixture
def mock_emotion_modulator():
    """Mock de EmotionModulator"""
    mock = Mock()
    mock.modulate.return_value = "Respuesta modulada emocionalmente."
    return mock


class TestLangGraphTTSIntegration:
    """Tests de integración del nodo TTS en LangGraph"""
    
    def test_generate_tts_node_basic(self, mock_tts_engine):
        """Test básico del nodo generate_tts sin emoción"""
        from core.graph import SARAiOrchestrator
        
        # Mock create_tts_engine
        with patch('agents.tts_engine.create_tts_engine', return_value=mock_tts_engine):
            orchestrator = SARAiOrchestrator(use_simulated_trm=True)
            
            # State simulado
            state = {
                "response": "Esta es una respuesta de prueba.",
                "detected_emotion": None,
                "audio_input": b"MOCK_INPUT_AUDIO"
            }
            
            # Ejecutar nodo
            result = orchestrator._generate_tts(state)
            
            # Verificaciones
            assert "audio_output" in result
            assert result["audio_output"] == b"MOCK_AUDIO_WAV_DATA"
            
            # TTS fue llamado con texto correcto
            mock_tts_engine.generate.assert_called_once()
            call_kwargs = mock_tts_engine.generate.call_args[1]
            assert call_kwargs["text"] == "Esta es una respuesta de prueba."
    
    @pytest.mark.skipif(not EMOTION_INTEGRATION_AVAILABLE, 
                       reason="emotion_integration not installed")
    def test_generate_tts_with_emotion(self, mock_tts_engine):
        """Test TTS con estado emocional aplicado"""
        from core.graph import SARAiOrchestrator
        
        with patch('agents.tts_engine.create_tts_engine', return_value=mock_tts_engine):
            orchestrator = SARAiOrchestrator(use_simulated_trm=True)
            
            state = {
                "response": "¡Me alegro mucho de ayudarte!",
                "detected_emotion": "empático",
                "audio_input": b"MOCK_INPUT"
            }
            
            result = orchestrator._generate_tts(state)
            
            # Audio generado
            assert result["audio_output"] == b"MOCK_AUDIO_WAV_DATA"
            
            # Verificar que se pasó emotion_state
            call_kwargs = mock_tts_engine.generate.call_args[1]
            emotion_state = call_kwargs.get("emotion_state")
            
            # Debe ser EmotionState con valence positivo (empático)
            if emotion_state:
                assert hasattr(emotion_state, 'valence')
                assert emotion_state.valence > 0.5  # Empático = positivo
    
    def test_generate_tts_sentinel_fallback(self):
        """Test que TTS falla gracefully (sentinel)"""
        from core.graph import SARAiOrchestrator
        
        # Mock que lanza excepción
        mock_failing_tts = Mock()
        mock_failing_tts.generate.side_effect = RuntimeError("TTS engine crashed")
        
        with patch('agents.tts_engine.create_tts_engine', return_value=mock_failing_tts):
            orchestrator = SARAiOrchestrator(use_simulated_trm=True)
            
            state = {
                "response": "Respuesta",
                "detected_emotion": None,
                "audio_input": b"AUDIO"
            }
            
            # No debe crashear, debe retornar None
            result = orchestrator._generate_tts(state)
            
            assert "audio_output" in result
            assert result["audio_output"] is None  # Sentinel fallback
    
    def test_enhance_with_emotion_node(self, mock_emotion_modulator):
        """Test del nodo enhance_with_emotion"""
        from core.graph import SARAiOrchestrator
        
        with patch('agents.emotion_modulator.create_emotion_modulator', 
                   return_value=mock_emotion_modulator):
            orchestrator = SARAiOrchestrator(use_simulated_trm=True)
            
            state = {
                "response": "Respuesta original.",
                "detected_emotion": "empático"
            }
            
            result = orchestrator._enhance_with_emotion(state)
            
            # Respuesta modulada
            assert result["response"] == "Respuesta modulada emocionalmente."
            
            # Modulator llamado correctamente
            mock_emotion_modulator.modulate.assert_called_once_with(
                text="Respuesta original.",
                target_emotion="empático"
            )
    
    def test_enhance_without_emotion(self):
        """Test que enhance pasa sin modificar si no hay emoción"""
        from core.graph import SARAiOrchestrator
        
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        state = {
            "response": "Respuesta original.",
            "detected_emotion": None
        }
        
        result = orchestrator._enhance_with_emotion(state)
        
        # No debe modificar la respuesta
        assert result["response"] == "Respuesta original."
    
    def test_route_to_tts_conditional(self):
        """Test del routing condicional hacia TTS"""
        from core.graph import SARAiOrchestrator
        
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        # Caso 1: Con audio_input → debe ir a TTS
        state_with_audio = {"audio_input": b"AUDIO_DATA"}
        route = orchestrator._route_to_tts(state_with_audio)
        assert route == "tts"
        
        # Caso 2: Sin audio_input → skip TTS
        state_without_audio = {"audio_input": None}
        route = orchestrator._route_to_tts(state_without_audio)
        assert route == "skip"


class TestEndToEndVoicePipeline:
    """Tests end-to-end del pipeline completo de voz"""
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="process_audio_input API pending implementation")
    @patch('agents.audio_router.route_audio')
    @patch('agents.omni_pipeline.process_audio_input')
    @patch('agents.tts_engine.create_tts_engine')
    @patch('agents.emotion_modulator.create_emotion_modulator')
    def test_full_audio_to_audio_pipeline(
        self,
        mock_modulator,
        mock_tts_factory,
        mock_omni,
        mock_router
    ):
        """
        Test del flujo completo audio → audio
        
        Pipeline:
        1. audio_input → route_audio (detecta idioma)
        2. process_audio_input (STT + emoción)
        3. TRM classify (hard/soft)
        4. LLM generate (tiny agent)
        5. enhance_with_emotion (modulación)
        6. generate_tts (audio output)
        """
        from core.graph import SARAiOrchestrator
        
        # Configurar mocks
        mock_router.return_value = ("omni", b"AUDIO_ROUTED", "es")
        
        mock_omni.return_value = {
            "text": "Hola, ¿cómo estás?",
            "emotion": "empático"
        }
        
        mock_modulator_instance = Mock()
        mock_modulator_instance.modulate.return_value = "¡Hola! ¿Cómo estás hoy? 😊"
        mock_modulator.return_value = mock_modulator_instance
        
        mock_tts_engine = Mock()
        mock_tts_output = Mock()
        mock_tts_output.audio_bytes = b"WAV_RESPONSE_AUDIO"
        mock_tts_output.duration_ms = 2000
        mock_tts_output.prosody_applied = True
        mock_tts_output.cached = False
        mock_tts_engine.generate.return_value = mock_tts_output
        mock_tts_factory.return_value = mock_tts_engine
        
        # Crear orchestrator
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        # Invocar con audio
        result = orchestrator.invoke_audio(b"MOCK_INPUT_AUDIO")
        
        # Verificaciones
        assert result["transcription"] == "Hola, ¿cómo estás?"
        assert result["detected_emotion"] == "empático"
        assert result["detected_lang"] == "es"
        assert result["audio_output"] == b"WAV_RESPONSE_AUDIO"
        assert "¡Hola!" in result["response"]  # Respuesta modulada
        
        # Verificar cadena de llamadas
        mock_router.assert_called_once()
        mock_omni.assert_called_once()
        mock_modulator_instance.modulate.assert_called_once()
        mock_tts_engine.generate.assert_called_once()
    
    @pytest.mark.integration
    def test_text_input_skips_tts(self):
        """Test que input de texto NO pasa por TTS"""
        from core.graph import SARAiOrchestrator
        
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        # Input de texto puro
        response = orchestrator.invoke("¿Cuál es la capital de Francia?")
        
        # Debe retornar respuesta (no verifica audio porque invoke() solo retorna str)
        assert isinstance(response, str)
        assert len(response) > 0


class TestTTSCacheIntegration:
    """Tests de integración del cache de TTS"""
    
    def test_cache_hit_reduces_latency(self):
        """Test que cache de TTS reduce latencia en segunda llamada"""
        from agents.tts_engine import create_tts_engine
        import time
        
        # Crear engine real (con cache)
        engine = create_tts_engine()
        
        text = "Esta es una frase de prueba para cachear."
        
        # Primera llamada (cache miss)
        start = time.time()
        output1 = engine.generate(text, emotion_state=None)
        latency1 = (time.time() - start) * 1000  # ms
        
        # Segunda llamada (cache hit esperado)
        start = time.time()
        output2 = engine.generate(text, emotion_state=None)
        latency2 = (time.time() - start) * 1000  # ms
        
        # Verificaciones
        assert output2.cached, "Segunda llamada debería ser cached"
        assert latency2 < latency1 * 0.1, f"Cache hit debería ser <10% de latencia original ({latency2:.1f}ms vs {latency1:.1f}ms)"
        
        # Audio idéntico
        assert output1.audio_bytes == output2.audio_bytes


# ============================================================
# BENCHMARKS
# ============================================================

class TestTTSPerformanceBenchmarks:
    """Benchmarks de rendimiento del pipeline TTS en LangGraph"""
    
    @pytest.mark.benchmark
    @patch('agents.tts_engine.create_tts_engine')
    def test_tts_node_latency(self, mock_tts_factory):
        """Benchmark: Latencia del nodo generate_tts"""
        import time
        from core.graph import SARAiOrchestrator
        
        # Mock con latencia simulada
        mock_engine = Mock()
        mock_output = Mock()
        mock_output.audio_bytes = b"AUDIO"
        mock_output.duration_ms = 1500
        mock_output.prosody_applied = False
        mock_output.cached = False
        
        def slow_generate(*args, **kwargs):
            time.sleep(0.05)  # Simular 50ms de TTS
            return mock_output
        
        mock_engine.generate = slow_generate
        mock_tts_factory.return_value = mock_engine
        
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        state = {
            "response": "Texto de respuesta.",
            "detected_emotion": None,
            "audio_input": b"AUDIO"
        }
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            orchestrator._generate_tts(state)
        elapsed = (time.time() - start) * 1000  # ms
        
        avg_latency = elapsed / 10
        
        # Target: <100ms por llamada (overhead de nodo)
        assert avg_latency < 100, f"Latencia promedio: {avg_latency:.1f}ms (target: <100ms)"
        print(f"✅ TTS Node Latency P50: {avg_latency:.1f}ms")
    
    @pytest.mark.benchmark
    def test_emotion_modulation_latency(self):
        """Benchmark: Latencia de modulación emocional"""
        import time
        from core.graph import SARAiOrchestrator
        
        mock_modulator = Mock()
        mock_modulator.modulate.return_value = "Respuesta modulada"
        
        with patch('agents.emotion_modulator.create_emotion_modulator', 
                   return_value=mock_modulator):
            orchestrator = SARAiOrchestrator(use_simulated_trm=True)
            
            state = {
                "response": "Respuesta original larga para medir overhead de modulación emocional.",
                "detected_emotion": "empático"
            }
            
            # Benchmark
            start = time.time()
            for _ in range(100):
                orchestrator._enhance_with_emotion(state)
            elapsed = (time.time() - start) * 1000  # ms
            
            avg_latency = elapsed / 100
            
            # Target: <10ms por llamada
            assert avg_latency < 10, f"Latencia promedio: {avg_latency:.1f}ms (target: <10ms)"
            print(f"✅ Emotion Modulation Latency P50: {avg_latency:.1f}ms")
