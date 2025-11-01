"""
Tests de integración de layers (v2.13) con el grafo principal

Verifica:
- Layer1: Detección de emoción en nodo classify
- Layer2: Memoria de tono en nodo compute_weights
- Layer3: Tone bridge en nodo enhance_with_emotion
"""

import pytest
import numpy as np
from pathlib import Path
from core.graph import SARAiOrchestrator, State
from core.layer1_io.audio_emotion_lite import detect_emotion
from core.layer2_memory.tone_memory import ToneMemoryBuffer, get_tone_memory_buffer
from core.layer3_fluidity.tone_bridge import ToneStyleBridge, get_tone_bridge


class TestLayer1EmotionDetection:
    """Tests para Layer1: detección de emoción"""
    
    def test_emotion_detection_from_audio(self):
        """Verifica que detect_emotion retorna estructura correcta"""
        # Simular audio (16000 Hz, 3 segundos)
        sample_rate = 16000
        duration = 3
        audio_array = np.random.randn(sample_rate * duration).astype(np.float32)
        
        emotion = detect_emotion(audio_array, sr=sample_rate)
        
        # Verificar campos requeridos
        assert "label" in emotion
        assert emotion["label"] in ["neutral", "happy", "sad", "angry", "fearful"]
        assert "valence" in emotion
        assert "arousal" in emotion
        assert 0.0 <= emotion["valence"] <= 1.0
        assert 0.0 <= emotion["arousal"] <= 1.0
    
    def test_graph_stores_emotion_in_state(self):
        """Verifica que el grafo guarda emoción en state"""
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        # Simular audio input
        audio_array = np.random.randn(16000 * 3).astype(np.float32)
        
        state: State = {
            "input": "Test query",
            "input_type": "audio",
            "audio_input": audio_array.tobytes(),
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "image_path": None,
            "video_path": None,
            "fps": None,
            "emotion": None,
            "tone_style": None,
            "filler_hint": None,
            "enable_reflection": False,
            "omni_loop_iterations": None,
            "auto_corrected": False,
            "hard": 0.0,
            "soft": 0.0,
            "web_query": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "skill_used": None,
            "agent_used": "tiny",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        # Ejecutar nodo classify
        result = orchestrator._classify_intent(state)
        
        # Verificar que emotion fue detectada
        assert state.get("emotion") is not None
        assert "label" in state["emotion"]
        assert "valence" in state["emotion"]


class TestLayer2ToneMemory:
    """Tests para Layer2: memoria de tono"""
    
    def test_tone_memory_append_and_recent(self):
        """Verifica que tone_memory guarda y recupera entradas"""
        tone_memory = ToneMemoryBuffer(
            storage_path=Path("state/test_layer2_tone_memory.jsonl"),
            max_entries=10
        )
        
        # Limpiar buffer
        tone_memory.clear()
        
        # Agregar entradas
        tone_memory.append({
            "label": "happy",
            "valence": 0.8,
            "arousal": 0.6
        })
        
        tone_memory.append({
            "label": "sad",
            "valence": 0.2,
            "arousal": 0.4
        })
        
        # Recuperar recientes
        recent = tone_memory.recent(limit=2)
        
        assert len(recent) == 2
        assert recent[-1]["label"] == "sad"
        assert recent[-2]["label"] == "happy"
        
        # Limpiar
        tone_memory.clear()
    
    def test_tone_memory_persistence(self):
        """Verifica persistencia en disco"""
        storage_path = Path("state/test_layer2_tone_memory_persistence.jsonl")
        
        # Buffer 1: agregar entrada
        tone_memory1 = ToneMemoryBuffer(storage_path=storage_path, max_entries=10)
        tone_memory1.clear()
        tone_memory1.append({
            "label": "happy",
            "valence": 0.8,
            "arousal": 0.6
        })
        
        # Buffer 2: cargar desde disco
        tone_memory2 = ToneMemoryBuffer(storage_path=storage_path, max_entries=10)
        recent = tone_memory2.recent(limit=1)
        
        assert len(recent) == 1
        assert recent[0]["label"] == "happy"
        
        # Limpiar
        tone_memory2.clear()
        if storage_path.exists():
            storage_path.unlink()
    
    def test_mcp_adjusts_beta_on_negative_tone(self):
        """Verifica que MCP aumenta β (empatía) si tono es negativo"""
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        tone_memory = get_tone_memory_buffer()
        tone_memory.clear()
        
        # Agregar historial de tono negativo
        for _ in range(5):
            tone_memory.append({
                "label": "sad",
                "valence": 0.2,  # Muy negativo
                "arousal": 0.3
            })
        
        state: State = {
            "input": "Test query",
            "input_type": "text",
            "audio_input": None,
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "image_path": None,
            "video_path": None,
            "fps": None,
            "emotion": {
                "label": "sad",
                "valence": 0.2,
                "arousal": 0.3
            },
            "tone_style": None,
            "filler_hint": None,
            "enable_reflection": False,
            "omni_loop_iterations": None,
            "auto_corrected": False,
            "hard": 0.5,
            "soft": 0.5,
            "web_query": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "skill_used": None,
            "agent_used": "tiny",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        # Ejecutar nodo compute_weights
        result = orchestrator._compute_weights(state)
        
        # β debe haber aumentado por tono negativo persistente
        assert result["beta"] > 0.5  # Mayor que balance inicial
        
        # Limpiar
        tone_memory.clear()


class TestLayer3ToneBridge:
    """Tests para Layer3: tone bridge"""
    
    def test_tone_bridge_smoothing(self):
        """Verifica smoothing de transiciones"""
        bridge = ToneStyleBridge(smoothing=0.25)
        
        # Primera actualización (neutral)
        profile1 = bridge.update("neutral", valence=0.5, arousal=0.5)
        assert profile1.valence_avg == 0.5
        assert profile1.arousal_avg == 0.5
        
        # Segunda actualización (sad - cambio abrupto)
        profile2 = bridge.update("sad", valence=0.2, arousal=0.3)
        
        # Smoothing debe evitar cambio abrupto
        assert profile2.valence_avg > 0.2  # No salta directamente a 0.2
        assert profile2.valence_avg < 0.5  # Pero se mueve hacia abajo
    
    def test_tone_bridge_style_inference(self):
        """Verifica inferencia de estilos"""
        bridge = ToneStyleBridge(smoothing=0.25)
        bridge.reset()
        
        # Energético y positivo
        profile = bridge.update("happy", valence=0.8, arousal=0.7)
        assert profile.style == "energetic_positive"
        
        # Reset
        bridge.reset()
        
        # Triste y baja energía
        profile = bridge.update("sad", valence=0.2, arousal=0.3)
        assert profile.style == "soft_support"
    
    def test_graph_stores_tone_style_in_state(self):
        """Verifica que el grafo guarda tone_style y filler_hint"""
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        state: State = {
            "input": "Test query",
            "input_type": "text",
            "audio_input": None,
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "image_path": None,
            "video_path": None,
            "fps": None,
            "emotion": {
                "label": "happy",
                "valence": 0.8,
                "arousal": 0.6
            },
            "tone_style": None,
            "filler_hint": None,
            "enable_reflection": False,
            "omni_loop_iterations": None,
            "auto_corrected": False,
            "hard": 0.5,
            "soft": 0.5,
            "web_query": 0.0,
            "alpha": 0.5,
            "beta": 0.5,
            "skill_used": None,
            "agent_used": "tiny",
            "response": "Test response",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        # Ejecutar nodo enhance_with_emotion
        result = orchestrator._enhance_with_emotion(state)
        
        # Verificar que tone_style y filler_hint fueron asignados
        assert state.get("tone_style") is not None
        assert state.get("filler_hint") is not None


class TestLayerIntegrationEndToEnd:
    """Tests end-to-end de integración de layers"""
    
    def test_full_pipeline_with_audio_input(self):
        """Verifica pipeline completo: audio → Layer1 → Layer2 → Layer3"""
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        tone_memory = get_tone_memory_buffer()
        tone_memory.clear()
        
        # Simular audio
        audio_array = np.random.randn(16000 * 3).astype(np.float32)
        
        # Estado inicial
        state: State = {
            "input": "Estoy muy triste hoy",
            "input_type": "audio",
            "audio_input": audio_array.tobytes(),
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "image_path": None,
            "video_path": None,
            "fps": None,
            "emotion": None,
            "tone_style": None,
            "filler_hint": None,
            "enable_reflection": False,
            "omni_loop_iterations": None,
            "auto_corrected": False,
            "hard": 0.0,
            "soft": 0.0,
            "web_query": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "skill_used": None,
            "agent_used": "tiny",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        # 1. Classify (Layer1: emotion detection)
        orchestrator._classify_intent(state)
        assert state.get("emotion") is not None
        
        # 2. Compute weights (Layer2: tone memory)
        orchestrator._compute_weights(state)
        assert len(tone_memory.recent(limit=1)) > 0  # Guardado en memoria
        
        # 3. Generate (simular respuesta)
        state["response"] = "Lo siento mucho. ¿En qué puedo ayudarte?"
        
        # 4. Enhance (Layer3: tone bridge)
        orchestrator._enhance_with_emotion(state)
        assert state.get("tone_style") is not None
        assert state.get("filler_hint") is not None
        
        # Limpiar
        tone_memory.clear()
    
    def test_text_input_skips_layer1(self):
        """Verifica que input de texto omite Layer1 (emotion detection)"""
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        state: State = {
            "input": "¿Qué es Python?",
            "input_type": "text",  # Texto, no audio
            "audio_input": None,
            "audio_output": None,
            "detected_emotion": None,
            "detected_lang": None,
            "voice_metadata": {},
            "image_path": None,
            "video_path": None,
            "fps": None,
            "emotion": None,
            "tone_style": None,
            "filler_hint": None,
            "enable_reflection": False,
            "omni_loop_iterations": None,
            "auto_corrected": False,
            "hard": 0.0,
            "soft": 0.0,
            "web_query": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "skill_used": None,
            "agent_used": "expert",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        # Ejecutar classify
        orchestrator._classify_intent(state)
        
        # emotion NO debe estar presente
        assert state.get("emotion") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
