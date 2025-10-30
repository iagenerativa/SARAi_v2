#!/usr/bin/env python3
"""
Test rápido de Best-of-Breed routing
Valida que imagen/video rutean correctamente a Qwen3-VL-4B
"""

import sys
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import SARAiOrchestrator, State


def test_image_routing():
    """Test 1: Imagen debe rutear a vision agent"""
    print("\n" + "="*60)
    print("TEST 1: Image Routing")
    print("="*60)
    
    orchestrator = SARAiOrchestrator(use_simulated_trm=True)
    
    state: State = {
        "input": "Describe los objetos en esta imagen",
        "image_path": "/home/noel/vision_test_images/pelicula.jpg",
        # Campos requeridos por TypedDict
        "input_type": "image",
        "audio_input": None,
        "audio_output": None,
        "detected_emotion": None,
        "detected_lang": None,
        "voice_metadata": {},
        "video_path": None,
        "fps": None,
        "hard": 0.0,
        "soft": 0.0,
        "web_query": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "agent_used": "vision",
        "response": "",
        "feedback": 0.0,
        "rag_metadata": {}
    }
    
    # Verificar routing inicial
    detected_type = orchestrator._detect_input_type(state)
    print(f"✓ Input type detectado: {detected_type}")
    
    assert detected_type["input_type"] == "image", "❌ FALLO: Debería detectar imagen"
    
    # Verificar routing condicional
    route = orchestrator._route_by_input_type(state)
    print(f"✓ Ruta seleccionada: {route}")
    
    assert route == "vision", "❌ FALLO: Debería rutear a vision"
    
    print("✅ TEST 1 PASSED: Imagen rutea a Qwen3-VL-4B")


def test_video_routing():
    """Test 2: Video debe rutear a vision agent"""
    print("\n" + "="*60)
    print("TEST 2: Video Routing")
    print("="*60)
    
    orchestrator = SARAiOrchestrator(use_simulated_trm=True)
    
    state: State = {
        "input": "¿Qué acciones ocurren en este video?",
        "video_path": "/path/to/video.mp4",
        # Campos requeridos
        "input_type": "video",
        "audio_input": None,
        "audio_output": None,
        "detected_emotion": None,
        "detected_lang": None,
        "voice_metadata": {},
        "image_path": None,
        "fps": 2.0,
        "hard": 0.0,
        "soft": 0.0,
        "web_query": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "agent_used": "vision",
        "response": "",
        "feedback": 0.0,
        "rag_metadata": {}
    }
    
    detected_type = orchestrator._detect_input_type(state)
    print(f"✓ Input type detectado: {detected_type}")
    
    assert detected_type["input_type"] == "video", "❌ FALLO: Debería detectar video"
    
    route = orchestrator._route_by_input_type(state)
    print(f"✓ Ruta seleccionada: {route}")
    
    assert route == "vision", "❌ FALLO: Debería rutear a vision"
    
    print("✅ TEST 2 PASSED: Video rutea a Qwen3-VL-4B")


def test_audio_routing():
    """Test 3: Audio debe rutear a process_voice"""
    print("\n" + "="*60)
    print("TEST 3: Audio Routing")
    print("="*60)
    
    orchestrator = SARAiOrchestrator(use_simulated_trm=True)
    
    state: State = {
        "input": "",  # Aún no transcrito
        "audio_input": b"fake_audio_bytes",
        # Campos requeridos
        "input_type": "audio",
        "audio_output": None,
        "detected_emotion": None,
        "detected_lang": None,
        "voice_metadata": {},
        "image_path": None,
        "video_path": None,
        "fps": None,
        "hard": 0.0,
        "soft": 0.0,
        "web_query": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "agent_used": "omni",
        "response": "",
        "feedback": 0.0,
        "rag_metadata": {}
    }
    
    detected_type = orchestrator._detect_input_type(state)
    print(f"✓ Input type detectado: {detected_type}")
    
    assert detected_type["input_type"] == "audio", "❌ FALLO: Debería detectar audio"
    
    route = orchestrator._route_by_input_type(state)
    print(f"✓ Ruta seleccionada: {route}")
    
    assert route == "audio", "❌ FALLO: Debería rutear a process_voice"
    
    print("✅ TEST 3 PASSED: Audio rutea a Omni-3B pipeline")


def test_text_routing():
    """Test 4: Texto debe rutear a classify"""
    print("\n" + "="*60)
    print("TEST 4: Text Routing")
    print("="*60)
    
    orchestrator = SARAiOrchestrator(use_simulated_trm=True)
    
    state: State = {
        "input": "¿Cómo funciona un motor de combustión?",
        # Campos requeridos
        "input_type": "text",
        "audio_input": None,
        "audio_output": None,
        "detected_emotion": None,
        "detected_lang": None,
        "voice_metadata": {},
        "image_path": None,
        "video_path": None,
        "fps": None,
        "hard": 0.0,
        "soft": 0.0,
        "web_query": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "agent_used": "expert",
        "response": "",
        "feedback": 0.0,
        "rag_metadata": {}
    }
    
    detected_type = orchestrator._detect_input_type(state)
    print(f"✓ Input type detectado: {detected_type}")
    
    assert detected_type["input_type"] == "text", "❌ FALLO: Debería detectar texto"
    
    route = orchestrator._route_by_input_type(state)
    print(f"✓ Ruta seleccionada: {route}")
    
    assert route == "text", "❌ FALLO: Debería rutear a classify"
    
    print("✅ TEST 4 PASSED: Texto rutea a TRM-Router")


def test_priority_order():
    """Test 5: Prioridad de routing (vision > audio > text)"""
    print("\n" + "="*60)
    print("TEST 5: Priority Order")
    print("="*60)
    
    orchestrator = SARAiOrchestrator(use_simulated_trm=True)
    
    # Caso 1: image_path + audio_input → Debe priorizar imagen
    state_mixed: State = {
        "input": "Test prioridad",
        "image_path": "/path/to/image.jpg",
        "audio_input": b"audio_bytes",
        # Campos requeridos
        "input_type": "image",
        "audio_output": None,
        "detected_emotion": None,
        "detected_lang": None,
        "voice_metadata": {},
        "video_path": None,
        "fps": None,
        "hard": 0.0,
        "soft": 0.0,
        "web_query": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "agent_used": "vision",
        "response": "",
        "feedback": 0.0,
        "rag_metadata": {}
    }
    
    detected = orchestrator._detect_input_type(state_mixed)
    print(f"✓ Con imagen + audio: {detected['input_type']}")
    
    assert detected["input_type"] == "image", "❌ FALLO: Debe priorizar imagen sobre audio"
    
    # Caso 2: video_path + image_path → Debe priorizar video
    state_visual: State = {
        "input": "Test visual priority",
        "video_path": "/path/to/video.mp4",
        "image_path": "/path/to/image.jpg",
        # Campos requeridos
        "input_type": "video",
        "audio_input": None,
        "audio_output": None,
        "detected_emotion": None,
        "detected_lang": None,
        "voice_metadata": {},
        "fps": None,
        "hard": 0.0,
        "soft": 0.0,
        "web_query": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "agent_used": "vision",
        "response": "",
        "feedback": 0.0,
        "rag_metadata": {}
    }
    
    detected = orchestrator._detect_input_type(state_visual)
    print(f"✓ Con video + imagen: {detected['input_type']}")
    
    assert detected["input_type"] == "video", "❌ FALLO: Debe priorizar video sobre imagen"
    
    print("✅ TEST 5 PASSED: Prioridades correctas (vision > audio > text)")


if __name__ == "__main__":
    print("\n🧪 BEST-OF-BREED ROUTING TESTS")
    print("Validando arquitectura Omni-3B + Qwen3-VL-4B\n")
    
    try:
        test_image_routing()
        test_video_routing()
        test_audio_routing()
        test_text_routing()
        test_priority_order()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS TESTS PASARON")
        print("="*60)
        print("\n✓ Imagen → Qwen3-VL-4B (vision agent)")
        print("✓ Video → Qwen3-VL-4B (vision agent)")
        print("✓ Audio → Omni-3B (process_voice)")
        print("✓ Texto → TRM-Router (classify)")
        print("✓ Prioridades: vision > audio > text")
        print("\n🎉 Best-of-Breed routing VALIDADO\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n💥 ERROR INESPERADO: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
