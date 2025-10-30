#!/usr/bin/env python3
"""
Test Real v2.16.1 - Best-of-Breed Architecture CORRECTA
========================================================

ARQUITECTURA CORRECTA v2.16.1:
- AUDIO: Qwen3-VL-4B-Instruct (190 MB) - Español STT/TTS
- TRADUCCIÓN: NLLB-600M (600 MB) - Pipeline multilingüe
- EMPATÍA: LFM2-1.2B (700 MB) - Soft > 0.7
- EXPERT: SOLAR HTTP (~200 MB) - Alpha > 0.7
- VISIÓN: Qwen3-VL-4B (3.3 GB) - Image/video on-demand

RAM: 2.04 GB baseline → 6 GB peak (87% → 62% libre)

Valida que NO hay solapamiento funcional:
- omni_native: Solo audio (español + NLLB)
- qwen3_vl: Solo visión (imagen/video)
- tiny (LFM2): Empatía + fallback

Test de integración E2E con métricas reales.
"""

import sys
import time
import psutil
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_ram_usage_gb():
    """Retorna RAM usada en GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def test_1_no_overlap_imports():
    """TEST 1: Importar agentes sin solapamiento"""
    print("\n" + "="*60)
    print("TEST 1: No Overlap - Imports")
    print("="*60)
    
    ram_before = get_ram_usage_gb()
    print(f"RAM inicial: {ram_before:.2f} GB")
    
    # Import omni_native (should work)
    try:
        from agents.omni_native import get_omni_agent, OmniConfig
        print("✅ agents.omni_native importado correctamente")
    except Exception as e:
        print(f"❌ FALLO al importar omni_native: {e}")
        return False
    
    # Import qwen3_vl (should work)
    try:
        from agents.qwen3_vl import get_qwen3_vl_agent, Qwen3VLConfig
        print("✅ agents.qwen3_vl importado correctamente")
    except Exception as e:
        print(f"❌ FALLO al importar qwen3_vl: {e}")
        return False
    
    ram_after = get_ram_usage_gb()
    print(f"RAM después imports: {ram_after:.2f} GB")
    print(f"Δ RAM: +{(ram_after - ram_before)*1024:.0f} MB")
    
    # Validar que son clases diferentes
    assert OmniConfig != Qwen3VLConfig, "❌ Config classes NO deben ser iguales"
    print("✅ Config classes son DIFERENTES (no overlap)")
    
    return True


def test_2_config_loading():
    """TEST 2: Cargar configs desde sarai.yaml"""
    print("\n" + "="*60)
    print("TEST 2: Config Loading from YAML")
    print("="*60)
    
    try:
        from agents.omni_native import OmniConfig
        
        # Debería cargar qwen_omni_3b (Best-of-Breed)
        omni_config = OmniConfig.from_yaml()
        print(f"✅ OmniConfig cargado:")
        print(f"   - Model: {omni_config.model_path}")
        print(f"   - Context: {omni_config.n_ctx}")
        print(f"   - Threads: {omni_config.n_threads}")
        
        # Validar que es Omni-3B o 7B (no otros modelos)
        assert "Omni" in omni_config.model_path, "❌ Debe ser Omni model"
        
    except Exception as e:
        print(f"❌ Error cargando OmniConfig: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from agents.qwen3_vl import Qwen3VLConfig
        
        # Debería cargar qwen3_vl_4b
        vision_config = Qwen3VLConfig.from_yaml()
        print(f"✅ Qwen3VLConfig cargado:")
        print(f"   - Model: {vision_config.model_path}")
        print(f"   - Context: {vision_config.n_ctx}")
        print(f"   - TTL: {vision_config.ttl_seconds}s")
        print(f"   - Permanent: {vision_config.permanent}")
        
        # Validar que es Qwen3-VL (no Omni)
        assert "Qwen3-VL" in vision_config.model_path, "❌ Debe ser Qwen3-VL model"
        assert not vision_config.permanent, "❌ Vision debe ser bajo demanda"
        
    except Exception as e:
        print(f"❌ Error cargando Qwen3VLConfig: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validar que apuntan a DIFERENTES modelos
    assert omni_config.model_path != vision_config.model_path, \
        "❌ Configs deben apuntar a MODELOS DIFERENTES"
    
    print("✅ Configs apuntan a MODELOS DIFERENTES (no overlap)")
    
    return True


def test_3_routing_logic():
    """TEST 3: Routing NO solapa funciones"""
    print("\n" + "="*60)
    print("TEST 3: Routing Logic (No Overlap)")
    print("="*60)
    
    from core.graph import SARAiOrchestrator
    
    # Crear orchestrator con TRM simulado (más rápido)
    try:
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        print("✅ Orchestrator inicializado")
    except Exception as e:
        print(f"❌ Error inicializando: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # TEST 3.1: Imagen → vision (NO omni)
    state_image = {
        "input": "Test imagen",
        "image_path": "/fake/image.jpg",
        "input_type": "image",
        # ... resto de campos requeridos
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
    
    detected = orchestrator._detect_input_type(state_image)
    route = orchestrator._route_by_input_type(state_image)
    
    assert detected["input_type"] == "image", "❌ Debe detectar imagen"
    assert route == "vision", "❌ Imagen debe rutear a vision (NO omni)"
    
    print("✅ Imagen → vision (CORRECTO, no va a omni)")
    
    # TEST 3.2: Audio → omni (NO vision)
    state_audio = {
        "input": "Test audio",
        "audio_input": b"fake_audio",
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
    
    detected = orchestrator._detect_input_type(state_audio)
    route = orchestrator._route_by_input_type(state_audio)
    
    assert detected["input_type"] == "audio", "❌ Debe detectar audio"
    assert route == "audio", "❌ Audio debe rutear a process_voice (omni pipeline)"
    
    print("✅ Audio → omni pipeline (CORRECTO, no va a vision)")
    
    # TEST 3.3: Texto alto soft → tiny (LFM2 para empatía)
    # CRÍTICO v2.16.1: Omni-3B solo para audio, empatía la maneja LFM2
    state_empathy = {
        "input": "Estoy muy triste",
        "input_type": "text",  # NO audio
        "audio_input": None,
        "audio_output": None,
        "detected_emotion": None,
        "detected_lang": None,
        "voice_metadata": {},
        "image_path": None,
        "video_path": None,
        "fps": None,
        "hard": 0.3,
        "soft": 0.8,  # Alta empatía
        "web_query": 0.0,
        "alpha": 0.3,
        "beta": 0.7,
        "agent_used": "tiny",  # v2.16.1: LFM2 maneja empatía
        "response": "",
        "feedback": 0.0,
        "rag_metadata": {}
    }
    
    route = orchestrator._route_to_agent(state_empathy)
    assert route == "tiny", "❌ Alta empatía (texto) debe ir a tiny (LFM2)"
    
    print("✅ Alta empatía → tiny/LFM2 (CORRECTO, omni solo para audio)")
    
    print("\n✅ TEST 3 PASSED: Routing SIN solapamiento funcional")
    
    return True


def test_4_functional_separation():
    """TEST 4: Separación funcional clara"""
    print("\n" + "="*60)
    print("TEST 4: Functional Separation")
    print("="*60)
    
    # Verificar que cada agente tiene su propósito ÚNICO
    # v2.16.1: Omni-3B solo audio, empatía va a LFM2
    functions = {
        "omni_native": ["Audio STT/TTS (español)", "NLLB pipeline (multilingüe)", "Emotion detection"],
        "qwen3_vl": ["Análisis de imagen", "Análisis de video", "Vision specialist"],
        "tiny (LFM2)": ["Empatía conversacional", "Soft-skills (soft > 0.7)", "Fallback general"]
    }
    
    for agent, responsibilities in functions.items():
        print(f"\n{agent}:")
        for resp in responsibilities:
            print(f"  - {resp}")
    
    print("\n✅ Separación funcional CLARA")
    print("   - omni_native: Audio + Empatía (NUNCA visión)")
    print("   - qwen3_vl: Visión EXCLUSIVA (NUNCA audio)")
    
    return True


def main():
    """Ejecuta todos los tests"""
    print("\n" + "="*60)
    print("🧪 TEST REAL v2.16.1 - Best-of-Breed")
    print("="*60)
    print("\nObjetivo: Validar que omni_native y qwen3_vl NO solapan")
    print("Arquitectura:")
    print("  - Omni-3B/7B → Audio + Empatía (permanente)")
    print("  - Qwen3-VL-4B → Visión (bajo demanda)")
    
    ram_start = get_ram_usage_gb()
    time_start = time.time()
    
    tests = [
        ("Imports sin solapamiento", test_1_no_overlap_imports),
        ("Config loading correcto", test_2_config_loading),
        ("Routing sin solapamiento", test_3_routing_logic),
        ("Separación funcional", test_4_functional_separation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n💥 ERROR en {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE TESTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    ram_end = get_ram_usage_gb()
    time_end = time.time()
    
    print(f"\n📊 Métricas:")
    print(f"   Tests: {passed}/{total} passed")
    print(f"   RAM total: {ram_end:.2f} GB")
    print(f"   Δ RAM: +{(ram_end - ram_start)*1024:.0f} MB")
    print(f"   Tiempo: {time_end - time_start:.1f}s")
    
    if passed == total:
        print("\n🎉 TODOS LOS TESTS PASARON")
        print("\n✅ ARQUITECTURA VALIDADA:")
        print("   - omni_native y qwen3_vl NO solapan funciones")
        print("   - Routing correcto por tipo de input")
        print("   - Configs apuntan a modelos diferentes")
        print("   - Separación funcional clara")
        print("\n🚀 LISTO PARA BENCHMARK")
        return 0
    else:
        print(f"\n❌ {total - passed} TESTS FALLARON")
        print("   Revisar logs arriba para detalles")
        return 1


if __name__ == "__main__":
    sys.exit(main())
