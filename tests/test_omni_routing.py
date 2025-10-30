#!/usr/bin/env python3
"""
Test de Routing Omni-7B en LangGraph v2.16
Valida que las queries se enruten correctamente al agente apropiado
"""

import os
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_routing_logic():
    """Test 1: Lógica de routing sin ejecutar LLMs"""
    print("🧪 Test 1: Validando lógica de routing...")
    
    from core.graph import SARAiOrchestrator, State
    
    # Crear orquestador con TRM simulado (no requiere modelos reales)
    orchestrator = SARAiOrchestrator(use_simulated_trm=True)
    
    # Casos de prueba
    test_cases = [
        # (state, expected_route, description)
        (
            {"web_query": 0.8, "alpha": 0.5, "soft": 0.3, "input_type": "text"},
            "rag",
            "Alta web_query → RAG"
        ),
        (
            {"web_query": 0.2, "alpha": 0.5, "soft": 0.8, "input_type": "text"},
            "omni",
            "Alta empatía (soft > 0.7) → Omni"
        ),
        (
            {"web_query": 0.2, "alpha": 0.5, "soft": 0.3, "input_type": "audio"},
            "omni",
            "Input de audio → Omni"
        ),
        (
            {"web_query": 0.2, "alpha": 0.9, "soft": 0.2, "input_type": "text"},
            "expert",
            "Alta técnica (alpha > 0.7) → Expert (SOLAR)"
        ),
        (
            {"web_query": 0.2, "alpha": 0.5, "soft": 0.3, "input_type": "text"},
            "tiny",
            "Valores medios → Tiny (fallback)"
        ),
    ]
    
    passed = 0
    failed = 0
    
    for state, expected, description in test_cases:
        result = orchestrator._route_to_agent(state)
        
        if result == expected:
            print(f"  ✅ {description}: {result}")
            passed += 1
        else:
            print(f"  ❌ {description}: esperado '{expected}', obtenido '{result}'")
            failed += 1
    
    print(f"\n📊 Resultados: {passed} pasados, {failed} fallidos")
    
    return failed == 0


def test_omni_node_structure():
    """Test 2: Verificar que el nodo Omni existe y es invocable"""
    print("\n🧪 Test 2: Validando nodo Omni-7B...")
    
    from core.graph import SARAiOrchestrator
    
    # Crear orquestador (sin simular TRM para cargar Omni real)
    try:
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        # Verificar que el nodo existe
        assert hasattr(orchestrator, '_generate_omni'), "Nodo _generate_omni no existe"
        print("  ✅ Nodo _generate_omni existe")
        
        # Verificar que omni_agent está cargado
        assert hasattr(orchestrator, 'omni_agent'), "omni_agent no está inicializado"
        print("  ✅ omni_agent está inicializado")
        
        # Verificar tipo del agente
        from agents.omni_native import OmniNativeAgent
        assert isinstance(orchestrator.omni_agent, OmniNativeAgent), "omni_agent no es OmniNativeAgent"
        print("  ✅ omni_agent es instancia de OmniNativeAgent")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_typing():
    """Test 3: Verificar que State incluye 'omni' en agent_used"""
    print("\n🧪 Test 3: Validando tipado de State...")
    
    from core.graph import State
    import typing
    
    # Obtener tipo de agent_used
    agent_used_type = State.__annotations__.get('agent_used')
    
    if agent_used_type:
        # Extraer valores literales del Literal type
        if hasattr(agent_used_type, '__args__'):
            allowed_values = agent_used_type.__args__
            
            if "omni" in allowed_values:
                print(f"  ✅ 'omni' está en agent_used: {allowed_values}")
                return True
            else:
                print(f"  ❌ 'omni' NO está en agent_used: {allowed_values}")
                return False
        else:
            print(f"  ⚠️  agent_used no es Literal: {agent_used_type}")
            return False
    else:
        print("  ❌ agent_used no está en State")
        return False


def test_omni_generation_mock():
    """Test 4: Simular generación con Omni (sin llamar al modelo real)"""
    print("\n🧪 Test 4: Simulando generación con Omni...")
    
    from core.graph import SARAiOrchestrator
    
    try:
        orchestrator = SARAiOrchestrator(use_simulated_trm=True)
        
        # Estado de prueba (alta empatía → debe enrutar a Omni)
        test_state = {
            "input": "Estoy muy triste, necesito ayuda",
            "hard": 0.2,
            "soft": 0.9,  # Alta empatía
            "web_query": 0.1,
            "alpha": 0.3,
            "beta": 0.7,
            "input_type": "text"
        }
        
        # Verificar routing
        route = orchestrator._route_to_agent(test_state)
        assert route == "omni", f"Routing incorrecto: esperado 'omni', obtenido '{route}'"
        print(f"  ✅ Routing correcto: {route}")
        
        # NOTA: No llamamos a _generate_omni() porque requeriría el modelo cargado
        # El test completo de generación está en test_omni_langchain.py
        
        print("  ✅ Omni-7B se enrutaría correctamente para queries empáticas")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecuta todos los tests"""
    print("=" * 60)
    print("TEST DE ROUTING OMNI-7B v2.16")
    print("=" * 60)
    
    tests = [
        ("Lógica de routing", test_routing_logic),
        ("Estructura del nodo", test_omni_node_structure),
        ("Tipado de State", test_state_typing),
        ("Generación simulada", test_omni_generation_mock)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crasheó: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTests pasados: {passed}/{total}")
    
    if passed == total:
        print("✅ TODOS LOS TESTS PASARON")
        return 0
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        return 1


if __name__ == "__main__":
    sys.exit(main())
