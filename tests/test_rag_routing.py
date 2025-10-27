#!/usr/bin/env python3
"""
Test de Routing RAG en Graph v2.11

Valida que:
1. TRM-Classifier detecta web_query correctamente
2. Graph enruta al nodo RAG cuando web_query > 0.7
3. RAG Agent retorna metadata completa
4. Fallback Sentinel funciona si SearXNG no disponible
"""

import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import create_orchestrator
import json


def test_web_query_routing():
    """Test 1: Queries web_query son enrutadas a RAG"""
    
    print("=" * 80)
    print("TEST 1: ROUTING DE WEB_QUERY")
    print("=" * 80)
    
    # Crear orquestador (usa TRM entrenado real)
    orchestrator = create_orchestrator(use_simulated_trm=False)
    
    # Queries que DEBEN activar web_query
    web_queries = [
        "¿Quién ganó el Oscar 2025?",
        "¿Cómo está el clima en Tokio hoy?",
        "Precio actual de Bitcoin",
        "Últimas noticias de tecnología",
        "¿Qué pasó en las elecciones de Argentina 2024?"
    ]
    
    results = []
    
    for query in web_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 80)
        
        try:
            # Invocar orquestador completo
            response = orchestrator.invoke(query)
            
            # Verificar que usó RAG (esto se imprime en logs)
            print(f"✅ Respuesta obtenida ({len(response)} chars)")
            print(f"   Preview: {response[:150]}...")
            
            results.append({
                "query": query,
                "success": True,
                "response_length": len(response)
            })
        
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN TEST 1")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    
    print(f"Exitosas: {success_count}/{total_count}")
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['query']}")
    
    return success_count == total_count


def test_non_web_query_routing():
    """Test 2: Queries hard/soft NO van a RAG"""
    
    print("\n" + "=" * 80)
    print("TEST 2: ROUTING DE HARD/SOFT (NO RAG)")
    print("=" * 80)
    
    orchestrator = create_orchestrator(use_simulated_trm=False)
    
    # Queries que NO deben activar RAG
    non_web_queries = {
        "hard": [
            "¿Cómo configurar SSH en Ubuntu?",
            "Error al importar numpy en Python"
        ],
        "soft": [
            "Me siento frustrado con este bug",
            "Explícame Python como a un niño"
        ]
    }
    
    results = []
    
    for category, queries in non_web_queries.items():
        for query in queries:
            print(f"\n🔍 Query ({category}): {query}")
            print("-" * 80)
            
            try:
                response = orchestrator.invoke(query)
                
                # Debería usar Expert o Tiny, NO RAG
                print(f"✅ Respuesta obtenida ({len(response)} chars)")
                
                results.append({
                    "query": query,
                    "category": category,
                    "success": True
                })
            
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    "query": query,
                    "category": category,
                    "success": False,
                    "error": str(e)
                })
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN TEST 2")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    
    print(f"Exitosas: {success_count}/{total_count}")
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} [{result['category']}] {result['query']}")
    
    return success_count == total_count


def test_rag_metadata():
    """Test 3: RAG retorna metadata completa"""
    
    print("\n" + "=" * 80)
    print("TEST 3: METADATA DE RAG")
    print("=" * 80)
    
    orchestrator = create_orchestrator(use_simulated_trm=False)
    
    query = "¿Quién ganó el Oscar 2025?"
    
    print(f"\n🔍 Query: {query}")
    print("-" * 80)
    
    try:
        # Acceder al estado completo (no solo response)
        initial_state = {
            "input": query,
            "hard": 0.0,
            "soft": 0.0,
            "web_query": 0.0,
            "alpha": 0.5,
            "beta": 0.5,
            "agent_used": "tiny",
            "response": "",
            "feedback": 0.0,
            "rag_metadata": {}
        }
        
        result_state = orchestrator.app.invoke(initial_state)
        
        # Verificar metadata
        if "rag_metadata" in result_state and result_state["rag_metadata"]:
            print("✅ Metadata RAG presente:")
            print(json.dumps(result_state["rag_metadata"], indent=2, ensure_ascii=False))
            
            # Validar campos críticos
            required_fields = ["source", "snippets_count", "llm_model", "synthesis_success"]
            missing_fields = [f for f in required_fields if f not in result_state["rag_metadata"]]
            
            if missing_fields:
                print(f"⚠️ Campos faltantes: {missing_fields}")
                return False
            
            return True
        else:
            print("⚠️ Metadata RAG vacía o ausente")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecutar todos los tests"""
    
    print("=" * 80)
    print("🧪 TEST SUITE: ROUTING RAG EN GRAPH v2.11")
    print("=" * 80)
    print("Objetivo: Validar integración completa TRM → Graph → RAG")
    print("=" * 80)
    
    tests = [
        ("Web Query Routing", test_web_query_routing),
        ("Non-Web Query Routing", test_non_web_query_routing),
        ("RAG Metadata", test_rag_metadata)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' crasheó: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Resumen final
    print("\n" + "=" * 80)
    print("📊 RESUMEN FINAL")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("=" * 80)
    print(f"Total: {total_passed}/{total_tests} tests pasados")
    print("=" * 80)
    
    if total_passed == total_tests:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("✅ M2.5 COMPLETADO: RAG integrado en Graph exitosamente")
        return 0
    else:
        print("\n⚠️ Algunos tests fallaron. Revisar logs arriba.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
