#!/usr/bin/env python3
"""
Test de Omni Native Agent v2.16
Valida arquitectura LangChain + memoria permanente
"""

import os
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import():
    """Test 1: Importación limpia"""
    print("🧪 Test 1: Importando módulo...")
    try:
        from agents.omni_native import OmniConfig, OmniNativeAgent, get_omni_agent
        print("✅ Importación exitosa")
        return True
    except Exception as e:
        print(f"❌ Error de importación: {e}")
        return False


def test_config():
    """Test 2: Carga de configuración"""
    print("\n🧪 Test 2: Cargando configuración...")
    try:
        from agents.omni_native import OmniConfig
        config = OmniConfig.from_yaml("config/sarai.yaml")
        
        print(f"  Model path: {config.model_path}")
        print(f"  Context: {config.n_ctx}")
        print(f"  Threads: {config.n_threads}")
        
        assert config.n_ctx == 8192, f"n_ctx esperado 8192, obtenido {config.n_ctx}"
        assert Path(config.model_path).suffix == ".gguf", "Archivo no es GGUF"
        
        print("✅ Configuración válida")
        return True
    except Exception as e:
        print(f"❌ Error de configuración: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_singleton():
    """Test 3: Patrón singleton"""
    print("\n🧪 Test 3: Validando singleton...")
    try:
        from agents.omni_native import get_omni_agent
        
        # Primera instancia
        agent1 = get_omni_agent()
        print(f"  Instancia 1: {id(agent1)}")
        
        # Segunda instancia (debe ser la misma)
        agent2 = get_omni_agent()
        print(f"  Instancia 2: {id(agent2)}")
        
        assert agent1 is agent2, "Singleton no funciona (instancias diferentes)"
        print("✅ Singleton funciona correctamente")
        return True
    except Exception as e:
        print(f"❌ Error de singleton: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_invoke():
    """Test 4: Generación de texto"""
    print("\n🧪 Test 4: Generando respuesta...")
    
    # Verificar si modelo existe (usa capitalización correcta del archivo real)
    model_path = Path("models/gguf/Qwen2.5-Omni-7B-Q4_K_M.gguf")
    if not model_path.exists():
        print(f"⚠️  Modelo no encontrado: {model_path}")
        print("   (Esto es esperado si no has descargado el modelo aún)")
        return True  # No falla el test
    
    try:
        from agents.omni_native import get_omni_agent
        agent = get_omni_agent()
        
        query = "Responde solo con un número: 2+2="
        print(f"\n📝 Query: {query}")
        
        # Generar con límite bajo para test rápido (solo 10 tokens)
        response = agent.invoke(query, max_tokens=10)
        print(f"🤖 Response: {response}")
        
        assert len(response) > 0, "Respuesta vacía"
        print("✅ Generación exitosa")
        return True
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   (Descarga el modelo para habilitar este test)")
        return True
    except Exception as e:
        print(f"❌ Error de generación: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecuta todos los tests"""
    print("=" * 60)
    print("TEST DE OMNI NATIVE AGENT v2.16")
    print("Arquitectura: LangChain + Memoria Permanente")
    print("=" * 60)
    
    tests = [
        test_import,
        test_config,
        test_singleton,
        test_invoke
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests pasados: {passed}/{total}")
    
    if passed == total:
        print("✅ TODOS LOS TESTS PASARON")
        return 0
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        return 1


if __name__ == "__main__":
    sys.exit(main())
