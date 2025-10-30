#!/usr/bin/env python3
"""
Test rápido de conexión al servidor Ollama
Valida configuración de .env para desarrollo
"""

import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.solar_ollama import SolarOllama


def main():
    print("🔧 Probando conexión a servidor Ollama de desarrollo...")
    print("="*70)
    
    try:
        # Crear cliente (lee .env automáticamente)
        client = SolarOllama(verbose=True)
        
        # Test simple
        print("\n📝 Enviando query de prueba...")
        response = client.generate(
            prompt="Di 'Hola' en una palabra",
            max_tokens=10,
            temperature=0.1
        )
        
        print(f"\n✅ Respuesta recibida: {response.strip()}")
        print("\n✅ Servidor Ollama funcionando correctamente")
        print(f"   Puedes usar SOLAR en desarrollo desde {client.base_url}")
        
        return 0
    
    except ConnectionError as e:
        print(f"\n❌ Error de conexión:")
        print(f"   {e}")
        print(f"\n💡 Soluciones:")
        print(f"   1. Verifica que el servidor Ollama esté corriendo:")
        print(f"      ssh user@192.168.0.251 'systemctl status ollama'")
        print(f"   2. Verifica la IP en .env (OLLAMA_BASE_URL)")
        print(f"   3. Prueba acceso HTTP:")
        print(f"      curl http://192.168.0.251:11434/api/tags")
        return 1
    
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
