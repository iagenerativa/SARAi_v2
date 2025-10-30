"""
SARAi v2.17 - Test Completo Capa 1
Test de integración: Canal IN + Canal OUT en modo full-duplex
"""

import sys
from pathlib import Path

# Añadir path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.layer1_io import Layer1Orchestrator


def main():
    """
    Test completo de la Capa 1
    
    Pipeline testeado:
        Audio (mic) → VAD → Vosk STT → BERT → LoRA Router →
        → [TRM Cache / LFM2 / NLLB] → MeloTTS → Audio (speakers)
    
    Características verificadas:
        ✓ Streaming real (chunks de 100ms)
        ✓ Full-duplex (entrada y salida independientes)
        ✓ TRM cache (< 50ms para respuestas comunes)
        ✓ LLM generación (para consultas complejas)
        ✓ Espera inteligente (no interrumpe al usuario)
        ✓ Estadísticas en tiempo real
    """
    
    print("\n" + "=" * 70)
    print("   🧪 TEST CAPA 1 - I/O FULL-DUPLEX")
    print("=" * 70)
    
    print("\n📋 Configuración del test:")
    print("  • Sample rate: 16kHz")
    print("  • Chunks: 100ms")
    print("  • VAD threshold: 0.02")
    print("  • Silencio fin frase: 500ms")
    print("  • Espera usuario: 300ms")
    
    print("\n🎯 Objetivos del test:")
    print("  1. Verificar captura de audio continua")
    print("  2. Validar Vosk STT streaming")
    print("  3. Comprobar routing LoRA (TRM/LLM/Traducir)")
    print("  4. Testear TRM cache (latencia < 50ms)")
    print("  5. Validar LLM generación (LFM2)")
    print("  6. Verificar MeloTTS streaming")
    print("  7. Comprobar coordinación full-duplex")
    
    print("\n" + "=" * 70)
    
    # Crear orquestador
    orchestrator = Layer1Orchestrator(
        sample_rate=16000,
        chunk_duration_ms=100,
        vad_energy_threshold=0.02,
        silence_timeout=0.5,
        user_silence_threshold=0.3
    )
    
    # Cargar componentes
    try:
        orchestrator.load_components()
    except Exception as e:
        print(f"\n❌ Error al cargar componentes: {e}")
        print("\nVerifica que:")
        print("  • Vosk modelo está en: models/vosk/vosk-model-small-es-0.42/")
        print("  • LFM2 modelo está en: models/gguf/LFM2-1.2B-Q4_K_M.gguf")
        print("  • MeloTTS se descargará automáticamente (checkpoint_es.pth)")
        print("  • BERT-es se descargará automáticamente de HuggingFace")
        return 1
    
    # Iniciar sistema
    orchestrator.start()
    
    print("\n💡 Ejemplos de consultas para probar:")
    print("\n  🔹 TRM Cache (respuestas rápidas):")
    print("     • \"Hola\"")
    print("     • \"¿Cómo estás?\"")
    print("     • \"Gracias\"")
    print("     • \"Buenos días\"")
    
    print("\n  🔹 LLM Generación (consultas complejas):")
    print("     • \"¿Qué es la inteligencia artificial?\"")
    print("     • \"Explícame Python\"")
    print("     • \"¿Cómo funciona el aprendizaje profundo?\"")
    
    print("\n  🔹 Traducir (idiomas no-español):")
    print("     • \"Hello, how are you?\" (inglés)")
    print("     • \"Bonjour, comment ça va?\" (francés)")
    
    print("\n" + "=" * 70)
    print("\n⏳ Test en ejecución...")
    print("   Habla naturalmente para probar el sistema")
    print("   Presiona Ctrl+C para finalizar el test\n")
    
    # Ejecutar interactivo
    orchestrator.run_interactive()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
