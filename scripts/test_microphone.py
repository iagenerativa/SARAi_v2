#!/usr/bin/env python3
"""
scripts/test_microphone.py

Script helper para ejecutar tests interactivos con micrófono

Uso:
    python scripts/test_microphone.py --audio-router    # Test de detección de idioma
    python scripts/test_microphone.py --emotion         # Test de detección emocional
    python scripts/test_microphone.py --all             # Ambos tests

Author: SARAi Team
Date: 2025-10-28
"""

import argparse
import subprocess
import sys


def check_dependencies():
    """Verifica que pyaudio y scipy estén instalados"""
    try:
        import pyaudio
        import scipy
        return True
    except ImportError as e:
        print(f"❌ Error: Dependencia faltante - {e}")
        print("\n📦 Instalar con:")
        print("   pip install pyaudio scipy")
        return False


def run_audio_router_test():
    """Ejecuta test de detección de idioma"""
    print("\n" + "="*70)
    print("🎤 INICIANDO: Test de Detección de Idioma")
    print("="*70)
    
    cmd = [
        "pytest",
        "tests/test_audio_router.py::TestLanguageDetector::test_detect_with_real_microphone",
        "-s",
        "-v",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def run_emotion_test():
    """Ejecuta test de detección emocional"""
    print("\n" + "="*70)
    print("🎭 INICIANDO: Test de Detección Emocional")
    print("="*70)
    
    cmd = [
        "pytest",
        "tests/test_emotion_modulator.py::TestEmotionModulationIntegration::test_emotion_detection_with_real_microphone",
        "-s",
        "-v",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Test interactivo de audio con micrófono",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/test_microphone.py --audio-router
  python scripts/test_microphone.py --emotion
  python scripts/test_microphone.py --all

Notas:
  - Requiere micrófono funcional
  - Instalar: pip install pyaudio scipy
  - Los tests son interactivos (requieren confirmación manual)
        """
    )
    
    parser.add_argument(
        "--audio-router",
        action="store_true",
        help="Test de detección de idioma (LanguageDetector)"
    )
    
    parser.add_argument(
        "--emotion",
        action="store_true",
        help="Test de detección emocional (EmotionModulator)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ejecutar ambos tests"
    )
    
    args = parser.parse_args()
    
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    # Si no se especifica nada, mostrar ayuda
    if not (args.audio_router or args.emotion or args.all):
        parser.print_help()
        sys.exit(0)
    
    results = []
    
    # Ejecutar tests
    if args.all or args.audio_router:
        ret = run_audio_router_test()
        results.append(("Audio Router (Language Detection)", ret))
    
    if args.all or args.emotion:
        ret = run_emotion_test()
        results.append(("Emotion Detection", ret))
    
    # Resumen
    print("\n" + "="*70)
    print("📊 RESUMEN DE TESTS INTERACTIVOS")
    print("="*70)
    
    all_passed = True
    for test_name, returncode in results:
        status = "✅ PASSED" if returncode == 0 else "❌ FAILED"
        print(f"   {test_name:40} {status}")
        if returncode != 0:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 Todos los tests pasaron exitosamente")
        sys.exit(0)
    else:
        print("\n⚠️  Algunos tests fallaron (revisar output arriba)")
        sys.exit(1)


if __name__ == "__main__":
    main()
