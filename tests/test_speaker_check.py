#!/usr/bin/env python3
"""
Test de Altavoces - Verificación Rápida

Reproduce un tono de prueba para verificar que los altavoces funcionan correctamente.
"""

import sys
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("❌ sounddevice no instalado")
    print("Instalando...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice"])
    import sounddevice as sd

def test_speaker():
    """Reproduce un tono de prueba"""
    print("\n" + "="*60)
    print("🔊 TEST DE ALTAVOCES")
    print("="*60)
    
    # Parámetros
    sample_rate = 44100  # Hz
    duration = 2  # segundos
    frequency = 440  # Hz (Nota LA)
    
    print(f"\n🎵 Generando tono de {frequency}Hz (Nota LA)")
    print(f"⏱️  Duración: {duration} segundos")
    
    # Generar onda sinusoidal
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = 0.3 * np.sin(2 * np.pi * frequency * t)  # Volumen 30%
    
    print("\n🔊 Reproduciendo en 3 segundos...")
    for i in range(3, 0, -1):
        print(f"   {i}...", flush=True)
        time.sleep(1)
    
    print("\n▶️  REPRODUCIENDO AHORA")
    print("    (Deberías oír un tono constante)")
    
    # Reproducir
    sd.play(wave, sample_rate)
    sd.wait()
    
    print("\n✅ Reproducción completada")
    print("\n¿Escuchaste el tono? (s/n): ", end='', flush=True)
    response = input().strip().lower()
    
    if response == 's' or response == 'si' or response == 'sí':
        print("\n✅ ¡Altavoces funcionando correctamente!")
        return True
    else:
        print("\n⚠️  Problema detectado. Verificar:")
        print("   1. Altavoces conectados")
        print("   2. Volumen del sistema no en mute")
        print("   3. Dispositivo de salida correcto")
        print("\n   Listar dispositivos:")
        print("   python -c \"import sounddevice as sd; print(sd.query_devices())\"")
        return False

def test_speaker_voice():
    """Reproduce un mensaje de voz de prueba"""
    print("\n" + "="*60)
    print("🔊 TEST DE VOZ (TTS)")
    print("="*60)
    
    try:
        import pyttsx3
    except ImportError:
        print("⚠️  pyttsx3 no instalado")
        print("Instalando...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyttsx3"])
        import pyttsx3
    
    print("\n🔧 Inicializando TTS...")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    
    message = "Hola, soy SARAi. Este es un test de altavoces. ¿Me escuchas correctamente?"
    
    print(f"\n📢 Mensaje: \"{message}\"")
    print("\n🔊 Reproduciendo en 3 segundos...")
    for i in range(3, 0, -1):
        print(f"   {i}...", flush=True)
        time.sleep(1)
    
    print("\n▶️  REPRODUCIENDO AHORA")
    engine.say(message)
    engine.runAndWait()
    
    print("\n✅ Reproducción completada")
    print("\n¿Escuchaste el mensaje claramente? (s/n): ", end='', flush=True)
    response = input().strip().lower()
    
    if response == 's' or response == 'si' or response == 'sí':
        print("\n✅ ¡TTS funcionando correctamente!")
        print("   Listo para usar test_voice_quick.py")
        return True
    else:
        print("\n⚠️  Problema con TTS. Verificar configuración de audio.")
        return False

def main():
    print("\n🎵 TEST DE AUDIO - SARAi v2.16.3\n")
    
    # Test 1: Tono simple
    print("TEST 1: Tono de prueba")
    tone_ok = test_speaker()
    
    if not tone_ok:
        print("\n❌ Test de tono falló. Corregir antes de continuar.")
        return
    
    # Test 2: Voz (TTS)
    print("\n" + "-"*60)
    print("\nTEST 2: Síntesis de voz (TTS)")
    print("¿Quieres probar TTS también? (s/n): ", end='', flush=True)
    response = input().strip().lower()
    
    if response == 's' or response == 'si' or response == 'sí':
        voice_ok = test_speaker_voice()
        
        if voice_ok:
            print("\n" + "="*60)
            print("✅ TODOS LOS TESTS PASADOS")
            print("="*60)
            print("\n🚀 Listo para ejecutar:")
            print("   python tests/test_voice_quick.py")
    else:
        print("\n✅ Test de tono OK. Puedes continuar con test_voice_quick.py")

if __name__ == "__main__":
    main()
