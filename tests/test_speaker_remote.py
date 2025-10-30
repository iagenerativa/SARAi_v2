#!/usr/bin/env python3
"""
Test de Altavoces - Versión para Servidor Remoto

Genera archivos de audio WAV que puedes descargar y reproducir localmente.
Útil cuando estás conectado por SSH sin acceso directo a altavoces.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.io import wavfile

def generate_test_tone():
    """Genera un tono de prueba y lo guarda a archivo"""
    print("\n" + "="*60)
    print("🔊 GENERANDO TONO DE PRUEBA")
    print("="*60)
    
    # Parámetros
    sample_rate = 44100  # Hz
    duration = 2  # segundos
    frequency = 440  # Hz (Nota LA)
    
    print(f"\n🎵 Tono: {frequency}Hz (Nota LA)")
    print(f"⏱️  Duración: {duration} segundos")
    
    # Generar onda sinusoidal
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = 0.3 * np.sin(2 * np.pi * frequency * t)  # Volumen 30%
    
    # Convertir a int16
    wave_int16 = (wave * 32767).astype(np.int16)
    
    # Guardar
    output_dir = Path("state/audio_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "test_tone.wav"
    
    wavfile.write(str(output_file), sample_rate, wave_int16)
    
    print(f"\n✅ Archivo generado: {output_file}")
    print(f"   Tamaño: {output_file.stat().st_size / 1024:.1f} KB")
    
    return output_file

def generate_voice_message():
    """Genera un mensaje de voz de prueba"""
    print("\n" + "="*60)
    print("🔊 GENERANDO MENSAJE DE VOZ")
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
    
    output_dir = Path("state/audio_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "test_voice.wav"
    
    print(f"\n📢 Mensaje: \"{message}\"")
    print(f"💾 Guardando en: {output_file}")
    
    engine.save_to_file(message, str(output_file))
    engine.runAndWait()
    
    print(f"\n✅ Archivo generado: {output_file}")
    
    if output_file.exists():
        print(f"   Tamaño: {output_file.stat().st_size / 1024:.1f} KB")
        return output_file
    else:
        print("❌ Error: archivo no generado")
        return None

def main():
    print("\n🎵 TEST DE AUDIO PARA SERVIDOR REMOTO - SARAi v2.16.3\n")
    print("Este script genera archivos WAV que puedes descargar y reproducir.")
    
    # Test 1: Tono simple
    tone_file = generate_test_tone()
    
    # Test 2: Voz (TTS)
    print("\n" + "-"*60)
    print("\n¿Generar también mensaje de voz? (s/n): ", end='', flush=True)
    response = input().strip().lower()
    
    if response == 's' or response == 'si' or response == 'sí':
        voice_file = generate_voice_message()
    else:
        voice_file = None
    
    # Resumen
    print("\n" + "="*60)
    print("✅ ARCHIVOS GENERADOS")
    print("="*60)
    print(f"\n📁 Ubicación: state/audio_test/")
    print(f"\n   1. {tone_file.name} - Tono de prueba (440Hz)")
    if voice_file:
        print(f"   2. {voice_file.name} - Mensaje de voz")
    
    print("\n📥 CÓMO DESCARGAR:")
    print("\n   Opción 1 - SCP (desde tu máquina local):")
    print(f"   scp noel@agi1:~/SARAi_v2/state/audio_test/*.wav .")
    
    print("\n   Opción 2 - SFTP:")
    print(f"   sftp noel@agi1")
    print(f"   get ~/SARAi_v2/state/audio_test/*.wav")
    
    print("\n   Opción 3 - Navegador (si tienes servidor web):")
    print(f"   http://localhost:8000/state/audio_test/")
    
    print("\n🔊 REPRODUCIR:")
    print("\n   Linux/Mac:")
    print("   aplay test_tone.wav")
    print("   mpg123 test_tone.wav")
    
    print("\n   Windows:")
    print("   start test_tone.wav")
    
    print("\n   Cualquier sistema:")
    print("   vlc test_tone.wav")
    
    print("\n" + "="*60)
    print("\n✅ Listo! Descarga los archivos y reprodúcelos localmente.")

if __name__ == "__main__":
    main()
