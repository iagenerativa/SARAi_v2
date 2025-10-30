#!/usr/bin/env python3
"""
Script para probar diferentes configuraciones de voz de Kitten TTS
Uso: python3 scripts/test_voice_config.py [voice] [speed]
"""

import sys
sys.path.append('.')
from agents.kitten_tts import KittenTTSEngine
import soundfile as sf
import time
import subprocess

def test_voice(voice="expr-voice-4-f", speed=1.2):
    """Prueba una configuración de voz y velocidad"""
    print(f"\n{'='*60}")
    print(f"Probando: {voice} con velocidad {speed}x")
    print(f"{'='*60}\n")
    
    engine = KittenTTSEngine(voice=voice, speed=speed)
    
    text = "Hola, soy SARAi. Esta es mi nueva voz española. ¿Qué te parece?"
    
    print("Generando audio...")
    start = time.perf_counter()
    audio = engine.synthesize(text)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"✅ Audio generado en {elapsed:.0f}ms")
    print(f"   Shape: {audio.shape}")
    print(f"   Duración real: {len(audio)/24000:.2f}s")
    
    output_file = f"/tmp/test_{voice}_speed{speed}.wav"
    sf.write(output_file, audio, 24000)
    
    print(f"\n🔊 Reproduciendo...")
    subprocess.run(["aplay", "-q", output_file])
    
    print(f"\n💾 Archivo guardado: {output_file}")
    return elapsed


def main():
    """Prueba varias configuraciones"""
    
    if len(sys.argv) > 1:
        # Probar configuración específica
        voice = sys.argv[1] if len(sys.argv) > 1 else "expr-voice-4-f"
        speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2
        test_voice(voice, speed)
    else:
        # Probar todas las voces femeninas españolas
        print("\n🎤 PRUEBA DE VOCES ESPAÑOLAS\n")
        
        voices = ["expr-voice-3-f", "expr-voice-4-f", "expr-voice-5-f"]
        speeds = [1.0, 1.2, 1.4]
        
        print("Voces disponibles:")
        for i, voice in enumerate(voices, 1):
            print(f"  {i}. {voice}")
        
        print("\nVelocidades:")
        for i, speed in enumerate(speeds, 1):
            print(f"  {i}. {speed}x")
        
        # Probar configuración por defecto
        print("\n" + "="*60)
        print("CONFIGURACIÓN ACTUAL (por defecto)")
        print("="*60)
        test_voice("expr-voice-4-f", 1.2)
        
        print("\n\n¿Quieres probar otras configuraciones?")
        print("Uso: python3 scripts/test_voice_config.py <voice> <speed>")
        print("Ejemplo: python3 scripts/test_voice_config.py expr-voice-5-f 1.4")


if __name__ == "__main__":
    main()
