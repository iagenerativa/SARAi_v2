#!/usr/bin/env python3
"""
Script para probar MeloTTS con diferentes configuraciones
Uso: python3 scripts/test_melo_config.py [speed]
"""

import sys
sys.path.append('.')
from agents.melo_tts import MeloTTSEngine
import soundfile as sf
import time
import subprocess

def test_melo(speed=1.0):
    """Prueba MeloTTS con una velocidad específica"""
    print(f"\n{'='*60}")
    print(f"Probando: MeloTTS Español con velocidad {speed}x")
    print(f"{'='*60}\n")
    
    engine = MeloTTSEngine(language='ES', speaker='ES', speed=speed)
    
    texts = [
        "Hola, soy SARAi. Esta es mi voz española nativa.",
        "Puedo ayudarte con cualquier consulta que tengas.",
        "Mi pronunciación es natural y expresiva."
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] \"{text}\"")
        
        start = time.perf_counter()
        audio = engine.synthesize(text)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"   ✅ Generado en {elapsed:.0f}ms")
        print(f"   Duración: {len(audio)/44100:.2f}s")
        
        output_file = f"/tmp/test_melo_{i}.wav"
        sf.write(output_file, audio, 44100)
        
        print(f"   🔊 Reproduciendo...")
        subprocess.run(["aplay", "-q", output_file])
    
    print(f"\n{'='*60}")
    print("✅ Test completado")
    print(f"{'='*60}\n")


def main():
    """Prueba varias configuraciones"""
    
    if len(sys.argv) > 1:
        # Probar velocidad específica
        speed = float(sys.argv[1])
        test_melo(speed)
    else:
        # Probar configuración por defecto
        print("\n🎤 PRUEBA DE MELOTTS ESPAÑOL\n")
        print("Velocidad disponibles:")
        print("  1. 0.8x (más lenta, más clara)")
        print("  2. 1.0x (normal) [DEFAULT]")
        print("  3. 1.2x (más rápida)")
        
        print("\n" + "="*60)
        print("CONFIGURACIÓN ACTUAL (por defecto)")
        print("="*60)
        test_melo(1.0)
        
        print("\n\n¿Quieres probar otras velocidades?")
        print("Uso: python3 scripts/test_melo_config.py <speed>")
        print("Ejemplo: python3 scripts/test_melo_config.py 1.2")


if __name__ == "__main__":
    main()
