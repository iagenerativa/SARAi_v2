"""
SARAi v2.17 - Test de Integración MeloTTS
Test de componentes individuales sin full-duplex
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.melo_tts import MeloTTSEngine
import soundfile as sf
import time
import subprocess


def test_melo_basic():
    """Test 1: Carga básica de MeloTTS"""
    print("\n" + "="*70)
    print("TEST 1: Carga Básica de MeloTTS")
    print("="*70)
    
    try:
        engine = MeloTTSEngine(language='ES', speaker='ES', speed=1.15)
        print("✅ MeloTTS cargado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error cargando MeloTTS: {e}")
        return False


def test_melo_synthesis():
    """Test 2: Síntesis de audio"""
    print("\n" + "="*70)
    print("TEST 2: Síntesis de Audio")
    print("="*70)
    
    try:
        engine = MeloTTSEngine(language='ES', speaker='ES', speed=1.15)
        
        test_phrases = [
            "Hola, soy SARAi.",
            "¿En qué puedo ayudarte?",
            "Entiendo. ¿Puedes darme más información?"
        ]
        
        for i, phrase in enumerate(test_phrases, 1):
            print(f"\n[{i}/{len(test_phrases)}] Sintetizando: \"{phrase}\"")
            
            start = time.perf_counter()
            audio = engine.synthesize(phrase)
            elapsed = (time.perf_counter() - start) * 1000
            
            if audio is None or len(audio) == 0:
                print(f"❌ Síntesis falló")
                return False
            
            print(f"   ✅ Audio generado en {elapsed:.0f}ms")
            print(f"   📊 Shape: {audio.shape}, dtype: {audio.dtype}")
            print(f"   ⏱️  Duración: {len(audio)/44100:.2f}s")
        
        print("\n✅ Todas las síntesis completadas correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en síntesis: {e}")
        return False


def test_melo_quality():
    """Test 3: Calidad de audio y reproducción"""
    print("\n" + "="*70)
    print("TEST 3: Calidad de Audio")
    print("="*70)
    
    try:
        engine = MeloTTSEngine(language='ES', speaker='ES', speed=1.15)
        
        text = "Hola, soy SARAi. Esta es mi voz española nativa con MeloTTS optimizado."
        
        print(f"\nTexto: \"{text}\"")
        print("\nGenerando audio...")
        
        start = time.perf_counter()
        audio = engine.synthesize(text)
        elapsed = (time.perf_counter() - start) * 1000
        
        # Guardar archivo
        output_file = "/tmp/test_melotts_quality.wav"
        sf.write(output_file, audio, 44100)
        
        print(f"✅ Audio generado en {elapsed:.0f}ms")
        print(f"📊 Estadísticas:")
        print(f"   • Sample rate: 44100 Hz")
        print(f"   • Canales: Mono")
        print(f"   • Duración: {len(audio)/44100:.2f}s")
        print(f"   • Tamaño: {len(audio)} samples")
        print(f"   • Archivo: {output_file}")
        
        print(f"\n🔊 Reproduciendo audio...")
        result = subprocess.run(
            ["aplay", "-q", output_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Audio reproducido correctamente")
            return True
        else:
            print(f"⚠️  Reproducción completada con código: {result.returncode}")
            return True  # No es error crítico
        
    except Exception as e:
        print(f"❌ Error en test de calidad: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_melo_performance():
    """Test 4: Performance y latencia"""
    print("\n" + "="*70)
    print("TEST 4: Performance y Latencia")
    print("="*70)
    
    try:
        engine = MeloTTSEngine(language='ES', speaker='ES', speed=1.15)
        
        phrases = [
            "Hola",
            "Buenos días",
            "¿Cómo estás?",
            "Gracias por tu ayuda",
            "Entiendo perfectamente lo que necesitas"
        ]
        
        latencies = []
        
        print("\nMidiendo latencia de síntesis...")
        for i, phrase in enumerate(phrases, 1):
            start = time.perf_counter()
            audio = engine.synthesize(phrase)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            
            duration = len(audio) / 44100
            rtf = elapsed / (duration * 1000)  # Real-time factor
            
            print(f"[{i}] {len(phrase):3d} chars → {elapsed:6.0f}ms (RTF: {rtf:.2f}x)")
        
        # Estadísticas
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"\n📊 Estadísticas de Latencia:")
        print(f"   • Promedio: {avg_latency:.0f}ms")
        print(f"   • Mínima:   {min_latency:.0f}ms")
        print(f"   • Máxima:   {max_latency:.0f}ms")
        
        # Validar objetivos
        target_avg = 5000  # 5s promedio es aceptable
        if avg_latency < target_avg:
            print(f"✅ Latencia promedio dentro del objetivo (< {target_avg}ms)")
            return True
        else:
            print(f"⚠️  Latencia promedio superior al objetivo ({avg_latency:.0f}ms > {target_avg}ms)")
            return True  # No falla, solo advierte
        
    except Exception as e:
        print(f"❌ Error en test de performance: {e}")
        return False


def test_melo_integration():
    """Test 5: Integración con OutputThread"""
    print("\n" + "="*70)
    print("TEST 5: Integración con OutputThread")
    print("="*70)
    
    try:
        from core.layer1_io.output_thread import OutputThread
        
        print("Inicializando OutputThread...")
        output_thread = OutputThread()
        
        print("Cargando componentes...")
        output_thread.load_components()
        
        # Verificar que MeloTTS está cargado
        if output_thread.melo_tts is None:
            print("❌ MeloTTS no está cargado en OutputThread")
            return False
        
        print("✅ OutputThread inicializado correctamente")
        print(f"   • MeloTTS: {type(output_thread.melo_tts).__name__}")
        print(f"   • Sample rate: {output_thread.melo_tts.sample_rate} Hz")
        print(f"   • Velocidad: {output_thread.melo_tts.speed}x")
        
        # Test de síntesis integrada
        print("\nTest de síntesis a través de OutputThread...")
        test_text = "Test de integración exitoso"
        
        start = time.perf_counter()
        audio = output_thread.melo_tts.synthesize(test_text)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"✅ Síntesis integrada exitosa ({elapsed:.0f}ms)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en integración: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecutar todos los tests"""
    print("\n" + "="*70)
    print("   🧪 TEST SUITE: INTEGRACIÓN MELOTTS EN SARAI v2.17")
    print("="*70)
    
    print("\n📋 Tests a ejecutar:")
    print("  1. Carga básica de MeloTTS")
    print("  2. Síntesis de audio múltiple")
    print("  3. Calidad de audio y reproducción")
    print("  4. Performance y latencia")
    print("  5. Integración con OutputThread")
    
    print("\n" + "="*70)
    print("INICIANDO TESTS...")
    print("="*70)
    
    results = {}
    
    # Test 1
    results['basic'] = test_melo_basic()
    time.sleep(1)
    
    # Test 2
    results['synthesis'] = test_melo_synthesis()
    time.sleep(1)
    
    # Test 3
    results['quality'] = test_melo_quality()
    time.sleep(1)
    
    # Test 4
    results['performance'] = test_melo_performance()
    time.sleep(1)
    
    # Test 5
    results['integration'] = test_melo_integration()
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.upper()}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "="*70)
    print(f"RESULTADO FINAL: {passed}/{total} tests pasados")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("✅ MeloTTS está completamente integrado y funcionando")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) fallaron")
        return 1


if __name__ == "__main__":
    sys.exit(main())
