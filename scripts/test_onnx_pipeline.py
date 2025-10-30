#!/usr/bin/env python3
"""
Test Suite para Qwen3-Omni-3B ONNX Pipeline
Validación completa del modelo real agi_audio_core.onnx
"""

import os
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from agents.audio_omni_pipeline import get_audio_omni_pipeline, AudioOmniConfig
    import yaml
except ImportError as e:
    print(f"❌ Error de import: {e}")
    print("Instalar dependencias: pip install onnxruntime librosa soundfile pyyaml")
    sys.exit(1)


def create_test_audio(filename: str = "test_sine.wav", duration: float = 1.0):
    """Crear archivo de audio sintético para testing"""
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mezcla de tonos (más realista que sine puro)
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +     # La (440Hz)
        0.2 * np.sin(2 * np.pi * 880 * t) +     # La octava alta
        0.1 * np.sin(2 * np.pi * 220 * t)       # La octava baja
    )
    
    # Aplicar envelope para evitar clicks
    envelope = np.exp(-t / duration)
    audio = audio * envelope
    
    sf.write(filename, audio, sample_rate)
    return filename, len(audio), sample_rate


def test_model_loading():
    """Test 1: Verificar carga del modelo ONNX"""
    print("🧪 Test 1: Carga del modelo...")
    
    start_time = time.time()
    try:
        pipeline = get_audio_omni_pipeline()
        load_time = time.time() - start_time
        
        print(f"✅ Modelo cargado en {load_time:.2f}s")
        print(f"   Config: {pipeline.config.model_path}")
        print(f"   RAM budget: {pipeline.config.max_memory_mb} MB")
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False


def test_model_inference():
    """Test 2: Inferencia con audio sintético"""
    print("\n🧪 Test 2: Inferencia ONNX...")
    
    # Crear audio de prueba
    audio_file, samples, sr = create_test_audio()
    print(f"   Audio creado: {samples} samples @ {sr} Hz")
    
    try:
        pipeline = get_audio_omni_pipeline()
        
        # Leer audio como bytes
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        print(f"   Audio bytes: {len(audio_bytes)} bytes")
        
        # Procesar
        start_time = time.time()
        result = pipeline.process_audio(audio_bytes)
        inference_time = time.time() - start_time
        
        print(f"✅ Inferencia exitosa en {inference_time:.3f}s")
        
        # Validar outputs
        assert "mel_features" in result
        assert "audio_codes" in result  
        assert "text" in result
        assert "metadata" in result
        
        mel_shape = result["mel_features"].shape
        codes_shape = result["audio_codes"].shape
        
        print(f"   Mel features: {mel_shape} ({result['mel_features'].dtype})")
        print(f"   Audio codes: {codes_shape} ({result['audio_codes'].dtype})")
        print(f"   Text output: {result['text']}")
        print(f"   Metadata: {result['metadata']}")
        
        # Validar shapes esperadas
        assert mel_shape == (1, 2048, 245760), f"Shape incorrecta: {mel_shape}"
        assert codes_shape == (1, 16, 128), f"Shape incorrecta: {codes_shape}"
        
        return True
        
    except Exception as e:
        print(f"❌ Error en inferencia: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Limpiar archivo temporal
        if os.path.exists(audio_file):
            os.remove(audio_file)


def test_config_loading():
    """Test 3: Carga de configuración desde YAML"""
    print("\n🧪 Test 3: Configuración YAML...")
    
    try:
        config_path = Path(__file__).parent.parent / "config" / "sarai.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        config = AudioOmniConfig.from_yaml(yaml_config)
        
        print(f"✅ Config cargada desde YAML")
        print(f"   Model path: {config.model_path}")
        print(f"   Max memory: {config.max_memory_mb} MB")
        print(f"   Sample rate: {config.sample_rate} Hz")
        print(f"   Threads: {config.n_threads}")
        
        # Validar valores esperados para modelo INT8
        assert "agi_audio_core_int8.onnx" in config.model_path, f"Modelo incorrecto: {config.model_path}"
        assert config.max_memory_mb == 1200, f"Max memory incorrecto: {config.max_memory_mb} (esperado 1200)"
        assert config.sample_rate == 22050, f"Sample rate incorrecto: {config.sample_rate}"
        
        return True
        
    except Exception as e:
        print(f"❌ Error en config: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_validation():
    """Test 4: Validar archivos del modelo"""
    print("\n🧪 Test 4: Validación de archivos...")
    
    try:
        model_path = "models/onnx/agi_audio_core.onnx"
        data_path = model_path + ".data"
        
        # Verificar archivos existen
        assert os.path.exists(model_path), f"Archivo faltante: {model_path}"
        assert os.path.exists(data_path), f"Archivo faltante: {data_path}"
        
        # Verificar tamaños
        model_size = os.path.getsize(model_path)
        data_size = os.path.getsize(data_path)
        
        print(f"✅ Archivos validados:")
        print(f"   Modelo: {model_size:,} bytes (~{model_size/1024:.1f} KB)")
        print(f"   Datos: {data_size:,} bytes (~{data_size/(1024**3):.1f} GB)")
        
        # Validar tamaños esperados
        assert model_size > 5000, f"Modelo muy pequeño: {model_size}"  # >5KB
        assert data_size > 4e9, f"Datos muy pequeños: {data_size}"     # >4GB
        
        return True
        
    except Exception as e:
        print(f"❌ Error validando archivos: {e}")
        return False


def benchmark_performance():
    """Benchmark: Medir latencia y throughput"""
    print("\n📊 Benchmark de performance...")
    
    try:
        pipeline = get_audio_omni_pipeline()
        
        # Test con diferentes duraciones de audio
        durations = [0.5, 1.0, 2.0, 3.0]  # segundos
        results = []
        
        for duration in durations:
            print(f"   Testing audio {duration}s...")
            
            # Crear audio de prueba
            audio_file, samples, sr = create_test_audio(
                filename=f"bench_{duration}s.wav", 
                duration=duration
            )
            
            try:
                with open(audio_file, 'rb') as f:
                    audio_bytes = f.read()
                
                # Múltiples runs para promediar
                times = []
                for _ in range(3):
                    start_time = time.time()
                    result = pipeline.process_audio(audio_bytes)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                throughput = duration / avg_time  # ratio real-time
                
                results.append({
                    'duration': duration,
                    'avg_time': avg_time,
                    'throughput': throughput,
                    'samples': samples
                })
                
                print(f"     Latencia: {avg_time:.3f}s (throughput: {throughput:.2f}x)")
                
            finally:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
        
        print(f"\n📈 Resumen benchmark:")
        for r in results:
            print(f"   {r['duration']}s audio: {r['avg_time']:.3f}s latencia, {r['throughput']:.2f}x throughput")
        
        # Calcular métricas agregadas
        avg_latency = np.mean([r['avg_time'] for r in results])
        avg_throughput = np.mean([r['throughput'] for r in results])
        
        print(f"\n🎯 Métricas finales:")
        print(f"   Latencia promedio: {avg_latency:.3f}s")
        print(f"   Throughput promedio: {avg_throughput:.2f}x")
        
        # Validar contra objetivos
        target_latency = 0.350  # 350ms objetivo
        if avg_latency <= target_latency:
            print(f"   ✅ Latencia OK (≤{target_latency}s)")
        else:
            print(f"   ⚠️  Latencia alta (>{target_latency}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en benchmark: {e}")
        return False


def main():
    """Ejecutar suite completa de tests"""
    print("🚀 SARAi v2.16.1 - Test Suite ONNX")
    print("=" * 50)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Model Inference", test_model_inference), 
        ("Config Loading", test_config_loading),
        ("File Validation", test_file_validation),
        ("Performance Benchmark", benchmark_performance)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🔍 Ejecutando: {name}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"💥 Test falló: {e}")
            results.append((name, False))
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE TESTS:")
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Score: {passed}/{len(tests)} tests pasaron")
    
    if passed == len(tests):
        print("🏆 TODOS LOS TESTS PASARON - MODELO ONNX LISTO")
        return 0
    else:
        print("⚠️  ALGUNOS TESTS FALLARON - REVISAR ERRORES")
        return 1


if __name__ == "__main__":
    sys.exit(main())