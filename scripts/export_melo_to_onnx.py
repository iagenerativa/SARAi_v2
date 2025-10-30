#!/usr/bin/env python3
"""
Script para exportar MeloTTS a formato ONNX optimizado.

ONNX Runtime ofrece:
- Instrucciones AVX/AVX2/AVX512 en CPU
- Graph optimization automático
- Cuantización INT8 (opcional)
- Latencia 2-3x menor que PyTorch CPU

Uso:
    python3 scripts/export_melo_to_onnx.py
"""

import sys
import os
import torch
import onnx
from pathlib import Path
import time

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def export_melo_to_onnx(
    output_dir: str = "models/onnx",
    language: str = "ES",
    optimize: bool = True
):
    """
    Exporta MeloTTS a ONNX con optimizaciones.
    
    Args:
        output_dir: Directorio de salida para modelos ONNX
        language: Idioma del modelo (ES, EN, etc.)
        optimize: Si True, aplica optimizaciones ONNX
    """
    print("🚀 Exportando MeloTTS a ONNX...")
    print(f"   Idioma: {language}")
    print(f"   Optimización: {'Sí' if optimize else 'No'}\n")
    
    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Importar MeloTTS
    try:
        from melo.api import TTS
    except ImportError:
        print("❌ Error: MeloTTS no instalado")
        print("   Ejecuta: pip3 install git+https://github.com/myshell-ai/MeloTTS.git")
        return False
    
    # Cargar modelo MeloTTS
    print("📥 Cargando modelo MeloTTS...")
    start_load = time.perf_counter()
    
    device = 'cpu'
    tts_model = TTS(language=language, device=device)
    speaker_id = tts_model.hps.data.spk2id[language]
    
    load_time = (time.perf_counter() - start_load) * 1000
    print(f"✅ Modelo cargado en {load_time:.0f}ms\n")
    
    # Preparar el modelo para exportación
    # MeloTTS tiene dos componentes principales:
    # 1. Text encoder (BERT-like)
    # 2. Vocoder (HiFi-GAN)
    
    print("🔧 Extrayendo componentes del modelo...")
    
    # Obtener el modelo interno (SynthesizerTrn)
    synth_model = tts_model.model
    synth_model.eval()
    
    # Inputs de ejemplo para el trazado ONNX
    # text_seq: secuencia de tokens de texto
    # text_length: longitud de la secuencia
    # speaker_id: ID del hablante
    
    dummy_text_len = 20
    dummy_text = torch.randint(0, 100, (1, dummy_text_len), dtype=torch.long)
    dummy_text_lengths = torch.LongTensor([dummy_text_len])
    dummy_speaker_id = torch.LongTensor([speaker_id])
    dummy_scales = torch.FloatTensor([1.0, 1.0, 0.667])  # noise_scale, length_scale, speed
    
    print(f"   Entrada de ejemplo: text shape={dummy_text.shape}")
    
    # Exportar modelo completo
    onnx_path = os.path.join(output_dir, f"melo_tts_{language.lower()}.onnx")
    
    print(f"\n📤 Exportando a {onnx_path}...")
    start_export = time.perf_counter()
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                synth_model,
                (
                    dummy_text,           # x (text tokens)
                    dummy_text_lengths,   # x_lengths
                    dummy_speaker_id,     # sid (speaker ID)
                    dummy_scales[0],      # noise_scale
                    dummy_scales[1],      # length_scale
                    dummy_scales[2],      # noise_scale_w (speed)
                ),
                onnx_path,
                export_params=True,
                opset_version=14,  # ONNX opset 14 tiene mejor soporte
                do_constant_folding=True,
                input_names=['text', 'text_lengths', 'speaker_id', 'noise_scale', 'length_scale', 'speed'],
                output_names=['audio'],
                dynamic_axes={
                    'text': {0: 'batch_size', 1: 'text_length'},
                    'text_lengths': {0: 'batch_size'},
                    'audio': {0: 'batch_size', 2: 'audio_length'}
                }
            )
        
        export_time = (time.perf_counter() - start_export) * 1000
        print(f"✅ Exportación completada en {export_time:.0f}ms")
        
    except Exception as e:
        print(f"⚠️ No se pudo exportar el modelo completo: {e}")
        print("   Intentando exportación alternativa...")
        
        # Plan B: Exportar solo el vocoder (componente más pesado)
        try:
            export_vocoder_only(tts_model, output_dir, language)
        except Exception as e2:
            print(f"❌ Exportación alternativa también falló: {e2}")
            return False
    
    # Verificar modelo ONNX
    if os.path.exists(onnx_path):
        print(f"\n🔍 Verificando modelo ONNX...")
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"✅ Modelo válido")
            print(f"   Tamaño: {file_size:.1f} MB")
            
            # Optimizar si se solicita
            if optimize:
                optimize_onnx_model(onnx_path, output_dir, language)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Advertencia al verificar: {e}")
            return False
    
    return False


def export_vocoder_only(tts_model, output_dir: str, language: str):
    """
    Exporta solo el vocoder (HiFi-GAN) a ONNX.
    El vocoder es el componente más pesado computacionalmente.
    """
    print("\n🔧 Exportando solo el vocoder (HiFi-GAN)...")
    
    # Extraer vocoder
    vocoder = tts_model.model.dec
    vocoder.eval()
    
    # Input de ejemplo para el vocoder
    # El vocoder toma mel-spectrograms y genera audio
    dummy_mel = torch.randn(1, 80, 100)  # (batch, n_mels, time_steps)
    
    vocoder_path = os.path.join(output_dir, f"melo_vocoder_{language.lower()}.onnx")
    
    with torch.no_grad():
        torch.onnx.export(
            vocoder,
            dummy_mel,
            vocoder_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['mel_spectrogram'],
            output_names=['audio'],
            dynamic_axes={
                'mel_spectrogram': {0: 'batch_size', 2: 'time_steps'},
                'audio': {0: 'batch_size', 2: 'audio_length'}
            }
        )
    
    file_size = os.path.getsize(vocoder_path) / (1024 * 1024)
    print(f"✅ Vocoder exportado: {file_size:.1f} MB")
    return vocoder_path


def optimize_onnx_model(onnx_path: str, output_dir: str, language: str):
    """
    Optimiza el modelo ONNX para inferencia en CPU.
    """
    print("\n⚡ Optimizando modelo ONNX...")
    
    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
        
        # Configurar optimizaciones
        opt_options = FusionOptions('bert')  # MeloTTS usa BERT-like encoder
        opt_options.enable_gelu_approximation = True
        
        # Optimizar
        optimized_path = os.path.join(output_dir, f"melo_tts_{language.lower()}_optimized.onnx")
        
        optimization_options = optimizer.OptimizationOptions()
        optimization_options.enable_gelu_approximation = True
        
        optimizer_instance = optimizer.optimize_model(
            onnx_path,
            model_type='bert',
            num_heads=0,  # Auto-detect
            hidden_size=0,  # Auto-detect
            optimization_options=optimization_options
        )
        
        optimizer_instance.save_model_to_file(optimized_path)
        
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
        
        print(f"✅ Optimización completada")
        print(f"   Original: {original_size:.1f} MB")
        print(f"   Optimizado: {optimized_size:.1f} MB")
        print(f"   Reducción: {(1 - optimized_size/original_size)*100:.1f}%")
        
        return optimized_path
        
    except ImportError:
        print("⚠️ onnxruntime.transformers no disponible")
        print("   Instalando: pip3 install onnxruntime-tools")
        return None
    except Exception as e:
        print(f"⚠️ No se pudo optimizar: {e}")
        return None


def test_onnx_inference(onnx_path: str, language: str = "ES"):
    """
    Test de inferencia con ONNX Runtime.
    """
    print("\n🧪 Probando inferencia ONNX Runtime...")
    
    import onnxruntime as ort
    import numpy as np
    
    # Crear sesión ONNX Runtime
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = os.cpu_count()
    
    session = ort.InferenceSession(onnx_path, sess_options)
    
    # Preparar inputs de prueba
    dummy_text = np.random.randint(0, 100, (1, 20), dtype=np.int64)
    dummy_text_lengths = np.array([20], dtype=np.int64)
    dummy_speaker_id = np.array([0], dtype=np.int64)
    dummy_noise_scale = np.array([1.0], dtype=np.float32)
    dummy_length_scale = np.array([1.0], dtype=np.float32)
    dummy_speed = np.array([0.667], dtype=np.float32)
    
    inputs = {
        'text': dummy_text,
        'text_lengths': dummy_text_lengths,
        'speaker_id': dummy_speaker_id,
        'noise_scale': dummy_noise_scale,
        'length_scale': dummy_length_scale,
        'speed': dummy_speed
    }
    
    # Warmup
    for _ in range(3):
        _ = session.run(None, inputs)
    
    # Benchmark
    times = []
    for _ in range(10):
        start = time.perf_counter()
        outputs = session.run(None, inputs)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"✅ Inferencia ONNX Runtime (10 iteraciones):")
    print(f"   Media: {avg_time:.0f}ms")
    print(f"   Min: {min_time:.0f}ms")
    print(f"   Max: {max_time:.0f}ms")
    
    return avg_time


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Exportar MeloTTS a ONNX")
    parser.add_argument("--language", default="ES", help="Idioma del modelo (ES, EN, etc.)")
    parser.add_argument("--output-dir", default="models/onnx", help="Directorio de salida")
    parser.add_argument("--no-optimize", action="store_true", help="No optimizar el modelo")
    parser.add_argument("--test", action="store_true", help="Probar inferencia después de exportar")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 MeloTTS → ONNX Exporter")
    print("=" * 60 + "\n")
    
    success = export_melo_to_onnx(
        output_dir=args.output_dir,
        language=args.language,
        optimize=not args.no_optimize
    )
    
    if success and args.test:
        # Intentar test de inferencia
        onnx_path = os.path.join(args.output_dir, f"melo_tts_{args.language.lower()}.onnx")
        if os.path.exists(onnx_path):
            try:
                test_onnx_inference(onnx_path, args.language)
            except Exception as e:
                print(f"⚠️ No se pudo probar inferencia: {e}")
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Exportación completada exitosamente")
        print(f"   Los modelos ONNX están en: {args.output_dir}/")
    else:
        print("⚠️ Exportación completada con advertencias")
    print("=" * 60)
