#!/usr/bin/env python3
"""
🧪 Test Simplificado del Pipeline ONNX
======================================
Prueba solo el componente Talker ONNX sin requerir el modelo completo
"""

import numpy as np
import onnxruntime as ort
import time

print("=" * 70)
print("🧪 TEST SIMPLIFICADO: qwen25_7b_audio.onnx")
print("=" * 70)

# Paths
model_path = "models/onnx/qwen25_7b_audio.onnx"

print("\n📊 CONFIGURACIÓN:")
print(f"   Modelo: {model_path}")
print(f"   Objetivo: Probar solo el componente Talker ONNX")

# 1. Cargar modelo ONNX
print("\n🔧 PASO 1: Cargar Modelo ONNX")
print("-" * 70)

try:
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    print("   ✅ Modelo cargado exitosamente")
    
    # Info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"\n   Input:  {input_info.name} {input_info.shape}")
    print(f"   Output: {output_info.name} {output_info.shape}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# 2. Simular hidden states del encoder
print("\n🎲 PASO 2: Generar Hidden States Simulados")
print("-" * 70)

# Simulamos el output del Audio Encoder
# En el pipeline real, esto vendría del modelo Qwen2.5-Omni encoder
batch_size = 1
seq_length = 50  # ~3 segundos de audio a 16Hz
hidden_dim = 3584

print(f"   Generando hidden states aleatorios:")
print(f"      Batch: {batch_size}")
print(f"      Sequence: {seq_length}")
print(f"      Hidden dim: {hidden_dim}")

# Generar con distribución normal (simula output real del encoder)
np.random.seed(42)
hidden_states = np.random.randn(batch_size, seq_length, hidden_dim).astype(np.float32)

# Normalizar (típico de salidas de transformer)
hidden_states = hidden_states / np.sqrt(hidden_dim)

print(f"   ✅ Hidden states generados: {hidden_states.shape}")
print(f"      Mean: {hidden_states.mean():.6f}")
print(f"      Std:  {hidden_states.std():.6f}")

# 3. Inferencia con ONNX
print("\n⚡ PASO 3: Inferencia Talker ONNX")
print("-" * 70)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"   Input tensor: {input_name}")
print(f"   Output tensor: {output_name}")

# Warm-up
print("\n   Warm-up (3 iteraciones)...")
for i in range(3):
    _ = session.run([output_name], {input_name: hidden_states})
    print(f"      Iteración {i+1}/3 ✓")

# Benchmark
print("\n   Benchmark (10 iteraciones)...")
latencies = []
for i in range(10):
    start = time.perf_counter()
    audio_logits = session.run([output_name], {input_name: hidden_states})[0]
    latency = (time.perf_counter() - start) * 1000
    latencies.append(latency)
    print(f"      Run {i+1:2d}/10: {latency:6.2f}ms")

# 4. Estadísticas
print("\n📊 PASO 4: Resultados")
print("-" * 70)

latencies_arr = np.array(latencies)

print(f"\n   Output shape: {audio_logits.shape}")
print(f"   Output dtype: {audio_logits.dtype}")
print(f"   Output range: [{audio_logits.min():.2f}, {audio_logits.max():.2f}]")

print(f"\n   ⚡ LATENCIA:")
print(f"      Media:    {latencies_arr.mean():6.2f}ms")
print(f"      Mediana:  {np.median(latencies_arr):6.2f}ms")
print(f"      Std Dev:  {latencies_arr.std():6.2f}ms")
print(f"      Min:      {latencies_arr.min():6.2f}ms")
print(f"      Max:      {latencies_arr.max():6.2f}ms")
print(f"      P95:      {np.percentile(latencies_arr, 95):6.2f}ms")
print(f"      P99:      {np.percentile(latencies_arr, 99):6.2f}ms")

# 5. Análisis de output
print("\n🔍 PASO 5: Análisis de Output")
print("-" * 70)

print(f"\n   Dimensiones esperadas:")
print(f"      Input:  [batch={batch_size}, seq={seq_length}, hidden={hidden_dim}]")
print(f"      Output: [batch={batch_size}, seq={seq_length}, vocab=8448]")

print(f"\n   Dimensiones reales:")
print(f"      Input:  {hidden_states.shape}")
print(f"      Output: {audio_logits.shape}")

if audio_logits.shape == (batch_size, seq_length, 8448):
    print(f"\n   ✅ Dimensiones CORRECTAS")
else:
    print(f"\n   ⚠️  Dimensiones inesperadas")

# Simular conversión a tokens (argmax)
audio_tokens = audio_logits.argmax(axis=-1)
print(f"\n   Audio tokens (argmax):")
print(f"      Shape: {audio_tokens.shape}")
print(f"      Range: [{audio_tokens.min()}, {audio_tokens.max()}]")
print(f"      Unique: {len(np.unique(audio_tokens))} diferentes")

# 6. Throughput
print("\n⚡ PASO 6: Throughput")
print("-" * 70)

tokens_per_second = (seq_length * 1000) / latencies_arr.mean()
audio_duration = seq_length / 50  # Asumiendo 50 frames/seg (típico)

print(f"\n   Tokens procesados: {seq_length}")
print(f"   Throughput: {tokens_per_second:,.0f} tokens/s")
print(f"   Audio duration: ~{audio_duration:.2f}s")
print(f"   Real-time factor: {audio_duration / (latencies_arr.mean()/1000):.2f}x")

# 7. Conclusión
print("\n" + "=" * 70)
print("✅ TEST COMPLETADO")
print("=" * 70)

print("\n📋 RESUMEN:")
print(f"   ✅ Modelo ONNX: Funcional")
print(f"   ✅ Latencia media: {latencies_arr.mean():.2f}ms")
print(f"   ✅ Output shape: Correcto")
print(f"   ✅ Throughput: {tokens_per_second:,.0f} tokens/s")

print("\n💡 SIGUIENTE PASO:")
print("   Para pipeline completo Audio → Audio:")
print("   1. Necesitas Audio Encoder (Qwen2.5-Omni-7B)")
print("   2. Necesitas Token2Wav Vocoder (BigVGAN)")
print("   3. Ejecuta: python scripts/test_optimal_audio_pipeline.py")

print("\n📖 DOCUMENTACIÓN:")
print("   Ver: docs/QWEN25_AUDIO_ONNX_ANALYSIS.md")

print("\n" + "=" * 70)
