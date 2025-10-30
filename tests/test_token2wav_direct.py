#!/usr/bin/env python3
"""
Test directo de Token2Wav para verificar parámetros correctos
"""

import torch
import numpy as np
from pathlib import Path

print("🔧 Test Token2Wav Directo\n")

# 1. Cargar modelo
print("[1/3] Cargando Token2Wav (FP16)...")
checkpoint = torch.load(
    'models/onnx/token2wav_fp16.pt',  # ← Usar FP16 por ahora
    map_location='cpu',
    weights_only=False
)

model = checkpoint['model']
model.eval()
print("✓ Modelo cargado\n")

# 2. Preparar datos de prueba
print("[2/3] Preparando datos de prueba...")
batch_size = 1
seq_len = 100
code_dim = 8448  # Output de Talker qwen25_7b
mel_dim = 80  # Dimensión estándar de mel-spectrogram
conditioning_dim = 768  # Dimensión típica de conditioning vector

# Simular audio_logits del Talker (en FP16 para coincidir con el modelo)
code = torch.randn(batch_size, seq_len, code_dim, dtype=torch.float16) * 0.1

# Crear mel-spectrogram de referencia dummy (FP16)
reference_mel = torch.randn(batch_size, 30000, mel_dim, dtype=torch.float16) * 0.1

# Crear conditioning vector dummy (FP16)
conditioning = torch.randn(batch_size, conditioning_dim, dtype=torch.float16) * 0.1

print(f"✓ Code shape: {code.shape} dtype: {code.dtype}")
print(f"✓ Reference mel shape: {reference_mel.shape} dtype: {reference_mel.dtype}")
print(f"✓ Conditioning shape: {conditioning.shape} dtype: {conditioning.dtype}\n")

# 3. Generar audio
print("[3/3] Generando audio...")
print("   Parámetros:")
print("   • code: [1, 100, 8448]")
print(f"   • conditioning: [1, {conditioning_dim}]")
print(f"   • reference_mel: [1, 30000, {mel_dim}]")
print("   • num_steps: 3")
print("   • guidance_scale: 1.0")

try:
    import time
    start = time.perf_counter()
    
    with torch.no_grad():
        waveform = model(
            code=code,
            conditioning=conditioning,  # ← Ahora con conditioning real
            reference_mel=reference_mel,
            num_steps=3,
            guidance_scale=1.0
        )
    
    latency = (time.perf_counter() - start) * 1000
    
    print(f"\n✅ ÉXITO")
    print(f"   Latencia: {latency:.1f}ms")
    print(f"   Output shape: {waveform.shape}")
    print(f"   Output dtype: {waveform.dtype}")
    print(f"   Output range: [{waveform.min():.3f}, {waveform.max():.3f}]")
    
    # Guardar audio de prueba
    import scipy.io.wavfile as wavfile
    
    audio_array = waveform.cpu().numpy()
    if len(audio_array.shape) > 1:
        audio_array = audio_array[0]
    
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    output_file = "logs/test_token2wav_output.wav"
    Path("logs").mkdir(exist_ok=True)
    wavfile.write(output_file, 24000, audio_int16)
    
    print(f"\n🔊 Audio guardado: {output_file}")
    print(f"   Duración: {len(audio_int16)/24000:.2f}s")
    print(f"   Sample rate: 24kHz")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
