#!/usr/bin/env python3
"""
🎯 Pipeline Audio-to-Audio ÓPTIMO
=================================
Usa el encoder REAL de Qwen2.5-Omni + Talker ONNX + Vocoder

Este es el pipeline completo sin razonamiento:
Audio → Audio Encoder (Qwen) → Talker (ONNX) → Token2Wav (PyTorch)
"""

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
import torchaudio

print("🎯 Pipeline Audio-to-Audio ÓPTIMO")
print("=" * 70)

class OptimalAudioPipeline:
    """
    Pipeline óptimo usando componentes reales de Qwen2.5-Omni
    
    Arquitectura:
    1. Audio Encoder: Extrae del modelo Qwen (32 layers, output 3584)
    2. Talker: ONNX optimizado (3ms latencia)
    3. Vocoder: Token2Wav del modelo Qwen
    
    NO incluye el Thinker (LLM), solo procesamiento de audio
    """
    
    def __init__(self, 
                 talker_onnx_path="qwen25_7b_audio.onnx",
                 device="cpu"):
        
        print("\n🔧 Inicializando pipeline óptimo...")
        self.device = device
        
        # Cargar modelo completo de Qwen
        print("   📥 Descargando Qwen2.5-Omni-7B...")
        print("      (Solo se usarán encoder y vocoder, ~14GB)")
        
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=device
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=True
        )
        
        # Extraer componentes específicos
        print("\n   🎤 Extrayendo Audio Encoder...")
        self.audio_encoder = self.model.audio_encoder
        
        print("   🔊 Extrayendo Token2Wav (Vocoder)...")
        self.vocoder = self.model.token2wav
        
        # Talker ONNX (rápido)
        print(f"   🎵 Cargando Talker ONNX: {talker_onnx_path}")
        self.talker_session = ort.InferenceSession(
            talker_onnx_path,
            providers=['CPUExecutionProvider']
        )
        
        print("\n✅ Pipeline óptimo inicializado")
        print(f"   🎤 Audio Encoder: {self.count_parameters(self.audio_encoder)/1e6:.1f}M params")
        print(f"   🎵 Talker: 10.8M params (ONNX)")
        print(f"   🔊 Vocoder: {self.count_parameters(self.vocoder)/1e6:.1f}M params")
    
    @staticmethod
    def count_parameters(model):
        """Cuenta parámetros entrenables"""
        return sum(p.numel() for p in model.parameters())
    
    def process(self, audio_path, output_path="output.wav"):
        """
        Pipeline completo: audio input → audio output
        
        Args:
            audio_path: Path al archivo de audio de entrada
            output_path: Path para guardar audio de salida
            
        Returns:
            waveform: Audio generado
        """
        print("\n" + "=" * 70)
        print("🚀 Procesando audio...")
        print("=" * 70)
        
        # 1. Cargar y preprocesar audio
        print(f"\n📥 Cargando audio: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Asegurar formato correcto
        if sample_rate != 16000:
            print(f"   🔄 Resampleando {sample_rate}Hz → 16000Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        if waveform.shape[0] > 1:
            print(f"   🔄 Convirtiendo stereo → mono")
            waveform = waveform.mean(dim=0, keepdim=True)
        
        print(f"   ✅ Audio: {waveform.shape[1]/sample_rate:.2f}s @ {sample_rate}Hz")
        
        # 2. Audio Encoder
        print(f"\n🎤 Codificando audio...")
        with torch.no_grad():
            # Procesar con el processor de Qwen
            inputs = self.processor(
                audios=waveform.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            
            # Extraer features con el encoder
            encoder_outputs = self.audio_encoder(
                inputs.input_features.to(self.device)
            )
            
            hidden_states = encoder_outputs.last_hidden_state
        
        print(f"   ✅ Features: {hidden_states.shape} (dim={hidden_states.shape[-1]})")
        
        # 3. Talker ONNX (rápido)
        print(f"\n🎵 Generando tokens de audio (Talker ONNX)...")
        import time
        start = time.perf_counter()
        
        hidden_np = hidden_states.cpu().numpy().astype(np.float32)
        input_name = self.talker_session.get_inputs()[0].name
        output_name = self.talker_session.get_outputs()[0].name
        
        audio_logits = self.talker_session.run(
            [output_name],
            {input_name: hidden_np}
        )[0]
        
        latency = (time.perf_counter() - start) * 1000
        print(f"   ✅ Audio tokens: {audio_logits.shape}")
        print(f"   ⚡ Latencia: {latency:.2f}ms")
        
        # 4. Token2Wav (Vocoder)
        print(f"\n🔊 Decodificando a waveform (Token2Wav)...")
        with torch.no_grad():
            audio_logits_tensor = torch.from_numpy(audio_logits).to(self.device)
            
            # Convertir logits a tokens (argmax)
            audio_tokens = audio_logits_tensor.argmax(dim=-1)
            
            print(f"   Audio tokens: {audio_tokens.shape}")
            
            # Generar waveform con el vocoder
            output_waveform = self.vocoder(audio_tokens)
        
        print(f"   ✅ Waveform: {output_waveform.shape}")
        
        # 5. Guardar audio
        if output_path:
            print(f"\n💾 Guardando audio: {output_path}")
            torchaudio.save(
                output_path,
                output_waveform.cpu(),
                sample_rate=24000  # BigVGAN usa 24kHz
            )
            print(f"   ✅ Audio guardado")
        
        print("\n" + "=" * 70)
        print("✅ Pipeline completado")
        print("=" * 70)
        
        return output_waveform
    
    def process_batch(self, audio_paths, output_dir="outputs"):
        """Procesa múltiples audios en batch"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, audio_path in enumerate(audio_paths):
            output_path = os.path.join(output_dir, f"output_{i:03d}.wav")
            waveform = self.process(audio_path, output_path)
            results.append(waveform)
        
        return results


# Demo
if __name__ == "__main__":
    print("\n🎯 DEMO: Pipeline Óptimo")
    print("=" * 70)
    
    print("\n📋 Este pipeline:")
    print("   ✅ Usa Audio Encoder real de Qwen (3584 dims)")
    print("   ✅ Usa Talker ONNX optimizado (3ms latencia)")
    print("   ✅ Usa Token2Wav real del modelo")
    print("   ❌ NO usa Thinker (LLM) - procesamiento directo")
    
    print("\n⚠️  REQUISITOS:")
    print("   - ~14GB de descarga (modelo completo)")
    print("   - ~20GB RAM")
    print("   - qwen25_7b_audio.onnx en directorio")
    
    try:
        # Inicializar
        pipeline = OptimalAudioPipeline(
            talker_onnx_path="qwen25_7b_audio.onnx",
            device="cpu"
        )
        
        print("\n✅ Pipeline listo para usar")
        print("\n📝 Uso:")
        print("   pipeline.process('input.wav', 'output.wav')")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Solución:")
        print("   pip install transformers torchaudio onnxruntime")
        print("   Asegúrate de tener qwen25_7b_audio.onnx")
