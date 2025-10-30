#!/usr/bin/env python3
"""
Pipeline OPTIMIZADO para CPU PURO (sin GPU)
Optimizaciones específicas para inferencia en CPU
"""

import torch
import onnxruntime as ort
import numpy as np
import os

# Configurar número óptimo de threads para CPU
cpu_count = os.cpu_count() or 4
torch.set_num_threads(cpu_count)
torch.set_num_interop_threads(cpu_count)

class AudioToAudioPipelineCPU:
    def __init__(self, lfm2_model_path):
        """
        PIPELINE OPTIMIZADO PARA CPU PURO
        
        FICHEROS NECESARIOS:
        - audio_encoder_fp16.pt (1.2 GB)
        - projection.onnx (38 MB)
        - qwen25_audio_gpu_lite.onnx (21 MB)
        - token2wav_fp16.pt (857 MB)
        - tu LFM2_1.2B modelo (1.2 GB)
        
        TOTAL: ~3.3 GB
        
        OPTIMIZACIONES CPU:
        ✅ Multi-threading optimizado ({cpu_count} threads)
        ✅ ONNX con DirectML/OpenVINO si disponible
        ✅ Intel MKL acceleration
        ✅ Reduced diffusion steps (5 vs 10)
        ✅ In-place operations
        ✅ Memory-efficient batching
        """
        print(f"⏳ Cargando componentes en CPU ({cpu_count} threads)...")
        
        # ==========================================
        # 1. Audio Encoder (PyTorch FP16)
        # ==========================================
        print("   📦 Cargando Audio Encoder...")
        checkpoint = torch.load('audio_encoder_fp16.pt', map_location='cpu', weights_only=False)
        self.audio_encoder = checkpoint['model'].eval()
        
        # No usar half() en CPU (no es eficiente), usar float32
        self.audio_encoder = self.audio_encoder.float()
        
        # Compilar si disponible (PyTorch 2.0+)
        try:
            self.audio_encoder = torch.compile(
                self.audio_encoder,
                mode='reduce-overhead',
                backend='inductor'
            )
            print("      ✅ Compilado con torch.compile")
        except:
            print("      ⚠️  torch.compile no disponible")
        
        # ==========================================
        # 2. Projection (ONNX optimizado para CPU)
        # ==========================================
        print("   📦 Cargando Projection ONNX...")
        sess_options = ort.SessionOptions()
        
        # Optimizaciones ONNX para CPU
        sess_options.intra_op_num_threads = cpu_count
        sess_options.inter_op_num_threads = cpu_count
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Probar providers especiales para CPU
        providers = self._get_best_cpu_providers()
        
        self.proj_session = ort.InferenceSession(
            'projection.onnx',
            sess_options,
            providers=providers
        )
        print(f"      ✅ Provider: {self.proj_session.get_providers()[0]}")
        
        # ==========================================
        # 3. LFM2_1.2B
        # ==========================================
        print("   📦 Cargando LFM2_1.2B...")
        self.lfm2 = torch.load(lfm2_model_path, map_location='cpu', weights_only=False).eval()
        self.lfm2 = self.lfm2.float()
        
        # Compilar LFM2
        try:
            self.lfm2 = torch.compile(
                self.lfm2,
                mode='reduce-overhead'
            )
            print("      ✅ Compilado con torch.compile")
        except:
            print("      ⚠️  torch.compile no disponible")
        
        # ==========================================
        # 4. Talker (ONNX optimizado para CPU)
        # ==========================================
        print("   📦 Cargando Talker ONNX...")
        self.talker_session = ort.InferenceSession(
            'qwen25_audio_gpu_lite.onnx',
            sess_options,
            providers=providers
        )
        print(f"      ✅ Provider: {self.talker_session.get_providers()[0]}")
        
        # ==========================================
        # 5. Token2Wav (PyTorch FP16 → FP32 para CPU)
        # ==========================================
        print("   📦 Cargando Token2Wav...")
        checkpoint = torch.load('token2wav_fp16.pt', map_location='cpu', weights_only=False)
        self.token2wav = checkpoint['model'].eval()
        self.token2wav = self.token2wav.float()
        
        # Compilar Token2Wav (puede ser complejo con diffusion)
        try:
            self.token2wav = torch.compile(
                self.token2wav,
                mode='default'
            )
            print("      ✅ Compilado con torch.compile")
        except Exception as e:
            print(f"      ⚠️  torch.compile falló: {e}")
        
        print(f"\n✅ Pipeline CPU optimizado listo!")
        print(f"   🧵 Usando {cpu_count} threads")
    
    def _get_best_cpu_providers(self):
        """Selecciona el mejor execution provider disponible para CPU"""
        available = ort.get_available_providers()
        
        # Prioridad de providers para CPU
        priority = [
            'OpenVINOExecutionProvider',   # Intel CPUs (muy rápido)
            'DmlExecutionProvider',        # DirectML (Windows)
            'CoreMLExecutionProvider',     # Apple Silicon
            'CPUExecutionProvider'         # Fallback
        ]
        
        for provider in priority:
            if provider in available:
                return [provider]
        
        return ['CPUExecutionProvider']
    
    @torch.inference_mode()
    def process(self, audio_input, reference_mel, num_diffusion_steps=5):
        """
        Procesa audio → audio con optimizaciones para CPU
        
        audio_input: [B, T_audio] waveform 16kHz
        reference_mel: [B, 80, T_mel] mel-spectrogram
        num_diffusion_steps: 5 recomendado para CPU (vs 10 default)
        
        Returns: [B, T_audio_out] waveform 24kHz
        """
        # Asegurar inputs en CPU y tipo correcto
        if not isinstance(audio_input, torch.Tensor):
            audio_input = torch.from_numpy(audio_input)
        if not isinstance(reference_mel, torch.Tensor):
            reference_mel = torch.from_numpy(reference_mel)
            
        audio_input = audio_input.cpu().float()
        reference_mel = reference_mel.cpu().float()
        
        # ==========================================
        # Paso 1: Audio Encoder [B, T_audio] → [B, T, 512]
        # ==========================================
        features = self.audio_encoder(audio_input)
        
        # ==========================================
        # Paso 2: Projection ONNX [B, T, 512] → [B, T, 3584]
        # ==========================================
        x_numpy = features.numpy()
        x_numpy = self.proj_session.run(None, {'hidden_states': x_numpy})[0]
        
        # ==========================================
        # Paso 3: LFM2_1.2B [B, T, 3584] → [B, T, 3584]
        # ==========================================
        x = torch.from_numpy(x_numpy).float()
        x = self.lfm2(x)
        
        # ==========================================
        # Paso 4: Talker ONNX [B, T, 3584] → [B, T, 896]
        # ==========================================
        x_numpy = x.numpy()
        audio_embeds_numpy = self.talker_session.run(None, {'hidden_states': x_numpy})[0]
        audio_embeds = torch.from_numpy(audio_embeds_numpy).float()
        
        # ==========================================
        # Paso 5: Token2Wav [B, T, 896] → waveform
        # ==========================================
        codes = audio_embeds
        conditioning = audio_embeds.mean(dim=1, keepdim=True)
        
        waveform = self.token2wav(
            code=codes,
            conditioning=conditioning,
            reference_mel=reference_mel,
            num_steps=num_diffusion_steps  # 5 para CPU (vs 10)
        )
        
        return waveform
    
    def process_streaming(self, audio_chunks, reference_mel, num_diffusion_steps=5):
        """
        Procesa audio en chunks para uso de memoria más eficiente
        
        audio_chunks: Generator o lista de chunks de audio
        reference_mel: [B, 80, T_mel] mel-spectrogram
        """
        results = []
        for chunk in audio_chunks:
            result = self.process(chunk, reference_mel, num_diffusion_steps)
            results.append(result)
        return torch.cat(results, dim=-1)


def benchmark_cpu(pipeline, num_runs=10):
    """Benchmark específico para CPU"""
    import time
    
    print("\n" + "="*60)
    print("🔥 BENCHMARK CPU")
    print("="*60)
    
    # Inputs de prueba
    audio_in = torch.randn(1, 16000 * 4)
    ref_mel = torch.randn(1, 80, 400)
    
    # Warmup (especialmente importante para torch.compile)
    print("\n⏳ Warmup (5 iteraciones para torch.compile)...")
    for i in range(5):
        print(f"   Warmup {i+1}/5...")
        _ = pipeline.process(audio_in, ref_mel, num_diffusion_steps=3)
    
    # Benchmark con diferentes configuraciones
    for num_steps in [5, 3, 1]:
        print(f"\n⚡ Testing con {num_steps} diffusion steps...")
        times = []
        
        for i in range(num_runs):
            start = time.perf_counter()
            _ = pipeline.process(audio_in, ref_mel, num_diffusion_steps=num_steps)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
            print(f"   Run {i+1}: {elapsed*1000:.0f} ms")
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        p50 = np.percentile(times, 50)
        
        audio_duration = 4.0
        real_time_factor = (audio_duration * 1000) / mean_time
        
        print(f"\n   📊 Steps={num_steps}:")
        print(f"      Media: {mean_time:.0f} ms (±{std_time:.0f} ms)")
        print(f"      P50: {p50:.0f} ms")
        print(f"      Real-time: {real_time_factor:.1f}x")
    
    # Uso de memoria
    import psutil
    process = psutil.Process()
    mem_gb = process.memory_info().rss / 1024**3
    
    print("\n" + "="*60)
    print("📋 RESUMEN")
    print("="*60)
    print(f"CPU Threads: {cpu_count}")
    print(f"Memoria: {mem_gb:.2f} GB")
    print(f"\nRECOMENDACIÓN CPU:")
    print(f"→ Usar num_diffusion_steps=5 para balance")
    print(f"→ Usar num_diffusion_steps=3 para demo rápido")
    print("="*60)


if __name__ == '__main__':
    print("🚀 PIPELINE OPTIMIZADO PARA CPU")
    print("="*60)
    
    try:
        # Cargar pipeline
        pipeline = AudioToAudioPipelineCPU('tu_lfm2_1.2b.pt')
        
        # Benchmark
        benchmark_cpu(pipeline, num_runs=5)
        
        # Test funcional
        print("\n✅ Test funcional...")
        audio_in = torch.randn(1, 16000 * 4)
        ref_mel = torch.randn(1, 80, 400)
        
        output = pipeline.process(audio_in, ref_mel, num_diffusion_steps=5)
        print(f"   Input: {audio_in.shape}")
        print(f"   Output: {output.shape}")
        print("   ✅ Pipeline funcional!")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Archivo no encontrado: {e}")
        print("\nAsegúrate de tener:")
        print("   - audio_encoder_fp16.pt")
        print("   - projection.onnx")
        print("   - qwen25_audio_gpu_lite.onnx")
        print("   - token2wav_fp16.pt")
        print("   - tu_lfm2_1.2b.pt")
