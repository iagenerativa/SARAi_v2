#!/usr/bin/env python3
"""
Test de Voz con Pipeline Completo ONNX + PyTorch

Usa la arquitectura completa:
  Audio → Encoder → Projection → LFM2 → Talker → Token2Wav → Audio

Objetivo: Medir latencias reales del pipeline end-to-end
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyaudio
    import numpy as np
    import torch
    import onnxruntime as ort
except ImportError as e:
    print(f"❌ Dependencias faltantes: {e}")
    print("   Instalar con: pip install pyaudio numpy torch onnxruntime")
    sys.exit(1)


class CompletePipelineVoiceTest:
    """Test con pipeline completo de audio"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.format = pyaudio.paInt16
        self.chunk = 1024
        
        self.audio = pyaudio.PyAudio()
        
        # Componentes del pipeline
        self.audio_encoder = None
        self.processor = None  # AudioProcessor necesario
        self.proj_session = None
        self.talker_session = None
        self.token2wav = None
        
        # Cargar pipeline completo
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Carga todos los componentes del pipeline"""
        base_path = Path("models/onnx")
        
        print("🔧 Cargando Pipeline Completo...")
        print("="*60)
        
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(cpu_count)
        
        # ==========================================
        # 1. Audio Encoder (PyTorch)
        # ==========================================
        encoder_path = base_path / "audio_encoder_int8.pt"  # Usar INT8 para velocidad
        if not encoder_path.exists():
            encoder_path = base_path / "audio_encoder_fp16.pt"
        
        if encoder_path.exists():
            print(f"📦 1/5 Cargando Audio Encoder + Processor ({encoder_path.name}, {encoder_path.stat().st_size / 1024**2:.0f}MB)...")
            print(f"    ⏳ Esto puede tomar 10-30 segundos...")
            start = time.time()
            
            try:
                # Cargar processor (necesario para preparar audio)
                from transformers import AutoProcessor
                print(f"    📦 Cargando AudioProcessor...")
                self.processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2.5-Omni-7B",
                    cache_dir="models/cache"
                )
                
                checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    self.audio_encoder = checkpoint['model'].eval()
                else:
                    self.audio_encoder = checkpoint.eval()
                
                self.audio_encoder = self.audio_encoder.float()
                
                load_time = (time.time() - start) * 1000
                print(f"    ✅ Cargado en {load_time/1000:.1f}s")
            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            print(f"    ❌ No encontrado: {encoder_path}")
            return
        
        # ==========================================
        # 2. Projection (ONNX)
        # ==========================================
        proj_path = base_path / "projection.onnx"
        if proj_path.exists():
            print(f"📦 2/5 Cargando Projection ONNX...")
            start = time.time()
            
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = cpu_count
            sess_options.inter_op_num_threads = cpu_count
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.proj_session = ort.InferenceSession(
                str(proj_path),
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
            load_time = (time.time() - start) * 1000
            print(f"    ✅ Cargado en {load_time:.0f}ms")
        else:
            print(f"    ❌ No encontrado: {proj_path}")
            return
        
        # ==========================================
        # 3. Talker (ONNX)
        # ==========================================
        talker_path = base_path / "qwen25_audio_gpu_lite.onnx"
        if talker_path.exists():
            print(f"📦 3/5 Cargando Talker ONNX...")
            start = time.time()
            
            self.talker_session = ort.InferenceSession(
                str(talker_path),
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
            load_time = (time.time() - start) * 1000
            print(f"    ✅ Cargado en {load_time:.0f}ms")
        else:
            print(f"    ❌ No encontrado: {talker_path}")
            return
        
        # ==========================================
        # 4. Token2Wav (PyTorch)
        # ==========================================
        t2w_path = base_path / "token2wav_int8.pt"
        if not t2w_path.exists():
            t2w_path = base_path / "token2wav_fp16.pt"
        
        if t2w_path.exists():
            print(f"📦 4/5 Cargando Token2Wav ({t2w_path.name}, {t2w_path.stat().st_size / 1024**2:.0f}MB)...")
            print(f"    ⏳ Esto puede tomar 10-30 segundos...")
            start = time.time()
            
            try:
                checkpoint = torch.load(t2w_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    self.token2wav = checkpoint['model'].eval()
                else:
                    self.token2wav = checkpoint.eval()
                
                self.token2wav = self.token2wav.float()
                
                load_time = (time.time() - start) * 1000
                print(f"    ✅ Cargado en {load_time/1000:.1f}s")
            except Exception as e:
                print(f"    ❌ Error: {e}")
                return
        else:
            print(f"    ❌ No encontrado: {t2w_path}")
            return
        
        print("="*60)
        print(f"✅ Pipeline completo cargado ({cpu_count} threads CPU)")
        print()
    
    def is_loaded(self):
        """Verifica si el pipeline está completamente cargado"""
        return all([
            self.audio_encoder is not None,
            self.processor is not None,
            self.proj_session is not None,
            self.talker_session is not None,
            self.token2wav is not None
        ])
    
    def record_audio(self, duration=5):
        """Graba audio del micrófono"""
        print(f"🎤 Grabando {duration}s...")
        
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        frames = []
        for _ in range(int(self.sample_rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        audio_bytes = b''.join(frames)
        print(f"✅ Grabación completa ({len(audio_bytes)} bytes)")
        
        return audio_bytes
    
    @torch.inference_mode()
    def process_pipeline(self, audio_bytes):
        """
        Procesa audio completo: Audio → hidden → LLM → audio
        
        Returns:
            {
                "success": bool,
                "audio_output": np.ndarray o None,
                "encoder_ms": float,
                "projection_ms": float,
                "talker_ms": float,
                "token2wav_ms": float,
                "total_ms": float,
                "error": str o None
            }
        """
        if not self.is_loaded():
            return {
                "success": False,
                "error": "Pipeline no está completamente cargado",
                "total_ms": 0
            }
        
        try:
            total_start = time.time()
            
            # Convertir bytes raw a waveform
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            waveform = audio_array.astype(np.float32) / 32768.0
            
            print(f"\n🔄 Procesando pipeline completo...")
            print(f"   Audio: {len(waveform)} samples @ 16kHz")
            
            # ==========================================
            # Paso 1: Audio Encoder con Processor
            # ==========================================
            print(f"   1/4 Audio Encoder + Processor...")
            step_start = time.time()
            
            # Usar processor para preparar el audio
            inputs = self.processor(
                audios=waveform,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            # Pasar por el encoder
            encoder_outputs = self.audio_encoder(inputs.input_features)
            features = encoder_outputs.last_hidden_state  # [1, T', D]
            
            encoder_ms = (time.time() - step_start) * 1000
            print(f"       ✅ {encoder_ms:.0f}ms → shape {features.shape}")
            
            # ==========================================
            # Paso 2: Projection ONNX
            # ==========================================
            print(f"   2/4 Projection ONNX...")
            step_start = time.time()
            
            features_np = features.numpy()
            input_name = self.proj_session.get_inputs()[0].name
            hidden_states = self.proj_session.run(None, {input_name: features_np})[0]
            
            projection_ms = (time.time() - step_start) * 1000
            print(f"       ✅ {projection_ms:.0f}ms → shape {hidden_states.shape}")
            
            # ==========================================
            # Paso 3: Talker ONNX
            # ==========================================
            print(f"   3/4 Talker ONNX...")
            step_start = time.time()
            
            input_name = self.talker_session.get_inputs()[0].name
            audio_embeds = self.talker_session.run(None, {input_name: hidden_states})[0]
            
            talker_ms = (time.time() - step_start) * 1000
            print(f"       ✅ {talker_ms:.0f}ms → shape {audio_embeds.shape}")
            
            # ==========================================
            # Paso 4: Token2Wav
            # ==========================================
            print(f"   4/4 Token2Wav...")
            step_start = time.time()
            
            codes = torch.from_numpy(audio_embeds)
            conditioning = codes.mean(dim=1, keepdim=True)
            
            # Generar mel de referencia (simplificado)
            ref_mel = torch.randn(1, 80, 400)
            
            # Token2Wav con diffusion steps reducidos para velocidad
            audio_output = self.token2wav(
                code=codes,
                conditioning=conditioning,
                reference_mel=ref_mel,
                num_steps=3  # Reducido para velocidad en CPU
            )
            
            token2wav_ms = (time.time() - step_start) * 1000
            print(f"       ✅ {token2wav_ms:.0f}ms → shape {audio_output.shape}")
            
            total_ms = (time.time() - total_start) * 1000
            
            return {
                "success": True,
                "audio_output": audio_output.numpy(),
                "encoder_ms": encoder_ms,
                "projection_ms": projection_ms,
                "talker_ms": talker_ms,
                "token2wav_ms": token2wav_ms,
                "total_ms": total_ms,
                "error": None
            }
            
        except Exception as e:
            print(f"❌ Error en pipeline: {e}")
            import traceback
            traceback.print_exc()
            
            total_ms = (time.time() - total_start) * 1000
            return {
                "success": False,
                "error": str(e),
                "total_ms": total_ms
            }
    
    def play_audio(self, audio_data, sample_rate=24000):
        """Reproduce audio generado"""
        try:
            # Convertir a int16 para reproducción
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            print(f"\n🔊 Reproduciendo audio generado ({sample_rate}Hz)...")
            
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )
            
            stream.write(audio_int16.tobytes())
            stream.stop_stream()
            stream.close()
            
            print(f"✅ Reproducción completa")
            
        except Exception as e:
            print(f"⚠️  Error reproduciendo audio: {e}")
    
    def run_test(self, turns=3):
        """Ejecuta test completo"""
        print(f"\n{'='*60}")
        print(f"🎙️  TEST PIPELINE COMPLETO")
        print(f"{'='*60}\n")
        
        if not self.is_loaded():
            print(f"❌ No se puede ejecutar: Pipeline incompleto")
            return
        
        results = []
        
        for turn in range(1, turns + 1):
            print(f"\n{'='*60}")
            print(f"TURNO {turn}/{turns}")
            print(f"{'='*60}")
            
            # 1. Grabar
            audio_bytes = self.record_audio(duration=5)
            
            # 2. Procesar
            result = self.process_pipeline(audio_bytes)
            results.append(result)
            
            # 3. Mostrar latencias
            if result['success']:
                print(f"\n{'─'*60}")
                print(f"⏱️  LATENCIAS DETALLADAS:")
                print(f"   Encoder:     {result['encoder_ms']:>6.0f}ms")
                print(f"   Projection:  {result['projection_ms']:>6.0f}ms")
                print(f"   Talker:      {result['talker_ms']:>6.0f}ms")
                print(f"   Token2Wav:   {result['token2wav_ms']:>6.0f}ms")
                print(f"   {'─'*40}")
                print(f"   TOTAL E2E:   {result['total_ms']:>6.0f}ms")
                print(f"{'─'*60}")
                
                # 4. Reproducir audio generado
                if result['audio_output'] is not None:
                    self.play_audio(result['audio_output'].squeeze())
            else:
                print(f"\n❌ Error: {result['error']}")
            
            # 5. Continuar?
            if turn < turns:
                try:
                    input(f"\n¿Continuar? (Enter=Sí, Ctrl+C=No) ")
                except KeyboardInterrupt:
                    print(f"\n\n⚠️  Test interrumpido")
                    break
        
        # Resumen
        self._print_summary(results)
        
        # Cleanup
        self.audio.terminate()
    
    def _print_summary(self, results):
        """Imprime resumen de resultados"""
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN")
        print(f"{'='*60}")
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\nTurnos: {len(results)}")
        print(f"  Exitosos: {len(successful)}")
        print(f"  Fallidos: {len(failed)}")
        
        if successful:
            print(f"\n⏱️  Latencias Promedio:")
            
            for component in ['encoder_ms', 'projection_ms', 'talker_ms', 'token2wav_ms', 'total_ms']:
                values = [r[component] for r in successful]
                avg = np.mean(values)
                min_v = np.min(values)
                max_v = np.max(values)
                
                label = component.replace('_ms', '').replace('_', ' ').title()
                print(f"   {label:15s}: {avg:6.0f}ms (min={min_v:.0f}, max={max_v:.0f})")
        
        if failed:
            print(f"\n❌ Errores:")
            for i, r in enumerate(failed, 1):
                print(f"  {i}. {r['error']}")


def main():
    """Punto de entrada"""
    test = CompletePipelineVoiceTest()
    
    if test.is_loaded():
        test.run_test(turns=3)
    else:
        print("\n❌ No se pudo cargar el pipeline completo")
        print("   Verifica que existan los archivos en models/onnx/:")
        print("   - audio_encoder_int8.pt o audio_encoder_fp16.pt")
        print("   - projection.onnx")
        print("   - qwen25_audio_gpu_lite.onnx")
        print("   - token2wav_int8.pt o token2wav_fp16.pt")


if __name__ == "__main__":
    main()
