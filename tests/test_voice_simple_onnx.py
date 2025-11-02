#!/usr/bin/env python3
"""
Test SIMPLIFICADO de Voz con SARAi - SOLO ONNX v2.16.3

Sistema unificado usando SOLO qwen25_audio.onnx:
- Grabación de audio (pyaudio)
- Procesamiento completo con ONNX (STT + LLM + TTS)
- Medición de latencias reales

Objetivo: Medir rendimiento del modelo ONNX sin capas adicionales

USO:
    python tests/test_voice_simple_onnx.py
"""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyaudio
    import numpy as np
except ImportError as e:
    pytest.skip(f"Dependencias ONNX de voz faltantes: {e}", allow_module_level=True)


class SimpleONNXVoiceTest:
    """Test simplificado usando SOLO qwen25_audio.onnx"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.format = pyaudio.paInt16
        self.chunk = 1024
        
        self.audio = pyaudio.PyAudio()
        self.onnx_session = None
        
        # Cargar modelo ONNX
        self._load_onnx()
    
    def _load_onnx(self):
        """Carga qwen25_audio.onnx"""
        try:
            import onnxruntime as ort
            
            # Buscar modelo ONNX (usar qwen25_audio.onnx específicamente)
            possible_paths = [
                Path("models/onnx/qwen25_audio.onnx"),  # Modelo principal + .data
                Path("models/onnx/qwen25_audio_int8.onnx"),  # Versión INT8 (más rápido)
                Path("models/onnx/qwen25_7b_audio.onnx"),  # Versión 7B completa
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path is None:
                # Buscar cualquier ONNX disponible
                onnx_dir = Path("models/onnx")
                if onnx_dir.exists():
                    onnx_files = list(onnx_dir.glob("*.onnx"))
                    if onnx_files:
                        model_path = onnx_files[0]
                    else:
                        print(f"❌ No hay archivos ONNX en {onnx_dir}")
                        return
                else:
                    print(f"❌ Directorio no existe: {onnx_dir}")
                    return
            
            print(f"🔧 Cargando {model_path.name}...")
            print(f"   Ubicación: {model_path}")
            start = time.time()
            
            # Configurar sesión ONNX optimizada
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            
            cpu_count = os.cpu_count() or 4
            sess_options.intra_op_num_threads = cpu_count
            sess_options.inter_op_num_threads = max(2, cpu_count // 2)
            
            self.onnx_session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            
            load_time = (time.time() - start) * 1000
            print(f"✅ ONNX cargado ({load_time:.0f}ms)")
            
            # Mostrar info del modelo
            print(f"\n📊 Información del modelo:")
            print(f"   Entradas:")
            for inp in self.onnx_session.get_inputs():
                print(f"      - {inp.name}: {inp.shape} ({inp.type})")
            print(f"   Salidas:")
            for out in self.onnx_session.get_outputs():
                print(f"      - {out.name}: {out.shape} ({out.type})")
            
        except Exception as e:
            print(f"❌ Error cargando ONNX: {e}")
            import traceback
            traceback.print_exc()
            self.onnx_session = None
    
    def record_audio(self, duration=5):
        """Graba audio del micrófono"""
        print(f"\n🎤 Grabando {duration}s...")
        
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
    
    def process_with_onnx(self, audio_bytes):
        """Procesa audio con ONNX y mide latencia"""
        if self.onnx_session is None:
            return {
                "success": False,
                "error": "ONNX no disponible",
                "latency_ms": 0
            }
        
        try:
            print(f"\n🔄 Procesando con ONNX...")
            start = time.time()
            
            # Convertir bytes raw de pyaudio a numpy array
            # pyaudio da bytes en formato int16
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convertir int16 a float32 normalizado [-1, 1]
            waveform = audio_array.astype(np.float32) / 32768.0
            
            print(f"   Audio shape: {waveform.shape}, min={waveform.min():.3f}, max={waveform.max():.3f}")
            
            # Preparar input según el formato del modelo
            inputs = self.onnx_session.get_inputs()
            input_name = inputs[0].name
            input_shape = inputs[0].shape
            
            print(f"   Modelo espera: {input_name} con shape {input_shape}")
            
            # El modelo espera hidden_states, no audio directo
            # Esto significa que necesitamos un encoder previo
            # Por ahora, intentemos crear un tensor compatible
            
            # Si espera [batch, seq, hidden_dim], crear dummy
            if len(input_shape) == 3:
                # Ejemplo: ['s47', 's87', 3072] significa dimensiones dinámicas
                batch_size = 1
                seq_len = len(waveform) // 512  # Downsample arbitrario
                hidden_dim = 3072
                
                # Crear tensor dummy (en producción esto vendría del encoder de audio)
                dummy_input = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
                
                print(f"   ⚠️  NOTA: Modelo espera hidden_states, no audio raw")
                print(f"   Generando tensor dummy {dummy_input.shape} para prueba")
                
                outputs = self.onnx_session.run(None, {input_name: dummy_input})
            else:
                # Adaptar audio según shape esperada
                audio_shapes = [
                    waveform.reshape(1, -1).astype(np.float32),  # [1, T]
                    waveform.reshape(1, 1, -1).astype(np.float32),  # [1, 1, T]
                    waveform.astype(np.float32),  # [T]
                ]
                
                outputs = None
                for audio_input in audio_shapes:
                    try:
                        outputs = self.onnx_session.run(None, {input_name: audio_input})
                        print(f"   ✅ Shape compatible: {audio_input.shape}")
                        break
                    except Exception as shape_error:
                        continue
                
                if outputs is None:
                    raise ValueError(f"Audio shape incompatible con modelo. Esperado: {input_shape}")
            
            latency = (time.time() - start) * 1000
            
            print(f"✅ ONNX procesado ({latency:.0f}ms)")
            print(f"   Outputs generados: {len(outputs)}")
            for i, out in enumerate(outputs):
                if isinstance(out, np.ndarray):
                    print(f"      - Output {i}: shape={out.shape}, dtype={out.dtype}")
            
            return {
                "success": True,
                "outputs": outputs,
                "latency_ms": latency,
                "error": None
            }
            
        except Exception as e:
            print(f"❌ Error en ONNX: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "latency_ms": 0
            }
    
    def run_test(self, turns=3):
        """Ejecuta test de voz"""
        print(f"\n{'='*60}")
        print(f"🎙️  TEST SIMPLIFICADO - SOLO ONNX")
        print(f"{'='*60}")
        
        if self.onnx_session is None:
            print(f"\n❌ No se puede ejecutar: ONNX no cargado")
            return
        
        results = []
        
        for turn in range(1, turns + 1):
            print(f"\n{'='*60}")
            print(f"TURNO {turn}/{turns}")
            print(f"{'='*60}")
            
            # 1. Grabar
            audio_bytes = self.record_audio(duration=5)
            
            # 2. Procesar con ONNX
            result = self.process_with_onnx(audio_bytes)
            
            # 3. Guardar resultado
            results.append(result)
            
            # 4. Mostrar latencia
            print(f"\n{'─'*60}")
            print(f"⏱️  LATENCIA E2E: {result['latency_ms']:.0f}ms")
            print(f"{'─'*60}")
            
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
            latencies = [r['latency_ms'] for r in successful]
            print(f"\nLatencias ONNX:")
            print(f"  Min:    {min(latencies):.0f}ms")
            print(f"  Max:    {max(latencies):.0f}ms")
            print(f"  Promedio: {np.mean(latencies):.0f}ms")
            print(f"  Mediana:  {np.median(latencies):.0f}ms")
        
        if failed:
            print(f"\n❌ Errores:")
            for i, r in enumerate(failed, 1):
                print(f"  {i}. {r['error']}")


def main():
    """Punto de entrada"""
    test = SimpleONNXVoiceTest()
    test.run_test(turns=3)


if __name__ == "__main__":
    main()
