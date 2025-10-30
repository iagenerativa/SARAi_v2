#!/usr/bin/env python3
"""
Test Rápido de Voz con SARAi - SIMPLIFICADO v2.16.3

Sistema unificado usando SOLO qwen25_audio.onnx:
- pyaudio (para grabar micrófono)
- qwen25_audio.onnx (STT + LLM + TTS todo en uno)
- Reproducción directa del audio generado

Objetivo: Medir latencias reales sin capas intermedias

USO:
    python tests/test_voice_quick.py
"""

import os
import sys
import time
import wave
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyaudio
    import numpy as np
except ImportError:
    print("❌ pyaudio no instalado. Instalar con: pip install pyaudio")
    sys.exit(1)


class QuickVoiceTest:
    """Test simplificado usando SOLO qwen25_audio.onnx"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 5  # Duración de grabación
        self.format = pyaudio.paInt16
        self.channels = 1
        
        self.audio = pyaudio.PyAudio()
        
        # Cargar SOLO el modelo ONNX completo
        self._load_onnx_pipeline()
    
    def _load_onnx_pipeline(self):
        """Cargar qwen25_audio.onnx como sistema completo"""
        try:
            import onnxruntime as ort
            
            model_path = Path("models/onnx/qwen25_audio.onnx")
            
            if not model_path.exists():
                print(f"❌ Modelo no encontrado: {model_path}")
                print(f"   Buscando en models/onnx/...")
                # Buscar archivo ONNX disponible
                onnx_dir = Path("models/onnx")
                if onnx_dir.exists():
                    onnx_files = list(onnx_dir.glob("*.onnx"))
                    if onnx_files:
                        model_path = onnx_files[0]
                        print(f"   Usando: {model_path}")
                    else:
                        print(f"   No hay archivos ONNX en {onnx_dir}")
                        self.onnx_session = None
                        return
                else:
                    self.onnx_session = None
                    return
            
            print(f"🔧 Cargando {model_path.name}...")
            start = time.time()
            
            # Configurar sesión ONNX optimizada
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            
            import os
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
            print(f"   Entradas: {[inp.name for inp in self.onnx_session.get_inputs()]}")
            print(f"   Salidas: {[out.name for out in self.onnx_session.get_outputs()]}")
            
        except Exception as e:
            print(f"❌ Error cargando ONNX: {e}")
            import traceback
            traceback.print_exc()
            self.onnx_session = None
    
    def record(self, duration: int = 5) -> bytes:
        """Graba audio del micrófono"""
        print(f"\n🎤 Grabando {duration}s...")
        
        chunk = 1024
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=chunk
        )
        
        frames = []
        for i in range(0, int(self.sample_rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        print("✅ Grabación completa")
        return b''.join(frames)
    
    def save_wav(self, audio_bytes: bytes, filename: str):
        """Guarda audio a WAV"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_bytes)
    
    def process_audio_with_onnx(self, audio_bytes: bytes) -> dict:
        """
        Procesa audio completo con ONNX: STT + LLM + TTS
        
        Returns:
            {
                "transcription": str,
                "response_text": str,
                "response_audio": bytes,
                "stt_time_ms": float,
                "llm_time_ms": float,
                "tts_time_ms": float,
                "total_time_ms": float
            }
        """
        if self.onnx_session is None:
            return {
                "transcription": "[ONNX no disponible]",
                "response_text": "[ONNX no disponible]",
                "response_audio": b"",
                "stt_time_ms": 0,
                "llm_time_ms": 0,
                "tts_time_ms": 0,
                "total_time_ms": 0
            }
        
        import time
        import numpy as np
        import io
        import soundfile as sf
        
        total_start = time.time()
        
        # 1. Convertir audio bytes a numpy array
        audio_io = io.BytesIO(audio_bytes)
        waveform, sr = sf.read(audio_io, dtype='float32')
        
        # Resamplear si es necesario
        if sr != 16000:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        
        # 2. Preparar input para ONNX (según el formato que requiera)
        # Esto depende de las entradas del modelo
        inputs = self.onnx_session.get_inputs()
        print(f"\n📊 Formato de entrada ONNX:")
        for inp in inputs:
            print(f"   - {inp.name}: shape={inp.shape}, type={inp.type}")
        
        # Por ahora, intentar inferencia básica
        try:
            # Preparar audio como input
            audio_input = waveform.reshape(1, -1).astype(np.float32)
            
            # Inferencia ONNX
            print(f"\n🔄 Procesando con ONNX...")
            onnx_start = time.time()
            
            # Obtener nombres de inputs y outputs
            input_name = inputs[0].name
            outputs = self.onnx_session.run(None, {input_name: audio_input})
            
            onnx_time = (time.time() - onnx_start) * 1000
            
            print(f"✅ ONNX procesado ({onnx_time:.0f}ms)")
            print(f"   Outputs: {len(outputs)} tensores")
            for i, out in enumerate(outputs):
                if isinstance(out, np.ndarray):
                    print(f"   - Output {i}: shape={out.shape}, dtype={out.dtype}")
            
            # Por ahora, retornar resultados simplificados
            total_time = (time.time() - total_start) * 1000
            
            return {
                "transcription": "[Audio procesado - implementar decodificación]",
                "response_text": "[Respuesta del modelo - implementar decodificación]",
                "response_audio": b"",  # TODO: convertir output a audio
                "stt_time_ms": onnx_time / 3,  # Estimación
                "llm_time_ms": onnx_time / 3,  # Estimación
                "tts_time_ms": onnx_time / 3,  # Estimación
                "total_time_ms": total_time
            }
            
        except Exception as e:
            print(f"❌ Error en inferencia ONNX: {e}")
            import traceback
            traceback.print_exc()
            
            total_time = (time.time() - total_start) * 1000
            return {
                "transcription": f"[Error: {e}]",
                "response_text": f"[Error: {e}]",
                "response_audio": b"",
                "stt_time_ms": 0,
                "llm_time_ms": 0,
                "tts_time_ms": 0,
                "total_time_ms": total_time
            }
    
    def generate_response(self, user_text: str) -> Tuple[str, float]:
        """Genera respuesta con LFM2"""
        if self.llm is None:
            # Respuestas por defecto
            if "hola" in user_text.lower():
                return "Hola! ¿En qué puedo ayudarte?", 50
            return "Interesante. Cuéntame más.", 50
        
        print("🤔 Razonando...")
        start = time.time()
        
        response = self.llm.create_completion(
            user_text,
            max_tokens=50,
            temperature=0.7,
            stop=["\n\n"]
        )
        
        latency = (time.time() - start) * 1000
        text = response['choices'][0]['text'].strip()
        
        print(f"✅ Respuesta: \"{text}\" ({latency:.0f}ms)")
        return text, latency
    
    def speak(self, text: str) -> float:
        """Sintetiza y reproduce voz con pipeline ONNX"""
        if self.audio_pipeline is None:
            print(f"🔇 Pipeline ONNX no disponible. Respuesta: \"{text}\"")
            return 0
        
        print("🔊 Hablando con TTS ONNX...")
        start = time.time()
        
        # Usar el pipeline ONNX para TTS
        self.audio_pipeline.speak(text)
        
        latency = (time.time() - start) * 1000
        print(f"✅ TTS ONNX ({latency:.0f}ms)")
        
        return latency
    
    def single_turn(self, turn: int):
        """Un turno de conversación"""
        print(f"\n{'='*60}")
        print(f"TURNO {turn}")
        print(f"{'='*60}")
        
        turn_start = time.time()
        
        # 1. Grabar
        audio_bytes = self.record(duration=5)
        temp_file = "temp_recording.wav"
        self.save_wav(audio_bytes, temp_file)
        
        # 2. STT
        user_text, stt_time = self.transcribe(temp_file)
        
        # 3. LLM
        response_text, llm_time = self.generate_response(user_text)
        
        # 4. TTS
        tts_time = self.speak(response_text)
        
        # Métricas
        total_time = (time.time() - turn_start) * 1000
        
        print(f"\n{'─'*60}")
        print(f"⏱️  LATENCIAS:")
        print(f"   STT:  {stt_time:.0f}ms")
        print(f"   LLM:  {llm_time:.0f}ms")
        print(f"   TTS:  {tts_time:.0f}ms")
        print(f"   E2E:  {total_time:.0f}ms")
        print(f"{'─'*60}")
        
        # Limpiar
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return {
            "user": user_text,
            "response": response_text,
            "stt_ms": stt_time,
            "llm_ms": llm_time,
            "tts_ms": tts_time,
            "e2e_ms": total_time
        }
    
    def run(self, turns: int = 3):
        """Ejecuta conversación"""
        print("\n" + "="*60)
        print("🎙️  TEST RÁPIDO DE VOZ - SARAi")
        print("="*60)
        
        results = []
        
        for i in range(1, turns + 1):
            try:
                result = self.single_turn(i)
                results.append(result)
                
                if i < turns:
                    print("\n¿Continuar? (Enter=Sí, Ctrl+C=No)")
                    input()
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Test interrumpido")
                break
        
        # Resumen
        if results:
            print("\n" + "="*60)
            print("📊 RESUMEN")
            print("="*60)
            
            avg_stt = np.mean([r["stt_ms"] for r in results])
            avg_llm = np.mean([r["llm_ms"] for r in results])
            avg_tts = np.mean([r["tts_ms"] for r in results])
            avg_e2e = np.mean([r["e2e_ms"] for r in results])
            
            print(f"\nTurnos: {len(results)}")
            print(f"\nLatencias promedio:")
            print(f"  STT:  {avg_stt:.0f}ms")
            print(f"  LLM:  {avg_llm:.0f}ms")
            print(f"  TTS:  {avg_tts:.0f}ms")
            print(f"  E2E:  {avg_e2e:.0f}ms")
            
            if avg_e2e < 2000:
                print("\n✅ Rendimiento excelente!")
            elif avg_e2e < 3000:
                print("\n✅ Rendimiento bueno")
            else:
                print("\n⚠️  Latencia alta, considerar optimizaciones")
        
        self.cleanup()
    
    def cleanup(self):
        """Limpieza"""
        self.audio.terminate()


if __name__ == "__main__":
    import numpy as np
    
    test = QuickVoiceTest()
    test.run(turns=3)
