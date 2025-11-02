#!/usr/bin/env python3
"""
SARAi v2.16.3 - Test Interactivo de Voz en Tiempo Real
======================================================

Test E2E del pipeline completo con audio REAL del micrófono.
Mide latencias reales de conversación.

Usa este script para probar directamente con tu voz.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import pytest

try:
    import numpy as np
    import torch
    import onnxruntime as ort
    import pyaudio
    import wave
except ImportError as e:
    pytest.skip(
        f"Dependencias de voz en tiempo real faltantes: {e}",
        allow_module_level=True,
    )

# Añadir directorio raíz al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Colores
class C:
    G = '\033[92m'
    Y = '\033[93m'
    R = '\033[91m'
    B = '\033[94m'
    M = '\033[95m'
    C = '\033[96m'
    E = '\033[0m'
    BOLD = '\033[1m'


class RealTimeVoicePipeline:
    """Pipeline de voz para pruebas en tiempo real"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.components = {}
        self.session_start = time.time()
        self.turn_count = 0
        self.all_latencies = []
        
    def load_components(self):
        """Carga componentes del pipeline"""
        
        print(f"\n{C.B}{C.BOLD}🔧 Cargando Componentes del Pipeline...{C.E}\n")
        
        # 1. Projection
        print(f"{C.C}[1/4] Projection ONNX...{C.E}", end=' ')
        projection_path = self.base_path / "models/onnx/projection.onnx"
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.components['projection'] = ort.InferenceSession(
            str(projection_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        print(f"{C.G}✓{C.E}")
        
        # 2. Talker
        print(f"{C.C}[2/4] Talker ONNX (qwen25_7b)...{C.E}", end=' ')
        talker_path = self.base_path / "models/onnx/qwen25_7b_audio.onnx"
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.components['talker'] = ort.InferenceSession(
            str(talker_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        print(f"{C.G}✓{C.E}")
        
        # 3. LFM2
        print(f"{C.C}[3/4] LFM2-1.2B...{C.E}", end=' ')
        lfm2_path = self.base_path / "models/lfm2/LFM2-1.2B-Q4_K_M.gguf"
        
        from llama_cpp import Llama
        
        self.components['lfm2'] = Llama(
            model_path=str(lfm2_path),
            n_ctx=512,
            n_threads=4,
            use_mmap=True,
            use_mlock=False,
            verbose=False
        )
        print(f"{C.G}✓{C.E}")
        
        # 4. Piper TTS (Voz en Español)
        print(f"{C.C}[4/4] Piper TTS (Voz Española)...{C.E}", end=' ')
        
        from agents.piper_tts import PiperTTSEngine
        self.components['piper_tts'] = PiperTTSEngine()
        
        print(f"{C.G}✓{C.E}")
        
        print(f"\n{C.G}✅ Pipeline listo para conversación CON VOZ{C.E}\n")
    
    def record_audio(self, duration: int = 5) -> bytes:
        """Graba audio del micrófono"""
        
        print(f"\n{C.Y}🎤 Grabando {duration}s...{C.E}")
        print(f"{C.Y}   (Habla ahora){C.E}")
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Indicador visual
            if i % 5 == 0:
                print('.', end='', flush=True)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        audio_bytes = b''.join(frames)
        
        print(f"\n{C.G}✓ Grabación completa{C.E}")
        
        return audio_bytes
    
    def _save_audio_piper(self, audio_array: np.ndarray, filename: str):
        """Guarda audio de Piper TTS como archivo WAV"""
        
        import scipy.io.wavfile as wavfile
        
        # Piper genera a 22050 Hz
        SAMPLE_RATE = 22050
        
        # Asegurar que es mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array[0]  # Tomar primer canal
        
        # Normalizar a int16
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        wavfile.write(filename, SAMPLE_RATE, audio_int16)
    
    def _save_audio_output(self, audio_array: np.ndarray, filename: str):
        """Guarda audio de salida como archivo WAV"""
        
        import scipy.io.wavfile as wavfile
        
        # Token2Wav genera a 24kHz
        SAMPLE_RATE = 24000
        
        # Asegurar que es mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array[0]  # Tomar primer canal
        
        # Normalizar a int16
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        wavfile.write(filename, SAMPLE_RATE, audio_int16)
    
    def save_audio(self, audio_bytes: bytes, filename: str):
        """Guarda audio como archivo WAV"""
        
        CHANNELS = 1
        RATE = 16000
        SAMPLE_WIDTH = 2  # 16-bit
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(RATE)
            wf.writeframes(audio_bytes)
    
    def process_turn(self, save_audio: bool = False):
        """Procesa un turno completo de conversación"""
        
        self.turn_count += 1
        
        print(f"\n{C.B}{C.BOLD}{'='*70}{C.E}")
        print(f"{C.B}{C.BOLD}   TURNO {self.turn_count}   {C.E}")
        print(f"{C.B}{C.BOLD}{'='*70}{C.E}")
        
        latencies = {}
        
        # 1. Grabar audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        start_total = time.perf_counter()
        
        audio_bytes = self.record_audio(5)
        latencies['recording'] = (time.perf_counter() - start_total) * 1000
        
        if save_audio:
            audio_file = f"logs/audio_input_{timestamp}.wav"
            self.save_audio(audio_bytes, audio_file)
            print(f"{C.C}   → Audio guardado: {audio_file}{C.E}")
        
        # 2. Generar features sintéticas (en el futuro: Audio Encoder real)
        print(f"\n{C.C}[1/5] Audio → Features (sintético)...{C.E}", end=' ')
        start = time.perf_counter()
        
        seq_len = 100
        features = np.random.randn(1, seq_len, 512).astype(np.float32) * 0.1
        features = np.clip(features, -0.5, 0.5)
        
        latencies['encoder'] = (time.perf_counter() - start) * 1000
        print(f"{C.G}{latencies['encoder']:.1f}ms{C.E}")
        
        # 3. Projection
        print(f"{C.C}[2/5] Projection (512→3584)...{C.E}", end=' ')
        start = time.perf_counter()
        
        projection_input = self.components['projection'].get_inputs()[0].name
        projection_output = self.components['projection'].get_outputs()[0].name
        
        hidden_states = self.components['projection'].run(
            [projection_output],
            {projection_input: features}
        )[0]
        
        latencies['projection'] = (time.perf_counter() - start) * 1000
        print(f"{C.G}{latencies['projection']:.1f}ms{C.E}")
        
        # 4. LFM2 Razonamiento
        print(f"{C.C}[3/5] LFM2 Razonamiento...{C.E}", end=' ')
        start = time.perf_counter()
        
        # Reset para evitar problemas de contexto
        self.components['lfm2'].reset()
        
        # Prompt simulado (en el futuro: transcripción del audio)
        prompt = f"Usuario: [Audio turno {self.turn_count}]\nAsistente:"
        
        response = self.components['lfm2'].create_completion(
            prompt,
            max_tokens=30,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        
        response_text = response['choices'][0]['text'].strip()
        
        latencies['lfm2'] = (time.perf_counter() - start) * 1000
        print(f"{C.G}{latencies['lfm2']:.1f}ms{C.E}")
        
        # 5. Talker
        print(f"{C.C}[4/5] Talker (3584→8448)...{C.E}", end=' ')
        start = time.perf_counter()
        
        talker_input = self.components['talker'].get_inputs()[0].name
        talker_output = self.components['talker'].get_outputs()[0].name
        
        audio_logits = self.components['talker'].run(
            [talker_output],
            {talker_input: hidden_states}
        )[0]
        
        latencies['talker'] = (time.perf_counter() - start) * 1000
        print(f"{C.G}{latencies['talker']:.1f}ms{C.E}")
        
        # 6. Piper TTS (Generación de Voz en Español)
        print(f"{C.C}[5/5] Piper TTS → Voz en Español...{C.E}", end=' ')
        start = time.perf_counter()
        
        # Sintetizar con Piper TTS usando el texto generado por LFM2
        audio_output = self.components['piper_tts'].synthesize(response_text)
        
        latencies['piper_tts'] = (time.perf_counter() - start) * 1000
        print(f"{C.G}{latencies['piper_tts']:.1f}ms{C.E}")
        
        # 7. Guardar audio de salida
        output_file = f"logs/audio_output_{timestamp}.wav"
        self._save_audio_piper(audio_output, output_file)
        print(f"{C.M}   🔊 Audio guardado: {output_file}{C.E}")
        
        # Total (excluyendo grabación)
        latencies['total_processing'] = sum([
            latencies['encoder'],
            latencies['projection'],
            latencies['lfm2'],
            latencies['talker'],
            latencies['piper_tts']
        ])
        
        latencies['total_e2e'] = (time.perf_counter() - start_total) * 1000
        
        # Mostrar resultados
        self._print_turn_results(latencies, response_text)
        
        # Guardar para estadísticas
        self.all_latencies.append(latencies)
        
        return latencies, response_text
    
    def _print_turn_results(self, latencies: dict, response: str):
        """Imprime resultados del turno"""
        
        print(f"\n{C.M}📊 Resultados:{C.E}")
        print(f"   Procesamiento: {latencies['total_processing']:.1f}ms")
        print(f"   Total E2E:     {latencies['total_e2e']:.1f}ms")
        
        print(f"\n{C.M}💬 Respuesta LLM:{C.E}")
        print(f"   {response}")
        
        # Evaluación
        if latencies['total_processing'] <= 1100:
            status = f"{C.G}✓ EXCELENTE{C.E}"
        elif latencies['total_processing'] <= 1500:
            status = f"{C.Y}✓ BUENO{C.E}"
        else:
            status = f"{C.R}⚠ MEJORABLE{C.E}"
        
        print(f"\n   Estado: {status}")
    
    def print_session_stats(self):
        """Imprime estadísticas de la sesión"""
        
        if not self.all_latencies:
            return
        
        print(f"\n{C.B}{C.BOLD}{'='*70}{C.E}")
        print(f"{C.B}{C.BOLD}   ESTADÍSTICAS DE LA SESIÓN   {C.E}")
        print(f"{C.B}{C.BOLD}{'='*70}{C.E}\n")
        
        total_times = [lat['total_processing'] for lat in self.all_latencies]
        lfm2_times = [lat['lfm2'] for lat in self.all_latencies]
        talker_times = [lat['talker'] for lat in self.all_latencies]
        
        print(f"{C.M}Turnos procesados:{C.E} {self.turn_count}")
        print(f"{C.M}Duración sesión:{C.E} {(time.time() - self.session_start):.1f}s")
        
        print(f"\n{C.M}Latencia Total (procesamiento):{C.E}")
        print(f"   Min:    {min(total_times):.1f}ms")
        print(f"   Max:    {max(total_times):.1f}ms")
        print(f"   Avg:    {np.mean(total_times):.1f}ms")
        print(f"   P50:    {np.percentile(total_times, 50):.1f}ms")
        
        print(f"\n{C.M}LFM2 Razonamiento:{C.E}")
        print(f"   Min:    {min(lfm2_times):.1f}ms")
        print(f"   Max:    {max(lfm2_times):.1f}ms")
        print(f"   Avg:    {np.mean(lfm2_times):.1f}ms")
        print(f"   % del total: {(np.mean(lfm2_times) / np.mean(total_times) * 100):.1f}%")
        
        print(f"\n{C.M}Talker ONNX:{C.E}")
        print(f"   Min:    {min(talker_times):.1f}ms")
        print(f"   Max:    {max(talker_times):.1f}ms")
        print(f"   Avg:    {np.mean(talker_times):.1f}ms")


def main():
    """Función principal"""
    
    print(f"\n{C.M}{C.BOLD}{'='*70}{C.E}")
    print(f"{C.M}{C.BOLD}   SARAi v2.16.3 - Test Interactivo de Voz en Tiempo Real   {C.E}")
    print(f"{C.M}{C.BOLD}{'='*70}{C.E}")
    
    pipeline = RealTimeVoicePipeline()
    
    # Cargar componentes
    pipeline.load_components()
    
    # Configuración
    print(f"{C.C}Opciones:{C.E}")
    print(f"  • Cada turno graba 5s de audio")
    print(f"  • Presiona Enter para iniciar cada turno")
    print(f"  • Escribe 'q' para salir y ver estadísticas")
    
    save_audio = input(f"\n{C.C}¿Guardar audios grabados? [s/N]: {C.E}").strip().lower() == 's'
    
    if save_audio:
        Path("logs").mkdir(exist_ok=True)
        print(f"{C.G}✓ Audios se guardarán en logs/{C.E}")
    
    print(f"\n{C.G}{'='*70}{C.E}")
    print(f"{C.G}🎙️  Sistema listo para conversación en tiempo real{C.E}")
    print(f"{C.G}{'='*70}{C.E}\n")
    
    # Loop de conversación
    while True:
        user_input = input(f"{C.C}Presiona Enter para nuevo turno (o 'q' para salir): {C.E}").strip()
        
        if user_input.lower() == 'q':
            break
        
        try:
            latencies, response = pipeline.process_turn(save_audio)
        except KeyboardInterrupt:
            print(f"\n{C.Y}⚠ Turno interrumpido{C.E}")
            continue
        except Exception as e:
            print(f"\n{C.R}❌ Error: {e}{C.E}")
            continue
    
    # Estadísticas finales
    pipeline.print_session_stats()
    
    print(f"\n{C.G}{C.BOLD}{'='*70}{C.E}")
    print(f"{C.G}{C.BOLD}✓ Sesión Finalizada{C.E}")
    print(f"{C.G}{C.BOLD}{'='*70}{C.E}\n")
    
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{C.Y}⚠ Sesión interrumpida{C.E}")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n{C.R}❌ Error fatal: {e}{C.E}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
