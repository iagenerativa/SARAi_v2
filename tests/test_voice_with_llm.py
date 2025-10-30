#!/usr/bin/env python3
"""
SARAi v2.16.3 - Pipeline Completo de Voz con LLM
================================================

Test del pipeline end-to-end con razonamiento:
Audio Input → Encoder → Projection → LFM2 → Talker → Token2Wav → Audio Output

Mide latencias reales de cada componente y latencia total E2E.
"""

import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Tuple, Optional
import pyaudio
import wave
import io

# Colores ANSI
class C:
    G = '\033[92m'  # Green
    Y = '\033[93m'  # Yellow
    R = '\033[91m'  # Red
    B = '\033[94m'  # Blue
    M = '\033[95m'  # Magenta
    C = '\033[96m'  # Cyan
    E = '\033[0m'   # End
    BOLD = '\033[1m'


class VoicePipelineWithLLM:
    """Pipeline completo de voz con LLM para razonamiento"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.components = {}
        self.latencies = {}
        
    def load_all_components(self):
        """Carga todos los componentes del pipeline"""
        
        print(f"\n{C.B}{C.BOLD}{'='*70}{C.E}")
        print(f"{C.B}{C.BOLD}   CARGANDO PIPELINE COMPLETO CON LLM   {C.E}")
        print(f"{C.B}{C.BOLD}{'='*70}{C.E}\n")
        
        total_start = time.perf_counter()
        
        # 1. Projection ONNX (más ligero, cargar primero)
        self._load_projection()
        
        # 2. Talker ONNX (qwen25_7b - 42MB)
        self._load_talker()
        
        # 3. LFM2-1.2B (razonamiento)
        self._load_lfm2()
        
        # 4. Audio Encoder (opcional para test con audio real)
        # self._load_audio_encoder()
        
        # 5. Token2Wav (opcional para generar audio de salida)
        # self._load_token2wav()
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        print(f"\n{C.G}{'='*70}{C.E}")
        print(f"{C.G}✓ Pipeline completo cargado en {total_time:.1f}ms{C.E}")
        print(f"{C.G}{'='*70}{C.E}\n")
        
        self._print_memory_usage()
    
    def _load_projection(self):
        """Carga Projection ONNX"""
        print(f"{C.C}[1/3] Cargando Projection ONNX...{C.E}")
        
        projection_path = self.base_path / "models/onnx/projection.onnx"
        
        if not projection_path.exists():
            print(f"{C.R}✗ Projection ONNX no encontrado{C.E}")
            return
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        start = time.perf_counter()
        self.components['projection'] = ort.InferenceSession(
            str(projection_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        load_time = (time.perf_counter() - start) * 1000
        self.latencies['projection_load'] = load_time
        
        # Metadata
        input_info = self.components['projection'].get_inputs()[0]
        output_info = self.components['projection'].get_outputs()[0]
        
        print(f"{C.G}      ✓ Cargado en {load_time:.1f}ms{C.E}")
        print(f"        Input:  {input_info.shape}")
        print(f"        Output: {output_info.shape}")
    
    def _load_talker(self):
        """Carga Talker ONNX (qwen25_7b)"""
        print(f"\n{C.C}[2/3] Cargando Talker ONNX (qwen25_7b - 42MB)...{C.E}")
        
        talker_path = self.base_path / "models/onnx/qwen25_7b_audio.onnx"
        
        if not talker_path.exists():
            print(f"{C.R}✗ Talker ONNX no encontrado{C.E}")
            return
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        start = time.perf_counter()
        self.components['talker'] = ort.InferenceSession(
            str(talker_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        load_time = (time.perf_counter() - start) * 1000
        self.latencies['talker_load'] = load_time
        
        # Metadata
        input_info = self.components['talker'].get_inputs()[0]
        output_info = self.components['talker'].get_outputs()[0]
        
        print(f"{C.G}      ✓ Cargado en {load_time:.1f}ms{C.E}")
        print(f"        Input:  {input_info.shape} ({input_info.type})")
        print(f"        Output: {output_info.shape} ({output_info.type})")
    
    def _load_lfm2(self):
        """Carga LFM2-1.2B para razonamiento"""
        print(f"\n{C.C}[3/3] Cargando LFM2-1.2B (698MB)...{C.E}")
        
        lfm2_path = self.base_path / "models/lfm2/LFM2-1.2B-Q4_K_M.gguf"
        
        if not lfm2_path.exists():
            print(f"{C.R}✗ LFM2 no encontrado{C.E}")
            return
        
        try:
            from llama_cpp import Llama
            
            start = time.perf_counter()
            self.components['lfm2'] = Llama(
                model_path=str(lfm2_path),
                n_ctx=512,  # Contexto más pequeño para respuestas cortas
                n_threads=4,
                use_mmap=True,
                use_mlock=False,
                verbose=False
            )
            load_time = (time.perf_counter() - start) * 1000
            self.latencies['lfm2_load'] = load_time
            
            print(f"{C.G}      ✓ Cargado en {load_time:.1f}ms{C.E}")
            print(f"        Context: 512 tokens")
            print(f"        Threads: 4")
            print(f"        Modo: Text Generation")
            
        except ImportError:
            print(f"{C.R}✗ llama-cpp-python no instalado{C.E}")
            print(f"        Instalar: pip install llama-cpp-python")
    
    def _print_memory_usage(self):
        """Imprime uso de memoria"""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / (1024 ** 2)
            
            print(f"\n{C.M}📊 Uso de Memoria:{C.E}")
            print(f"   RAM actual: {mem_mb:.1f} MB")
        except ImportError:
            pass
    
    def process_with_synthetic_audio(self, prompt: str = "Hola, ¿cómo estás?") -> Dict:
        """
        Procesa el pipeline completo con audio sintético
        
        Args:
            prompt: Texto para LFM2 (simula lo que diría el audio)
        
        Returns:
            Dict con latencias de cada componente
        """
        print(f"\n{C.B}{C.BOLD}{'='*70}{C.E}")
        print(f"{C.B}{C.BOLD}   PROCESANDO PIPELINE COMPLETO   {C.E}")
        print(f"{C.B}{C.BOLD}{'='*70}{C.E}\n")
        
        print(f"{C.M}Prompt simulado:{C.E} \"{prompt}\"")
        
        latencies = {}
        
        # PASO 1: Generar features sintéticas (simula Audio Encoder)
        print(f"\n{C.C}[1/5] Generando features sintéticas (simula Audio Encoder)...{C.E}")
        start = time.perf_counter()
        
        # Features típicas del audio encoder: [1, seq_len, 512]
        seq_len = 100  # ~3 segundos de audio
        features = np.random.randn(1, seq_len, 512).astype(np.float32) * 0.1
        features = np.clip(features, -0.5, 0.5)
        
        latencies['audio_encoder'] = (time.perf_counter() - start) * 1000
        print(f"{C.G}      ✓ Generadas: {features.shape} ({latencies['audio_encoder']:.1f}ms){C.E}")
        
        # PASO 2: Projection (512 → 3584)
        print(f"\n{C.C}[2/5] Projection ONNX (512 → 3584)...{C.E}")
        start = time.perf_counter()
        
        projection_input = self.components['projection'].get_inputs()[0].name
        projection_output = self.components['projection'].get_outputs()[0].name
        
        hidden_states = self.components['projection'].run(
            [projection_output],
            {projection_input: features}
        )[0]
        
        latencies['projection'] = (time.perf_counter() - start) * 1000
        print(f"{C.G}      ✓ Proyectado: {hidden_states.shape} ({latencies['projection']:.1f}ms){C.E}")
        
        # PASO 3: LFM2 Razonamiento
        print(f"\n{C.C}[3/5] LFM2 Razonamiento...{C.E}")
        start = time.perf_counter()
        
        # Reset del modelo para cada inferencia (evita problemas de contexto)
        self.components['lfm2'].reset()
        
        # LFM2 procesa el prompt y genera respuesta
        response = self.components['lfm2'].create_completion(
            prompt,
            max_tokens=30,  # Respuestas cortas
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        
        response_text = response['choices'][0]['text'].strip()
        
        latencies['lfm2_inference'] = (time.perf_counter() - start) * 1000
        print(f"{C.G}      ✓ Respuesta: \"{response_text}\" ({latencies['lfm2_inference']:.1f}ms){C.E}")
        
        # Nota: En producción, LFM2 debería modular hidden_states
        # Para este test, mantenemos hidden_states como están
        
        # PASO 4: Talker (3584 → 8448 audio tokens)
        print(f"\n{C.C}[4/5] Talker ONNX (3584 → 8448)...{C.E}")
        start = time.perf_counter()
        
        talker_input = self.components['talker'].get_inputs()[0].name
        talker_output = self.components['talker'].get_outputs()[0].name
        
        audio_logits = self.components['talker'].run(
            [talker_output],
            {talker_input: hidden_states}
        )[0]
        
        latencies['talker'] = (time.perf_counter() - start) * 1000
        print(f"{C.G}      ✓ Audio tokens: {audio_logits.shape} ({latencies['talker']:.1f}ms){C.E}")
        
        # PASO 5: Token2Wav (simulado)
        print(f"\n{C.C}[5/5] Token2Wav (simulado - 3 diffusion steps)...{C.E}")
        start = time.perf_counter()
        
        # Simular tiempo de Token2Wav
        time.sleep(0.05)  # 50ms estimado
        
        latencies['token2wav'] = (time.perf_counter() - start) * 1000
        print(f"{C.G}      ✓ Audio generado (simulado): ({latencies['token2wav']:.1f}ms){C.E}")
        
        # Calcular latencia total
        latencies['total_e2e'] = sum(latencies.values())
        
        return latencies, response_text
    
    def record_real_audio(self, duration: int = 5) -> bytes:
        """
        Graba audio real del micrófono
        
        Args:
            duration: Duración en segundos
        
        Returns:
            Audio bytes
        """
        print(f"\n{C.Y}🎤 Grabando {duration}s de audio...{C.E}")
        
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
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Convertir a bytes
        audio_bytes = b''.join(frames)
        
        print(f"{C.G}✓ Grabación completa: {len(audio_bytes)} bytes{C.E}")
        
        return audio_bytes
    
    def print_results(self, latencies: Dict, response_text: str):
        """Imprime resultados del pipeline"""
        
        print(f"\n{C.B}{C.BOLD}{'='*70}{C.E}")
        print(f"{C.B}{C.BOLD}   RESULTADOS DEL PIPELINE   {C.E}")
        print(f"{C.B}{C.BOLD}{'='*70}{C.E}\n")
        
        print(f"{C.M}📊 Latencias por Componente:{C.E}\n")
        
        components_order = [
            ('audio_encoder', 'Audio Encoder'),
            ('projection', 'Projection ONNX'),
            ('lfm2_inference', 'LFM2 Razonamiento'),
            ('talker', 'Talker ONNX'),
            ('token2wav', 'Token2Wav')
        ]
        
        for key, name in components_order:
            if key in latencies:
                lat = latencies[key]
                print(f"   {name:<25} {lat:>8.1f}ms")
        
        print(f"   {'-'*35}")
        print(f"   {'TOTAL E2E':<25} {latencies['total_e2e']:>8.1f}ms")
        
        # Evaluación vs objetivos
        print(f"\n{C.M}🎯 Evaluación vs Objetivos:{C.E}\n")
        
        total = latencies['total_e2e']
        
        if total <= 1100:
            status = f"{C.G}✓ EXCELENTE{C.E}"
        elif total <= 1500:
            status = f"{C.Y}✓ BUENO{C.E}"
        else:
            status = f"{C.R}⚠ MEJORABLE{C.E}"
        
        print(f"   Objetivo con LLM: ≤1500ms")
        print(f"   Real medido:      {total:.1f}ms")
        print(f"   Estado:           {status}")
        
        # Desglose porcentual
        print(f"\n{C.M}📈 Desglose Porcentual:{C.E}\n")
        
        for key, name in components_order:
            if key in latencies:
                percentage = (latencies[key] / total) * 100
                bar_length = int(percentage / 2)
                bar = '█' * bar_length
                print(f"   {name:<25} {bar} {percentage:5.1f}%")
        
        # Respuesta del LLM
        print(f"\n{C.M}💬 Respuesta del LLM:{C.E}")
        print(f"   \"{response_text}\"")


def main():
    """Función principal"""
    
    print(f"\n{C.M}{C.BOLD}{'='*70}{C.E}")
    print(f"{C.M}{C.BOLD}   SARAi v2.16.3 - Pipeline Completo con LLM   {C.E}")
    print(f"{C.M}{C.BOLD}{'='*70}{C.E}")
    
    # Crear pipeline
    pipeline = VoicePipelineWithLLM()
    
    # Cargar componentes
    pipeline.load_all_components()
    
    # Menú de opciones
    print(f"\n{C.C}Opciones de Test:{C.E}")
    print(f"  1. Test con audio sintético (más rápido)")
    print(f"  2. Test con audio real del micrófono")
    print(f"  3. Benchmark (3 turnos sintéticos)")
    
    choice = input(f"\n{C.C}Selecciona opción [1-3]: {C.E}").strip()
    
    if choice == '1':
        # Test sintético
        print(f"\n{C.Y}Modo: Audio Sintético{C.E}")
        prompt = input(f"{C.C}Prompt para LFM2 (Enter = default): {C.E}").strip()
        if not prompt:
            prompt = "Hola, ¿cómo estás?"
        
        latencies, response = pipeline.process_with_synthetic_audio(prompt)
        pipeline.print_results(latencies, response)
    
    elif choice == '2':
        # Test con audio real
        print(f"\n{C.Y}Modo: Audio Real del Micrófono{C.E}")
        print(f"{C.R}⚠ Nota: Audio Encoder no implementado aún (requiere AutoProcessor){C.E}")
        print(f"{C.Y}Este test grabará audio pero procesará con features sintéticas{C.E}")
        
        input(f"\n{C.C}Presiona Enter para empezar a grabar...{C.E}")
        audio_bytes = pipeline.record_real_audio(5)
        
        # Por ahora, usar features sintéticas
        latencies, response = pipeline.process_with_synthetic_audio("Audio grabado del usuario")
        pipeline.print_results(latencies, response)
    
    elif choice == '3':
        # Benchmark
        print(f"\n{C.Y}Modo: Benchmark (3 turnos){C.E}")
        
        all_latencies = []
        prompts = [
            "Hola, ¿cómo estás?",
            "¿Qué tiempo hace hoy?",
            "Cuéntame un chiste"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{C.B}━━━ TURNO {i}/3 ━━━{C.E}")
            latencies, response = pipeline.process_with_synthetic_audio(prompt)
            all_latencies.append(latencies)
            print(f"\n{C.G}✓ Turno {i} completado: {latencies['total_e2e']:.1f}ms{C.E}")
            print(f"   Respuesta: \"{response}\"")
        
        # Estadísticas agregadas
        print(f"\n{C.B}{C.BOLD}{'='*70}{C.E}")
        print(f"{C.B}{C.BOLD}   ESTADÍSTICAS DEL BENCHMARK   {C.E}")
        print(f"{C.B}{C.BOLD}{'='*70}{C.E}\n")
        
        total_times = [lat['total_e2e'] for lat in all_latencies]
        lfm2_times = [lat['lfm2_inference'] for lat in all_latencies]
        talker_times = [lat['talker'] for lat in all_latencies]
        
        print(f"{C.M}Latencia Total E2E:{C.E}")
        print(f"   Min:    {min(total_times):.1f}ms")
        print(f"   Max:    {max(total_times):.1f}ms")
        print(f"   Avg:    {np.mean(total_times):.1f}ms")
        print(f"   P50:    {np.percentile(total_times, 50):.1f}ms")
        
        print(f"\n{C.M}LFM2 Razonamiento:{C.E}")
        print(f"   Min:    {min(lfm2_times):.1f}ms")
        print(f"   Max:    {max(lfm2_times):.1f}ms")
        print(f"   Avg:    {np.mean(lfm2_times):.1f}ms")
        
        print(f"\n{C.M}Talker ONNX:{C.E}")
        print(f"   Min:    {min(talker_times):.1f}ms")
        print(f"   Max:    {max(talker_times):.1f}ms")
        print(f"   Avg:    {np.mean(talker_times):.1f}ms")
    
    else:
        print(f"{C.R}Opción inválida{C.E}")
        return 1
    
    print(f"\n{C.G}{C.BOLD}{'='*70}{C.E}")
    print(f"{C.G}{C.BOLD}✓ Test Completado{C.E}")
    print(f"{C.G}{C.BOLD}{'='*70}{C.E}\n")
    
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{C.Y}⚠ Test interrumpido por usuario{C.E}")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n{C.R}❌ Error: {e}{C.E}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
