#!/usr/bin/env python3
"""
Benchmark de Latencia Audio - Comparación Qwen3-VL-4B-Instruct vs Qwen3-Omni-30B

Uso:
    # Test con Qwen3-VL-4B-Instruct (cuando lo tengas)
    python scripts/benchmark_audio_latency.py --model models/onnx/Qwen3-VL-4B-Instruct.onnx
    
    # Test con Qwen3-Omni-30B (actual)
    python scripts/benchmark_audio_latency.py --model models/onnx/agi_audio_core_int8.onnx
    
    # Comparación lado a lado
    python scripts/benchmark_audio_latency.py --compare

Objetivo: Validar si Qwen3-VL-4B-Instruct alcanza latencia <240ms en la práctica
"""

import argparse
import time
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List

# Añadir path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import onnxruntime as ort
except ImportError:
    print("❌ ERROR: onnxruntime no instalado")
    print("   Ejecuta: pip install onnxruntime")
    sys.exit(1)


class AudioLatencyBenchmark:
    """Benchmark de latencia para modelos audio ONNX"""
    
    def __init__(self, model_path: str, model_name: str = None):
        self.model_path = Path(model_path)
        self.model_name = model_name or self.model_path.stem
        self.session = None
        self.results = []
    
    def load_model(self):
        """Carga el modelo ONNX con optimizaciones"""
        print(f"\n{'='*60}")
        print(f"🔧 Cargando modelo: {self.model_name}")
        print(f"   Path: {self.model_path}")
        print(f"{'='*60}\n")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        
        # Configuración de sesión (mismas opts que producción)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        
        cpu_count = os.cpu_count() or 4
        sess_options.intra_op_num_threads = cpu_count
        sess_options.inter_op_num_threads = max(2, cpu_count // 2)
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_reuse = True
        
        providers = [
            ('CPUExecutionProvider', {
                'arena_extend_strategy': 'kSameAsRequested',
                'enable_cpu_mem_arena': True,
                'use_arena_allocator': True,
                'arena_size': 256 * 1024 * 1024,
            })
        ]
        
        start_time = time.time()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        load_time = time.time() - start_time
        
        # Mostrar info del modelo
        print(f"✅ Modelo cargado en {load_time:.2f}s")
        print(f"\nInputs:")
        for inp in self.session.get_inputs():
            print(f"  - {inp.name}: {inp.shape} ({inp.type})")
        print(f"\nOutputs:")
        for out in self.session.get_outputs():
            print(f"  - {out.name}: {out.shape} ({out.type})")
        
        return load_time
    
    def warmup(self, num_passes: int = 3):
        """Warmup del modelo (compilar kernels)"""
        print(f"\n🔥 Warmup ({num_passes} passes)...")
        
        # Crear dummy input basado en los inputs del modelo
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        input_type = self.session.get_inputs()[0].type
        
        # Reemplazar dims dinámicas con valores típicos para audio
        resolved_shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim is None or dim < 0:
                # Para audio típico: batch=1, seq_len variable, features fijos
                resolved_shape.append(1 if len(resolved_shape) == 0 else 
                                    512 if len(resolved_shape) == 1 else 
                                    dim)
            else:
                resolved_shape.append(dim)
        
        # Determinar dtype según el tipo de tensor
        if 'float' in input_type:
            dummy_input = np.random.randn(*resolved_shape).astype(np.float32)
        else:
            dummy_input = np.random.randint(0, 300, size=resolved_shape, dtype=np.int64)
        
        warmup_times = []
        for i in range(num_passes):
            start = time.time()
            _ = self.session.run(None, {input_name: dummy_input})
            elapsed = time.time() - start
            warmup_times.append(elapsed)
            print(f"  Pass {i+1}: {elapsed:.3f}s")
        
        avg_warmup = np.mean(warmup_times)
        print(f"✅ Warmup completado: {avg_warmup:.3f}s promedio")
        return warmup_times
    
    def benchmark_inference(self, num_iterations: int = 10, audio_lengths: List[float] = None):
        """
        Benchmark de latencia de inferencia
        
        Args:
            num_iterations: Iteraciones por cada longitud de audio
            audio_lengths: Longitudes de audio a testear (en segundos)
        
        Returns:
            Dict con estadísticas de latencia
        """
        if audio_lengths is None:
            audio_lengths = [0.5, 1.0, 2.0, 3.0, 5.0]  # Duraciones típicas
        
        print(f"\n📊 Benchmark de Inferencia")
        print(f"   Iteraciones por longitud: {num_iterations}")
        print(f"   Longitudes de audio: {audio_lengths}s")
        print(f"{'='*60}\n")
        
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        input_type = self.session.get_inputs()[0].type
        
        # Reemplazar dims dinámicas
        resolved_shape = []
        for i, dim in enumerate(input_shape):
            if isinstance(dim, str) or dim is None or dim < 0:
                # Para audio: batch=1, seq_len variable, features fijos
                resolved_shape.append(1 if i == 0 else 
                                    512 if i == 1 else 
                                    dim)
            else:
                resolved_shape.append(dim)
        
        # Determinar dtype según el tipo de tensor
        if 'float' in input_type:
            audio_input = np.random.randn(*resolved_shape).astype(np.float32)
        else:
            audio_input = np.random.randint(0, 300, size=resolved_shape, dtype=np.int64)
        
        all_results = {}
        
        for audio_len in audio_lengths:
            print(f"🎵 Audio de {audio_len}s:")
            latencies = []
            
            # Simular audio de diferentes longitudes
            # (en producción, esto vendría del encoder de audio)
            for i in range(num_iterations):
                # Determinar dtype según el tipo de tensor
                if 'float' in input_type:
                    audio_input = np.random.randn(*resolved_shape).astype(np.float32)
                else:
                    audio_input = np.random.randint(0, 300, size=resolved_shape, dtype=np.int64)
                
                start = time.time()
                outputs = self.session.run(None, {input_name: audio_input})
                latency = time.time() - start
                
                latencies.append(latency * 1000)  # ms
            
            # Estadísticas
            latencies_np = np.array(latencies)
            stats = {
                'mean': np.mean(latencies_np),
                'median': np.median(latencies_np),
                'std': np.std(latencies_np),
                'min': np.min(latencies_np),
                'max': np.max(latencies_np),
                'p50': np.percentile(latencies_np, 50),
                'p90': np.percentile(latencies_np, 90),
                'p99': np.percentile(latencies_np, 99),
            }
            
            all_results[audio_len] = stats
            
            print(f"   P50:  {stats['p50']:.1f}ms")
            print(f"   P90:  {stats['p90']:.1f}ms")
            print(f"   P99:  {stats['p99']:.1f}ms")
            print(f"   Mean: {stats['mean']:.1f}ms ± {stats['std']:.1f}ms")
            print(f"   Range: [{stats['min']:.1f}ms - {stats['max']:.1f}ms]\n")
        
        return all_results
    
    def print_summary(self, load_time: float, warmup_times: List[float], inference_results: Dict):
        """Imprime resumen final"""
        print(f"\n{'='*60}")
        print(f"📋 RESUMEN - {self.model_name}")
        print(f"{'='*60}\n")
        
        print(f"⏱️  Tiempos de Carga:")
        print(f"   Carga modelo:  {load_time:.2f}s")
        print(f"   Warmup (avg):  {np.mean(warmup_times):.2f}s")
        print(f"   Warmup (total): {sum(warmup_times):.2f}s")
        
        print(f"\n⚡ Latencia de Inferencia:")
        
        # Calcular promedios globales
        all_p50 = [stats['p50'] for stats in inference_results.values()]
        all_p99 = [stats['p99'] for stats in inference_results.values()]
        
        print(f"   P50 promedio:  {np.mean(all_p50):.1f}ms")
        print(f"   P99 promedio:  {np.mean(all_p99):.1f}ms")
        
        # Objetivo de latencia
        target_latency = 240  # ms
        p50_avg = np.mean(all_p50)
        
        print(f"\n🎯 Objetivo: <{target_latency}ms")
        if p50_avg < target_latency:
            print(f"   ✅ CUMPLE (P50: {p50_avg:.1f}ms < {target_latency}ms)")
        else:
            print(f"   ❌ NO CUMPLE (P50: {p50_avg:.1f}ms > {target_latency}ms)")
            print(f"   Exceso: {p50_avg - target_latency:.1f}ms ({((p50_avg/target_latency - 1)*100):.1f}% más lento)")
        
        # Throughput estimado
        throughput = 1000 / p50_avg  # inferencias por segundo
        print(f"\n📈 Throughput estimado: {throughput:.2f} inferencias/segundo")
        
        return {
            'model_name': self.model_name,
            'load_time': load_time,
            'warmup_avg': np.mean(warmup_times),
            'latency_p50': p50_avg,
            'latency_p99': np.mean(all_p99),
            'meets_target': p50_avg < target_latency,
            'throughput': throughput
        }


def compare_models(model_a: str, model_b: str):
    """Compara dos modelos lado a lado"""
    print(f"\n{'#'*60}")
    print(f"# COMPARACIÓN DE MODELOS")
    print(f"{'#'*60}\n")
    
    results = []
    
    for model_path in [model_a, model_b]:
        if not Path(model_path).exists():
            print(f"⚠️  Modelo no encontrado: {model_path}")
            print(f"   Saltando...\n")
            continue
        
        bench = AudioLatencyBenchmark(model_path)
        load_time = bench.load_model()
        warmup_times = bench.warmup(num_passes=3)
        inference_results = bench.benchmark_inference(num_iterations=10)
        summary = bench.print_summary(load_time, warmup_times, inference_results)
        
        results.append(summary)
    
    if len(results) == 2:
        print(f"\n{'='*60}")
        print(f"📊 TABLA COMPARATIVA")
        print(f"{'='*60}\n")
        
        print(f"{'Métrica':<25} {'Modelo A':<20} {'Modelo B':<20} {'Diferencia'}")
        print(f"{'-'*80}")
        
        metrics = [
            ('Modelo', 'model_name', 'model_name', 'str'),
            ('Carga (s)', 'load_time', 'load_time', 'float'),
            ('Warmup (s)', 'warmup_avg', 'warmup_avg', 'float'),
            ('Latencia P50 (ms)', 'latency_p50', 'latency_p50', 'float'),
            ('Latencia P99 (ms)', 'latency_p99', 'latency_p99', 'float'),
            ('Cumple <240ms', 'meets_target', 'meets_target', 'bool'),
            ('Throughput (inf/s)', 'throughput', 'throughput', 'float'),
        ]
        
        for label, key_a, key_b, dtype in metrics:
            val_a = results[0][key_a]
            val_b = results[1][key_b]
            
            if dtype == 'str':
                print(f"{label:<25} {str(val_a):<20} {str(val_b):<20} {'N/A'}")
            elif dtype == 'bool':
                str_a = '✅ SÍ' if val_a else '❌ NO'
                str_b = '✅ SÍ' if val_b else '❌ NO'
                print(f"{label:<25} {str_a:<20} {str_b:<20} {'N/A'}")
            elif dtype == 'float':
                diff = val_b - val_a
                diff_pct = ((val_b / val_a - 1) * 100) if val_a > 0 else 0
                
                if 'Latencia' in label or 'Carga' in label or 'Warmup' in label:
                    # Menor es mejor
                    symbol = '⬇️' if diff < 0 else '⬆️'
                else:
                    # Mayor es mejor
                    symbol = '⬆️' if diff > 0 else '⬇️'
                
                print(f"{label:<25} {val_a:<20.2f} {val_b:<20.2f} {symbol} {diff:+.2f} ({diff_pct:+.1f}%)")
        
        print(f"\n{'='*60}")
        
        # Veredicto
        print(f"\n🎯 VEREDICTO:")
        
        if results[0]['meets_target'] and not results[1]['meets_target']:
            print(f"   ✅ Modelo A ({results[0]['model_name']}) CUMPLE objetivo (<240ms)")
            print(f"   ❌ Modelo B ({results[1]['model_name']}) NO cumple")
            print(f"   → Recomendación: Usar Modelo A")
        elif results[1]['meets_target'] and not results[0]['meets_target']:
            print(f"   ✅ Modelo B ({results[1]['model_name']}) CUMPLE objetivo (<240ms)")
            print(f"   ❌ Modelo A ({results[0]['model_name']}) NO cumple")
            print(f"   → Recomendación: Usar Modelo B")
        elif results[0]['meets_target'] and results[1]['meets_target']:
            faster_idx = 0 if results[0]['latency_p50'] < results[1]['latency_p50'] else 1
            print(f"   ✅ AMBOS modelos cumplen objetivo (<240ms)")
            print(f"   → Modelo {'A' if faster_idx == 0 else 'B'} es más rápido")
            print(f"   → Recomendación: Usar {'A' if faster_idx == 0 else 'B'} para latencia, o comparar calidad")
        else:
            print(f"   ❌ NINGÚN modelo cumple objetivo (<240ms)")
            print(f"   → Necesario implementar sistema híbrido o buscar alternativa")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark de latencia para modelos audio ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Test individual
  python scripts/benchmark_audio_latency.py --model models/onnx/Qwen3-VL-4B-Instruct.onnx
  
  # Comparación
  python scripts/benchmark_audio_latency.py --compare \\
      --model-a models/onnx/Qwen3-VL-4B-Instruct.onnx \\
      --model-b models/onnx/agi_audio_core_int8.onnx
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path al modelo ONNX a testear'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Comparar dos modelos'
    )
    
    parser.add_argument(
        '--model-a',
        type=str,
        default='models/onnx/Qwen3-VL-4B-Instruct.onnx',
        help='Modelo A para comparación (default: Qwen3-VL-4B-Instruct.onnx)'
    )
    
    parser.add_argument(
        '--model-b',
        type=str,
        default='models/onnx/agi_audio_core_int8.onnx',
        help='Modelo B para comparación (default: agi_audio_core_int8.onnx)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Iteraciones por benchmark (default: 10)'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Modo comparación
        compare_models(args.model_a, args.model_b)
    
    elif args.model:
        # Modo individual
        bench = AudioLatencyBenchmark(args.model)
        load_time = bench.load_model()
        warmup_times = bench.warmup(num_passes=3)
        inference_results = bench.benchmark_inference(num_iterations=args.iterations)
        bench.print_summary(load_time, warmup_times, inference_results)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
