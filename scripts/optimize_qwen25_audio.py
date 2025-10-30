#!/usr/bin/env python3
"""
Optimizaciones ULTRA para Qwen2.5-Omni INT8

Objetivo: Reducir latencia de 262.6ms → <240ms (~9% mejora)

Técnicas aplicadas:
1. Inter-op threads optimizado (paralelización de operadores)
2. Execution mode ORT_PARALLEL (vs SEQUENTIAL)
3. Graph optimization ALL (vs EXTENDED)
4. Arena size ajustado dinámicamente
5. Disable profiling overhead
6. Optimized memory pattern
7. Quantized INT8 specific optimizations

Versión: 2.16.1 - Última oportunidad para 240ms
Fecha: 29 octubre 2025
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("❌ ERROR: onnxruntime no instalado")
    sys.exit(1)


class UltraOptimizedBenchmark:
    """
    Benchmark con optimizaciones ULTRA para INT8
    
    Estrategia:
    - Parallel execution para modelo pequeño (96MB)
    - Threads optimizados según CPU
    - Memory arena pre-allocada
    - Profiling deshabilitado
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.session = None
    
    def load_with_ultra_opts(self):
        """Carga con configuración ULTRA optimizada"""
        print("🚀 Cargando modelo con ULTRA optimizations...")
        print("=" * 60)
        
        sess_options = ort.SessionOptions()
        
        # 1. Graph optimization ALL (vs EXTENDED)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        print("✅ Graph optimization: ALL")
        
        # 2. Threading optimizado
        cpu_count = os.cpu_count() or 4
        
        # Para modelo pequeño (96MB), usar más inter-op threads
        sess_options.intra_op_num_threads = max(2, cpu_count // 2)  # Reducido
        sess_options.inter_op_num_threads = cpu_count  # Aumentado (paralelización)
        
        print(f"✅ Threads: intra_op={sess_options.intra_op_num_threads}, inter_op={sess_options.inter_op_num_threads}")
        
        # 3. Execution mode PARALLEL (modelo pequeño puede beneficiarse)
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        print("✅ Execution mode: PARALLEL")
        
        # 4. Memory optimizations
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_reuse = True
        
        # Arena más pequeña para modelo INT8 (96MB vs 384MB FP32)
        arena_size = 128 * 1024 * 1024  # 128MB (vs 256MB antes)
        print(f"✅ Arena size: {arena_size // (1024*1024)}MB")
        
        # 5. Disable profiling (reduce overhead)
        sess_options.enable_profiling = False
        
        # 6. INT8 specific optimizations
        providers = [
            ('CPUExecutionProvider', {
                'arena_extend_strategy': 'kSameAsRequested',
                'enable_cpu_mem_arena': True,
                'use_arena_allocator': True,
                'arena_size': arena_size,
                # INT8 specific
                'use_fp16_math': False,  # INT8 no necesita FP16
                'enable_fast_math': True,  # Aproximaciones rápidas
            })
        ]
        
        print("\n🔄 Cargando sesión...")
        start = time.time()
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        load_time = time.time() - start
        print(f"✅ Modelo cargado en {load_time:.2f}s\n")
        
        return self.session
    
    def warmup(self, num_passes=5):
        """Warmup extendido (5 passes vs 3)"""
        print("🔥 Warmup ULTRA (5 passes)...")
        
        input_meta = self.session.get_inputs()[0]
        
        # Shape fijo para warmup
        shape = [1, 512, 3072]  # Batch=1, seq_len=512, features=3072
        dummy_input = np.random.randn(*shape).astype(np.float32)
        
        warmup_times = []
        for i in range(num_passes):
            start = time.time()
            self.session.run(None, {input_meta.name: dummy_input})
            warmup_times.append(time.time() - start)
            print(f"  Pass {i+1}: {warmup_times[-1]*1000:.1f}ms")
        
        avg = np.mean(warmup_times)
        print(f"✅ Warmup promedio: {avg*1000:.1f}ms\n")
        
        return avg
    
    def benchmark_inference(self, num_iterations=50):
        """
        Benchmark intensivo (50 iteraciones vs 10)
        
        Objetivo: P50 < 240ms
        """
        print("📊 Benchmark ULTRA (50 iteraciones)...")
        print("=" * 60)
        
        input_meta = self.session.get_inputs()[0]
        
        # Test con longitud real de audio (1s = ~512 tokens)
        shape = [1, 512, 3072]
        audio_input = np.random.randn(*shape).astype(np.float32)
        
        latencies = []
        
        for i in range(num_iterations):
            start = time.time()
            outputs = self.session.run(None, {input_meta.name: audio_input})
            latencies.append(time.time() - start)
            
            # Progress cada 10 iteraciones
            if (i + 1) % 10 == 0:
                print(f"  Iteración {i+1}/{num_iterations}: {latencies[-1]*1000:.1f}ms")
        
        # Estadísticas
        latencies_ms = [l * 1000 for l in latencies]
        
        p50 = np.percentile(latencies_ms, 50)
        p90 = np.percentile(latencies_ms, 90)
        p99 = np.percentile(latencies_ms, 99)
        mean = np.mean(latencies_ms)
        std = np.std(latencies_ms)
        
        print("\n" + "=" * 60)
        print("📋 RESULTADOS ULTRA")
        print("=" * 60)
        print(f"\n⚡ Latencia:")
        print(f"   P50:  {p50:.1f}ms")
        print(f"   P90:  {p90:.1f}ms")
        print(f"   P99:  {p99:.1f}ms")
        print(f"   Mean: {mean:.1f}ms ± {std:.1f}ms")
        
        print(f"\n🎯 Objetivo: <240ms")
        if p50 < 240:
            print(f"   ✅ CUMPLE (P50: {p50:.1f}ms < 240ms)")
            print(f"   Margen: {240 - p50:.1f}ms")
        else:
            print(f"   ❌ NO CUMPLE (P50: {p50:.1f}ms > 240ms)")
            print(f"   Exceso: {p50 - 240:.1f}ms ({((p50-240)/240)*100:.1f}%)")
        
        # Comparación con benchmark anterior
        baseline_p50 = 262.6
        improvement = ((baseline_p50 - p50) / baseline_p50) * 100
        
        print(f"\n📈 Mejora vs baseline (262.6ms):")
        print(f"   Reducción: {baseline_p50 - p50:.1f}ms ({improvement:.1f}%)")
        
        return {
            'p50': p50,
            'p90': p90,
            'p99': p99,
            'mean': mean,
            'std': std,
            'improvement_pct': improvement
        }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ULTRA para Qwen2.5-Omni INT8"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/onnx/qwen25_audio_int8.onnx',
        help='Ruta al modelo INT8'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=50,
        help='Número de iteraciones (default: 50)'
    )
    
    args = parser.parse_args()
    
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "BENCHMARK ULTRA - Qwen2.5-Omni INT8 v2.16.1" + " " * 8 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("🎯 Objetivo: Reducir latencia 262.6ms → <240ms")
    print("📊 Técnicas: ORT_PARALLEL + Threads optimizados + Arena tuned")
    print()
    
    # Benchmark ULTRA
    benchmark = UltraOptimizedBenchmark(args.model)
    benchmark.load_with_ultra_opts()
    benchmark.warmup(num_passes=5)
    results = benchmark.benchmark_inference(num_iterations=args.iterations)
    
    # Veredicto final
    print("\n" + "=" * 60)
    print("🏆 VEREDICTO FINAL")
    print("=" * 60)
    
    if results['p50'] < 240:
        print("\n✅ MODELO VIABLE PARA PRODUCCIÓN")
        print(f"   Latencia P50: {results['p50']:.1f}ms < 240ms objetivo")
        print(f"   Mejora total: {results['improvement_pct']:.1f}% vs baseline")
        print("\n📝 Próximos pasos:")
        print("   1. Actualizar audio_omni_pipeline.py con estas opts")
        print("   2. Test con audio real (no dummy data)")
        print("   3. Validar WER y MOS en producción")
    else:
        print("\n⚠️  MODELO BORDERLINE")
        print(f"   Latencia P50: {results['p50']:.1f}ms > 240ms objetivo")
        print(f"   Exceso: {results['p50'] - 240:.1f}ms")
        print("\n💡 Opciones:")
        print("   A. Aceptar {:.1f}ms como suficiente (experiencia fluida)".format(results['p50']))
        print("   B. Implementar Whisper-small + Piper (~140ms garantizado)")
        print("   C. Sistema dual-speed (fast lane + normal lane)")
    
    print("\n" + "=" * 60)
    
    return 0 if results['p50'] < 240 else 1


if __name__ == "__main__":
    sys.exit(main())
