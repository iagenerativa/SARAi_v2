#!/usr/bin/env python3
"""
Fine-tuning de optimizaciones para Qwen2.5-Omni INT8

Descubrimiento: ORT_PARALLEL empeora (498ms vs 262ms SEQUENTIAL)

Nueva estrategia:
1. Mantener SEQUENTIAL execution
2. Ajustar threads intra_op (probar diferentes valores)
3. Graph optimization ALL vs EXTENDED
4. Arena size tuning
5. Memory pattern optimization

Objetivo: 262.6ms → <240ms (~9% mejora)
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("❌ ERROR: onnxruntime no instalado")
    sys.exit(1)


def test_configuration(model_path: str, config_name: str, sess_options: ort.SessionOptions, providers: list):
    """Test una configuración específica"""
    
    # Cargar modelo
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers
    )
    
    input_meta = session.get_inputs()[0]
    shape = [1, 512, 3072]
    dummy_input = np.random.randn(*shape).astype(np.float32)
    
    # Warmup (3 passes)
    for _ in range(3):
        session.run(None, {input_meta.name: dummy_input})
    
    # Benchmark (20 iteraciones)
    latencies = []
    for _ in range(20):
        start = time.time()
        session.run(None, {input_meta.name: dummy_input})
        latencies.append((time.time() - start) * 1000)
    
    p50 = np.percentile(latencies, 50)
    
    return p50


def main():
    model_path = Path("models/onnx/qwen25_audio_int8.onnx")
    
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "FINE-TUNING OPTIMIZACIONES - v2.16.1" + " " * 12 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("🎯 Objetivo: Optimizar desde 262.6ms → <240ms")
    print("📊 Estrategia: Grid search de configuraciones")
    print()
    
    cpu_count = os.cpu_count() or 4
    results = []
    
    # Configuración 1: BASELINE (actual)
    print("=" * 60)
    print("🧪 Config 1: BASELINE (SEQUENTIAL, EXTENDED, threads=cpu_count)")
    print("=" * 60)
    
    sess_opt_1 = ort.SessionOptions()
    sess_opt_1.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_opt_1.intra_op_num_threads = cpu_count
    sess_opt_1.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opt_1.enable_cpu_mem_arena = True
    
    providers_1 = [('CPUExecutionProvider', {'arena_size': 256 * 1024 * 1024})]
    
    p50_1 = test_configuration(model_path, "Baseline", sess_opt_1, providers_1)
    results.append(("Baseline (EXTENDED, threads=4)", p50_1))
    print(f"✅ P50: {p50_1:.1f}ms\n")
    
    # Configuración 2: Graph optimization ALL
    print("=" * 60)
    print("🧪 Config 2: Graph OPT ALL (vs EXTENDED)")
    print("=" * 60)
    
    sess_opt_2 = ort.SessionOptions()
    sess_opt_2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opt_2.intra_op_num_threads = cpu_count
    sess_opt_2.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opt_2.enable_cpu_mem_arena = True
    
    providers_2 = [('CPUExecutionProvider', {'arena_size': 256 * 1024 * 1024})]
    
    p50_2 = test_configuration(model_path, "Graph ALL", sess_opt_2, providers_2)
    results.append(("Graph ALL (threads=4)", p50_2))
    print(f"✅ P50: {p50_2:.1f}ms (Δ: {p50_2 - p50_1:+.1f}ms)\n")
    
    # Configuración 3: Reducir threads (menos contención)
    print("=" * 60)
    print("🧪 Config 3: Threads reducidos (2 vs 4)")
    print("=" * 60)
    
    sess_opt_3 = ort.SessionOptions()
    sess_opt_3.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_opt_3.intra_op_num_threads = 2
    sess_opt_3.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opt_3.enable_cpu_mem_arena = True
    
    providers_3 = [('CPUExecutionProvider', {'arena_size': 256 * 1024 * 1024})]
    
    p50_3 = test_configuration(model_path, "Threads=2", sess_opt_3, providers_3)
    results.append(("Threads=2 (EXTENDED)", p50_3))
    print(f"✅ P50: {p50_3:.1f}ms (Δ: {p50_3 - p50_1:+.1f}ms)\n")
    
    # Configuración 4: Arena más pequeña
    print("=" * 60)
    print("🧪 Config 4: Arena size 128MB (vs 256MB)")
    print("=" * 60)
    
    sess_opt_4 = ort.SessionOptions()
    sess_opt_4.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_opt_4.intra_op_num_threads = cpu_count
    sess_opt_4.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opt_4.enable_cpu_mem_arena = True
    
    providers_4 = [('CPUExecutionProvider', {'arena_size': 128 * 1024 * 1024})]
    
    p50_4 = test_configuration(model_path, "Arena=128MB", sess_opt_4, providers_4)
    results.append(("Arena 128MB (threads=4)", p50_4))
    print(f"✅ P50: {p50_4:.1f}ms (Δ: {p50_4 - p50_1:+.1f}ms)\n")
    
    # Configuración 5: Graph ALL + Threads=2
    print("=" * 60)
    print("🧪 Config 5: Graph ALL + Threads=2 (combo)")
    print("=" * 60)
    
    sess_opt_5 = ort.SessionOptions()
    sess_opt_5.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opt_5.intra_op_num_threads = 2
    sess_opt_5.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opt_5.enable_cpu_mem_arena = True
    
    providers_5 = [('CPUExecutionProvider', {'arena_size': 128 * 1024 * 1024})]
    
    p50_5 = test_configuration(model_path, "ALL+Threads2+Arena128", sess_opt_5, providers_5)
    results.append(("Graph ALL + Threads=2 + Arena=128MB", p50_5))
    print(f"✅ P50: {p50_5:.1f}ms (Δ: {p50_5 - p50_1:+.1f}ms)\n")
    
    # Configuración 6: Single thread (eliminar overhead)
    print("=" * 60)
    print("🧪 Config 6: Single thread (threads=1)")
    print("=" * 60)
    
    sess_opt_6 = ort.SessionOptions()
    sess_opt_6.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_opt_6.intra_op_num_threads = 1
    sess_opt_6.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opt_6.enable_cpu_mem_arena = True
    
    providers_6 = [('CPUExecutionProvider', {'arena_size': 256 * 1024 * 1024})]
    
    p50_6 = test_configuration(model_path, "Threads=1", sess_opt_6, providers_6)
    results.append(("Single thread (threads=1)", p50_6))
    print(f"✅ P50: {p50_6:.1f}ms (Δ: {p50_6 - p50_1:+.1f}ms)\n")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 60)
    print()
    
    # Ordenar por latencia (menor a mayor)
    results_sorted = sorted(results, key=lambda x: x[1])
    
    for i, (config, latency) in enumerate(results_sorted, 1):
        delta = latency - p50_1
        symbol = "🏆" if i == 1 else "✅" if latency < 240 else "⚠️"
        print(f"{symbol} {i}. {config}")
        print(f"     P50: {latency:.1f}ms (Δ baseline: {delta:+.1f}ms)")
        print()
    
    # Mejor configuración
    best_config, best_latency = results_sorted[0]
    
    print("=" * 60)
    print("🏆 MEJOR CONFIGURACIÓN")
    print("=" * 60)
    print(f"Config: {best_config}")
    print(f"Latencia: {best_latency:.1f}ms")
    
    if best_latency < 240:
        print(f"✅ CUMPLE objetivo (<240ms)")
        print(f"   Margen: {240 - best_latency:.1f}ms")
    else:
        print(f"❌ NO CUMPLE objetivo (<240ms)")
        print(f"   Exceso: {best_latency - 240:.1f}ms ({((best_latency-240)/240)*100:.1f}%)")
    
    improvement_vs_baseline = ((p50_1 - best_latency) / p50_1) * 100
    print(f"\n📈 Mejora vs baseline: {improvement_vs_baseline:.1f}%")
    
    print("\n" + "=" * 60)
    
    return 0 if best_latency < 240 else 1


if __name__ == "__main__":
    sys.exit(main())
