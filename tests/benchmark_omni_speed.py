#!/usr/bin/env python3
"""
Benchmark de Velocidad Omni-7B - Optimización CPU

Mide throughput real (tokens/segundo) con las optimizaciones aplicadas:
- n_batch=512 (vs 8 default)
- auto_reduce_context para queries cortas
- Configuraciones FP16, mmap, etc.

TARGET: 7-15 tok/s (vs 4 tok/s baseline)

Uso:
    python3 tests/benchmark_omni_speed.py
"""

import sys
import os
import time
from statistics import mean, median

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.omni_native import get_omni_agent


def estimate_tokens(text: str) -> int:
    """Estimación rápida de tokens (1 token ≈ 4 chars)"""
    return len(text) // 4


def benchmark_throughput(agent, queries: list, iterations: int = 3):
    """
    Mide throughput real (tokens/segundo)
    
    Args:
        agent: OmniNativeAgent
        queries: Lista de queries de test
        iterations: Iteraciones por query
    
    Returns:
        dict con métricas
    """
    results = []
    
    for idx, query in enumerate(queries):
        print(f"\n📝 Query {idx + 1}/{len(queries)}: \"{query[:50]}...\"")
        
        query_results = []
        
        for iteration in range(iterations):
            start = time.perf_counter()
            response = agent.invoke(query)
            end = time.perf_counter()
            
            latency = end - start
            tokens = estimate_tokens(response)
            tok_per_sec = tokens / latency if latency > 0 else 0
            
            query_results.append({
                "latency": latency,
                "tokens": tokens,
                "tok_per_sec": tok_per_sec
            })
            
            print(f"   Iter {iteration + 1}: {latency:.2f}s, {tokens} tokens, {tok_per_sec:.1f} tok/s")
        
        # Consolidar resultados de esta query
        avg_tok_per_sec = mean([r["tok_per_sec"] for r in query_results])
        avg_latency = mean([r["latency"] for r in query_results])
        avg_tokens = int(mean([r["tokens"] for r in query_results]))
        
        results.append({
            "query": query,
            "avg_latency": avg_latency,
            "avg_tokens": avg_tokens,
            "avg_tok_per_sec": avg_tok_per_sec
        })
    
    return results


def print_results_table(results: list):
    """Imprime tabla de resultados"""
    print("\n" + "="*80)
    print("📊 RESULTADOS DEL BENCHMARK")
    print("="*80)
    
    print(f"\n{'Query':<50} {'Avg Lat (s)':<15} {'Tokens':<10} {'Tok/s':<10}")
    print("-"*80)
    
    for result in results:
        query_preview = result['query'][:47] + "..." if len(result['query']) > 47 else result['query']
        print(f"{query_preview:<50} "
              f"{result['avg_latency']:<15.2f} "
              f"{result['avg_tokens']:<10} "
              f"{result['avg_tok_per_sec']:<10.1f}")
    
    # Métricas globales
    avg_global_tok_per_sec = mean([r["avg_tok_per_sec"] for r in results])
    median_global_tok_per_sec = median([r["avg_tok_per_sec"] for r in results])
    
    print("\n" + "="*80)
    print("🎯 MÉTRICAS GLOBALES")
    print("="*80)
    print(f"Promedio throughput:  {avg_global_tok_per_sec:.1f} tok/s")
    print(f"Mediana throughput:   {median_global_tok_per_sec:.1f} tok/s")
    
    # Validación KPI
    print("\n🎯 VALIDACIÓN KPI:")
    target_min = 7
    target_max = 15
    
    if avg_global_tok_per_sec >= target_min and avg_global_tok_per_sec <= target_max:
        status = "✅ CUMPLIDO"
    elif avg_global_tok_per_sec > target_max:
        status = f"⭐ SUPERADO ({avg_global_tok_per_sec - target_max:.1f} tok/s extra)"
    else:
        status = f"⚠️  BAJO TARGET ({target_min - avg_global_tok_per_sec:.1f} tok/s faltantes)"
    
    print(f"   Target: {target_min}-{target_max} tok/s")
    print(f"   Real:   {avg_global_tok_per_sec:.1f} tok/s")
    print(f"   {status}")
    
    # Comparación con baseline
    baseline = 4.0
    improvement = ((avg_global_tok_per_sec - baseline) / baseline) * 100
    
    print(f"\n📈 MEJORA vs BASELINE:")
    print(f"   Baseline (v2.16.0): {baseline} tok/s")
    print(f"   Optimizado (v2.16.1): {avg_global_tok_per_sec:.1f} tok/s")
    print(f"   Mejora: {improvement:+.1f}%")


def main():
    """Ejecuta benchmark completo"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                 BENCHMARK DE VELOCIDAD OMNI-7B v2.16.1                  ║
║                                                                          ║
║  Optimizaciones CPU aplicadas:                                          ║
║  - n_batch=512 (vs 8 default)                                           ║
║  - auto_reduce_context (queries cortas = 256 tokens)                    ║
║  - FP16 KV cache + mmap + sampling optimizado                           ║
║                                                                          ║
║  Target: 7-15 tok/s (vs 4 tok/s baseline)                               ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Queries de test (mix: cortas y largas)
    test_queries = [
        # Queries CORTAS (<100 chars) - auto_reduce activado (256 tokens)
        "Hola, ¿cómo estás?",
        "¿Qué tal el día?",
        "Cuéntame un chiste corto",
        
        # Queries LARGAS (>100 chars) - max_tokens completo (2048)
        "Explícame qué es la física cuántica de forma detallada pero accesible",
        "Dame consejos prácticos para mejorar mi productividad en el trabajo",
        "Cuéntame una historia corta sobre un gato que viaja en el tiempo"
    ]
    
    print("\n🚀 Cargando Omni-7B Agent...")
    agent = get_omni_agent()
    print("✅ Agente listo\n")
    
    print(f"📋 Configuración:")
    print(f"   Queries: {len(test_queries)}")
    print(f"   Iteraciones por query: 3")
    print(f"   Auto-reduce context: {agent.config.auto_reduce_context}")
    print(f"   n_batch: {agent.config.n_batch}")
    print(f"   n_threads: {agent.config.n_threads}")
    
    # Ejecutar benchmark
    results = benchmark_throughput(agent, test_queries, iterations=3)
    
    # Imprimir resultados
    print_results_table(results)
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    main()
