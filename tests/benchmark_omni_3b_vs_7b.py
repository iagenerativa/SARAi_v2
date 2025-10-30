#!/usr/bin/env python3
"""
Benchmark Comparativo: Omni-3B vs Omni-7B

Mide throughput REAL de ambos modelos con queries idénticas:
- Omni-3B (Fast): Target 9-12 tok/s
- Omni-7B (Quality): Baseline 4 tok/s

Valida la hipótesis: 3B es 2.3x más rápido que 7B

Uso:
    python3 tests/benchmark_omni_3b_vs_7b.py
"""

import sys
import os
import time
from statistics import mean, median
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.omni_fast import get_omni_fast_agent
from agents.omni_native import get_omni_agent


def estimate_tokens(text: str) -> int:
    """Estimación rápida de tokens (1 token ≈ 4 chars)"""
    return len(text) // 4


def benchmark_model(agent, model_name: str, queries: List[str], iterations: int = 3) -> List[Dict]:
    """
    Benchmarking de un modelo específico
    
    Args:
        agent: Agente (OmniFastAgent o OmniNativeAgent)
        model_name: Nombre del modelo ("Omni-3B" o "Omni-7B")
        queries: Lista de queries
        iterations: Iteraciones por query
    
    Returns:
        Lista de resultados por query
    """
    print(f"\n{'='*80}")
    print(f"🚀 BENCHMARKING: {model_name}")
    print(f"{'='*80}")
    
    results = []
    
    for idx, query in enumerate(queries):
        print(f"\n📝 Query {idx + 1}/{len(queries)}: \"{query[:60]}...\"")
        
        query_results = []
        
        for iteration in range(iterations):
            start = time.perf_counter()
            response = agent.invoke(query, max_tokens=100)  # Fijo 100 tokens para comparabilidad
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
        
        # Consolidar resultados
        avg_tok_per_sec = mean([r["tok_per_sec"] for r in query_results])
        avg_latency = mean([r["latency"] for r in query_results])
        avg_tokens = int(mean([r["tokens"] for r in query_results]))
        
        results.append({
            "query": query,
            "model": model_name,
            "avg_latency": avg_latency,
            "avg_tokens": avg_tokens,
            "avg_tok_per_sec": avg_tok_per_sec
        })
    
    return results


def print_comparison_table(results_3b: List[Dict], results_7b: List[Dict]):
    """Imprime tabla comparativa"""
    print("\n" + "="*100)
    print("📊 COMPARACIÓN OMNI-3B vs OMNI-7B")
    print("="*100)
    
    print(f"\n{'Query':<40} {'Modelo':<12} {'Lat (s)':<10} {'Tokens':<8} {'Tok/s':<10} {'vs Baseline':<12}")
    print("-"*100)
    
    for r3b, r7b in zip(results_3b, results_7b):
        query_preview = r3b['query'][:37] + "..." if len(r3b['query']) > 37 else r3b['query']
        
        # Omni-7B (baseline)
        print(f"{query_preview:<40} "
              f"{'7B (base)':<12} "
              f"{r7b['avg_latency']:<10.2f} "
              f"{r7b['avg_tokens']:<8} "
              f"{r7b['avg_tok_per_sec']:<10.1f} "
              f"{'---':<12}")
        
        # Omni-3B (fast)
        speedup = (r7b['avg_latency'] / r3b['avg_latency']) if r3b['avg_latency'] > 0 else 0
        improvement = ((r3b['avg_tok_per_sec'] - r7b['avg_tok_per_sec']) / r7b['avg_tok_per_sec'] * 100) if r7b['avg_tok_per_sec'] > 0 else 0
        
        print(f"{'':<40} "
              f"{'3B (fast)':<12} "
              f"{r3b['avg_latency']:<10.2f} "
              f"{r3b['avg_tokens']:<8} "
              f"{r3b['avg_tok_per_sec']:<10.1f} "
              f"{improvement:+.0f}% ({speedup:.1f}x)")
        
        print()
    
    # Métricas globales
    avg_3b = mean([r["avg_tok_per_sec"] for r in results_3b])
    avg_7b = mean([r["avg_tok_per_sec"] for r in results_7b])
    
    global_speedup = avg_3b / avg_7b if avg_7b > 0 else 0
    global_improvement = ((avg_3b - avg_7b) / avg_7b * 100) if avg_7b > 0 else 0
    
    print("="*100)
    print("🎯 MÉTRICAS GLOBALES")
    print("="*100)
    
    print(f"\nOmni-7B (Baseline):")
    print(f"  • Throughput promedio: {avg_7b:.1f} tok/s")
    print(f"  • Latencia promedio:   {mean([r['avg_latency'] for r in results_7b]):.1f} s")
    
    print(f"\nOmni-3B (Fast):")
    print(f"  • Throughput promedio: {avg_3b:.1f} tok/s")
    print(f"  • Latencia promedio:   {mean([r['avg_latency'] for r in results_3b]):.1f} s")
    
    print(f"\n🚀 MEJORA GLOBAL:")
    print(f"  • Speedup:     {global_speedup:.2f}x más rápido")
    print(f"  • Improvement: {global_improvement:+.1f}% throughput")
    
    # Validación KPI 3B
    print(f"\n🎯 VALIDACIÓN KPI OMNI-3B:")
    target_min = 7  # Mínimo del rango 7-15 tok/s
    target_max = 15
    
    if avg_3b >= target_min and avg_3b <= target_max:
        status = "✅ CUMPLIDO"
    elif avg_3b > target_max:
        status = f"⭐ SUPERADO ({avg_3b - target_max:.1f} tok/s extra)"
    else:
        status = f"⚠️  BAJO TARGET ({target_min - avg_3b:.1f} tok/s faltantes)"
    
    print(f"  • Target:      {target_min}-{target_max} tok/s")
    print(f"  • Real:        {avg_3b:.1f} tok/s")
    print(f"  • {status}")
    
    # Validación hipótesis
    expected_speedup = 2.3
    speedup_diff = abs(global_speedup - expected_speedup)
    
    print(f"\n📐 VALIDACIÓN HIPÓTESIS (3B = 2.3x más rápido que 7B):")
    print(f"  • Esperado:    {expected_speedup:.1f}x")
    print(f"  • Real:        {global_speedup:.2f}x")
    
    if speedup_diff < 0.5:
        print(f"  • ✅ HIPÓTESIS CONFIRMADA (diferencia {speedup_diff:.2f}x)")
    else:
        print(f"  • ⚠️  DESVIACIÓN SIGNIFICATIVA (diferencia {speedup_diff:.2f}x)")


def main():
    """Ejecuta benchmark comparativo"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           BENCHMARK COMPARATIVO: OMNI-3B vs OMNI-7B v2.16.1            ║
║                                                                          ║
║  Hipótesis a validar:                                                    ║
║  • Omni-3B: 9-12 tok/s (2.3x más rápido que 7B)                         ║
║  • Omni-7B: 4 tok/s (baseline)                                          ║
║                                                                          ║
║  Metodología:                                                            ║
║  • Queries idénticas para ambos modelos                                 ║
║  • max_tokens=100 fijo (comparabilidad)                                 ║
║  • 3 iteraciones por query (promedio)                                   ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Queries de test (mix conversacional)
    test_queries = [
        "Hola, ¿cómo estás hoy?",
        "Cuéntame un chiste corto",
        "¿Qué opinas del clima?",
        "Dame un consejo para estudiar mejor",
        "Explícame qué es la inteligencia artificial"
    ]
    
    print("\n📋 Configuración:")
    print(f"   Queries: {len(test_queries)}")
    print(f"   Iteraciones por query: 3")
    print(f"   Max tokens: 100 (fijo)")
    
    # Cargar agentes
    print("\n🚀 Cargando agentes...")
    print("\n[1/2] Cargando Omni-3B (Fast)...")
    agent_3b = get_omni_fast_agent()
    
    print("\n[2/2] Cargando Omni-7B (Quality)...")
    agent_7b = get_omni_agent()
    
    print("\n✅ Ambos agentes listos")
    
    # Benchmark Omni-3B
    results_3b = benchmark_model(agent_3b, "Omni-3B", test_queries, iterations=3)
    
    # Benchmark Omni-7B
    results_7b = benchmark_model(agent_7b, "Omni-7B", test_queries, iterations=3)
    
    # Comparación
    print_comparison_table(results_3b, results_7b)
    
    print("\n" + "="*100)
    print("✅ BENCHMARK COMPLETADO")
    print("="*100)


if __name__ == "__main__":
    main()
