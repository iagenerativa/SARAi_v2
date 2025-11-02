#!/usr/bin/env python3
"""
SARAi v2.14 - Benchmark Wrapper Overhead
=========================================

Mide la latencia del Unified Wrapper vs uso directo de modelos.

Objetivo: Confirmar overhead <5%

Casos de prueba:
1. Ollama: Wrapper vs requests directo
2. Embeddings: Wrapper vs AutoModel directo
3. GGUF: Wrapper vs llama-cpp-python directo (si disponible)
"""

import os
import time
import statistics
import sys
from typing import List, Dict
import requests
import numpy as np

# Add parent directory to path
sys.path.insert(0, '/home/noel/SARAi_v2')

from core.unified_model_wrapper import get_model


class BenchmarkRunner:
    """Ejecuta benchmarks con múltiples iteraciones"""
    
    def __init__(self, iterations: int = 5, warmup: int = 1):
        self.iterations = iterations
        self.warmup = warmup
        self.results = {}
    
    def run(self, name: str, func, *args, **kwargs) -> Dict[str, float]:
        """Ejecuta benchmark con warmup"""
        print(f"\n🔬 Benchmark: {name}")
        print(f"   Warmup: {self.warmup} | Iterations: {self.iterations}")
        
        # Warmup
        for i in range(self.warmup):
            func(*args, **kwargs)
            print(f"   Warmup {i+1}/{self.warmup} ✓")
        
        # Mediciones reales
        latencies = []
        for i in range(self.iterations):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            
            latency = (end - start) * 1000  # ms
            latencies.append(latency)
            print(f"   Iteration {i+1}/{self.iterations}: {latency:.2f} ms")
        
        # Estadísticas
        stats = {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'min': min(latencies),
            'max': max(latencies),
            'latencies': latencies
        }
        
        print(f"   Mean: {stats['mean']:.2f} ms | Median: {stats['median']:.2f} ms | StdDev: {stats['stdev']:.2f} ms")
        
        self.results[name] = stats
        return stats
    
    def compare(self, baseline: str, wrapper: str) -> float:
        """Compara wrapper vs baseline y calcula overhead"""
        baseline_stats = self.results[baseline]
        wrapper_stats = self.results[wrapper]
        
        overhead_ms = wrapper_stats['mean'] - baseline_stats['mean']
        overhead_pct = (overhead_ms / baseline_stats['mean']) * 100
        
        print(f"\n📊 Comparación: {wrapper} vs {baseline}")
        print(f"   Baseline:  {baseline_stats['mean']:.2f} ms")
        print(f"   Wrapper:   {wrapper_stats['mean']:.2f} ms")
        print(f"   Overhead:  {overhead_ms:.2f} ms ({overhead_pct:.2f}%)")
        
        if overhead_pct < 5:
            print(f"   ✅ Overhead <5% (objetivo cumplido)")
        else:
            print(f"   ⚠️ Overhead ≥5% (revisar optimización)")
        
        return overhead_pct


def benchmark_ollama_wrapper(prompt: str = "Hola") -> str:
    """Benchmark: Unified Wrapper con Ollama"""
    solar = get_model("solar_short")
    response = solar.invoke(prompt)
    return response


def benchmark_ollama_direct(prompt: str = "Hola") -> str:
    """Benchmark: requests directo a Ollama API"""
    api_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    response = requests.post(
        f"{api_url.rstrip('/')}/api/generate",
        json={
            "model": "hf.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF:Q4_K_M",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        },
        timeout=30
    )
    
    data = response.json()
    return data.get("response", "")


def benchmark_embeddings_wrapper(text: str = "SARAi es una AGI local") -> np.ndarray:
    """Benchmark: Unified Wrapper con Embeddings"""
    embeddings = get_model("embeddings")
    vector = embeddings.invoke(text)
    return vector


def benchmark_embeddings_direct(text: str = "SARAi es una AGI local") -> np.ndarray:
    """Benchmark: AutoModel directo (sin wrapper)"""
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    # Cargar modelo directamente
    repo_id = "google/embeddinggemma-300m-qat-q4_0-unquantized"
    model = AutoModel.from_pretrained(repo_id, cache_dir="models/cache/embeddings")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir="models/cache/embeddings")
    
    # Encode
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    # L2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()[0]


def main():
    """Ejecuta todos los benchmarks"""
    print("=" * 80)
    print("SARAi v2.14 - Benchmark Wrapper Overhead")
    print("=" * 80)
    
    runner = BenchmarkRunner(iterations=5, warmup=1)
    
    # Test prompt corto
    test_prompt = "Hola"
    test_text = "SARAi es una AGI local"
    
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Ollama (SOLAR)")
    print("=" * 80)
    
    try:
        # Baseline: requests directo
        runner.run(
            "ollama_direct",
            benchmark_ollama_direct,
            test_prompt
        )
        
        # Wrapper
        runner.run(
            "ollama_wrapper",
            benchmark_ollama_wrapper,
            test_prompt
        )
        
        # Comparar
        overhead_ollama = runner.compare("ollama_direct", "ollama_wrapper")
        
    except Exception as e:
        print(f"⚠️ Ollama benchmark falló: {e}")
        overhead_ollama = None
    
    print("\n" + "=" * 80)
    print("BENCHMARK 2: Embeddings (EmbeddingGemma-300M)")
    print("=" * 80)
    
    try:
        # NOTA: Este test carga el modelo 2 veces (directo + wrapper)
        # En producción, wrapper usa cache, aquí medimos overhead puro
        
        print("\n⚠️ NOTA: Este benchmark carga el modelo 2 veces (directo + wrapper)")
        print("         En producción, wrapper usa cache y es más eficiente.\n")
        
        # Baseline: AutoModel directo (primera carga)
        print("📥 Cargando modelo directo (primera vez)...")
        runner.run(
            "embeddings_direct",
            benchmark_embeddings_direct,
            test_text
        )
        
        # Wrapper (segunda carga)
        print("📥 Cargando modelo wrapper (segunda vez)...")
        runner.run(
            "embeddings_wrapper",
            benchmark_embeddings_wrapper,
            test_text
        )
        
        # Comparar
        overhead_embeddings = runner.compare("embeddings_direct", "embeddings_wrapper")
        
    except Exception as e:
        print(f"⚠️ Embeddings benchmark falló: {e}")
        overhead_embeddings = None
    
    # Reporte final
    print("\n" + "=" * 80)
    print("📊 REPORTE FINAL")
    print("=" * 80)
    
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│                  OVERHEAD SUMMARY                           │")
    print("├─────────────────────────────────────────────────────────────┤")
    
    if overhead_ollama is not None:
        status_ollama = "✅" if overhead_ollama < 5 else "⚠️"
        print(f"│  Ollama (SOLAR):       {overhead_ollama:6.2f}%  {status_ollama}                    │")
    else:
        print("│  Ollama (SOLAR):       N/A     ⚠️ (servidor no disponible) │")
    
    if overhead_embeddings is not None:
        status_emb = "✅" if overhead_embeddings < 5 else "⚠️"
        print(f"│  Embeddings (Gemma):   {overhead_embeddings:6.2f}%  {status_emb}                    │")
    else:
        print("│  Embeddings (Gemma):   N/A     ⚠️ (error en carga)         │")
    
    print("└─────────────────────────────────────────────────────────────┘")
    
    # Calcular promedio si ambos disponibles
    if overhead_ollama is not None and overhead_embeddings is not None:
        avg_overhead = (overhead_ollama + overhead_embeddings) / 2
        print(f"\n🎯 OVERHEAD PROMEDIO: {avg_overhead:.2f}%")
        
        if avg_overhead < 5:
            print("✅ OBJETIVO CUMPLIDO: Overhead <5%")
        else:
            print("⚠️ OBJETIVO NO CUMPLIDO: Overhead ≥5% (revisar optimización)")
    
    # Conclusiones
    print("\n" + "=" * 80)
    print("💡 CONCLUSIONES")
    print("=" * 80)
    
    print("""
1. **Ollama Overhead**: 
   - Medido contra requests.post directo
   - Incluye overhead de clase wrapper + ModelRegistry lookup
   - En producción, cache reduce overhead en llamadas subsecuentes

2. **Embeddings Overhead**:
   - Medido contra AutoModel directo
   - Wrapper añade: cache management + interface LangChain
   - Nota: Test carga modelo 2 veces (no representa uso real)

3. **Recomendaciones**:
   - Si overhead >5%: Considerar lazy loading más agresivo
   - Si overhead <5%: Wrapper es eficiente para abstracción proporcionada
   - En producción: Cache de ModelRegistry amortiza overhead
    """)
    
    print("\n✅ Benchmark completado")
    print("=" * 80)


if __name__ == "__main__":
    main()
