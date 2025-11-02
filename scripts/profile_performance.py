#!/usr/bin/env python3
"""
Performance Profiler para SARAi

FASE 5: Optimización - Profiling de CPU y Memoria
Fecha: 2 Noviembre 2025

Uso:
    python scripts/profile_performance.py --target graph --duration 60
    python scripts/profile_performance.py --target mcp --profile memory
    python scripts/profile_performance.py --target fast-lane --output reports/
"""

import os
import sys
import time
import cProfile
import pstats
import argparse
from pathlib import Path
from io import StringIO
from datetime import datetime

# Memory profiling (opcional, requiere memory_profiler)
try:
    from memory_profiler import profile as memory_profile
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    print("⚠️  memory_profiler no instalado. Instalar con: pip install memory-profiler")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class PerformanceProfiler:
    """Profiler unificado para CPU y memoria"""
    
    def __init__(self, output_dir: str = "reports/profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def profile_cpu(self, target_func, *args, **kwargs):
        """
        Profiling de CPU con cProfile
        
        Args:
            target_func: Función a perfilar
            *args, **kwargs: Argumentos para target_func
        
        Returns:
            Resultado de target_func
        """
        profiler = cProfile.Profile()
        
        print(f"🔍 Iniciando profiling de CPU: {target_func.__name__}")
        
        profiler.enable()
        result = target_func(*args, **kwargs)
        profiler.disable()
        
        # Generar reporte
        stats = pstats.Stats(profiler)
        
        # Guardar stats raw
        stats_file = self.output_dir / f"cpu_{target_func.__name__}_{self.timestamp}.prof"
        stats.dump_stats(str(stats_file))
        print(f"✅ Stats guardados: {stats_file}")
        
        # Generar reporte legible
        report_file = self.output_dir / f"cpu_{target_func.__name__}_{self.timestamp}.txt"
        with open(report_file, "w") as f:
            stream = StringIO()
            stats_sorted = pstats.Stats(profiler, stream=stream)
            
            # Top 50 funciones por tiempo total
            stats_sorted.sort_stats('cumulative')
            stats_sorted.print_stats(50)
            
            f.write("=" * 80 + "\n")
            f.write(f"CPU PROFILING REPORT: {target_func.__name__}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write("=" * 80 + "\n\n")
            f.write(stream.getvalue())
        
        print(f"📄 Reporte generado: {report_file}")
        
        # Mostrar top 10 en consola
        print("\n" + "=" * 80)
        print("🔥 TOP 10 FUNCIONES MÁS COSTOSAS (cumulative time)")
        print("=" * 80)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return result
    
    def profile_memory(self, target_func, *args, **kwargs):
        """
        Profiling de memoria con memory_profiler
        
        Requiere: pip install memory-profiler
        """
        if not HAS_MEMORY_PROFILER:
            print("❌ memory_profiler no disponible. Instalar con:")
            print("   pip install memory-profiler")
            return None
        
        print(f"🧠 Iniciando profiling de memoria: {target_func.__name__}")
        
        # Decorar función con @profile
        profiled_func = memory_profile(target_func)
        
        # Ejecutar con profiling
        result = profiled_func(*args, **kwargs)
        
        print(f"✅ Profiling de memoria completado")
        
        return result
    
    def profile_line_by_line(self, target_file: str):
        """
        Profiling línea por línea con line_profiler
        
        Requiere: pip install line-profiler
        """
        print(f"📝 Line-by-line profiling no implementado aún")
        print(f"   Usar: kernprof -l -v {target_file}")


def profile_graph_execution(duration: int = 60):
    """Perfilar ejecución del grafo SARAi"""
    from core.graph import compile_sarai_graph
    
    print(f"🔄 Perfiling graph execution ({duration}s)...")
    
    graph = compile_sarai_graph()
    
    test_queries = [
        "¿Cómo está el clima?",
        "Explícame la teoría de la relatividad",
        "Error 404 en servidor",
        "Me siento frustrado hoy",
        "Configura SSH en Linux",
    ]
    
    def run_graph_batch():
        """Ejecutar batch de queries"""
        for query in test_queries:
            state = {"input": query}
            for event in graph.stream(state):
                if "response" in event:
                    break
    
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < duration:
        run_graph_batch()
        iterations += 1
    
    elapsed = time.time() - start_time
    queries_per_sec = (iterations * len(test_queries)) / elapsed
    
    print(f"✅ Procesadas {iterations * len(test_queries)} queries en {elapsed:.2f}s")
    print(f"   Throughput: {queries_per_sec:.2f} queries/s")
    
    return iterations


def profile_mcp_decisions():
    """Perfilar MCP compute_weights"""
    from core.mcp import MCP
    
    print("🧠 Perfiling MCP decisions...")
    
    mcp = MCP()
    
    test_cases = [
        ({"hard": 0.9, "soft": 0.1}, "technical query"),
        ({"hard": 0.2, "soft": 0.8}, "emotional query"),
        ({"hard": 0.5, "soft": 0.5}, "hybrid query"),
    ]
    
    def run_mcp_batch():
        """Ejecutar batch de decisiones MCP"""
        for scores, context in test_cases:
            alpha, beta = mcp.compute_weights(scores, context)
    
    # Ejecutar 1000 iteraciones
    for _ in range(1000):
        run_mcp_batch()
    
    print("✅ MCP batch completado")


def profile_fast_lane():
    """Perfilar Fast Lane end-to-end"""
    print("🏃 Perfiling Fast Lane...")
    
    # Importar tests
    sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
    from test_fast_lane_latency import TestFastLaneLatency
    
    tester = TestFastLaneLatency()
    tester.setup()
    
    # Ejecutar 10 runs (reducido vs 30 normal)
    queries = tester.load_golden_queries()
    critical_queries = tester.filter_critical_queries(queries)
    
    def run_fast_lane():
        """Ejecutar Fast Lane con queries críticas"""
        for query_obj in critical_queries[:3]:  # Solo 3 queries
            query = query_obj["input"]
            _ = tester.measure_latency(query)
    
    run_fast_lane()
    
    print("✅ Fast Lane profiling completado")


def main():
    parser = argparse.ArgumentParser(
        description="Performance Profiler para SARAi"
    )
    parser.add_argument(
        "--target",
        choices=["graph", "mcp", "fast-lane", "all"],
        default="graph",
        help="Componente a perfilar"
    )
    parser.add_argument(
        "--profile",
        choices=["cpu", "memory", "both"],
        default="cpu",
        help="Tipo de profiling"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duración del profiling en segundos (solo para graph)"
    )
    parser.add_argument(
        "--output",
        default="reports/profiling",
        help="Directorio de output para reportes"
    )
    
    args = parser.parse_args()
    
    profiler = PerformanceProfiler(output_dir=args.output)
    
    print("=" * 80)
    print("🔬 SARAi Performance Profiler")
    print("=" * 80)
    print(f"Target:   {args.target}")
    print(f"Profile:  {args.profile}")
    print(f"Output:   {args.output}")
    print("=" * 80)
    print()
    
    # Seleccionar target
    target_funcs = {
        "graph": lambda: profile_graph_execution(args.duration),
        "mcp": profile_mcp_decisions,
        "fast-lane": profile_fast_lane,
    }
    
    targets = [args.target] if args.target != "all" else ["graph", "mcp", "fast-lane"]
    
    for target_name in targets:
        target_func = target_funcs[target_name]
        
        print(f"\n{'=' * 80}")
        print(f"🎯 Profiling: {target_name}")
        print(f"{'=' * 80}\n")
        
        if args.profile in ["cpu", "both"]:
            profiler.profile_cpu(target_func)
        
        if args.profile in ["memory", "both"]:
            if HAS_MEMORY_PROFILER:
                profiler.profile_memory(target_func)
            else:
                print("⚠️  Skipping memory profiling (memory_profiler no instalado)")
    
    print("\n" + "=" * 80)
    print("✅ Profiling completado")
    print(f"📂 Reportes en: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
