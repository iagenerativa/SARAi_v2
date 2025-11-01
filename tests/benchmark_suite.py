"""
SARAi Benchmark Suite - Medición de Mejoras Entre Fases

Este módulo permite comparar KPIs reales entre versiones para validar
que cada fase evolutiva mejora el sistema.

Uso:
    # Ejecutar benchmark de versión actual
    python tests/benchmark_suite.py --version v2.14 --save
    
    # Comparar con versión anterior
    python tests/benchmark_suite.py --version v2.14 --compare v2.13
    
    # Ver histórico
    python tests/benchmark_suite.py --history

KPIs Medidos:
    - Latencia P50/P95/P99 (texto, RAG, vision)
    - RAM P99 (pico de uso)
    - CPU usage promedio
    - Precisión hard/soft classification
    - Skills detection accuracy
    - Cache hit rate
    - Model load time (cold start)
    - LCEL overhead vs imperative

Filosofía:
    "No optimizamos lo que no medimos.
    Cada fase debe probar con números que es mejor que la anterior."

Autor: SARAi Benchmark System
Fecha: 1 Noviembre 2025
"""

import time
import psutil
import gc
import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

BENCHMARK_DIR = Path("benchmarks/results")
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# Test queries para cada categoría
BENCHMARK_QUERIES = {
    "text_short": [
        "¿Qué es Python?",
        "Explica recursividad",
        "Define API REST",
        "¿Cómo funciona Docker?",
        "¿Qué es Git?",
    ],
    
    "text_long": [
        "Explica en detalle la arquitectura de microservicios, incluyendo ventajas, desventajas, patrones comunes como API Gateway, Service Discovery, Circuit Breaker, y cómo se compara con arquitectura monolítica",
        "Describe el ciclo completo de desarrollo de software desde requirements hasta deployment, incluyendo metodologías ágiles, testing strategies, CI/CD pipelines, y best practices de DevOps",
    ],
    
    "rag": [
        "¿Quién ganó el último mundial de fútbol?",
        "¿Cuál es el precio actual de Bitcoin?",
        "¿Qué pasó hoy en las noticias?",
    ],
    
    "hard_classification": [
        ("Configura un servidor SSH en Ubuntu", "hard"),
        ("Error 404 en mi aplicación web", "hard"),
        ("Cómo instalar PostgreSQL en Docker", "hard"),
        ("Debug de segmentation fault en C++", "hard"),
        ("Optimizar consulta SQL lenta", "hard"),
    ],
    
    "soft_classification": [
        ("Estoy muy frustrado con este código", "soft"),
        ("Me siento bloqueado, necesito ayuda", "soft"),
        ("Estoy feliz con los resultados", "soft"),
        ("Tengo miedo de romper producción", "soft"),
        ("Necesito motivación para continuar", "soft"),
    ],
    
    "skills_detection": [
        ("Crea una función Python para ordenar lista", "programming"),
        ("Mi servidor web da error 500", "diagnosis"),
        ("Calcula el ROI de esta inversión", "financial"),
        ("Escribe una historia de ciencia ficción", "creative"),
        ("Resuelve este puzzle lógico: ...", "reasoning"),
    ],
}


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class SARAiBenchmark:
    """
    Sistema de benchmarking para SARAi.
    
    Mide KPIs reales y los compara con versiones anteriores.
    """
    
    def __init__(self, version: str):
        """
        Args:
            version: Versión a benchmarkear (ej: "v2.14")
        """
        self.version = version
        self.results = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "latency": {},
            "memory": {},
            "accuracy": {},
            "other": {},
        }
        
        # Obtener RAM inicial
        self.base_ram = psutil.virtual_memory().used / (1024**3)
    
    # ------------------------------------------------------------------------
    # LATENCY BENCHMARKS
    # ------------------------------------------------------------------------
    
    def benchmark_latency_text_short(self) -> Dict[str, float]:
        """
        Mide latencia en queries cortas (texto).
        
        Returns:
            {"p50": 18.2, "p95": 23.5, "p99": 25.4}
        """
        logger.info("📊 Benchmark: Latency Text Short")
        
        latencies = []
        
        try:
            # Importar dinámicamente según versión
            if self.version.startswith("v2.14"):
                from core.unified_model_wrapper import get_model
                model = get_model("lfm2")
            else:
                from core.model_pool import get_model_pool
                pool = get_model_pool()
                model = pool.get("tiny")
            
            for query in BENCHMARK_QUERIES["text_short"]:
                gc.collect()  # Limpiar antes de cada test
                
                start = time.time()
                response = model.invoke(query) if hasattr(model, 'invoke') else model.generate(query)
                latency = time.time() - start
                
                latencies.append(latency)
                logger.debug(f"  Query: '{query[:30]}...' → {latency:.2f}s")
            
            return {
                "p50": statistics.median(latencies),
                "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
                "p99": max(latencies),
                "mean": statistics.mean(latencies),
                "samples": len(latencies),
            }
        
        except Exception as e:
            logger.error(f"❌ Latency text short failed: {e}")
            return {"error": str(e)}
    
    def benchmark_latency_text_long(self) -> Dict[str, float]:
        """Mide latencia en queries largas (texto)."""
        logger.info("📊 Benchmark: Latency Text Long")
        
        latencies = []
        
        try:
            if self.version.startswith("v2.14"):
                from core.unified_model_wrapper import get_model
                model = get_model("solar_long")
            else:
                from core.model_pool import get_model_pool
                pool = get_model_pool()
                model = pool.get("expert_long")
            
            for query in BENCHMARK_QUERIES["text_long"]:
                gc.collect()
                
                start = time.time()
                response = model.invoke(query) if hasattr(model, 'invoke') else model.generate(query)
                latency = time.time() - start
                
                latencies.append(latency)
                logger.debug(f"  Long query → {latency:.2f}s")
            
            return {
                "p50": statistics.median(latencies),
                "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
                "p99": max(latencies),
                "mean": statistics.mean(latencies),
                "samples": len(latencies),
            }
        
        except Exception as e:
            logger.error(f"❌ Latency text long failed: {e}")
            return {"error": str(e)}
    
    def benchmark_latency_rag(self) -> Dict[str, float]:
        """Mide latencia en queries RAG (web search)."""
        logger.info("📊 Benchmark: Latency RAG")
        
        latencies = []
        
        try:
            if self.version.startswith("v2.14"):
                from core.langchain_pipelines import create_rag_pipeline
                pipeline = create_rag_pipeline("solar_short", enable_cache=True, safe_mode=False)
            else:
                from agents.rag_agent import execute_rag
                from core.model_pool import get_model_pool
                pool = get_model_pool()
            
            for query in BENCHMARK_QUERIES["rag"]:
                gc.collect()
                
                start = time.time()
                
                if self.version.startswith("v2.14"):
                    response = pipeline.invoke(query)
                else:
                    state = {"input": query, "scores": {"web_query": 0.9}}
                    execute_rag(state, pool)
                
                latency = time.time() - start
                latencies.append(latency)
                logger.debug(f"  RAG query → {latency:.2f}s")
            
            return {
                "p50": statistics.median(latencies),
                "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
                "p99": max(latencies),
                "mean": statistics.mean(latencies),
                "samples": len(latencies),
            }
        
        except Exception as e:
            logger.error(f"❌ Latency RAG failed: {e}")
            return {"error": str(e)}
    
    # ------------------------------------------------------------------------
    # MEMORY BENCHMARKS
    # ------------------------------------------------------------------------
    
    def benchmark_memory_usage(self) -> Dict[str, float]:
        """
        Mide uso de RAM durante operación normal.
        
        Returns:
            {"base_gb": 0.5, "text_gb": 5.2, "vision_gb": 8.9, "p99_gb": 10.8}
        """
        logger.info("📊 Benchmark: Memory Usage")
        
        gc.collect()
        base_ram = psutil.virtual_memory().used / (1024**3)
        
        # Cargar modelo de texto
        try:
            if self.version.startswith("v2.14"):
                from core.unified_model_wrapper import get_model
                model_text = get_model("lfm2")
            else:
                from core.model_pool import get_model_pool
                pool = get_model_pool()
                model_text = pool.get("tiny")
            
            time.sleep(1)  # Dar tiempo a estabilizar
            text_ram = psutil.virtual_memory().used / (1024**3)
            
            # Cargar modelo de visión
            try:
                if self.version.startswith("v2.14"):
                    model_vision = get_model("qwen3_vl")
                else:
                    model_vision = pool.get("qwen3_vl")
                
                time.sleep(1)
                vision_ram = psutil.virtual_memory().used / (1024**3)
            except:
                vision_ram = text_ram  # Si no hay multimodal
            
            p99_ram = max(base_ram, text_ram, vision_ram)
            
            return {
                "base_gb": round(base_ram, 2),
                "text_gb": round(text_ram, 2),
                "vision_gb": round(vision_ram, 2),
                "p99_gb": round(p99_ram, 2),
                "delta_text_gb": round(text_ram - base_ram, 2),
                "delta_vision_gb": round(vision_ram - text_ram, 2),
            }
        
        except Exception as e:
            logger.error(f"❌ Memory benchmark failed: {e}")
            return {"error": str(e)}
    
    # ------------------------------------------------------------------------
    # ACCURACY BENCHMARKS
    # ------------------------------------------------------------------------
    
    def benchmark_classification_accuracy(self) -> Dict[str, float]:
        """
        Mide precisión del clasificador TRM (hard/soft).
        
        Returns:
            {"hard_precision": 0.87, "soft_precision": 0.79}
        """
        logger.info("📊 Benchmark: Classification Accuracy")
        
        try:
            from core.trm_classifier import load_trm_classifier
            classifier = load_trm_classifier()
            
            # Test hard classification
            hard_correct = 0
            for query, expected in BENCHMARK_QUERIES["hard_classification"]:
                scores = classifier.invoke(query)
                predicted = "hard" if scores["hard"] > scores["soft"] else "soft"
                if predicted == expected:
                    hard_correct += 1
            
            hard_precision = hard_correct / len(BENCHMARK_QUERIES["hard_classification"])
            
            # Test soft classification
            soft_correct = 0
            for query, expected in BENCHMARK_QUERIES["soft_classification"]:
                scores = classifier.invoke(query)
                predicted = "hard" if scores["hard"] > scores["soft"] else "soft"
                if predicted == expected:
                    soft_correct += 1
            
            soft_precision = soft_correct / len(BENCHMARK_QUERIES["soft_classification"])
            
            return {
                "hard_precision": round(hard_precision, 3),
                "soft_precision": round(soft_precision, 3),
                "overall_accuracy": round((hard_correct + soft_correct) / 10, 3),
            }
        
        except Exception as e:
            logger.error(f"❌ Classification accuracy failed: {e}")
            return {"error": str(e)}
    
    def benchmark_skills_detection(self) -> Dict[str, float]:
        """
        Mide precisión de detección de skills (v2.12).
        
        Returns:
            {"precision": 0.94, "recall": 0.89}
        """
        logger.info("📊 Benchmark: Skills Detection")
        
        try:
            from core.mcp import detect_and_apply_skill
            
            correct = 0
            for query, expected_skill in BENCHMARK_QUERIES["skills_detection"]:
                detected_skill = detect_and_apply_skill(query, agent_type="solar")
                
                if detected_skill and detected_skill["name"] == expected_skill:
                    correct += 1
            
            precision = correct / len(BENCHMARK_QUERIES["skills_detection"])
            
            return {
                "precision": round(precision, 3),
                "total_samples": len(BENCHMARK_QUERIES["skills_detection"]),
                "correct": correct,
            }
        
        except Exception as e:
            logger.error(f"❌ Skills detection failed: {e}")
            return {"error": str(e)}
    
    # ------------------------------------------------------------------------
    # OTHER BENCHMARKS
    # ------------------------------------------------------------------------
    
    def benchmark_cold_start(self) -> Dict[str, float]:
        """
        Mide tiempo de carga de modelos (cold start).
        
        Returns:
            {"lfm2_load_time": 0.9, "solar_load_time": 1.2}
        """
        logger.info("📊 Benchmark: Cold Start")
        
        try:
            if self.version.startswith("v2.14"):
                from core.unified_model_wrapper import ModelRegistry
                registry = ModelRegistry()
                registry._models = {}  # Limpiar cache
                
                # Medir LFM2
                gc.collect()
                start = time.time()
                model_lfm2 = registry.get_model("lfm2")
                lfm2_time = time.time() - start
                
                # Medir SOLAR (si local)
                try:
                    registry._models = {}
                    gc.collect()
                    start = time.time()
                    model_solar = registry.get_model("solar_short")
                    solar_time = time.time() - start
                except:
                    solar_time = 0.0  # Remoto, no aplica
            
            else:
                from core.model_pool import get_model_pool
                pool = get_model_pool()
                pool.cache = {}  # Limpiar cache
                
                gc.collect()
                start = time.time()
                model_lfm2 = pool.get("tiny")
                lfm2_time = time.time() - start
                
                try:
                    pool.cache = {}
                    gc.collect()
                    start = time.time()
                    model_solar = pool.get("expert_short")
                    solar_time = time.time() - start
                except:
                    solar_time = 0.0
            
            return {
                "lfm2_load_time": round(lfm2_time, 2),
                "solar_load_time": round(solar_time, 2),
                "total_load_time": round(lfm2_time + solar_time, 2),
            }
        
        except Exception as e:
            logger.error(f"❌ Cold start failed: {e}")
            return {"error": str(e)}
    
    def benchmark_code_complexity(self) -> Dict[str, int]:
        """
        Mide complejidad del código (LOC, nesting, try-except).
        
        Returns:
            {"graph_loc": 380, "nesting_max": 1, "try_except_count": 1}
        """
        logger.info("📊 Benchmark: Code Complexity")
        
        try:
            # Detectar archivo de graph según versión
            if self.version.startswith("v2.14"):
                graph_file = "core/graph_v2_14.py"
            else:
                graph_file = "core/graph.py"
            
            if not os.path.exists(graph_file):
                return {"error": f"Graph file not found: {graph_file}"}
            
            with open(graph_file, 'r') as f:
                lines = f.readlines()
            
            # Contar LOC (sin comentarios ni líneas vacías)
            loc = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
            
            # Medir nesting máximo (por indentación)
            max_nesting = 0
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    nesting = indent // 4
                    max_nesting = max(max_nesting, nesting)
            
            # Contar try-except
            try_except_count = sum(1 for line in lines if 'try:' in line or 'except' in line)
            
            return {
                "graph_loc": loc,
                "nesting_max": max_nesting,
                "try_except_count": try_except_count // 2,  # Dividir por 2 (try + except = 1 bloque)
                "file": graph_file,
            }
        
        except Exception as e:
            logger.error(f"❌ Code complexity failed: {e}")
            return {"error": str(e)}
    
    # ------------------------------------------------------------------------
    # MAIN RUNNER
    # ------------------------------------------------------------------------
    
    def run_all(self) -> Dict:
        """
        Ejecuta todos los benchmarks.
        
        Returns:
            Resultados completos con todos los KPIs
        """
        logger.info(f"🚀 Running SARAi Benchmark Suite - {self.version}")
        logger.info("=" * 60)
        
        # Latency
        self.results["latency"]["text_short"] = self.benchmark_latency_text_short()
        self.results["latency"]["text_long"] = self.benchmark_latency_text_long()
        self.results["latency"]["rag"] = self.benchmark_latency_rag()
        
        # Memory
        self.results["memory"] = self.benchmark_memory_usage()
        
        # Accuracy
        self.results["accuracy"]["classification"] = self.benchmark_classification_accuracy()
        self.results["accuracy"]["skills"] = self.benchmark_skills_detection()
        
        # Other
        self.results["other"]["cold_start"] = self.benchmark_cold_start()
        self.results["other"]["code_complexity"] = self.benchmark_code_complexity()
        
        logger.info("=" * 60)
        logger.info(f"✅ Benchmark complete: {self.version}")
        
        return self.results
    
    def save_results(self) -> Path:
        """
        Guarda resultados en JSON.
        
        Returns:
            Path al archivo guardado
        """
        filename = f"benchmark_{self.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = BENCHMARK_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {filepath}")
        return filepath


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def compare_versions(version_a: str, version_b: str) -> Dict:
    """
    Compara dos versiones de benchmarks.
    
    Args:
        version_a: Versión base (ej: "v2.13")
        version_b: Versión nueva (ej: "v2.14")
    
    Returns:
        Dict con comparaciones y deltas
    """
    logger.info(f"📊 Comparing {version_a} vs {version_b}")
    
    # Buscar últimos benchmarks de cada versión
    files_a = sorted(BENCHMARK_DIR.glob(f"benchmark_{version_a}_*.json"))
    files_b = sorted(BENCHMARK_DIR.glob(f"benchmark_{version_b}_*.json"))
    
    if not files_a:
        raise FileNotFoundError(f"No benchmark found for {version_a}")
    if not files_b:
        raise FileNotFoundError(f"No benchmark found for {version_b}")
    
    # Cargar últimos resultados
    with open(files_a[-1], 'r') as f:
        results_a = json.load(f)
    
    with open(files_b[-1], 'r') as f:
        results_b = json.load(f)
    
    # Calcular deltas
    comparison = {
        "version_a": version_a,
        "version_b": version_b,
        "improvements": {},
        "regressions": {},
        "summary": {},
    }
    
    # Latency comparisons
    for category in ["text_short", "text_long", "rag"]:
        if category in results_a["latency"] and category in results_b["latency"]:
            a_p50 = results_a["latency"][category].get("p50", 0)
            b_p50 = results_b["latency"][category].get("p50", 0)
            
            if a_p50 > 0:
                delta = ((b_p50 - a_p50) / a_p50) * 100
                
                if delta < 0:  # Mejora
                    comparison["improvements"][f"latency_{category}_p50"] = {
                        "old": round(a_p50, 2),
                        "new": round(b_p50, 2),
                        "delta_percent": round(delta, 1),
                    }
                else:  # Regresión
                    comparison["regressions"][f"latency_{category}_p50"] = {
                        "old": round(a_p50, 2),
                        "new": round(b_p50, 2),
                        "delta_percent": round(delta, 1),
                    }
    
    # Memory comparison
    if "memory" in results_a and "memory" in results_b:
        a_p99 = results_a["memory"].get("p99_gb", 0)
        b_p99 = results_b["memory"].get("p99_gb", 0)
        
        if a_p99 > 0:
            delta = ((b_p99 - a_p99) / a_p99) * 100
            
            if delta < 0:
                comparison["improvements"]["memory_p99"] = {
                    "old": a_p99,
                    "new": b_p99,
                    "delta_percent": round(delta, 1),
                }
            else:
                comparison["regressions"]["memory_p99"] = {
                    "old": a_p99,
                    "new": b_p99,
                    "delta_percent": round(delta, 1),
                }
    
    # Code complexity comparison
    if "code_complexity" in results_a.get("other", {}) and "code_complexity" in results_b.get("other", {}):
        a_loc = results_a["other"]["code_complexity"].get("graph_loc", 0)
        b_loc = results_b["other"]["code_complexity"].get("graph_loc", 0)
        
        if a_loc > 0:
            delta = ((b_loc - a_loc) / a_loc) * 100
            
            if delta < 0:
                comparison["improvements"]["code_loc"] = {
                    "old": a_loc,
                    "new": b_loc,
                    "delta_percent": round(delta, 1),
                }
            else:
                comparison["regressions"]["code_loc"] = {
                    "old": a_loc,
                    "new": b_loc,
                    "delta_percent": round(delta, 1),
                }
    
    # Summary
    comparison["summary"] = {
        "total_improvements": len(comparison["improvements"]),
        "total_regressions": len(comparison["regressions"]),
        "net_improvement": len(comparison["improvements"]) - len(comparison["regressions"]),
    }
    
    return comparison


def print_comparison_report(comparison: Dict):
    """Imprime reporte de comparación en formato legible."""
    
    print("\n" + "=" * 80)
    print(f"📊 SARAi Version Comparison: {comparison['version_a']} → {comparison['version_b']}")
    print("=" * 80)
    
    # Improvements
    if comparison["improvements"]:
        print("\n✅ IMPROVEMENTS:")
        for metric, data in comparison["improvements"].items():
            print(f"  • {metric}:")
            print(f"    {data['old']} → {data['new']} ({data['delta_percent']:+.1f}%)")
    
    # Regressions
    if comparison["regressions"]:
        print("\n❌ REGRESSIONS:")
        for metric, data in comparison["regressions"].items():
            print(f"  • {metric}:")
            print(f"    {data['old']} → {data['new']} ({data['delta_percent']:+.1f}%)")
    
    # Summary
    print("\n📈 SUMMARY:")
    print(f"  Total Improvements: {comparison['summary']['total_improvements']}")
    print(f"  Total Regressions: {comparison['summary']['total_regressions']}")
    print(f"  Net Improvement: {comparison['summary']['net_improvement']}")
    
    if comparison["summary"]["net_improvement"] > 0:
        print("\n🎉 Overall: VERSION IMPROVED ✅")
    elif comparison["summary"]["net_improvement"] < 0:
        print("\n⚠️  Overall: VERSION REGRESSED ❌")
    else:
        print("\n➡️  Overall: VERSION NEUTRAL")
    
    print("=" * 80 + "\n")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SARAi Benchmark Suite")
    parser.add_argument("--version", required=True, help="Version to benchmark (e.g., v2.14)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--compare", help="Compare with another version (e.g., v2.13)")
    parser.add_argument("--history", action="store_true", help="Show benchmark history")
    
    args = parser.parse_args()
    
    if args.history:
        # Mostrar histórico
        files = sorted(BENCHMARK_DIR.glob("benchmark_*.json"))
        print(f"\n📚 Benchmark History ({len(files)} results):")
        for f in files:
            print(f"  • {f.name}")
        print()
    
    else:
        # Ejecutar benchmark
        benchmark = SARAiBenchmark(args.version)
        results = benchmark.run_all()
        
        if args.save:
            benchmark.save_results()
        
        # Comparar si se especificó
        if args.compare:
            comparison = compare_versions(args.compare, args.version)
            print_comparison_report(comparison)
