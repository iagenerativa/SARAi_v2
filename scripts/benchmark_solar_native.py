#!/usr/bin/env python3
"""
Benchmark: Ollama vs Native SOLAR
Comparación de rendimiento entre servidor Ollama y acceso directo GGUF

Author: SARAi v2.16 Integration Team
Date: 2025-10-28
"""

import time
import json
import psutil
from pathlib import Path
from typing import Dict, List

# Import our native wrapper
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.solar_native import SolarNative

# Ollama client
try:
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama client no disponible")


class BenchmarkRunner:
    """
    Ejecuta benchmark comparativo Ollama vs Native
    """
    
    def __init__(self):
        self.results = {
            "ollama": [],
            "native_short": [],
            "native_long": []
        }
        
        self.test_prompts = [
            {
                "name": "Short Technical",
                "prompt": "Pregunta: ¿Qué es backpropagation?\nRespuesta breve:",
                "max_tokens": 150,
                "context_mode": "short"
            },
            {
                "name": "Medium Explanation",
                "prompt": "Explica cómo funciona un transformer en deep learning:",
                "max_tokens": 300,
                "context_mode": "short"
            },
            {
                "name": "Long Detailed",
                "prompt": """Contexto: Los modelos transformer son la base de LLMs modernos.

Pregunta: Explica detalladamente la arquitectura transformer, incluyendo:
1. Encoder y decoder
2. Mecanismo de atención
3. Positional encoding
4. Feed-forward networks

Respuesta técnica completa:""",
                "max_tokens": 512,
                "context_mode": "long"
            }
        ]
    
    def benchmark_ollama(self) -> Dict:
        """
        Benchmark con Ollama server
        """
        if not OLLAMA_AVAILABLE:
            print("⚠️  Ollama no disponible. Skip.")
            return {"error": "Ollama not available"}
        
        print("\n" + "=" * 70)
        print("🔵 BENCHMARK: Ollama Server")
        print("=" * 70)
        
        client = Client(host="http://localhost:11434")
        results = []
        
        for i, test in enumerate(self.test_prompts, 1):
            print(f"\n📝 Test {i}/{len(self.test_prompts)}: {test['name']}")
            print("-" * 70)
            
            try:
                # Medir RAM antes
                process = psutil.Process()
                ram_before_mb = process.memory_info().rss / (1024**2)
                
                # Generación
                start_time = time.time()
                
                response = client.generate(
                    model="solar:10.7b",
                    prompt=test["prompt"],
                    options={
                        "num_predict": test["max_tokens"],
                        "temperature": 0.7,
                        "top_p": 0.95
                    },
                    stream=False
                )
                
                elapsed = time.time() - start_time
                
                # Medir RAM después
                ram_after_mb = process.memory_info().rss / (1024**2)
                
                # Calcular tokens (aproximado)
                tokens = len(response["response"].split())
                tok_per_sec = tokens / elapsed if elapsed > 0 else 0
                
                result = {
                    "test": test["name"],
                    "backend": "ollama",
                    "elapsed_sec": elapsed,
                    "tokens": tokens,
                    "tok_per_sec": tok_per_sec,
                    "ram_mb": ram_after_mb,
                    "ram_delta_mb": ram_after_mb - ram_before_mb,
                    "response_preview": response["response"][:200]
                }
                
                results.append(result)
                
                print(f"⏱️  Tiempo: {elapsed:.2f}s")
                print(f"🚀 Velocidad: {tok_per_sec:.2f} tok/s")
                print(f"💾 RAM: {ram_after_mb:.0f} MB (Δ {ram_delta_mb:+.0f} MB)")
                print(f"📄 Preview: {result['response_preview']}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    "test": test["name"],
                    "backend": "ollama",
                    "error": str(e)
                })
        
        self.results["ollama"] = results
        return results
    
    def benchmark_native(self, context_mode: str = "short") -> Dict:
        """
        Benchmark con Native wrapper
        """
        print("\n" + "=" * 70)
        print(f"🟢 BENCHMARK: Native SOLAR ({context_mode} context)")
        print("=" * 70)
        
        # Cargar modelo una vez
        solar = SolarNative(
            context_mode=context_mode,
            use_langchain=True,
            verbose=False
        )
        
        results = []
        
        # Filtrar tests por context_mode
        filtered_tests = [
            t for t in self.test_prompts
            if t["context_mode"] == context_mode
        ]
        
        for i, test in enumerate(filtered_tests, 1):
            print(f"\n📝 Test {i}/{len(filtered_tests)}: {test['name']}")
            print("-" * 70)
            
            try:
                # Medir RAM antes
                process = psutil.Process()
                ram_before_mb = process.memory_info().rss / (1024**2)
                
                # Generación
                start_time = time.time()
                
                response = solar.generate(
                    prompt=test["prompt"],
                    max_tokens=test["max_tokens"],
                    temperature=0.7,
                    top_p=0.95
                )
                
                elapsed = time.time() - start_time
                
                # Medir RAM después
                ram_after_mb = process.memory_info().rss / (1024**2)
                
                # Calcular tokens
                tokens = len(response.split())
                tok_per_sec = tokens / elapsed if elapsed > 0 else 0
                
                result = {
                    "test": test["name"],
                    "backend": f"native_{context_mode}",
                    "elapsed_sec": elapsed,
                    "tokens": tokens,
                    "tok_per_sec": tok_per_sec,
                    "ram_mb": ram_after_mb,
                    "ram_delta_mb": ram_after_mb - ram_before_mb,
                    "response_preview": response[:200]
                }
                
                results.append(result)
                
                print(f"⏱️  Tiempo: {elapsed:.2f}s")
                print(f"🚀 Velocidad: {tok_per_sec:.2f} tok/s")
                print(f"💾 RAM: {ram_after_mb:.0f} MB (Δ {ram_delta_mb:+.0f} MB)")
                print(f"📄 Preview: {result['response_preview']}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    "test": test["name"],
                    "backend": f"native_{context_mode}",
                    "error": str(e)
                })
        
        # Limpiar
        solar.unload()
        
        self.results[f"native_{context_mode}"] = results
        return results
    
    def generate_report(self):
        """
        Genera reporte comparativo
        """
        print("\n" + "=" * 70)
        print("📊 REPORTE COMPARATIVO: Ollama vs Native SOLAR")
        print("=" * 70)
        
        # Tabla comparativa
        print("\n📈 Velocidad de Inferencia (tok/s)")
        print("-" * 70)
        print(f"{'Test':<25} {'Ollama':>12} {'Native Short':>12} {'Native Long':>12}")
        print("-" * 70)
        
        # Agrupar por nombre de test
        test_names = list(set(
            r["test"] for results in self.results.values()
            for r in results if "error" not in r
        ))
        
        for test_name in test_names:
            row = [test_name[:24]]
            
            # Ollama
            ollama_result = next(
                (r for r in self.results.get("ollama", []) if r.get("test") == test_name),
                None
            )
            row.append(f"{ollama_result['tok_per_sec']:.2f}" if ollama_result and "error" not in ollama_result else "N/A")
            
            # Native short
            native_short_result = next(
                (r for r in self.results.get("native_short", []) if r.get("test") == test_name),
                None
            )
            row.append(f"{native_short_result['tok_per_sec']:.2f}" if native_short_result and "error" not in native_short_result else "N/A")
            
            # Native long
            native_long_result = next(
                (r for r in self.results.get("native_long", []) if r.get("test") == test_name),
                None
            )
            row.append(f"{native_long_result['tok_per_sec']:.2f}" if native_long_result and "error" not in native_long_result else "N/A")
            
            print(f"{row[0]:<25} {row[1]:>12} {row[2]:>12} {row[3]:>12}")
        
        # RAM usage
        print("\n💾 Uso de RAM (MB)")
        print("-" * 70)
        print(f"{'Test':<25} {'Ollama':>12} {'Native Short':>12} {'Native Long':>12}")
        print("-" * 70)
        
        for test_name in test_names:
            row = [test_name[:24]]
            
            # Ollama
            ollama_result = next(
                (r for r in self.results.get("ollama", []) if r.get("test") == test_name),
                None
            )
            row.append(f"{ollama_result['ram_mb']:.0f}" if ollama_result and "error" not in ollama_result else "N/A")
            
            # Native short
            native_short_result = next(
                (r for r in self.results.get("native_short", []) if r.get("test") == test_name),
                None
            )
            row.append(f"{native_short_result['ram_mb']:.0f}" if native_short_result and "error" not in native_short_result else "N/A")
            
            # Native long
            native_long_result = next(
                (r for r in self.results.get("native_long", []) if r.get("test") == test_name),
                None
            )
            row.append(f"{native_long_result['ram_mb']:.0f}" if native_long_result and "error" not in native_long_result else "N/A")
            
            print(f"{row[0]:<25} {row[1]:>12} {row[2]:>12} {row[3]:>12}")
        
        # Conclusiones
        print("\n🎯 CONCLUSIONES")
        print("-" * 70)
        
        # Calcular promedios
        ollama_avg_speed = sum(
            r["tok_per_sec"] for r in self.results.get("ollama", [])
            if "error" not in r
        ) / max(len([r for r in self.results.get("ollama", []) if "error" not in r]), 1)
        
        native_short_avg_speed = sum(
            r["tok_per_sec"] for r in self.results.get("native_short", [])
            if "error" not in r
        ) / max(len([r for r in self.results.get("native_short", []) if "error" not in r]), 1)
        
        native_long_avg_speed = sum(
            r["tok_per_sec"] for r in self.results.get("native_long", [])
            if "error" not in r
        ) / max(len([r for r in self.results.get("native_long", []) if "error" not in r]), 1)
        
        print(f"Velocidad promedio Ollama:       {ollama_avg_speed:.2f} tok/s")
        print(f"Velocidad promedio Native Short: {native_short_avg_speed:.2f} tok/s")
        print(f"Velocidad promedio Native Long:  {native_long_avg_speed:.2f} tok/s")
        
        # Mejor opción
        print("\n✅ RECOMENDACIÓN:")
        if native_short_avg_speed > ollama_avg_speed * 0.9:  # Native es ≥90% de Ollama
            print("   → Usar Native wrapper (mejor estabilidad, sin dependencia de servidor)")
        else:
            print(f"   → Ollama es {(ollama_avg_speed/native_short_avg_speed - 1)*100:.1f}% más rápido")
        
        # Guardar resultados
        output_path = Path("logs") / f"solar_benchmark_{int(time.time())}.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📁 Resultados guardados: {output_path}")


def main():
    """
    Ejecuta benchmark completo
    """
    print("=" * 70)
    print("SOLAR Benchmark: Ollama vs Native")
    print("=" * 70)
    
    runner = BenchmarkRunner()
    
    # Benchmark Ollama (si está disponible)
    if OLLAMA_AVAILABLE:
        try:
            runner.benchmark_ollama()
        except Exception as e:
            print(f"⚠️  Ollama benchmark falló: {e}")
    
    # Benchmark Native (short context)
    runner.benchmark_native(context_mode="short")
    
    # Benchmark Native (long context)
    runner.benchmark_native(context_mode="long")
    
    # Generar reporte
    runner.generate_report()
    
    print("\n" + "=" * 70)
    print("✅ Benchmark completado")
    print("=" * 70)


if __name__ == "__main__":
    main()
