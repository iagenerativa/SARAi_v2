#!/usr/bin/env python3
"""
Test de rendimiento de modelos nativos con llama-cpp-python
SARAi v2.16 - Stack completo sin Ollama
"""

import time
import psutil
from pathlib import Path
from typing import Dict, Any
import json

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ llama-cpp-python no instalado")
    print("   pip install llama-cpp-python")
    exit(1)


class ModelBenchmark:
    """Benchmark de modelos GGUF con métricas detalladas"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 6):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llm = None
        self.metrics = {}
    
    def load_model(self) -> Dict[str, Any]:
        """Carga modelo y mide tiempo/RAM"""
        print(f"\n{'='*60}")
        print(f"CARGANDO: {self.model_path.name}")
        print(f"{'='*60}")
        
        # Medir RAM antes
        process = psutil.Process()
        ram_before = process.memory_info().rss / (1024**3)  # GB
        
        # Medir tiempo de carga
        start_time = time.time()
        
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False
        )
        
        load_time = time.time() - start_time
        
        # Medir RAM después
        ram_after = process.memory_info().rss / (1024**3)
        ram_used = ram_after - ram_before
        
        self.metrics['load_time_s'] = round(load_time, 2)
        self.metrics['ram_used_gb'] = round(ram_used, 2)
        self.metrics['model_size_gb'] = round(self.model_path.stat().st_size / (1024**3), 2)
        
        print(f"⏱️  Tiempo de carga: {load_time:.2f}s")
        print(f"💾 RAM usada: {ram_used:.2f} GB")
        print(f"📦 Tamaño archivo: {self.metrics['model_size_gb']} GB")
        
        return self.metrics
    
    def benchmark_inference(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Ejecuta inferencia y mide rendimiento"""
        print(f"\n{'─'*60}")
        print(f"TEST DE INFERENCIA")
        print(f"{'─'*60}")
        print(f"Prompt: {prompt[:80]}...")
        
        start_time = time.time()
        
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False
        )
        
        inference_time = time.time() - start_time
        
        # Métricas de la respuesta
        text = response['choices'][0]['text'].strip()
        tokens_generated = response['usage']['completion_tokens']
        tokens_prompt = response['usage']['prompt_tokens']
        
        # Calcular tok/s
        tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0
        
        self.metrics['inference_time_s'] = round(inference_time, 2)
        self.metrics['tokens_generated'] = tokens_generated
        self.metrics['tokens_prompt'] = tokens_prompt
        self.metrics['tokens_per_sec'] = round(tokens_per_sec, 2)
        self.metrics['response_preview'] = text[:200]
        
        print(f"\n💬 Respuesta ({tokens_generated} tokens):")
        print(f"   {text[:150]}...")
        print(f"\n📊 Métricas:")
        print(f"   ⏱️  Tiempo: {inference_time:.2f}s")
        print(f"   🚀 Velocidad: {tokens_per_sec:.2f} tok/s")
        print(f"   📝 Prompt: {tokens_prompt} tokens")
        print(f"   📤 Generados: {tokens_generated} tokens")
        
        return self.metrics
    
    def unload(self):
        """Libera memoria del modelo"""
        if self.llm:
            del self.llm
            self.llm = None
    
    def get_results(self) -> Dict[str, Any]:
        """Retorna todas las métricas"""
        return {
            'model': self.model_path.name,
            'n_ctx': self.n_ctx,
            'n_threads': self.n_threads,
            **self.metrics
        }


def test_all_models():
    """Test completo del stack v2.16"""
    
    print("\n" + "="*70)
    print(" SARAi v2.16 - BENCHMARK NATIVO DE MODELOS (llama-cpp-python)")
    print("="*70)
    
    models_config = [
        {
            'name': 'SOLAR-10.7B',
            'path': 'models/solar/solar-10.7b-instruct-v1.0.Q4_K_M.gguf',
            'prompt': 'Explica backpropagation en deep learning con detalle técnico sobre la función de pérdida y el gradiente.',
            'role': 'Expert (Hard Skills)',
            'n_ctx': 2048
        },
        {
            'name': 'LFM2-1.2B',
            'path': 'models/lfm2/LFM2-1.2B-Q4_K_M.gguf',
            'prompt': 'Estoy muy frustrado porque mi código no funciona y llevo 3 horas intentándolo. No sé qué más hacer.',
            'role': 'Tiny Tier (Soft Skills)',
            'n_ctx': 2048
        },
        {
            'name': 'Qwen3-VL-4B-Instruct',
            'path': 'models/qwen_omni/Qwen3-VL-4B-Instruct-Q5_K_M.gguf',
            'prompt': 'Explica qué es un asistente multimodal y por qué es útil procesando audio y texto.',
            'role': 'Omni-Loop (Reflexión)',
            'n_ctx': 2048
        }
    ]
    
    results = []
    total_ram = 0
    
    for config in models_config:
        model_path = Path(config['path'])
        
        if not model_path.exists():
            print(f"\n⚠️  SKIP: {config['name']} no encontrado en {model_path}")
            print(f"   Descargar desde HuggingFace primero")
            continue
        
        print(f"\n{'#'*70}")
        print(f"# MODELO: {config['name']} - {config['role']}")
        print(f"{'#'*70}")
        
        benchmark = ModelBenchmark(
            model_path=str(model_path),
            n_ctx=config['n_ctx'],
            n_threads=6
        )
        
        # Cargar
        benchmark.load_model()
        
        # Test de inferencia
        benchmark.benchmark_inference(
            prompt=config['prompt'],
            max_tokens=200
        )
        
        # Guardar resultados
        result = benchmark.get_results()
        result['role'] = config['role']
        results.append(result)
        
        total_ram += result.get('ram_used_gb', 0)
        
        # Liberar memoria antes del siguiente
        benchmark.unload()
        
        time.sleep(2)  # Pausa entre modelos
    
    # Resumen final
    print(f"\n{'='*70}")
    print(" RESUMEN DEL STACK v2.16")
    print(f"{'='*70}\n")
    
    print(f"{'Modelo':<25} {'Rol':<20} {'RAM (GB)':<10} {'Tok/s':<10} {'Archivo (GB)':<12}")
    print("-" * 70)
    
    for r in results:
        print(
            f"{r['model'][:24]:<25} "
            f"{r['role'][:19]:<20} "
            f"{r.get('ram_used_gb', 0):<10.2f} "
            f"{r.get('tokens_per_sec', 0):<10.2f} "
            f"{r.get('model_size_gb', 0):<12.2f}"
        )
    
    print("-" * 70)
    print(f"{'TOTAL':<45} {total_ram:<10.2f} {'':10} {sum(r.get('model_size_gb', 0) for r in results):<12.2f}")
    
    # Guardar JSON
    output_file = f"logs/model_benchmark_{int(time.time())}.json"
    Path("logs").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'stack_version': 'v2.16',
            'backend': 'llama-cpp-python',
            'total_ram_gb': round(total_ram, 2),
            'total_size_gb': round(sum(r.get('model_size_gb', 0) for r in results), 2),
            'models': results
        }, f, indent=2)
    
    print(f"\n✅ Resultados guardados en: {output_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    try:
        results = test_all_models()
        
        # Validar KPIs v2.16
        print(f"\n{'='*70}")
        print(" VALIDACIÓN DE KPIs v2.16")
        print(f"{'='*70}\n")
        
        total_ram = sum(r.get('ram_used_gb', 0) for r in results)
        
        # KPI 1: RAM total ≤ 12GB
        if total_ram <= 12.0:
            print(f"✅ RAM Total: {total_ram:.2f} GB ≤ 12 GB")
        else:
            print(f"❌ RAM Total: {total_ram:.2f} GB > 12 GB (FALLO)")
        
        # KPI 2: Velocidad mínima
        expert_speed = next((r['tokens_per_sec'] for r in results if 'SOLAR' in r['model']), 0)
        tiny_speed = next((r['tokens_per_sec'] for r in results if 'LFM2' in r['model']), 0)
        
        if expert_speed >= 3.0:
            print(f"✅ Expert Speed: {expert_speed:.2f} tok/s ≥ 3.0 tok/s")
        else:
            print(f"⚠️  Expert Speed: {expert_speed:.2f} tok/s < 3.0 tok/s")
        
        if tiny_speed >= 8.0:
            print(f"✅ Tiny Speed: {tiny_speed:.2f} tok/s ≥ 8.0 tok/s")
        else:
            print(f"⚠️  Tiny Speed: {tiny_speed:.2f} tok/s < 8.0 tok/s")
        
        print("\n" + "="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
