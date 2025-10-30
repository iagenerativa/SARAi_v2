#!/usr/bin/env python3
"""
Test de Latencia DIRECTA - LFM2-1.2B GGUF

Carga LFM2 directamente desde archivo local sin ModelPool
para medir latencia real de carga e inferencia.
"""

import pytest
import time
from pathlib import Path
import psutil


class TestLFM2LatencyDirect:
    """Test de latencia con carga directa de LFM2"""
    
    def test_lfm2_load_and_inference(self):
        """Cargar LFM2 y medir latencia de inferencia real"""
        from llama_cpp import Llama
        
        print("\n" + "="*70)
        print("🧠 TEST DE LATENCIA: LFM2-1.2B (CARGA DIRECTA)")
        print("="*70)
        
        # Path del modelo local
        model_path = "models/lfm2/LFM2-1.2B-Q4_K_M.gguf"
        
        assert Path(model_path).exists(), f"❌ Modelo no encontrado: {model_path}"
        
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        print(f"\n📦 Modelo: {model_path}")
        print(f"📊 Tamaño: {size_mb:.1f} MB")
        
        # Medir RAM antes
        process = psutil.Process()
        ram_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # 1. LATENCIA DE CARGA
        print(f"\n[1] Cargando modelo...")
        start_load = time.perf_counter()
        
        llm = Llama(
            model_path=model_path,
            n_ctx=512,  # Contexto corto para test rápido
            n_threads=6,
            verbose=False
        )
        
        load_time_ms = (time.perf_counter() - start_load) * 1000
        
        # Medir RAM después
        ram_after = process.memory_info().rss / (1024 * 1024)  # MB
        ram_used = ram_after - ram_before
        
        print(f"✅ Cargado en: {load_time_ms:.2f} ms")
        print(f"💾 RAM usada: {ram_used:.1f} MB")
        
        # 2. LATENCIA DE INFERENCIA (Warm-up)
        print(f"\n[2] Warm-up (primera inferencia)...")
        prompt = "Hola, ¿cómo estás?"
        
        start_warmup = time.perf_counter()
        output_warmup = llm(
            prompt,
            max_tokens=10,
            temperature=0.7,
            echo=False
        )
        warmup_time_ms = (time.perf_counter() - start_warmup) * 1000
        
        print(f"✅ Warm-up: {warmup_time_ms:.2f} ms")
        print(f"📝 Output: {output_warmup['choices'][0]['text'][:50]}...")
        
        # 3. LATENCIA DE INFERENCIA (Promedio 5 runs)
        print(f"\n[3] Benchmarking inferencia (5 runs)...")
        
        test_prompts = [
            "¿Qué es Python?",
            "Explica qué es un LLM",
            "¿Cómo funciona la IA?",
            "Define machine learning",
            "¿Qué es un transformer?"
        ]
        
        inference_times = []
        tokens_generated = []
        
        for i, test_prompt in enumerate(test_prompts, 1):
            # Reset del contexto antes de cada prompt
            llm.reset()
            
            start_inf = time.perf_counter()
            
            output = llm(
                test_prompt,
                max_tokens=20,  # Respuesta corta
                temperature=0.7,
                echo=False
            )
            
            inf_time_ms = (time.perf_counter() - start_inf) * 1000
            inference_times.append(inf_time_ms)
            
            # Contar tokens generados (aproximado)
            tokens = len(output['choices'][0]['text'].split())
            tokens_generated.append(tokens)
            
            print(f"  Run {i}: {inf_time_ms:6.2f} ms ({tokens} tokens)")
        
        # Estadísticas
        avg_inference = sum(inference_times) / len(inference_times)
        min_inference = min(inference_times)
        max_inference = max(inference_times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        
        # Tokens por segundo
        total_tokens = sum(tokens_generated)
        total_time_s = sum(inference_times) / 1000
        tokens_per_sec = total_tokens / total_time_s if total_time_s > 0 else 0
        
        # RESUMEN
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN DE LATENCIAS LFM2-1.2B")
        print(f"{'='*70}")
        print(f"  Carga del modelo:        {load_time_ms:8.2f} ms")
        print(f"  Warm-up (1st run):       {warmup_time_ms:8.2f} ms")
        print(f"  Inferencia promedio:     {avg_inference:8.2f} ms")
        print(f"  Inferencia mínima:       {min_inference:8.2f} ms")
        print(f"  Inferencia máxima:       {max_inference:8.2f} ms")
        print(f"  RAM usada:               {ram_used:8.1f} MB")
        print(f"  Tokens/segundo:          {tokens_per_sec:8.1f} tok/s")
        print(f"{'='*70}")
        
        # Validaciones
        assert load_time_ms < 5000, f"Carga muy lenta: {load_time_ms:.0f}ms > 5000ms"
        assert avg_inference < 2000, f"Inferencia lenta: {avg_inference:.0f}ms > 2000ms"
        assert ram_used < 1300, f"RAM excesiva: {ram_used:.0f}MB > 1300MB"
        
        print(f"\n✅ Objetivos de latencia CUMPLIDOS")
        print(f"   Carga:      <5000 ms → {load_time_ms:.0f} ms ✅")
        print(f"   Inferencia: <2000 ms → {avg_inference:.0f} ms ✅")
        print(f"   RAM:        <1300 MB → {ram_used:.0f} MB ✅")
        
        return {
            'load_time_ms': load_time_ms,
            'warmup_time_ms': warmup_time_ms,
            'avg_inference_ms': avg_inference,
            'min_inference_ms': min_inference,
            'max_inference_ms': max_inference,
            'ram_used_mb': ram_used,
            'tokens_per_sec': tokens_per_sec
        }
    
    def test_lfm2_context_scaling(self):
        """Test de latencia con diferentes tamaños de contexto"""
        from llama_cpp import Llama
        
        print("\n" + "="*70)
        print("📏 TEST: ESCALABILIDAD DE CONTEXTO LFM2")
        print("="*70)
        
        model_path = "models/lfm2/LFM2-1.2B-Q4_K_M.gguf"
        assert Path(model_path).exists(), f"Modelo no encontrado: {model_path}"
        
        # Test con diferentes contextos
        contexts = [512, 1024, 2048]
        results = []
        
        for n_ctx in contexts:
            print(f"\n📐 Testeando n_ctx={n_ctx}...")
            
            start = time.perf_counter()
            llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=6,
                verbose=False
            )
            load_time = (time.perf_counter() - start) * 1000
            
            # Inferencia con prompt que llena parte del contexto
            prompt = "Explica " * (n_ctx // 10) + "qué es la IA?"
            
            start_inf = time.perf_counter()
            output = llm(prompt[:min(len(prompt), n_ctx//2)], max_tokens=20, echo=False)
            inf_time = (time.perf_counter() - start_inf) * 1000
            
            print(f"  ✅ Carga: {load_time:.2f} ms, Inferencia: {inf_time:.2f} ms")
            
            results.append({
                'n_ctx': n_ctx,
                'load_ms': load_time,
                'inference_ms': inf_time
            })
            
            del llm  # Liberar memoria
        
        # Resumen comparativo
        print(f"\n{'='*70}")
        print(f"📊 COMPARATIVA DE CONTEXTOS")
        print(f"{'='*70}")
        print(f"{'Contexto':>10} | {'Carga (ms)':>12} | {'Inferencia (ms)':>16}")
        print(f"{'-'*10}-+-{'-'*12}-+-{'-'*16}")
        
        for r in results:
            print(f"{r['n_ctx']:10d} | {r['load_ms']:12.2f} | {r['inference_ms']:16.2f}")
        
        print(f"{'='*70}")
        
        # Validar que n_ctx=2048 es manejable
        ctx_2048 = next(r for r in results if r['n_ctx'] == 2048)
        assert ctx_2048['load_ms'] < 8000, f"n_ctx=2048 muy lento: {ctx_2048['load_ms']:.0f}ms"
        
        print(f"\n✅ n_ctx=2048 manejable: {ctx_2048['load_ms']:.0f} ms carga")
