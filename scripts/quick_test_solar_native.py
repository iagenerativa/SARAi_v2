#!/usr/bin/env python3
"""
Test rápido de SOLAR Native con GGUF de Ollama
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.solar_native import SolarNative
import time

print("=" * 70)
print("🚀 Test Rápido: SOLAR Native (GGUF de Ollama)")
print("=" * 70)

# Test 1: API compatible Upstage
print("\n📝 TEST 1: API Upstage (max_new_tokens=64)")
print("-" * 70)

start = time.time()
solar = SolarNative(context_mode="short", use_langchain=True, verbose=True)
load_time = time.time() - start

text = "Hi, my name is "
print(f"Input: '{text}'")

output = solar.generate_upstage_style(text, max_new_tokens=64)
print(f"\nOutput: '{output}'")

# Test 2: Pregunta técnica
print("\n\n📝 TEST 2: Pregunta técnica (español)")
print("-" * 70)

prompt = "Pregunta: ¿Qué es backpropagation en deep learning?\nRespuesta:"
response = solar.generate(prompt, max_tokens=150, temperature=0.3)
print(f"\n{response}")

# Stats
stats = solar.get_stats()
print("\n" + "=" * 70)
print("📊 ESTADÍSTICAS")
print("=" * 70)
print(f"Tiempo de carga:  {load_time:.2f}s")
print(f"RAM actual:       {stats['ram_mb']:.0f} MB")
print(f"Contexto:         {stats['n_ctx']} tokens")
print(f"Backend:          {stats['backend']}")
print(f"GGUF path:        {stats['model_path']}")
print("=" * 70)

solar.unload()
print("\n✅ Test completado")
