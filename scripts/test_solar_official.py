#!/usr/bin/env python3
"""
Test SOLAR Native con GGUF oficial descargado
Usando wrapper mejorado con sugerencias de Upstage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.solar_native import SolarNative
import time

print("=" * 70)
print("🚀 SOLAR Native - Modelo Oficial de Hugging Face")
print("=" * 70)

# Ruta al GGUF oficial descargado
official_gguf = "models/solar/solar-10.7b-instruct-v1.0.Q4_K_M.gguf"

print(f"\n📦 Usando: {official_gguf}")
print(f"📊 Tamaño: 6.1 GB (Q4_K_M)")

# Test 1: API compatible Upstage
print("\n" + "=" * 70)
print("📝 TEST 1: API Upstage (max_new_tokens=64)")
print("=" * 70)

start = time.time()
solar = SolarNative(
    model_path=official_gguf,
    context_mode="short",
    use_langchain=True,
    verbose=True
)
load_time = time.time() - start

text = "Hi, my name is "
print(f"\nInput: '{text}'")

output = solar.generate_upstage_style(text, max_new_tokens=64)
print(f"\n✅ Output completo:")
print(f"'{output}'")
print(f"\n✅ Solo generado:")
print(f"'{output[len(text):]}'")

# Test 2: Pregunta técnica en español
print("\n" + "=" * 70)
print("📝 TEST 2: Pregunta Técnica (Español)")
print("=" * 70)

prompt = """Pregunta: ¿Qué es backpropagation en deep learning?
Responde de forma clara y técnica en máximo 3 líneas.

Respuesta:"""

print(f"Prompt:\n{prompt}\n")

response = solar.generate(prompt, max_tokens=200, temperature=0.3)
print(f"Respuesta:\n{response}")

# Test 3: Explicación detallada (long context)
print("\n" + "=" * 70)
print("📝 TEST 3: Explicación Detallada (Long Context)")
print("=" * 70)

solar.unload()  # Liberar short
print("Cambiando a modo long context (2048 tokens)...\n")

solar_long = SolarNative(
    model_path=official_gguf,
    context_mode="long",
    use_langchain=True,
    verbose=True
)

prompt_long = """Contexto: Los transformers revolucionaron el NLP en 2017.

Pregunta: Explica detalladamente la arquitectura transformer, incluyendo:
1. Mecanismo de atención (Query, Key, Value)
2. Multi-head attention
3. Positional encoding
4. Feed-forward networks

Respuesta técnica:"""

print(f"Prompt:\n{prompt_long}\n")

response_long = solar_long.generate(prompt_long, max_tokens=512, temperature=0.5)
print(f"Respuesta:\n{response_long}")

# Stats finales
print("\n" + "=" * 70)
print("📊 ESTADÍSTICAS FINALES")
print("=" * 70)

stats = solar_long.get_stats()
print(f"Tiempo de carga inicial: {load_time:.2f}s")
print(f"RAM actual:              {stats['ram_mb']:.0f} MB")
print(f"Modelo:                  {stats['model']}")
print(f"Contexto:                {stats['context_mode']} ({stats['n_ctx']} tokens)")
print(f"Backend:                 {stats['backend']}")
print(f"GGUF path:               {stats['model_path']}")

solar_long.unload()

print("\n" + "=" * 70)
print("✅ Test completado - Modelo oficial Upstage validado")
print("=" * 70)
