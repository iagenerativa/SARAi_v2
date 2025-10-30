#!/usr/bin/env python3
"""
Script de prueba interactiva para SARAi v2.17
Modo TEXTO (sin audio) para validar pipeline básico
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.layer1_io.bert_embedder import BERTEmbedder
from core.layer1_io.lora_router import LoRARouter
from llama_cpp import Llama
import json
import time


def load_trm_cache():
    """Carga cache de respuestas TRM"""
    cache_path = Path("state/trm_cache.json")
    
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            "hola": "¡Hola! ¿En qué puedo ayudarte?",
            "cómo estás": "Muy bien, gracias por preguntar. ¿Y tú?",
            "adiós": "¡Hasta luego! Que tengas un buen día.",
            "gracias": "De nada, es un placer ayudarte.",
            "buenos días": "¡Buenos días! ¿Cómo puedo asistirte hoy?",
        }


def generate_trm_response(text: str, cache: dict) -> str:
    """Genera respuesta desde cache TRM"""
    text_normalized = text.lower().strip()
    
    # Buscar coincidencia exacta
    if text_normalized in cache:
        return cache[text_normalized]
    
    # Buscar coincidencia parcial
    for key, response in cache.items():
        if key in text_normalized or text_normalized in key:
            return response
    
    # Fallback
    return "Entiendo. ¿Puedes darme más detalles?"


def generate_llm_response(text: str, lfm2) -> str:
    """Genera respuesta con LFM2"""
    prompt = f"Usuario: {text}\nAsistente:"
    
    lfm2.reset()
    
    response = lfm2.create_completion(
        prompt,
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
        stop=["Usuario:", "\n\n"]
    )
    
    return response['choices'][0]['text'].strip()


def main():
    print("=" * 70)
    print("   SARAi v2.17 - Prueba Interactiva (Modo Texto)")
    print("=" * 70)
    print("\nCargando componentes...\n")
    
    # 1. BERT Embedder
    print("[1/3] BERT Embedder...")
    start = time.perf_counter()
    bert = BERTEmbedder()
    print(f"    ✓ BERT cargado en {(time.perf_counter() - start) * 1000:.0f}ms")
    
    # 2. LoRA Router
    print("[2/3] LoRA Router...")
    start = time.perf_counter()
    router = LoRARouter.load("models/lora_router.pt")
    router.eval()
    print(f"    ✓ Router cargado en {(time.perf_counter() - start) * 1000:.0f}ms")
    
    # 3. LFM2
    print("[3/3] LFM2-1.2B...")
    start = time.perf_counter()
    lfm2 = Llama(
        model_path="models/gguf/LFM2-1.2B-Q4_K_M.gguf",
        n_ctx=512,
        n_threads=4,
        verbose=False
    )
    print(f"    ✓ LFM2 cargado en {(time.perf_counter() - start) * 1000:.0f}ms")
    
    # 4. TRM Cache
    print("\n📚 Cargando TRM cache...")
    trm_cache = load_trm_cache()
    print(f"    ✓ {len(trm_cache)} respuestas en caché")
    
    print("\n✅ Todos los componentes listos\n")
    print("=" * 70)
    print("💬 Escribe 'salir' para terminar")
    print("=" * 70)
    print()
    
    while True:
        try:
            # Input del usuario
            user_input = input("👤 Tú: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("\n👋 ¡Hasta luego!")
                break
            
            # 1. Embedding con BERT
            start_total = time.perf_counter()
            embedding = bert.encode(user_input)
            
            # 2. Routing con LoRA
            start_route = time.perf_counter()
            result = router.predict(embedding)
            decision = result["decision"]
            confidence = result["confidence"]
            scores_dict = result["scores"]
            route_time = (time.perf_counter() - start_route) * 1000
            
            print(f"\n🧠 Router → {decision} (confianza: {confidence:.2%})")
            print(f"   Scores: TRM={scores_dict['TRM']:.3f}, LLM={scores_dict['LLM']:.3f}, Traducir={scores_dict['Traducir']:.3f}")
            print(f"   ⏱️  Routing: {route_time:.1f}ms")
            
            # 3. Generar respuesta según ruta
            start_gen = time.perf_counter()
            
            if decision == "TRM":
                response = generate_trm_response(user_input, trm_cache)
                gen_time = (time.perf_counter() - start_gen) * 1000
                print(f"   ⚡ TRM cache: {gen_time:.1f}ms")
            
            elif decision == "LLM":
                response = generate_llm_response(user_input, lfm2)
                gen_time = (time.perf_counter() - start_gen) * 1000
                print(f"   🤖 LLM generación: {gen_time:.0f}ms")
            
            else:  # Traducir
                print("   ⚠️  Traducción no implementada, usando LLM...")
                response = generate_llm_response(user_input, lfm2)
                gen_time = (time.perf_counter() - start_gen) * 1000
            
            total_time = (time.perf_counter() - start_total) * 1000
            
            # Mostrar respuesta
            print(f"\n🤖 SARAi: {response}")
            print(f"   ⏱️  Total: {total_time:.0f}ms\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
