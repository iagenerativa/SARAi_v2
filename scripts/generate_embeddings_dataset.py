#!/usr/bin/env python3
"""
Genera dataset de TRM con embeddings reales de EmbeddingGemma (768-D)
Reemplaza TF-IDF por embeddings semánticos de calidad.
"""

import json
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embeddings import get_embedding_model


# Templates de queries (copiados de generate_trm_dataset.py original)
HARD_QUERIES = [
    "¿Cómo configurar SSH en Ubuntu 22.04?",
    "Error al importar numpy en Python 3.11",
    "Diferencia entre async/await y Promises en JavaScript",
    "¿Cómo optimizar una query SQL con JOIN?",
    "Bug en mi código: segmentation fault en C++",
    "Instalar Docker en Raspberry Pi 4",
    "¿Qué es una función recursiva en Python?",
    "Configurar firewall con iptables",
    "¿Cómo usar git rebase interactivo?",
    "Diferencia entre class y interface en Java",
]

SOFT_QUERIES = [
    "Me siento frustrado con este bug, llevo horas",
    "Gracias por tu ayuda, eres muy claro explicando",
    "Estoy perdido, no entiendo nada de este código",
    "Explícame Python como si fuera un niño de 10 años",
    "Estoy cansado de intentar y fallar",
    "¿Puedes ser más paciente conmigo?",
    "No me juzgues, soy principiante en esto",
    "Agradezco mucho tu tiempo",
    "Me siento motivado a seguir aprendiendo",
    "Esto es muy difícil para mí",
]

WEB_QUERIES = [
    "¿Quién ganó el Oscar 2025?",
    "¿Cuándo fue la final de la Copa del Mundo 2026?",
    "¿Cómo está el clima en Tokio hoy?",
    "Precio actual de Bitcoin",
    "Últimas noticias de tecnología",
    "¿Quién es el presidente actual de Francia?",
    "Resultados del partido Real Madrid vs Barcelona hoy",
    "¿Qué pasó en las elecciones de Argentina 2024?",
    "Stock price de Tesla ahora",
    "¿Cuándo sale la nueva película de Marvel?",
]


def generate_dataset_with_embeddings(
    n_samples: int,
    embedder,
    output_path: str = "data/trm_training_embeddings.npz"
):
    """
    Genera dataset con embeddings reales de EmbeddingGemma
    
    Returns:
        embeddings: np.array (n_samples, 768)
        labels_hard: np.array (n_samples,)
        labels_soft: np.array (n_samples,)
        labels_web: np.array (n_samples,)
        texts: List[str]
    """
    print(f"\n🔧 Generando {n_samples} ejemplos con EmbeddingGemma...")
    
    # Distribución: 40% hard, 30% soft, 30% web
    n_hard = int(n_samples * 0.4)
    n_soft = int(n_samples * 0.3)
    n_web = n_samples - n_hard - n_soft
    
    print(f"   Hard: {n_hard}, Soft: {n_soft}, Web: {n_web}")
    
    all_texts = []
    all_labels_hard = []
    all_labels_soft = []
    all_labels_web = []
    
    # Generar queries hard
    print("\n📝 Generando queries HARD...")
    for _ in tqdm(range(n_hard)):
        query = np.random.choice(HARD_QUERIES)
        all_texts.append(query)
        all_labels_hard.append(1.0)
        all_labels_soft.append(0.0)
        all_labels_web.append(0.0)
    
    # Generar queries soft
    print("\n📝 Generando queries SOFT...")
    for _ in tqdm(range(n_soft)):
        query = np.random.choice(SOFT_QUERIES)
        all_texts.append(query)
        all_labels_hard.append(0.0)
        all_labels_soft.append(1.0)
        all_labels_web.append(0.0)
    
    # Generar queries web
    print("\n📝 Generando queries WEB...")
    for _ in tqdm(range(n_web)):
        query = np.random.choice(WEB_QUERIES)
        all_texts.append(query)
        all_labels_hard.append(0.0)
        all_labels_soft.append(0.0)
        all_labels_web.append(1.0)
    
    # Generar embeddings (batch processing para eficiencia)
    print("\n🔄 Generando embeddings (esto puede tardar unos minutos)...")
    
    embeddings_list = []
    batch_size = 32
    
    for i in tqdm(range(0, len(all_texts), batch_size)):
        batch_texts = all_texts[i:i+batch_size]
        
        # Encodear batch
        batch_embeddings = []
        for text in batch_texts:
            emb = embedder.encode(text, normalize=True)
            batch_embeddings.append(emb)
        
        embeddings_list.extend(batch_embeddings)
    
    embeddings = np.array(embeddings_list, dtype=np.float32)
    labels_hard = np.array(all_labels_hard, dtype=np.float32)
    labels_soft = np.array(all_labels_soft, dtype=np.float32)
    labels_web = np.array(all_labels_web, dtype=np.float32)
    
    print(f"\n✅ Embeddings generados: {embeddings.shape}")
    print(f"   Dimensión: {embeddings.shape[1]}-D")
    
    # Guardar en formato numpy comprimido
    print(f"\n💾 Guardando dataset en: {output_path}")
    
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        labels_hard=labels_hard,
        labels_soft=labels_soft,
        labels_web=labels_web,
        texts=np.array(all_texts, dtype=object)
    )
    
    # También guardar JSONL para compatibilidad
    jsonl_path = output_path.replace('.npz', '.jsonl')
    print(f"💾 Guardando JSONL en: {jsonl_path}")
    
    with open(jsonl_path, 'w') as f:
        for text, hard, soft, web in zip(all_texts, labels_hard, labels_soft, labels_web):
            entry = {
                "text": text,
                "hard": float(hard),
                "soft": float(soft),
                "web_query": float(web)
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Dataset guardado exitosamente")
    print(f"   NPZ (con embeddings): {output_path}")
    print(f"   JSONL (sin embeddings): {jsonl_path}")
    
    return embeddings, labels_hard, labels_soft, labels_web, all_texts


def main():
    parser = argparse.ArgumentParser(
        description="Generar dataset TRM con EmbeddingGemma (768-D)"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=5000,
        help="Número de ejemplos a generar"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/trm_training_embeddings.npz",
        help="Ruta de salida (formato .npz)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 GENERADOR DE DATASET TRM CON EMBEDDINGGEMMA")
    print("="*70)
    print(f"Samples: {args.samples}")
    print(f"Output: {args.output}")
    print("="*70)
    
    # Cargar EmbeddingGemma
    print("\n📦 Cargando EmbeddingGemma...")
    try:
        embedder = get_embedding_model()
        print(f"✅ EmbeddingGemma cargado correctamente")
    except Exception as e:
        print(f"❌ Error cargando EmbeddingGemma: {e}")
        print("\nAsegúrate de:")
        print("  1. Estar autenticado: huggingface-cli login")
        print("  2. Aceptar términos: https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized")
        return 1
    
    # Generar dataset
    try:
        embeddings, hard, soft, web, texts = generate_dataset_with_embeddings(
            n_samples=args.samples,
            embedder=embedder,
            output_path=args.output
        )
        
        # Estadísticas
        print("\n📊 ESTADÍSTICAS:")
        print("="*70)
        print(f"Total ejemplos: {len(texts)}")
        print(f"Hard queries: {int(hard.sum())} ({hard.sum()/len(hard)*100:.1f}%)")
        print(f"Soft queries: {int(soft.sum())} ({soft.sum()/len(soft)*100:.1f}%)")
        print(f"Web queries: {int(web.sum())} ({web.sum()/len(web)*100:.1f}%)")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding dim: {embeddings.shape[1]}-D")
        print("="*70)
        
        print("\n✅ GENERACIÓN COMPLETADA")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error durante generación: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
