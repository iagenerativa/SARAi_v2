"""
Script para generar dataset sintético para entrenar TRM-Classifier
Usa un LLM grande para etiquetar intenciones hard/soft
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import random


def generate_prompts() -> List[str]:
    """Genera prompts diversos para etiquetar"""
    
    # Prompts técnicos (hard)
    technical_prompts = [
        "¿Cómo configuro SSH en Linux?",
        "Error al compilar código Python",
        "Explica el algoritmo de Dijkstra",
        "¿Qué es Docker y cómo se usa?",
        "Necesito depurar un memory leak en C++",
        "¿Cómo funciona el protocolo HTTPS?",
        "Configura un servidor Apache",
        "Resuelve esta ecuación diferencial",
        "¿Qué es una CNN en deep learning?",
        "Ayuda con error de base de datos SQL"
    ]
    
    # Prompts emocionales (soft)
    emotional_prompts = [
        "Estoy muy triste hoy",
        "Me siento frustrado con mi trabajo",
        "Necesito motivación para seguir",
        "Estoy confundido y no sé qué hacer",
        "Me siento solo",
        "Tengo ansiedad por el futuro",
        "Estoy emocionado por este proyecto",
        "Me da miedo fracasar",
        "Necesito alguien que me escuche",
        "Estoy orgulloso de mi logro"
    ]
    
    # Prompts híbridos
    hybrid_prompts = [
        "Explícame Python como si tuviera 12 años",
        "Ayúdame a entender git, estoy perdido",
        "¿Por qué mi código no funciona? Estoy frustrado",
        "Necesito aprender ML pero no sé por dónde empezar",
        "¿Puedes explicarme blockchain de forma amigable?",
        "Estoy confundido con las pointers en C",
        "Ayúdame a elegir una carrera en tech",
        "¿Cómo supero el síndrome del impostor programando?",
        "Necesito consejo sobre cambiar de trabajo a IA",
        "Explícame recursión de forma simple"
    ]
    
    all_prompts = technical_prompts + emotional_prompts + hybrid_prompts
    random.shuffle(all_prompts)
    
    return all_prompts


def label_prompt_simple(prompt: str) -> Dict[str, float]:
    """
    Etiqueta un prompt con scores hard/soft usando heurísticas
    (Versión simple sin LLM)
    """
    low = prompt.lower()
    
    # Keywords técnicas
    hard_kw = ["código", "error", "configurar", "algoritmo", "servidor", 
               "base de datos", "compilar", "depurar", "protocolo", "ssh",
               "docker", "python", "c++", "sql", "apache", "https"]
    
    # Keywords emocionales
    soft_kw = ["triste", "frustrado", "motivación", "confundido", "solo",
               "ansiedad", "miedo", "emocionado", "orgulloso", "siento",
               "escuche", "ayúdame", "perdido", "no sé"]
    
    hard_count = sum(1 for kw in hard_kw if kw in low)
    soft_count = sum(1 for kw in soft_kw if kw in low)
    
    # Normalizar
    hard = min(hard_count / 2.0, 1.0) if hard_count > 0 else 0.1
    soft = min(soft_count / 2.0, 1.0) if soft_count > 0 else 0.1
    
    # Asegurar que prompts híbridos tengan ambos altos
    if hard_count > 0 and soft_count > 0:
        hard = max(hard, 0.6)
        soft = max(soft, 0.6)
    
    return {"hard": round(hard, 2), "soft": round(soft, 2)}


def generate_dataset(num_samples: int, output_path: Path):
    """Genera dataset completo"""
    
    print(f"📝 Generando {num_samples} ejemplos...")
    
    prompts = generate_prompts()
    dataset = []
    
    # Repetir prompts si necesario
    while len(dataset) < num_samples:
        for prompt in prompts:
            if len(dataset) >= num_samples:
                break
            
            labels = label_prompt_simple(prompt)
            dataset.append({
                "input": prompt,
                "hard": labels["hard"],
                "soft": labels["soft"]
            })
    
    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Dataset guardado en {output_path}")
    print(f"📊 Ejemplos: {len(dataset)}")
    
    # Estadísticas
    avg_hard = sum(x["hard"] for x in dataset) / len(dataset)
    avg_soft = sum(x["soft"] for x in dataset) / len(dataset)
    print(f"📈 Promedio hard: {avg_hard:.2f}")
    print(f"📈 Promedio soft: {avg_soft:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Generar dataset sintético para TRM")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Número de ejemplos a generar")
    parser.add_argument("--output", type=str, default="data/trm_training.json",
                       help="Ruta de salida")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    generate_dataset(args.samples, output_path)


if __name__ == "__main__":
    main()
