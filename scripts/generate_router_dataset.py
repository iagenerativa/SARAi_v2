"""Generate training dataset for the LoRA Router (TRM vs LLM vs Traducción).

This script creates a labelled dataset of Spanish utterances for the router
classifier. Each utterance is embedded using the BERT Embedder so the training
process can run without requiring the original text inputs afterwards.

Output file: data/router_training.npz with keys:
    • embeddings: np.ndarray[float32] shape (N, 768)
    • labels: np.ndarray[int64] shape (N,)
    • phrases: np.ndarray[str] shape (N,) (metadata / debugging)
    • label_names: list[str] length 3 (order: TRM, LLM, Traducir)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure repository root is importable when running as a script
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import BERT embedder from the project
try:
    from core.layer1_io.bert_embedder import BERTEmbedder
except ModuleNotFoundError as exc:  # pragma: no cover - helpful message for CLI
    raise SystemExit(
        "BERT embedder not found. Verify PYTHONPATH or run from repository root"
    ) from exc

LABEL_TRM = 0
LABEL_LLM = 1
LABEL_TRANSLATE = 2
LABEL_NAMES = ["TRM", "LLM", "Traducir"]


def _expand_variants(phrases: Iterable[str]) -> List[str]:
    """Return a unique list containing the original phrases plus simple variants."""
    variants: List[str] = []
    for phrase in phrases:
        base = phrase.strip()
        if not base:
            continue
        candidates = {
            base,
            base.capitalize(),
            base.lower(),
            base.upper(),
        }
        variants.extend(candidates)
    # Preserve order while removing duplicates
    seen: set[str] = set()
    deduped: List[str] = []
    for item in variants:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _build_phrase_bank() -> Dict[int, List[str]]:
    """Create labelled phrase lists for each router class."""
    trm_base = [
        "hola",
        "hola sara",
        "hola asistente",
        "buenos días",
        "buenas tardes",
        "buenas noches",
        "cómo estás",
        "qué tal",
        "gracias",
        "muchas gracias",
        "de nada",
        "perdona",
        "lo siento",
        "dame un momento",
        "sí",
        "no",
        "tal vez",
        "claro",
        "por supuesto",
        "quién eres",
        "qué puedes hacer",
        "cuéntame algo",
        "despídete",
        "adiós",
        "hasta luego",
        "nos vemos",
        "me ayudas",
        "ayuda",
        "qué hora es",
        "qué día es hoy",
        "enciende la luz",
        "apaga la luz",
        "sube el volumen",
        "baja el volumen",
        "abre la puerta",
        "cierra la puerta",
        "pon una alarma",
        "qué tiempo hace",
        "qué temperatura hace",
        "muéstrame el calendario",
        "recuérdame una cita",
        "elige una canción",
        "pon música relajante",
        "pausa la música",
        "reanuda la canción",
        "siguiente canción",
        "anterior canción",
    ]

    llm_questions = [
        "explícame la teoría de la relatividad",
        "dame un resumen de la segunda guerra mundial",
        "cómo funciona el aprendizaje profundo",
        "qué diferencias hay entre python y javascript",
        "escribe un script en bash que haga una copia de seguridad",
        "genera un ejemplo de código en python para ordenar una lista",
        "cómo puedo entrenar una red neuronal",
        "qué opinas de la energía nuclear",
        "cuáles son las causas del cambio climático",
        "qué impactos tiene la inteligencia artificial en la sociedad",
        "redacta una carta formal de renuncia",
        "ayúdame a planificar un viaje de 5 días por madrid",
        "explica la teoría cuántica en términos sencillos",
        "por qué el cielo es azul",
        "cuál es la diferencia entre tcp y udp",
        "qué es kubernetes",
        "cómo implemento una cola de prioridad en c++",
        "dame una lista de ideas para contenido de redes sociales",
        "qué estrategias existen para reducir la deuda técnica",
        "cómo puedo mejorar la retención de clientes",
        "cuáles son los pasos para abrir una empresa en españa",
        "qué recomienda la guía nist para ciberseguridad",
        "analiza esta frase y dime su sentimiento",
        "qué implicaciones legales tiene el uso de drones",
        "qué técnica de machine learning es mejor para clasificación",
        "explica el método científico",
    ]

    translate_samples = [
        "tradúceme hola mundo al inglés",
        "traduce 'good morning' al español",
        "puedes traducir esto: je suis très content",
        "qué significa 'guten morgen'",
        "translate this sentence into spanish",
        "could you translate the following text to english",
        "bonjour, comment ça va",
        "ich möchte wissen, was das bedeutet",
        "traductor, dime qué significa 'arrivederci'",
        "help me translate 'feliz cumpleaños' to portuguese",
        "qué quiere decir 'buen appetit'",
        "cómo se dice 'gracias' en japonés",
        "dime cómo traducir 'see you later'",
        "me puedes traducir esta frase al francés",
        "qué significa 'buenos días' en alemán",
        "quiero que traduzcas este texto",
        "hello, can you translate this paragraph",
        "traduce al inglés: espero que tengas un buen día",
        "traduce al español: have a nice weekend",
        "please translate: guten tag, wie geht es dir",
        "qué significa la palabra 'tschüss'",
    ]

    return {
        LABEL_TRM: _expand_variants(trm_base),
        LABEL_LLM: _expand_variants(llm_questions),
        LABEL_TRANSLATE: _expand_variants(translate_samples),
    }


def _augment_with_templates(phrases: List[str], templates: Iterable[str]) -> List[str]:
    """Apply phrase templates by formatting each template with a phrase."""
    augmented = phrases[:]
    for template in templates:
        for phrase in phrases:
            augmented.append(template.format(phrase=phrase))
    return _expand_variants(augmented)


def build_dataset(embedder: BERTEmbedder, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    random.seed(seed)
    np.random.seed(seed)

    phrase_bank = _build_phrase_bank()

    # Simple templates to increase diversity
    trm_templates = [
        "{phrase}?",
        "oye, {phrase}",
        "disculpa, {phrase}",
        "por favor, {phrase}",
    ]
    llm_templates = [
        "necesito que me expliques {phrase}",
        "podrías analizar {phrase}?",
        "me ayudas a entender {phrase}?",
        "haz un resumen sobre {phrase}",
    ]
    translate_templates = [
        "sé traductor: {phrase}",
        "quiero saber cómo se dice: {phrase}",
        "podrías decirme qué significa: {phrase}",
    ]

    phrase_bank[LABEL_TRM] = _augment_with_templates(phrase_bank[LABEL_TRM], trm_templates)
    phrase_bank[LABEL_LLM] = _augment_with_templates(phrase_bank[LABEL_LLM], llm_templates)
    phrase_bank[LABEL_TRANSLATE] = _augment_with_templates(phrase_bank[LABEL_TRANSLATE], translate_templates)

    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    phrases: List[str] = []

    for label, phrase_list in phrase_bank.items():
        print(f"→ Generando embeddings para {LABEL_NAMES[label]} ({len(phrase_list)} frases)...")
        # Shuffle for diversity before encoding
        random.shuffle(phrase_list)
        for chunk_start in range(0, len(phrase_list), 32):
            chunk = phrase_list[chunk_start:chunk_start + 32]
            chunk_embeddings = embedder.encode_batch(chunk)
            embeddings.append(chunk_embeddings.astype(np.float32))
            labels.extend([label] * len(chunk))
            phrases.extend(chunk)

    # Concatenate all embeddings
    embeddings_array = np.vstack(embeddings)
    labels_array = np.asarray(labels, dtype=np.int64)
    phrases_array = np.asarray(phrases, dtype=object)

    print("✅ Dataset generado")
    print(f"   Total muestras: {embeddings_array.shape[0]}")
    print(f"   Dimensión embeddings: {embeddings_array.shape[1]}")

    return embeddings_array, labels_array, phrases_array.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data for the LoRA Router")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "router_training.npz",
        help="Ruta del archivo .npz de salida",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad",
    )
    args = parser.parse_args()

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("🧠 Cargando BERT Embedder...")
    embedder = BERTEmbedder()

    embeddings, labels, phrases = build_dataset(embedder, seed=args.seed)

    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        labels=labels,
        phrases=np.array(phrases, dtype=object),
        label_names=np.array(LABEL_NAMES, dtype=object),
    )

    print(f"\n✅ Dataset guardado en: {output_path}")
    print("   Distribución de clases:")
    total = len(labels)
    for label_id, name in enumerate(LABEL_NAMES):
        count = int(np.sum(labels == label_id))
        pct = 100.0 * count / max(total, 1)
        print(f"     • {name:<9}: {count:>4} muestras ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
