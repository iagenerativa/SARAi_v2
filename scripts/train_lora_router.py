"""Train the LoRA Router model using the dataset generated for layer 1.

Usage examples:

    # 1) Generate dataset (if not already available)
    python scripts/generate_router_dataset.py

    # 2) Train router and save weights to models/lora_router.pt
    python scripts/train_lora_router.py --epochs 12 --batch-size 64

The script prints training progress, evaluates on a validation split and saves
metrics next to the trained model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is importable when running as a script
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from core.layer1_io.lora_router import LoRARouter, train_lora_router
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Unable to import LoRARouter. Verify PYTHONPATH or run from repository root."
    ) from exc

DEFAULT_DATA_PATH = BASE_DIR / "data" / "router_training.npz"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "lora_router.pt"
DEFAULT_METRICS_PATH = BASE_DIR / "models" / "lora_router_metrics.json"


def _split_dataset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(embeddings))
    rng.shuffle(indices)

    split_idx = int(len(indices) * (1.0 - val_ratio))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    return (
        embeddings[train_idx],
        labels[train_idx],
        embeddings[val_idx],
        labels[val_idx],
    )


def _create_dataloader(
    embeddings: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(embeddings),
        torch.from_numpy(labels),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _evaluate(router: LoRARouter, loader: DataLoader, device: torch.device) -> float:
    router.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings_batch, labels_batch in loader:
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits = router.forward(embeddings_batch)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.numel()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar el LoRA Router")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Ruta al archivo .npz con embeddings y labels",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Número de épocas de entrenamiento",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamaño de batch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fracción para validación",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla de aleatoriedad",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta de salida para guardar el modelo entrenado",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Ruta para guardar métricas en JSON",
    )
    args = parser.parse_args()

    dataset_path: Path = args.dataset
    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset no encontrado en {dataset_path}. Ejecuta generate_router_dataset.py primero."
        )

    print(f"📦 Cargando dataset desde {dataset_path}...")
    data = np.load(dataset_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    label_names = data.get("label_names")

    # El router entrena de forma determinista en CPU (suficiente para el tamaño del modelo)
    device = torch.device("cpu")
    print(f"🧠 Dispositivo: {device}")

    train_embeddings, train_labels, val_embeddings, val_labels = _split_dataset(
        embeddings,
        labels,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    val_loader = _create_dataloader(val_embeddings, val_labels, batch_size=args.batch_size)

    router = LoRARouter()

    # Entrenamiento utilizando la función existente del módulo
    print("🚀 Iniciando entrenamiento del LoRA Router...")
    train_lora_router(
        router,
        train_embeddings,
        train_labels,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    print("📏 Evaluando en el conjunto de validación...")
    val_accuracy = _evaluate(router, val_loader, device)
    print(f"✅ Exactitud validación: {val_accuracy * 100:.2f}%")

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    router.save(output_path)
    print(f"💾 Modelo guardado en: {output_path}")

    metrics_path: Path = args.metrics_output
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        (
            "{\n"
            f"  \"epochs\": {args.epochs},\n"
            f"  \"batch_size\": {args.batch_size},\n"
            f"  \"learning_rate\": {args.lr},\n"
            f"  \"val_accuracy\": {val_accuracy:.4f},\n"
            f"  \"classes\": {label_names.tolist() if label_names is not None else ['TRM','LLM','Traducir']}\n"
            "}\n"
        ),
        encoding="utf-8",
    )
    print(f"🧾 Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
