"""Update the calibration file for EmotionAudioLite using labelled audio samples.

Usage:
    python scripts/update_emotion_calibration.py \
        --label neutral \
        --audio logs/audio_input_20251030_115740.wav

The script computes descriptive statistics over the supplied audio clips and
stores the aggregated metrics in `state/audio_emotion_calibration.json`. This
allows the synthetic data generator to align its baselines with real-world
recordings.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np

# Asegurar que el repositorio esté en sys.path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.layer1_io.audio_emotion_lite import extract_features

CALIBRATION_PATH = Path("state/audio_emotion_calibration.json")


def _load_existing_calibration(path: Path) -> Dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def _compute_metrics(audio_paths: List[Path]) -> Dict[str, List[float]]:
    metrics_acc: Dict[str, List[float]] = defaultdict(list)
    for audio_path in audio_paths:
        y, sr = librosa.load(str(audio_path), sr=16000)
        _, metrics = extract_features(y, sr=sr)
        if not metrics:
            print(f"⚠️  Audio sin suficientes datos para extraer métricas: {audio_path}")
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating)):
                metrics_acc[key].append(float(value))
    return metrics_acc


def _aggregate_statistics(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    sample_count = 0
    for key, values in metric_lists.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float32)
        aggregated[f"{key}_mean"] = float(arr.mean())
        aggregated[f"{key}_std"] = float(arr.std())
        sample_count = max(sample_count, len(values))
    aggregated["samples"] = sample_count
    return aggregated


def update_calibration(label: str, audio_paths: List[Path], output: Path) -> None:
    existing = _load_existing_calibration(output)
    metrics_acc = _compute_metrics(audio_paths)
    if not metrics_acc:
        raise SystemExit("No se pudieron calcular métricas a partir de los audios proporcionados.")

    aggregated = _aggregate_statistics(metrics_acc)
    aggregated["updated_at"] = datetime.utcnow().isoformat()
    aggregated["audio_samples"] = [path.name for path in audio_paths]

    existing.setdefault(label, {})
    existing[label].update(aggregated)

    # También guardar agregados globales (default) si no existen
    if "default" not in existing:
        existing["default"] = {k: v for k, v in aggregated.items() if k.endswith("_mean")}
        existing["default"]["source"] = label
        existing["default"]["updated_at"] = aggregated["updated_at"]

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, ensure_ascii=False, indent=2)

    print(f"✅ Calibración actualizada para etiqueta '{label}' usando {len(audio_paths)} archivo(s)")
    print(f"   Archivo: {output}")
    for key, value in aggregated.items():
        if key.endswith("_mean"):
            print(f"   • {key}: {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Actualiza la calibración de EmotionAudioLite")
    parser.add_argument("--label", default="neutral", help="Etiqueta emocional de los audios proporcionados")
    parser.add_argument(
        "--audio",
        nargs="+",
        required=True,
        help="Lista de archivos de audio (WAV) a utilizar para la calibración",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CALIBRATION_PATH,
        help="Ruta del archivo de calibración a generar/actualizar",
    )
    args = parser.parse_args()

    audio_paths = [Path(p) for p in args.audio]
    missing = [p for p in audio_paths if not p.exists()]
    if missing:
        raise SystemExit(f"No se encontraron los siguientes audios: {', '.join(str(p) for p in missing)}")

    update_calibration(args.label, audio_paths, args.output)


if __name__ == "__main__":
    main()
