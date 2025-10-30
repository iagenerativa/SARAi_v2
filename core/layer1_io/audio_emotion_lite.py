"""
SARAi v2.17 - Audio Emotion Lite
Análisis ligero de tono/emoción a partir de audio crudo para complementar
la señal semántica de BERT en el canal de entrada.

El clasificador está pensado para CPU sin dependencias pesadas. Utiliza
características clásicas (energía, pitch, MFCC, formantes, etc.) combinadas
con un RandomForest balanceado y se entrena sobre datos sintéticos
parametrizados por emoción.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ArrayLike = Union[np.ndarray, List[float]]


# ---------------------------------------------------------------------------
# 1. FEATURE EXTRACTION
# ---------------------------------------------------------------------------


def _safe_trim(audio: np.ndarray) -> np.ndarray:
    """Recorta silencios extremos manteniendo estabilidad numérica."""
    if audio.size == 0:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=20)
    return trimmed if trimmed.size > 0 else audio


def _compute_pitch(y: np.ndarray, sr: int) -> Tuple[float, float, float]:
    """Retorna (media, std, jitter) del pitch."""
    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    except Exception:
        return 0.0, 0.0, 0.0
    if f0 is None:
        return 0.0, 0.0, 0.0
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return 0.0, 0.0, 0.0
    pitch_mean = float(np.mean(f0))
    pitch_std = float(np.std(f0))
    jitter = 0.0
    if len(f0) > 1 and pitch_mean > 0:
        jitter = float(np.mean(np.abs(np.diff(f0))) / (pitch_mean + 1e-8))
    return pitch_mean, pitch_std, jitter


def _compute_formants(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """Aproxima F1 y F2 mediante LPC."""
    try:
        order = min(2 + int(sr / 1000), 20)
        if y.size <= order:
            return 500.0, 1500.0
        a = librosa.lpc(y + 1e-6, order=order)
        roots = np.roots(np.concatenate(([1.0], a)))
        roots = roots[np.imag(roots) != 0]
        formant_freqs = np.sort(np.abs(np.arctan2(np.imag(roots), np.real(roots)) * sr / (2 * np.pi)))
        f1 = float(formant_freqs[0]) if formant_freqs.size > 0 else 500.0
        f2 = float(formant_freqs[1]) if formant_freqs.size > 1 else 1500.0
        return f1, f2
    except Exception:
        return 500.0, 1500.0


def _compute_tempo(y: np.ndarray, sr: int) -> float:
    """Tempo aproximado para habla (puede ser ruidoso, pero informativo)."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except Exception:
        return 0.0


def _compute_mfcc_stats(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    return (
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
    )


def _compute_valence_arousal(energy: float, pitch_mean: float) -> Tuple[float, float]:
    valence = 0.5 + 0.3 * (pitch_mean - 150.0) / 200.0 + 0.3 * (energy - 0.02) / 0.04
    valence = float(np.clip(valence, 0.0, 1.0))
    # Arousal aproximado: normalizamos energía y variabilidad tonal
    arousal = np.clip((energy - 0.01) / 0.05 + (pitch_mean - 120.0) / 200.0, 0.0, 1.0)
    return valence, float(arousal)


def extract_features(
    audio: ArrayLike,
    sr: int = 16000,
    ensure_mono: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Extrae vector de características y métricas auxiliares.

    Args:
        audio: Audio crudo en float32 [-1, 1].
        sr: Sample rate asociado.
        ensure_mono: Si True, mezcla a mono (esperado en pipeline).

    Returns:
        Tuple(features, metrics). Si features es None significa que el audio
        es demasiado corto o silencioso.
    """
    if isinstance(audio, list):
        audio = np.asarray(audio, dtype=np.float32)
    elif not isinstance(audio, np.ndarray):
        raise TypeError("audio debe ser ndarray o lista")

    if audio.size == 0:
        return None, {}

    if ensure_mono and audio.ndim > 1:
        audio = librosa.to_mono(audio.T)

    audio = audio.astype(np.float32)
    audio = librosa.util.normalize(audio)
    audio = _safe_trim(audio)

    if audio.size < 512:
        return None, {}

    metrics: Dict[str, float] = {}

    rms = librosa.feature.rms(y=audio)[0]
    energy = float(np.mean(rms))
    energy_var = float(np.std(rms))
    metrics["energy"] = energy
    metrics["energy_var"] = energy_var

    pitch_mean, pitch_std, jitter = _compute_pitch(audio, sr)
    metrics["pitch_mean"] = pitch_mean
    metrics["pitch_std"] = pitch_std
    metrics["jitter"] = jitter

    mfcc_mean, mfcc_std, delta_mean = _compute_mfcc_stats(audio, sr)

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    metrics["zcr"] = zcr

    tempo = _compute_tempo(audio, sr)
    metrics["tempo"] = tempo

    f1, f2 = _compute_formants(audio, sr)
    metrics["f1"] = f1
    metrics["f2"] = f2

    shimmer = float(np.std(rms) / (np.mean(rms) + 1e-8))
    metrics["shimmer"] = shimmer

    valence, arousal = _compute_valence_arousal(energy, pitch_mean)
    metrics["valence"] = valence
    metrics["arousal"] = arousal

    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    metrics["spectral_centroid"] = spectral_centroid
    metrics["spectral_bandwidth"] = spectral_bandwidth

    feature_vector = np.hstack(
        [
            energy,
            energy_var,
            pitch_mean,
            pitch_std,
            jitter,
            mfcc_mean,
            mfcc_std,
            delta_mean,
            zcr,
            tempo,
            f1,
            f2,
            shimmer,
            valence,
            arousal,
            spectral_centroid,
            spectral_bandwidth,
        ]
    )

    return feature_vector.astype(np.float32), metrics


# ---------------------------------------------------------------------------
# 2. MODELO LIGERO
# ---------------------------------------------------------------------------


@dataclass
class EmotionResult:
    label: str
    confidence: float
    valence: float
    arousal: float
    probabilities: Dict[str, float]
    features: Dict[str, float]


class EmotionAudioLite:
    """Clasificador ligero de emociones basadas en tono."""

    EMOTIONS: List[str] = [
        "neutral",
        "alegría",
        "tristeza",
        "ira",
        "miedo",
        "sorpresa",
        "asco",
        "confianza",
        "desprecio",
        "ansiedad",
        "aburrimiento",
        "frustración",
        "calma",
        "emoción_pos",
        "emoción_neg",
    ]

    def __init__(
        self,
        model_path: Union[str, Path] = "models/audio_emotion_lite.joblib",
        calibration_path: Union[str, Path] = "state/audio_emotion_calibration.json",
        auto_train: bool = True,
        synthetic_size: int = 1200,
    ):
        self.model_path = Path(model_path)
        self.calibration_path = Path(calibration_path)
        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=320,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
        )
        self.synthetic_size = synthetic_size
        self._trained = False
        self.calibration = self._load_calibration()

        if self.model_path.exists():
            self._load()
        elif auto_train:
            self.train()
            self._save()

    # ---------------------------- Persistence ----------------------------
    def _save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "clf": self.clf}, self.model_path)

    def _load(self):
        data = joblib.load(self.model_path)
        self.scaler = data["scaler"]
        self.clf = data["clf"]
        self._trained = True

    def _load_calibration(self) -> Dict[str, Dict[str, float]]:
        if not self.calibration_path.exists():
            return {}
        try:
            with open(self.calibration_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def reload_calibration(self) -> None:
        """Recarga la calibración desde disco."""
        self.calibration = self._load_calibration()

    def _calibration_for(self, label: str) -> Dict[str, float]:
        return self.calibration.get(label) or self.calibration.get("default", {})

    # ------------------------- Synthetic Training -----------------------
    def _synthetic_data(self, n: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        n = n or self.synthetic_size
        rng = np.random.default_rng(42)
        X: List[List[float]] = []
        y: List[int] = []

        neutral_cal = self._calibration_for("neutral")
        base_energy = float(neutral_cal.get("energy_mean", 0.03))
        base_pitch = float(neutral_cal.get("pitch_mean", 190.0))
        base_pitch_std = float(neutral_cal.get("pitch_std", 25.0))
        base_zcr = float(neutral_cal.get("zcr_mean", 0.08))
        base_tempo = float(neutral_cal.get("tempo_mean", 120.0))
        base_valence = float(neutral_cal.get("valence", 0.5))
        base_arousal = float(neutral_cal.get("arousal", 0.5))
        base_centroid = float(neutral_cal.get("spectral_centroid_mean", 1800.0))
        base_bandwidth = float(neutral_cal.get("spectral_bandwidth_mean", 1200.0))

        def clamp(value: float, low: float, high: float) -> float:
            return float(np.clip(value, low, high))

        def sample_norm(base: float, rel_std: float, low: float, high: float) -> float:
            std = max(abs(base) * rel_std, rel_std)
            val = rng.normal(base, std)
            return clamp(val, low, high)

        def sample_energy(mult: float = 1.0) -> float:
            base = max(base_energy * mult, 0.003)
            std = max(base * 0.25, 0.0015)
            return clamp(rng.normal(base, std), 0.003, 0.12)

        def sample_pitch(offset: float = 0.0, mult: float = 1.0) -> float:
            base = base_pitch * mult + offset
            std = max(base_pitch_std * 1.1, 6.0)
            return clamp(rng.normal(base, std), 60.0, 480.0)

        for idx, emo in enumerate(self.EMOTIONS):
            samples_per_class = max(1, n // len(self.EMOTIONS))
            for _ in range(samples_per_class):
                energy = sample_energy()
                pitch = sample_pitch()
                pitch_std = clamp(rng.normal(base_pitch_std, base_pitch_std * 0.3), 4.0, 120.0)
                jitter = rng.uniform(0.005, 0.05)
                zcr = sample_norm(base_zcr, 0.25, 0.015, 0.25)
                tempo = sample_norm(base_tempo, 0.3, 40.0, 220.0)
                valence = clamp(rng.normal(base_valence, 0.2), 0.0, 1.0)
                arousal = clamp(rng.normal(base_arousal, 0.25), 0.0, 1.0)
                shimmer = rng.uniform(0.04, 0.2)
                spectral_centroid = sample_norm(base_centroid, 0.25, 500.0, 4000.0)
                spectral_bandwidth = sample_norm(base_bandwidth, 0.3, 400.0, 3200.0)

                if emo == "alegría":
                    energy = sample_energy(1.3)
                    pitch = sample_pitch(offset=35.0)
                    valence = clamp(max(valence, 0.82), 0.7, 0.95)
                    arousal = clamp(max(arousal, 0.75), 0.6, 0.95)
                elif emo == "tristeza":
                    energy = sample_energy(0.5)
                    pitch = sample_pitch(offset=-30.0)
                    valence = clamp(min(valence, 0.25), 0.05, 0.3)
                    arousal = clamp(min(arousal, 0.25), 0.05, 0.35)
                elif emo == "ira":
                    energy = sample_energy(1.6)
                    pitch = sample_pitch(offset=55.0)
                    pitch_std = clamp(pitch_std * 1.4, 8.0, 160.0)
                    jitter *= 1.3
                    arousal = clamp(max(arousal, 0.9), 0.75, 1.0)
                elif emo == "miedo":
                    pitch = sample_pitch(offset=65.0)
                    pitch_std = clamp(pitch_std * 1.55, 8.0, 170.0)
                    arousal = clamp(max(arousal, 0.88), 0.65, 0.98)
                elif emo == "sorpresa":
                    pitch = sample_pitch(offset=70.0)
                    pitch_std = clamp(pitch_std * 1.65, 10.0, 170.0)
                    arousal = clamp(max(arousal, 0.82), 0.6, 0.95)
                elif emo == "asco":
                    energy = sample_energy(0.7)
                    zcr = clamp(zcr * 1.25, 0.02, 0.28)
                    valence = clamp(min(valence, 0.15), 0.05, 0.25)
                elif emo == "confianza":
                    energy = sample_energy(1.1)
                    tempo = sample_norm(base_tempo - 10.0, 0.2, 40.0, 200.0)
                    valence = clamp(max(valence, 0.7), 0.6, 0.9)
                    arousal = clamp(min(arousal, 0.6), 0.3, 0.65)
                elif emo == "ansiedad":
                    pitch_std = clamp(pitch_std * 1.45, 8.0, 150.0)
                    tempo = sample_norm(base_tempo + 30.0, 0.25, 40.0, 220.0)
                    arousal = clamp(max(arousal, 0.78), 0.6, 0.95)
                elif emo == "aburrimiento":
                    energy = sample_energy(0.55)
                    tempo = sample_norm(base_tempo - 35.0, 0.2, 30.0, 200.0)
                    valence = clamp(min(valence, 0.3), 0.05, 0.4)
                    arousal = clamp(min(arousal, 0.35), 0.05, 0.45)
                elif emo == "frustración":
                    energy = sample_energy(1.25)
                    pitch_std = clamp(pitch_std * 1.25, 8.0, 160.0)
                    zcr = clamp(zcr * 1.15, 0.02, 0.28)
                    arousal = clamp(max(arousal, 0.72), 0.5, 0.92)
                elif emo == "calma":
                    energy = sample_energy(0.85)
                    pitch = clamp(sample_pitch() * 0.95, 70.0, 260.0)
                    pitch_std = clamp(pitch_std * 0.6, 4.0, 40.0)
                    valence = clamp(max(valence, 0.58), 0.5, 0.75)
                    arousal = clamp(min(arousal, 0.4), 0.1, 0.5)
                elif emo == "desprecio":
                    energy = sample_energy(0.65)
                    pitch = sample_pitch(offset=-25.0)
                    zcr = clamp(zcr * 1.05, 0.02, 0.28)
                    valence = clamp(min(valence, 0.35), 0.1, 0.35)
                elif emo == "emoción_pos":
                    energy = sample_energy(1.35)
                    pitch = sample_pitch(offset=28.0)
                    valence = clamp(max(valence, 0.78), 0.65, 0.92)
                    arousal = clamp(max(arousal, 0.68), 0.5, 0.9)
                elif emo == "emoción_neg":
                    energy = sample_energy(0.6)
                    pitch = sample_pitch(offset=-20.0)
                    valence = clamp(min(valence, 0.28), 0.08, 0.35)
                    arousal = clamp(min(arousal, 0.45), 0.15, 0.55)

                energy_var = max(1e-4, energy * rng.uniform(0.05, 0.22))
                mfcc_mean = rng.uniform(-12, 12, 13)
                mfcc_std = rng.uniform(1, 6, 13)
                delta_mean = rng.uniform(-2, 2, 13)

                f1 = rng.uniform(400, 900)
                f2 = rng.uniform(1100, 2000)

                vector = [
                    energy,
                    energy_var,
                    pitch,
                    pitch_std,
                    jitter,
                    *mfcc_mean,
                    *mfcc_std,
                    *delta_mean,
                    zcr,
                    tempo,
                    f1,
                    f2,
                    shimmer,
                    valence,
                    arousal,
                    spectral_centroid,
                    spectral_bandwidth,
                ]
                X.append(vector)
                y.append(idx)

        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)

    def train(self):
        X, y = self._synthetic_data()
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y)
        self._trained = True

    # ----------------------------- Inference -----------------------------
    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self._trained:
            self.train()
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.clf.predict_proba(features_scaled)[0]

    def analyze(
        self,
        audio: ArrayLike,
        sr: int = 16000,
        top_k: int = 3,
    ) -> EmotionResult:
        features, metrics = extract_features(audio, sr=sr)
        if features is None:
            return EmotionResult(
                label="silencio",
                confidence=1.0,
                valence=0.5,
                arousal=0.0,
                probabilities={"silencio": 1.0},
                features={"detected": False},
            )

        proba = self._predict_proba(features)
        sorted_idx = np.argsort(proba)[::-1]
        top_indices = sorted_idx[: max(1, top_k)]
        label_idx = int(sorted_idx[0])
        label = self.EMOTIONS[label_idx]
        confidence = float(proba[label_idx])
        probabilities = {self.EMOTIONS[i]: float(proba[i]) for i in top_indices}

        metrics_updated = {**metrics, "detected": True}

        return EmotionResult(
            label=label,
            confidence=confidence,
            valence=float(metrics_updated.get("valence", 0.5)),
            arousal=float(metrics_updated.get("arousal", 0.0)),
            probabilities=probabilities,
            features=metrics_updated,
        )

    def predict(
        self,
        audio: Union[str, ArrayLike],
        sr: int = 16000,
        proba: bool = False,
    ) -> Union[str, Tuple[str, float]]:
        if isinstance(audio, (str, Path)):
            y, real_sr = librosa.load(str(audio), sr=sr)
            result = self.analyze(y, sr=real_sr)
        else:
            result = self.analyze(audio, sr=sr)
        return (result.label, result.confidence) if proba else result.label


# ---------------------------------------------------------------------------
# 3. UTILIDAD LIGERA
# ---------------------------------------------------------------------------

def demo(audio_path: str):
    """Demostración rápida desde CLI."""
    model = EmotionAudioLite()
    audio, sr = librosa.load(audio_path, sr=16000)
    result = model.analyze(audio, sr=sr)
    print(json.dumps(
        {
            "label": result.label,
            "confidence": result.confidence,
            "valence": result.valence,
            "arousal": result.arousal,
            "probabilities": result.probabilities,
            "features": result.features,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo EmotionAudioLite")
    parser.add_argument("audio", help="Ruta a archivo WAV mono 16kHz")
    args = parser.parse_args()
    demo(args.audio)
