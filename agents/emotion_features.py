"""
agents/emotion_features.py

Extracción de features acústicas para detección emocional
Basado en técnicas de procesamiento de señales de voz

Características extraídas:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma features
- Spectral features (centroid, rolloff, contrast)
- ZCR (Zero-Crossing Rate)
- Energy envelope
- Pitch tracking

Referencias:
- Kaggle: Reconocimiento de emociones español
- LibROSA: Audio feature extraction
- OpenSmile: Acoustic features for emotion

Author: SARAi v2.11
Date: 2025-10-28
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class AcousticFeatures:
    """
    Conjunto de features acústicas extraídas
    
    Attributes:
        mfcc: MFCCs (13 coeficientes)
        chroma: Chroma features (12 bins)
        spectral_centroid: Centro espectral
        spectral_rolloff: Rolloff espectral
        spectral_contrast: Contraste espectral (7 bands)
        zcr: Zero-crossing rate
        rms_energy: Root mean square energy
        pitch_mean: Pitch promedio (Hz)
        pitch_std: Desviación estándar de pitch
    """
    mfcc: np.ndarray  # Shape: (13,)
    chroma: np.ndarray  # Shape: (12,)
    spectral_centroid: float
    spectral_rolloff: float
    spectral_contrast: np.ndarray  # Shape: (7,)
    zcr: float
    rms_energy: float
    pitch_mean: float
    pitch_std: float
    
    def to_vector(self) -> np.ndarray:
        """
        Convierte features a vector 1D para ML
        
        Returns:
            Vector de features (tamaño: 13+12+7+5 = 37 features)
        """
        return np.concatenate([
            self.mfcc,  # 13
            self.chroma,  # 12
            self.spectral_contrast,  # 7
            [self.spectral_centroid,  # 1
             self.spectral_rolloff,   # 1
             self.zcr,                # 1
             self.rms_energy,         # 1
             self.pitch_mean]         # 1
            # Total: 37 (sin pitch_std para evitar 38)
        ])


class EmotionFeatureExtractor:
    """
    Extractor de features acústicas para detección emocional
    
    Usa LibROSA para análisis de audio profesional
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Inicializa extractor
        
        Args:
            sample_rate: Sample rate del audio en Hz
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "LibROSA no disponible. Instalar con: pip install librosa"
            )
        
        self.sample_rate = sample_rate
        
        # Configuración de features
        self.n_mfcc = 13
        self.n_chroma = 12
        self.n_fft = 2048
        self.hop_length = 512
    
    def extract(self, audio: np.ndarray) -> AcousticFeatures:
        """
        Extrae features acústicas completas
        
        Args:
            audio: Waveform numpy array (mono, sample_rate Hz)
        
        Returns:
            AcousticFeatures con todos los descriptores
        
        Raises:
            ValueError: Si audio está vacío o es inválido
        """
        if len(audio) == 0:
            raise ValueError("Audio vacío")
        
        # Normalizar audio
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # 1. MFCCs (correlacionan con timbre/emoción)
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mfcc_mean = np.mean(mfccs, axis=1)  # Promedio temporal
        
        # 2. Chroma (correlaciona con tono/melodía)
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        chroma_mean = np.mean(chroma, axis=1)
        
        # 3. Spectral centroid (brillo del sonido)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        # 4. Spectral rolloff (frecuencia de corte)
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        # 5. Spectral contrast (diferencias espectrales)
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        
        # 6. Zero-crossing rate (correlaciona con pitch)
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        zcr_mean = np.mean(zcr)
        
        # 7. RMS Energy (volumen/intensidad)
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        rms_mean = np.mean(rms)
        
        # 8. Pitch tracking (fundamental frequency)
        pitch_mean, pitch_std = self._extract_pitch(audio)
        
        return AcousticFeatures(
            mfcc=mfcc_mean,
            chroma=chroma_mean,
            spectral_centroid=float(spectral_centroid_mean),
            spectral_rolloff=float(spectral_rolloff_mean),
            spectral_contrast=spectral_contrast_mean,
            zcr=float(zcr_mean),
            rms_energy=float(rms_mean),
            pitch_mean=pitch_mean,
            pitch_std=pitch_std
        )
    
    def _extract_pitch(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extrae pitch (F0) usando autocorrelación
        
        Args:
            audio: Waveform
        
        Returns:
            (pitch_mean, pitch_std) en Hz
        """
        try:
            # Estimar pitch con pyin (probabilistic YIN)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz (voz humana baja)
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz (voz humana alta)
                sr=self.sample_rate
            )
            
            # Filtrar frames no vocalizados
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 0:
                pitch_mean = float(np.nanmean(f0_voiced))
                pitch_std = float(np.nanstd(f0_voiced))
            else:
                pitch_mean = 0.0
                pitch_std = 0.0
        
        except Exception:
            # Fallback si pyin falla
            pitch_mean = 0.0
            pitch_std = 0.0
        
        return pitch_mean, pitch_std
    
    def extract_statistics(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extrae estadísticas adicionales del audio
        
        Args:
            audio: Waveform
        
        Returns:
            Dict con estadísticas globales
        """
        return {
            "duration_s": len(audio) / self.sample_rate,
            "mean_amplitude": float(np.mean(np.abs(audio))),
            "max_amplitude": float(np.max(np.abs(audio))),
            "std_amplitude": float(np.std(audio)),
            "skewness": float(self._compute_skewness(audio)),
            "kurtosis": float(self._compute_kurtosis(audio))
        }
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Calcula asimetría de la distribución"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Calcula kurtosis (curtosis) de la distribución"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4)) - 3.0


# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_emotion_features(
    audio: np.ndarray,
    sample_rate: int = 16000
) -> AcousticFeatures:
    """
    Helper para extraer features rápidamente
    
    Args:
        audio: Waveform
        sample_rate: Sample rate
    
    Returns:
        AcousticFeatures completos
    
    Example:
        >>> audio, sr = librosa.load("speech.wav", sr=16000)
        >>> features = extract_emotion_features(audio, sr)
        >>> feature_vector = features.to_vector()  # 37-D
    """
    extractor = EmotionFeatureExtractor(sample_rate)
    return extractor.extract(audio)


def features_to_emotion_heuristic(features: AcousticFeatures) -> Dict[str, float]:
    """
    Heurísticas básicas para mapear features → emociones
    
    Args:
        features: Features acústicas
    
    Returns:
        Dict con scores por emoción
    
    Algorithm:
        - Pitch alto + RMS alta → EXCITED/SURPRISED
        - Pitch bajo + RMS baja → SAD
        - RMS muy alta + ZCR alto → ANGRY
        - Spectral centroid bajo + RMS baja → CALM
    """
    scores = {
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "fearful": 0.0,
        "surprised": 0.0,
        "calm": 0.0,
        "excited": 0.0,
        "neutral": 0.2  # Baseline
    }
    
    # Normalizar features clave
    pitch_norm = min(features.pitch_mean / 200.0, 1.0)  # 200Hz = neutral
    energy_norm = min(features.rms_energy / 0.5, 1.0)
    zcr_norm = min(features.zcr / 0.2, 1.0)
    centroid_norm = min(features.spectral_centroid / 2000.0, 1.0)
    
    # Reglas heurísticas
    # EXCITED: pitch alto + energía alta
    scores["excited"] = (pitch_norm * 0.6 + energy_norm * 0.4) * 0.8
    
    # HAPPY: pitch medio-alto + energía media
    scores["happy"] = (pitch_norm * 0.5 + energy_norm * 0.3) * 0.7
    
    # SAD: pitch bajo + energía baja
    scores["sad"] = ((1.0 - pitch_norm) * 0.6 + (1.0 - energy_norm) * 0.4) * 0.6
    
    # ANGRY: energía muy alta + ZCR alto
    scores["angry"] = (energy_norm * 0.7 + zcr_norm * 0.3) * 0.7
    
    # SURPRISED: pitch muy alto + energía alta
    if pitch_norm > 0.7:
        scores["surprised"] = (pitch_norm * 0.8 + energy_norm * 0.2) * 0.6
    
    # FEARFUL: pitch alto + energía variable
    if pitch_norm > 0.6:
        scores["fearful"] = pitch_norm * 0.5
    
    # CALM: pitch bajo + energía baja + centroid bajo
    scores["calm"] = (
        (1.0 - pitch_norm) * 0.4 +
        (1.0 - energy_norm) * 0.4 +
        (1.0 - centroid_norm) * 0.2
    ) * 0.5
    
    # Normalizar a [0, 1]
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    
    return scores
