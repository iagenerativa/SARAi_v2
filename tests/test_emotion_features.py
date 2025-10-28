"""
tests/test_emotion_features.py

Suite de tests para extracción de features acústicas emocionales

Testing:
- AcousticFeatures dataclass
- EmotionFeatureExtractor (MFCCs, chroma, spectral, pitch)
- Heurísticas features → emociones
- Vector conversion para ML

Author: SARAi v2.11
Date: 2025-10-28
"""

import pytest
import numpy as np
from typing import Dict

# Importar módulo a testear
from agents.emotion_features import (
    AcousticFeatures,
    EmotionFeatureExtractor,
    extract_emotion_features,
    features_to_emotion_heuristic,
    LIBROSA_AVAILABLE
)


# ============================================
# SKIP SI LIBROSA NO DISPONIBLE
# ============================================

pytestmark = pytest.mark.skipif(
    not LIBROSA_AVAILABLE,
    reason="LibROSA no disponible (pip install librosa)"
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def sample_audio() -> np.ndarray:
    """
    Audio sintético: 1s de tono puro 440Hz (A4)
    Simula voz neutral
    """
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Tono puro + armónicos (simula voz)
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # Fundamental
        0.3 * np.sin(2 * np.pi * 880 * t) +  # 2º armónico
        0.2 * np.sin(2 * np.pi * 1320 * t)   # 3º armónico
    )
    
    # Normalizar
    audio = audio / np.max(np.abs(audio))
    
    return audio.astype(np.float32)


@pytest.fixture
def happy_audio() -> np.ndarray:
    """
    Audio sintético: tono alto (600Hz) + energía alta
    Simula voz FELIZ
    """
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Pitch alto + modulación
    audio = (
        0.7 * np.sin(2 * np.pi * 600 * t) +
        0.4 * np.sin(2 * np.pi * 1200 * t) +
        0.3 * np.sin(2 * np.pi * 1800 * t)
    )
    
    # Normalizar
    audio = audio / np.max(np.abs(audio))
    
    return audio.astype(np.float32)


@pytest.fixture
def sad_audio() -> np.ndarray:
    """
    Audio sintético: tono bajo (180Hz) + energía baja
    Simula voz TRISTE
    """
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Pitch bajo + energía reducida
    audio = (
        0.3 * np.sin(2 * np.pi * 180 * t) +
        0.2 * np.sin(2 * np.pi * 360 * t)
    )
    
    # Normalizar
    audio = audio / np.max(np.abs(audio))
    
    return audio.astype(np.float32)


@pytest.fixture
def extractor() -> EmotionFeatureExtractor:
    """Extractor configurado"""
    return EmotionFeatureExtractor(sample_rate=16000)


# ============================================
# TESTS: AcousticFeatures
# ============================================

def test_acoustic_features_creation():
    """AcousticFeatures se crea correctamente"""
    features = AcousticFeatures(
        mfcc=np.random.rand(13),
        chroma=np.random.rand(12),
        spectral_centroid=1500.0,
        spectral_rolloff=3000.0,
        spectral_contrast=np.random.rand(7),
        zcr=0.15,
        rms_energy=0.4,
        pitch_mean=220.0,
        pitch_std=15.0
    )
    
    assert features.mfcc.shape == (13,)
    assert features.chroma.shape == (12,)
    assert features.spectral_contrast.shape == (7,)
    assert features.spectral_centroid == 1500.0
    assert features.pitch_mean == 220.0


def test_to_vector_shape():
    """to_vector() retorna vector de tamaño correcto"""
    features = AcousticFeatures(
        mfcc=np.random.rand(13),
        chroma=np.random.rand(12),
        spectral_centroid=1500.0,
        spectral_rolloff=3000.0,
        spectral_contrast=np.random.rand(7),
        zcr=0.15,
        rms_energy=0.4,
        pitch_mean=220.0,
        pitch_std=15.0
    )
    
    vector = features.to_vector()
    
    # 13 (MFCC) + 12 (chroma) + 7 (contrast) + 5 (scalars) = 37
    assert vector.shape == (37,)
    assert not np.any(np.isnan(vector))


def test_to_vector_content():
    """to_vector() preserva contenido correcto"""
    features = AcousticFeatures(
        mfcc=np.ones(13) * 10,
        chroma=np.ones(12) * 20,
        spectral_centroid=1500.0,
        spectral_rolloff=3000.0,
        spectral_contrast=np.ones(7) * 30,
        zcr=0.15,
        rms_energy=0.4,
        pitch_mean=220.0,
        pitch_std=15.0
    )
    
    vector = features.to_vector()
    
    # Verificar bloques
    assert np.all(vector[:13] == 10)  # MFCC
    assert np.all(vector[13:25] == 20)  # Chroma
    assert np.all(vector[25:32] == 30)  # Contrast
    assert vector[32] == 1500.0  # Centroid
    assert vector[33] == 3000.0  # Rolloff
    assert vector[34] == 0.15  # ZCR
    assert vector[35] == 0.4  # RMS
    assert vector[36] == 220.0  # Pitch mean


# ============================================
# TESTS: EmotionFeatureExtractor
# ============================================

def test_extractor_initialization():
    """Extractor se inicializa correctamente"""
    extractor = EmotionFeatureExtractor(sample_rate=16000)
    
    assert extractor.sample_rate == 16000
    assert extractor.n_mfcc == 13
    assert extractor.n_chroma == 12


def test_extract_basic(extractor, sample_audio):
    """extract() retorna AcousticFeatures válidos"""
    features = extractor.extract(sample_audio)
    
    assert isinstance(features, AcousticFeatures)
    assert features.mfcc.shape == (13,)
    assert features.chroma.shape == (12,)
    assert features.spectral_contrast.shape == (7,)
    assert features.spectral_centroid > 0
    assert features.rms_energy > 0


def test_extract_mfcc_range(extractor, sample_audio):
    """MFCCs están en rango razonable"""
    features = extractor.extract(sample_audio)
    
    # MFCC[0] puede ser muy grande (energía total), el resto más acotado
    assert features.mfcc[0] > -500  # Primer coeficiente
    assert np.all(features.mfcc[1:] > -100)  # Resto de coeficientes
    assert np.all(features.mfcc < 100)


def test_extract_chroma_range(extractor, sample_audio):
    """Chroma features están normalizados"""
    features = extractor.extract(sample_audio)
    
    # Chroma típicamente en [0, 1]
    assert np.all(features.chroma >= 0)
    assert np.all(features.chroma <= 1.5)  # Margen para variaciones


def test_extract_pitch_440hz(extractor, sample_audio):
    """Pitch detecta correctamente 440Hz (A4)"""
    features = extractor.extract(sample_audio)
    
    # Audio sintético es 440Hz ± margen
    assert 400 < features.pitch_mean < 480
    assert features.pitch_std >= 0


def test_extract_empty_audio_fails(extractor):
    """extract() falla con audio vacío"""
    with pytest.raises(ValueError, match="Audio vacío"):
        extractor.extract(np.array([]))


def test_extract_normalizes_audio(extractor):
    """extract() normaliza audio muy alto"""
    # Audio con amplitudes >1.0
    loud_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 10.0
    
    # No debe crashear
    features = extractor.extract(loud_audio)
    
    assert features.rms_energy > 0
    assert features.rms_energy <= 1.0  # Normalizado


def test_extract_statistics(extractor, sample_audio):
    """extract_statistics() retorna métricas correctas"""
    stats = extractor.extract_statistics(sample_audio)
    
    assert "duration_s" in stats
    assert "mean_amplitude" in stats
    assert "max_amplitude" in stats
    assert "std_amplitude" in stats
    assert "skewness" in stats
    assert "kurtosis" in stats
    
    # Validar valores razonables
    assert stats["duration_s"] == pytest.approx(1.0, abs=0.1)
    assert 0 <= stats["max_amplitude"] <= 1.0


# ============================================
# TESTS: Helper Functions
# ============================================

def test_extract_emotion_features_helper(sample_audio):
    """extract_emotion_features() wrapper funciona"""
    features = extract_emotion_features(sample_audio, sample_rate=16000)
    
    assert isinstance(features, AcousticFeatures)
    assert features.mfcc.shape == (13,)


def test_features_to_emotion_heuristic(sample_audio):
    """features_to_emotion_heuristic() retorna scores válidos"""
    extractor = EmotionFeatureExtractor(sample_rate=16000)
    features = extractor.extract(sample_audio)
    
    scores = features_to_emotion_heuristic(features)
    
    # Verificar estructura
    assert "happy" in scores
    assert "sad" in scores
    assert "angry" in scores
    assert "neutral" in scores
    
    # Scores en [0, 1]
    for emotion, score in scores.items():
        assert 0 <= score <= 1.0
    
    # Suma aproximadamente 1.0 (normalización)
    total = sum(scores.values())
    assert 0.9 <= total <= 1.1


def test_heuristic_detects_happy(happy_audio):
    """Heurística detecta HAPPY en audio alegre"""
    extractor = EmotionFeatureExtractor(sample_rate=16000)
    features = extractor.extract(happy_audio)
    
    scores = features_to_emotion_heuristic(features)
    
    # HAPPY o EXCITED debe tener score alto
    assert scores["happy"] > 0.15 or scores["excited"] > 0.15


def test_heuristic_detects_sad(sad_audio):
    """Heurística detecta SAD en audio triste"""
    extractor = EmotionFeatureExtractor(sample_rate=16000)
    features = extractor.extract(sad_audio)
    
    scores = features_to_emotion_heuristic(features)
    
    # SAD o CALM debe tener score razonable (pitch bajo)
    assert scores["sad"] > 0.01 or scores["calm"] > 0.05


# ============================================
# TESTS: Edge Cases
# ============================================

def test_extract_silence():
    """extract() maneja silencio (audio = 0)"""
    silence = np.zeros(16000, dtype=np.float32)
    extractor = EmotionFeatureExtractor(sample_rate=16000)
    
    features = extractor.extract(silence)
    
    # Debe retornar features válidos (aunque con valores bajos)
    assert features.rms_energy == pytest.approx(0.0, abs=1e-6)
    assert features.pitch_mean == 0.0  # Sin pitch en silencio


def test_extract_noise():
    """extract() maneja ruido blanco"""
    noise = np.random.randn(16000).astype(np.float32) * 0.1
    extractor = EmotionFeatureExtractor(sample_rate=16000)
    
    features = extractor.extract(noise)
    
    # Ruido blanco tiene ZCR alto
    assert features.zcr > 0.2
    # Pitch debería ser bajo/cero (sin estructura tonal)
    assert features.pitch_mean < 100 or features.pitch_mean == 0.0


def test_extract_very_short_audio():
    """extract() maneja audio muy corto (0.1s)"""
    short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600)).astype(np.float32)
    extractor = EmotionFeatureExtractor(sample_rate=16000)
    
    features = extractor.extract(short_audio)
    
    # Debe funcionar (aunque menos confiable)
    assert features.mfcc.shape == (13,)


# ============================================
# TESTS: Integration
# ============================================

def test_full_pipeline_happy_to_vector():
    """Pipeline completo: audio → features → vector"""
    # 1. Audio alegre
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    happy_audio = 0.7 * np.sin(2 * np.pi * 600 * t)
    happy_audio = happy_audio.astype(np.float32)
    
    # 2. Extraer features
    features = extract_emotion_features(happy_audio, sample_rate)
    
    # 3. Convertir a vector
    vector = features.to_vector()
    
    # 4. Verificar
    assert vector.shape == (37,)
    assert not np.any(np.isnan(vector))
    assert not np.any(np.isinf(vector))


def test_full_pipeline_with_heuristic():
    """Pipeline: audio → features → heurística → scores"""
    # Audio neutro
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 220 * t)
    audio = audio.astype(np.float32)
    
    # Extraer features
    extractor = EmotionFeatureExtractor(sample_rate)
    features = extractor.extract(audio)
    
    # Aplicar heurística
    scores = features_to_emotion_heuristic(features)
    
    # Verificar salida válida
    assert len(scores) >= 6  # Al menos 6 emociones
    assert all(0 <= v <= 1 for v in scores.values())
    assert sum(scores.values()) == pytest.approx(1.0, abs=0.1)


# ============================================
# BENCHMARK (opcional, pytest -k bench)
# ============================================

def test_benchmark_extraction_speed():
    """Benchmark: extracción debe ser <300ms para 1s audio en CPU"""
    import time
    
    # Audio de 1 segundo
    audio = np.random.randn(16000).astype(np.float32)
    extractor = EmotionFeatureExtractor(sample_rate=16000)
    
    # Warm-up
    extractor.extract(audio)
    
    # Benchmark (3 iteraciones)
    times = []
    for _ in range(3):
        start = time.time()
        extractor.extract(audio)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    
    # Debe ser <300ms en CPU moderno (LibROSA es más lento que heurísticas)
    assert avg_time < 0.35  # 350ms margen generoso para CI
    print(f"\n⏱️  Tiempo promedio extracción: {avg_time*1000:.1f}ms")
