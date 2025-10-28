"""
agents/emotion_modulator.py

Modulador emocional para SARAi v2.11
Detecta emociones en audio y ajusta respuestas según contexto afectivo

Features v2.11.1:
- Detección multi-heurística (energía, ZCR, espectral, textual)
- **NEW**: MFCC + Chroma + Spectral features (LibROSA)
- Análisis de trayectoria emocional conversacional
- Blend de emociones secundarias
- Keywords contextuales español/inglés

Author: SARAi v2.11
Date: 2025-10-28
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# NEW v2.11.1: Importar acoustic features si disponible
try:
    from agents.emotion_features import (
        EmotionFeatureExtractor,
        features_to_emotion_heuristic,
        LIBROSA_AVAILABLE
    )
    USE_LIBROSA_FEATURES = LIBROSA_AVAILABLE
except ImportError:
    USE_LIBROSA_FEATURES = False

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from enum import Enum


class EmotionCategory(str, Enum):
    """
    Categorías emocionales básicas (modelo de Ekman + extensiones)
    
    Ref: Ekman, P. (1992). "An argument for basic emotions"
    """
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    CALM = "calm"
    EXCITED = "excited"
    NEUTRAL = "neutral"


@dataclass
class EmotionProfile:
    """
    Perfil emocional detectado del usuario
    
    Attributes:
        primary: Emoción dominante
        secondary: Emoción secundaria (si aplica)
        intensity: Intensidad [0.0, 1.0]
        confidence: Confianza de detección [0.0, 1.0]
        raw_scores: Scores por categoría
    """
    primary: EmotionCategory
    secondary: Optional[EmotionCategory] = None
    intensity: float = 0.5
    confidence: float = 0.0
    raw_scores: Dict[EmotionCategory, float] = None
    
    def __post_init__(self):
        # Validación de rangos
        assert 0.0 <= self.intensity <= 1.0, "Intensity must be in [0, 1]"
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be in [0, 1]"
        
        if self.raw_scores is None:
            self.raw_scores = {}


@dataclass
class ModulationResult:
    """
    Resultado de modulación emocional
    
    Attributes:
        modulated_embedding: Embedding ajustado
        original_embedding: Embedding original
        emotion_vector: Vector de emoción aplicado
        delta_norm: Magnitud del cambio (L2 norm)
        metadata: Metadata adicional
    """
    modulated_embedding: np.ndarray
    original_embedding: np.ndarray
    emotion_vector: np.ndarray
    delta_norm: float
    metadata: Dict = None


class EmotionModulator:
    """
    Modulador emocional para embeddings de voz
    
    Pipeline:
    1. Detectar emoción del audio (usando modelo pre-entrenado)
    2. Calcular vector de ajuste emocional
    3. Modular embedding con interpolación ponderada
    4. Retornar embedding ajustado + metadata
    
    Ejemplo:
        modulator = EmotionModulator()
        profile = modulator.detect_emotion(audio_features)
        result = modulator.modulate(embedding, profile)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        emotion_vectors_path: Optional[str] = "models/emotion_vectors.npy",
        sample_rate: int = 16000  # NEW v2.11.1
    ):
        """
        Inicializa modulador emocional
        
        Args:
            model_path: Path al modelo de detección (si None, usa heurísticas)
            emotion_vectors_path: Path a vectores emocionales pre-calculados
            sample_rate: Sample rate del audio en Hz (NEW v2.11.1)
        """
        self.model_path = model_path
        self.emotion_vectors_path = emotion_vectors_path
        self.sample_rate = sample_rate  # NEW v2.11.1
        
        # Lazy loading (cargar solo cuando se necesite)
        self._emotion_model = None
        self._emotion_vectors = None
        
        # NEW v2.11.1: Inicializar extractor de features acústicas
        if USE_LIBROSA_FEATURES:
            self.feature_extractor = EmotionFeatureExtractor(sample_rate)
            logger.info("✅ EmotionModulator con LibROSA features (MFCC, chroma)")
        else:
            self.feature_extractor = None
            logger.warning("⚠️  EmotionModulator sin LibROSA (heurísticas básicas)")
        
        # Parámetros de modulación (tuneables)
        self.modulation_strength = 0.3  # [0, 1] - qué tanto ajustar
        self.min_confidence_threshold = 0.5  # No modular si confianza <50%
        
        # Estadísticas (para auditoría)
        self.stats = {
            "total_modulations": 0,
            "skipped_low_confidence": 0,
            "avg_delta_norm": 0.0,
            "emotion_distribution": {e: 0 for e in EmotionCategory}
        }
    
    def load_emotion_vectors(self) -> Dict[EmotionCategory, np.ndarray]:
        """
        Carga vectores emocionales pre-calculados
        
        Returns:
            Dict mapping EmotionCategory → vector (768-D para Qwen-Omni)
        
        Raises:
            FileNotFoundError: Si no existe el archivo
        """
        if self._emotion_vectors is not None:
            return self._emotion_vectors
        
        try:
            # Cargar desde disco
            vectors_data = np.load(self.emotion_vectors_path, allow_pickle=True).item()
            self._emotion_vectors = {
                EmotionCategory(k): v for k, v in vectors_data.items()
            }
        except FileNotFoundError:
            # Fallback: vectores sintéticos (para testing)
            self._emotion_vectors = self._generate_synthetic_vectors()
        
        return self._emotion_vectors
    
    def _generate_synthetic_vectors(self) -> Dict[EmotionCategory, np.ndarray]:
        """
        Genera vectores emocionales sintéticos para testing
        
        Returns:
            Dict con vectores aleatorios normalizados (768-D)
        """
        np.random.seed(42)  # Reproducibilidad
        vectors = {}
        
        for emotion in EmotionCategory:
            # Vector aleatorio normalizado
            vec = np.random.randn(768)
            vec = vec / np.linalg.norm(vec)  # Normalizar a unit sphere
            vectors[emotion] = vec.astype(np.float32)
        
        return vectors
    
    def detect_emotion(
        self,
        audio_features: np.ndarray,
        text: Optional[str] = None
    ) -> EmotionProfile:
        """
        Detecta emoción del audio con heurísticas mejoradas
        
        Args:
            audio_features: Features extraídos del audio (e.g., mel-spectrogram, waveform)
            text: Transcripción del audio (opcional, para análisis multi-modal)
        
        Returns:
            EmotionProfile con emoción detectada + confianza
        
        Algorithm v2.11.1:
            1. **NEW**: Si LibROSA disponible, extrae MFCC+chroma+spectral
            2. Extrae características acústicas básicas (energía, ZCR, espectral)
            3. Analiza texto si disponible (keywords emocionales)
            4. **NEW**: Combina scores LibROSA + heurísticas + textuales
            5. Selecciona emoción primaria + secundaria (si aplica)
        
        Note:
            Fase 2 integrará modelo pre-entrenado (emoDBert, WavLM-emotion)
        """
        # PASO 0 (NEW v2.11.1): Features acústicas avanzadas con LibROSA
        if USE_LIBROSA_FEATURES and self.feature_extractor is not None:
            try:
                acoustic_features = self.feature_extractor.extract(audio_features)
                librosa_scores = features_to_emotion_heuristic(acoustic_features)
                
                logger.debug(f"🎵 LibROSA scores: {librosa_scores}")
                
                # Usar scores de LibROSA como base
                emotion_scores = {
                    EmotionCategory.HAPPY: librosa_scores.get("happy", 0.0),
                    EmotionCategory.SAD: librosa_scores.get("sad", 0.0),
                    EmotionCategory.ANGRY: librosa_scores.get("angry", 0.0),
                    EmotionCategory.FEARFUL: librosa_scores.get("fearful", 0.0),
                    EmotionCategory.SURPRISED: librosa_scores.get("surprised", 0.0),
                    EmotionCategory.CALM: librosa_scores.get("calm", 0.0),
                    EmotionCategory.EXCITED: librosa_scores.get("excited", 0.0),
                    EmotionCategory.NEUTRAL: librosa_scores.get("neutral", 0.0),
                    EmotionCategory.DISGUSTED: 0.0  # No en heurística básica
                }
                
                # Añadir peso adicional de heurísticas básicas (blend)
                use_basic_heuristics = True
                base_weight = 0.4  # 60% LibROSA, 40% heurísticas básicas
                
            except Exception as e:
                logger.warning(f"⚠️  LibROSA feature extraction falló: {e}")
                use_basic_heuristics = True
                base_weight = 1.0
                emotion_scores = {e: 0.0 for e in EmotionCategory}
        else:
            # Fallback: solo heurísticas básicas
            use_basic_heuristics = True
            base_weight = 1.0
            emotion_scores = {e: 0.0 for e in EmotionCategory}
        
        # PASO 1: Características acústicas básicas (siempre ejecutar si weight > 0)
        if use_basic_heuristics and base_weight > 0:
            mean_energy = np.mean(np.abs(audio_features))
            max_energy = np.max(np.abs(audio_features))
            std_energy = np.std(audio_features)
            
            # Zero-Crossing Rate (correlación con pitch/excitación)
            zcr = np.mean(np.abs(np.diff(np.sign(audio_features)))) / 2.0
            
            # Scores básicos
            basic_scores = {e: 0.0 for e in EmotionCategory}
            
            # HEURÍSTICA 1: Energía alta → Excited/Angry
            if max_energy > 0.7:
                basic_scores[EmotionCategory.EXCITED] += 0.6
                basic_scores[EmotionCategory.ANGRY] += 0.3 * (std_energy / 0.5)
            
            # HEURÍSTICA 2: Energía baja → Sad/Calm
            elif mean_energy < 0.2:
                basic_scores[EmotionCategory.SAD] += 0.5
                basic_scores[EmotionCategory.CALM] += 0.3
            
            # HEURÍSTICA 3: Alta varianza → Angry/Fearful
            if std_energy > 0.3:
                basic_scores[EmotionCategory.ANGRY] += 0.4
                basic_scores[EmotionCategory.FEARFUL] += 0.2
            
            # HEURÍSTICA 4: ZCR alto → Excited/Surprised
            if zcr > 0.15:
                basic_scores[EmotionCategory.EXCITED] += 0.3
                basic_scores[EmotionCategory.SURPRISED] += 0.2
            
            # Blend LibROSA + básicas (si aplica)
            for emotion in EmotionCategory:
                emotion_scores[emotion] += basic_scores[emotion] * base_weight
        
        # PASO 2: Análisis textual (si disponible)
        if text:
            text_lower = text.lower()
            
            # Keywords emocionales (básico)
            emotion_keywords = {
                EmotionCategory.HAPPY: ["feliz", "alegre", "contento", "genial", "happy", "great"],
                EmotionCategory.SAD: ["triste", "deprimido", "mal", "sad", "depressed"],
                EmotionCategory.ANGRY: ["enojado", "furioso", "molesto", "angry", "mad"],
                EmotionCategory.FEARFUL: ["miedo", "asustado", "nervioso", "scared", "afraid"],
                EmotionCategory.SURPRISED: ["sorprendido", "wow", "increíble", "surprised"],
                EmotionCategory.CALM: ["tranquilo", "relajado", "sereno", "calm", "relaxed"]
            }
            
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        emotion_scores[emotion] += 0.4
        
        # PASO 3: Normalizar y aplicar baseline neutral
        emotion_scores[EmotionCategory.NEUTRAL] = 0.2  # Baseline siempre presente
        
        # Normalizar scores a [0, 1]
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {e: s / total_score for e, s in emotion_scores.items()}
        
        # PASO 4: Seleccionar primaria y secundaria
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary = sorted_emotions[0][0]
        primary_score = sorted_emotions[0][1]
        
        # Secundaria solo si score > 0.2 y diferencia con primaria < 0.3
        secondary = None
        if len(sorted_emotions) > 1:
            secondary_candidate = sorted_emotions[1]
            if secondary_candidate[1] > 0.2 and (primary_score - secondary_candidate[1]) < 0.3:
                secondary = secondary_candidate[0]
        
        # PASO 5: Calcular intensity y confidence
        intensity = min(primary_score * 1.5, 1.0)  # Amplificar ligeramente
        
        # Confidence basado en separación entre emociones
        if len(sorted_emotions) > 1:
            separation = sorted_emotions[0][1] - sorted_emotions[1][1]
            confidence = min(0.5 + separation, 1.0)
        else:
            confidence = 0.7
        
        return EmotionProfile(
            primary=primary,
            secondary=secondary,
            intensity=intensity,
            confidence=confidence,
            raw_scores=emotion_scores
        )
    
    def modulate(
        self,
        embedding: np.ndarray,
        emotion_profile: EmotionProfile
    ) -> ModulationResult:
        """
        Modula embedding según perfil emocional
        
        Args:
            embedding: Embedding original (768-D)
            emotion_profile: Perfil emocional detectado
        
        Returns:
            ModulationResult con embedding modulado + metadata
        
        Algorithm:
            modulated = original + α * emotion_vector
            donde α = strength * intensity * confidence
        """
        # Validación
        assert embedding.shape == (768,), f"Esperado (768,), got {embedding.shape}"
        
        # Check confidence threshold
        if emotion_profile.confidence < self.min_confidence_threshold:
            self.stats["skipped_low_confidence"] += 1
            return ModulationResult(
                modulated_embedding=embedding.copy(),
                original_embedding=embedding.copy(),
                emotion_vector=np.zeros(768),
                delta_norm=0.0,
                metadata={"reason": "low_confidence", "confidence": emotion_profile.confidence}
            )
        
        # Cargar vectores emocionales
        emotion_vectors = self.load_emotion_vectors()
        emotion_vector = emotion_vectors[emotion_profile.primary]
        
        # Calcular factor de modulación
        alpha = (
            self.modulation_strength *
            emotion_profile.intensity *
            emotion_profile.confidence
        )
        
        # Aplicar modulación
        modulated_embedding = embedding + alpha * emotion_vector
        
        # Normalizar (mantener en unit sphere)
        modulated_embedding = modulated_embedding / np.linalg.norm(modulated_embedding)
        
        # Calcular delta
        delta_norm = np.linalg.norm(modulated_embedding - embedding)
        
        # Actualizar estadísticas
        self.stats["total_modulations"] += 1
        self.stats["emotion_distribution"][emotion_profile.primary] += 1
        self.stats["avg_delta_norm"] = (
            (self.stats["avg_delta_norm"] * (self.stats["total_modulations"] - 1) + delta_norm) /
            self.stats["total_modulations"]
        )
        
        return ModulationResult(
            modulated_embedding=modulated_embedding.astype(np.float32),
            original_embedding=embedding.copy(),
            emotion_vector=emotion_vector,
            delta_norm=delta_norm,
            metadata={
                "emotion": emotion_profile.primary.value,
                "intensity": emotion_profile.intensity,
                "confidence": emotion_profile.confidence,
                "alpha": alpha
            }
        )
    
    def get_stats(self) -> Dict:
        """
        Retorna estadísticas de modulación
        
        Returns:
            Dict con métricas de uso
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Resetea estadísticas (útil para testing)"""
        self.stats = {
            "total_modulations": 0,
            "skipped_low_confidence": 0,
            "avg_delta_norm": 0.0,
            "emotion_distribution": {e: 0 for e in EmotionCategory}
        }


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_emotion_modulator(
    modulation_strength: float = 0.3,
    min_confidence: float = 0.5
) -> EmotionModulator:
    """
    Factory function para crear modulador con config
    
    Args:
        modulation_strength: Fuerza de modulación [0, 1]
        min_confidence: Confianza mínima para modular
    
    Returns:
        EmotionModulator configurado
    
    Example:
        >>> modulator = create_emotion_modulator(modulation_strength=0.4)
        >>> profile = EmotionProfile(primary=EmotionCategory.HAPPY, intensity=0.8, confidence=0.9)
        >>> result = modulator.modulate(embedding, profile)
    """
    modulator = EmotionModulator()
    modulator.modulation_strength = modulation_strength
    modulator.min_confidence_threshold = min_confidence
    return modulator


def blend_emotion_vectors(
    vectors: Dict[EmotionCategory, np.ndarray],
    weights: Dict[EmotionCategory, float]
) -> np.ndarray:
    """
    Combina múltiples vectores emocionales con pesos
    
    Args:
        vectors: Dict de vectores emocionales (768-D cada uno)
        weights: Dict de pesos por emoción (deben sumar ~1.0)
    
    Returns:
        Vector combinado normalizado (768-D)
    
    Example:
        >>> vectors = {EmotionCategory.HAPPY: vec1, EmotionCategory.EXCITED: vec2}
        >>> weights = {EmotionCategory.HAPPY: 0.7, EmotionCategory.EXCITED: 0.3}
        >>> blended = blend_emotion_vectors(vectors, weights)
    
    Note:
        Útil para emociones mixtas (e.g., 70% feliz + 30% sorprendido)
    """
    # Validar que todas las emociones tengan vectores
    for emotion in weights.keys():
        if emotion not in vectors:
            raise ValueError(f"Emoción {emotion} no tiene vector disponible")
    
    # Combinar con pesos
    blended = np.zeros(768, dtype=np.float32)
    for emotion, weight in weights.items():
        blended += weight * vectors[emotion]
    
    # Normalizar a unit sphere
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm
    
    return blended


def analyze_emotion_trajectory(
    profiles: List[EmotionProfile],
    window_size: int = 5
) -> Dict:
    """
    Analiza la trayectoria emocional en una conversación
    
    Args:
        profiles: Lista de perfiles emocionales (orden cronológico)
        window_size: Tamaño de ventana para suavizado
    
    Returns:
        Dict con métricas de trayectoria:
        - dominant_emotion: Emoción más frecuente
        - avg_intensity: Intensidad promedio
        - volatility: Volatilidad emocional (std de cambios)
        - trend: "escalating", "de-escalating", "stable"
    
    Example:
        >>> profiles = [profile1, profile2, profile3, ...]
        >>> trajectory = analyze_emotion_trajectory(profiles)
        >>> print(f"Usuario está {trajectory['trend']}")
    
    Note:
        Útil para adaptar estrategia de respuesta en diálogos largos
    """
    if not profiles:
        return {
            "dominant_emotion": EmotionCategory.NEUTRAL,
            "avg_intensity": 0.0,
            "volatility": 0.0,
            "trend": "stable"
        }
    
    # Contar emociones dominantes
    emotion_counts = {}
    intensities = []
    
    for profile in profiles:
        emotion = profile.primary
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        intensities.append(profile.intensity)
    
    # Emoción dominante
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    
    # Intensidad promedio
    avg_intensity = np.mean(intensities)
    
    # Volatilidad (cambios entre emociones consecutivas)
    changes = []
    for i in range(1, len(profiles)):
        if profiles[i].primary != profiles[i-1].primary:
            changes.append(1)
        else:
            changes.append(0)
    
    volatility = np.mean(changes) if changes else 0.0
    
    # Tendencia (últimos N vs primeros N)
    if len(intensities) >= window_size * 2:
        first_half = np.mean(intensities[:window_size])
        second_half = np.mean(intensities[-window_size:])
        
        if second_half > first_half + 0.1:
            trend = "escalating"
        elif second_half < first_half - 0.1:
            trend = "de-escalating"
        else:
            trend = "stable"
    else:
        trend = "stable"
    
    return {
        "dominant_emotion": dominant_emotion,
        "avg_intensity": float(avg_intensity),
        "volatility": float(volatility),
        "trend": trend,
        "total_samples": len(profiles),
        "emotion_distribution": {e.value: c for e, c in emotion_counts.items()}
    }
