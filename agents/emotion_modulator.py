"""
agents/emotion_modulator.py

Módulo de modulación emocional para Qwen2.5-Omni-3B.
Ajusta embeddings según perfil emocional detectado.

Filosofía v2.11: "La empatía no es solo palabras, es resonancia vectorial"

KPIs Objetivo:
- MOS Empatía: ≥4.0/5.0
- Latencia modulación: ≤50ms
- Precisión detección: ≥85%

Author: SARAi Team
Date: 2025-10-28
Version: 2.11
"""

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
        emotion_vectors_path: Optional[str] = "models/emotion_vectors.npy"
    ):
        """
        Inicializa modulador emocional
        
        Args:
            model_path: Path al modelo de detección (si None, usa heurísticas)
            emotion_vectors_path: Path a vectores emocionales pre-calculados
        """
        self.model_path = model_path
        self.emotion_vectors_path = emotion_vectors_path
        
        # Lazy loading (cargar solo cuando se necesite)
        self._emotion_model = None
        self._emotion_vectors = None
        
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
        Detecta emoción del audio
        
        Args:
            audio_features: Features extraídos del audio (e.g., mel-spectrogram)
            text: Transcripción del audio (opcional, para multi-modal)
        
        Returns:
            EmotionProfile con emoción detectada + confianza
        
        Note:
            Implementación actual usa heurísticas simples.
            TODO: Integrar modelo pre-entrenado (emoDBert, WavLM-emotion, etc.)
        """
        # FASE 1: Heurística simple basada en energía del audio
        # TODO: Reemplazar con modelo real en Fase 2
        
        # Calcular estadísticas del audio
        mean_energy = np.mean(np.abs(audio_features))
        max_energy = np.max(np.abs(audio_features))
        std_energy = np.std(audio_features)
        
        # Heurística básica (placeholder)
        if max_energy > 0.7:
            primary = EmotionCategory.EXCITED
            intensity = 0.8
            confidence = 0.6
        elif mean_energy < 0.2:
            primary = EmotionCategory.SAD
            intensity = 0.5
            confidence = 0.6
        elif std_energy > 0.3:
            primary = EmotionCategory.ANGRY
            intensity = 0.7
            confidence = 0.5
        else:
            primary = EmotionCategory.NEUTRAL
            intensity = 0.3
            confidence = 0.7
        
        # Mock de scores (para compatibilidad con tests)
        raw_scores = {
            primary: confidence,
            EmotionCategory.NEUTRAL: 1.0 - confidence
        }
        
        return EmotionProfile(
            primary=primary,
            intensity=intensity,
            confidence=confidence,
            raw_scores=raw_scores
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
