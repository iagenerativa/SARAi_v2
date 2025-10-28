"""
tests/test_emotion_modulator.py

Test suite para emotion_modulator.py

Coverage objetivo: ≥90%
Tests: 8 básicos (skeleton Fase 1)

Author: SARAi Team
Date: 2025-10-28
"""

import pytest
import numpy as np
from agents.emotion_modulator import (
    EmotionCategory,
    EmotionProfile,
    ModulationResult,
    EmotionModulator,
    create_emotion_modulator
)


class TestEmotionProfile:
    """Tests para EmotionProfile dataclass"""
    
    def test_create_basic_profile(self):
        """Test creación básica de perfil"""
        profile = EmotionProfile(
            primary=EmotionCategory.HAPPY,
            intensity=0.8,
            confidence=0.9
        )
        
        assert profile.primary == EmotionCategory.HAPPY
        assert profile.intensity == 0.8
        assert profile.confidence == 0.9
        assert profile.secondary is None
        assert isinstance(profile.raw_scores, dict)
    
    def test_profile_validation_intensity(self):
        """Test validación de rangos de intensity"""
        # Debería fallar con intensity > 1.0
        with pytest.raises(AssertionError, match="Intensity must be in"):
            EmotionProfile(
                primary=EmotionCategory.NEUTRAL,
                intensity=1.5,
                confidence=0.5
            )
        
        # Debería fallar con intensity < 0.0
        with pytest.raises(AssertionError, match="Intensity must be in"):
            EmotionProfile(
                primary=EmotionCategory.NEUTRAL,
                intensity=-0.2,
                confidence=0.5
            )
    
    def test_profile_validation_confidence(self):
        """Test validación de rangos de confidence"""
        # Debería fallar con confidence > 1.0
        with pytest.raises(AssertionError, match="Confidence must be in"):
            EmotionProfile(
                primary=EmotionCategory.CALM,
                intensity=0.5,
                confidence=1.2
            )
    
    def test_profile_with_secondary_emotion(self):
        """Test perfil con emoción secundaria"""
        profile = EmotionProfile(
            primary=EmotionCategory.HAPPY,
            secondary=EmotionCategory.EXCITED,
            intensity=0.7,
            confidence=0.85,
            raw_scores={
                EmotionCategory.HAPPY: 0.6,
                EmotionCategory.EXCITED: 0.3,
                EmotionCategory.NEUTRAL: 0.1
            }
        )
        
        assert profile.secondary == EmotionCategory.EXCITED
        assert len(profile.raw_scores) == 3
        assert profile.raw_scores[EmotionCategory.HAPPY] == 0.6


class TestEmotionModulator:
    """Tests para EmotionModulator class"""
    
    def test_modulator_initialization(self):
        """Test inicialización del modulador"""
        modulator = EmotionModulator()
        
        assert modulator.modulation_strength == 0.3
        assert modulator.min_confidence_threshold == 0.5
        assert modulator.stats["total_modulations"] == 0
        assert modulator._emotion_vectors is None  # Lazy loading
    
    def test_load_synthetic_vectors(self):
        """Test generación de vectores sintéticos"""
        modulator = EmotionModulator(emotion_vectors_path="nonexistent.npy")
        vectors = modulator.load_emotion_vectors()
        
        # Verificar todas las categorías
        assert len(vectors) == len(EmotionCategory)
        
        # Verificar dimensiones (768-D para Qwen-Omni)
        for emotion, vec in vectors.items():
            assert vec.shape == (768,)
            assert vec.dtype == np.float32
            # Verificar normalización (unit sphere)
            assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
    
    def test_detect_emotion_heuristic(self):
        """Test detección heurística de emoción"""
        modulator = EmotionModulator()
        
        # Test 1: Audio con energía alta → EXCITED
        high_energy_audio = np.random.randn(16000) * 0.8
        profile = modulator.detect_emotion(high_energy_audio)
        
        assert profile.primary == EmotionCategory.EXCITED
        assert profile.confidence >= 0.5
        assert 0.0 <= profile.intensity <= 1.0
        
        # Test 2: Audio con energía baja → SAD
        low_energy_audio = np.random.randn(16000) * 0.1
        profile = modulator.detect_emotion(low_energy_audio)
        
        assert profile.primary == EmotionCategory.SAD
        assert profile.confidence >= 0.5
    
    def test_modulate_basic(self):
        """Test modulación básica de embedding"""
        modulator = EmotionModulator()
        
        # Embedding sintético (768-D normalizado)
        original_embedding = np.random.randn(768).astype(np.float32)
        original_embedding = original_embedding / np.linalg.norm(original_embedding)
        
        # Perfil con alta confianza
        profile = EmotionProfile(
            primary=EmotionCategory.HAPPY,
            intensity=0.8,
            confidence=0.9
        )
        
        result = modulator.modulate(original_embedding, profile)
        
        # Verificaciones
        assert isinstance(result, ModulationResult)
        assert result.modulated_embedding.shape == (768,)
        assert result.delta_norm > 0.0  # Hubo cambio
        assert np.isclose(np.linalg.norm(result.modulated_embedding), 1.0, atol=1e-5)  # Normalizado
        assert result.metadata["emotion"] == "happy"
    
    def test_modulate_low_confidence_skip(self):
        """Test que baja confianza NO modula"""
        modulator = EmotionModulator()
        
        original_embedding = np.random.randn(768).astype(np.float32)
        original_embedding = original_embedding / np.linalg.norm(original_embedding)
        
        # Perfil con BAJA confianza (< threshold 0.5)
        profile = EmotionProfile(
            primary=EmotionCategory.ANGRY,
            intensity=0.9,
            confidence=0.3  # ❌ Menor que threshold
        )
        
        result = modulator.modulate(original_embedding, profile)
        
        # Verificar que NO se moduló
        assert result.delta_norm == 0.0
        assert np.allclose(result.modulated_embedding, original_embedding)
        assert result.metadata["reason"] == "low_confidence"
        assert modulator.stats["skipped_low_confidence"] == 1
    
    def test_modulate_updates_stats(self):
        """Test que modulación actualiza estadísticas"""
        modulator = EmotionModulator()
        modulator.reset_stats()
        
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        profile = EmotionProfile(
            primary=EmotionCategory.CALM,
            intensity=0.6,
            confidence=0.8
        )
        
        # Modular 3 veces
        for _ in range(3):
            modulator.modulate(embedding, profile)
        
        stats = modulator.get_stats()
        
        assert stats["total_modulations"] == 3
        assert stats["emotion_distribution"][EmotionCategory.CALM] == 3
        assert stats["avg_delta_norm"] > 0.0


class TestHelperFunctions:
    """Tests para funciones helper"""
    
    def test_create_emotion_modulator_factory(self):
        """Test factory function con configuración custom"""
        modulator = create_emotion_modulator(
            modulation_strength=0.5,
            min_confidence=0.7
        )
        
        assert modulator.modulation_strength == 0.5
        assert modulator.min_confidence_threshold == 0.7
        assert isinstance(modulator, EmotionModulator)


# ============================================
# INTEGRATION TESTS (opcional para Fase 1)
# ============================================

class TestEmotionModulationIntegration:
    """Tests de integración end-to-end"""
    
    @pytest.mark.skip(reason="Requiere audio real - implementar en Fase 2")
    def test_full_pipeline_with_real_audio(self):
        """Test pipeline completo con audio real"""
        # TODO: Implementar cuando tengamos audio samples
        pass
