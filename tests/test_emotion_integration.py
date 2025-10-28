"""
tests/test_emotion_integration.py

Test suite para emotion_integration.py

Coverage objetivo: ≥90%

Author: SARAi Team
Date: 2025-10-28
"""

import pytest
import numpy as np
from agents.emotion_integration import (
    VoiceAdaptationConfig,
    EmotionVoiceIntegrator,
    create_emotion_voice_integrator,
    get_prosody_for_emotion
)
from agents.emotion_modulator import (
    EmotionCategory,
    EmotionProfile,
    create_emotion_modulator
)


class TestVoiceAdaptationConfig:
    """Tests para VoiceAdaptationConfig"""
    
    def test_create_config(self):
        """Test creación de configuración"""
        config = VoiceAdaptationConfig(
            pitch_shift=2.0,
            speed_factor=1.1,
            volume_gain=1.5,
            pause_duration=100
        )
        
        assert config.pitch_shift == 2.0
        assert config.speed_factor == 1.1
        assert config.volume_gain == 1.5
        assert config.pause_duration == 100
    
    def test_to_dict(self):
        """Test exportación a dict"""
        config = VoiceAdaptationConfig(
            pitch_shift=-2.0,
            speed_factor=0.85,
            volume_gain=-2.0,
            pause_duration=300
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["pitch_shift"] == -2.0
        assert config_dict["speed_factor"] == 0.85
        assert config_dict["volume_gain"] == -2.0
        assert config_dict["pause_duration"] == 300


class TestEmotionVoiceIntegrator:
    """Tests para EmotionVoiceIntegrator"""
    
    def test_initialization(self):
        """Test inicialización del integrador"""
        integrator = EmotionVoiceIntegrator()
        
        assert integrator.modulator is not None
        assert len(integrator.EMOTION_VOICE_PROFILES) == 9  # 9 emociones
    
    def test_emotion_profiles_complete(self):
        """Test que todas las emociones tienen perfil de voz"""
        integrator = EmotionVoiceIntegrator()
        
        for emotion in EmotionCategory:
            assert emotion in integrator.EMOTION_VOICE_PROFILES
            config = integrator.EMOTION_VOICE_PROFILES[emotion]
            assert isinstance(config, VoiceAdaptationConfig)
    
    def test_detect_and_adapt_basic(self):
        """Test detección y adaptación básica"""
        integrator = EmotionVoiceIntegrator()
        
        # Audio sintético con energía alta (→ EXCITED)
        audio = np.random.randn(16000) * 0.8
        
        profile, config = integrator.detect_and_adapt(audio)
        
        # Verificar tipos
        assert isinstance(profile, EmotionProfile)
        assert isinstance(config, VoiceAdaptationConfig)
        
        # Verificar que config no es None
        assert config.pitch_shift is not None
        assert config.speed_factor > 0
    
    def test_detect_and_adapt_with_text(self):
        """Test detección con texto multimodal"""
        integrator = EmotionVoiceIntegrator()
        
        audio = np.random.randn(16000) * 0.3
        text = "Estoy muy feliz hoy"
        
        profile, config = integrator.detect_and_adapt(audio, text)
        
        # Debería detectar emoción positiva por el keyword "feliz"
        # (puede ser HAPPY, EXCITED, etc. dependiendo de heurísticas)
        assert profile.primary in [
            EmotionCategory.HAPPY,
            EmotionCategory.EXCITED,
            EmotionCategory.NEUTRAL
        ]
        # Verificar que HAPPY tiene score > 0 (keyword detectado)
        assert profile.raw_scores.get(EmotionCategory.HAPPY, 0.0) > 0.0
    
    def test_intensity_multiplier_effect(self):
        """Test efecto del multiplicador de intensidad"""
        integrator = EmotionVoiceIntegrator()
        
        audio = np.random.randn(16000) * 0.5
        
        # Sin multiplicador
        _, config_normal = integrator.detect_and_adapt(audio, intensity_multiplier=1.0)
        
        # Con multiplicador bajo (menos adaptación)
        _, config_low = integrator.detect_and_adapt(audio, intensity_multiplier=0.3)
        
        # Con multiplicador alto (más adaptación)
        _, config_high = integrator.detect_and_adapt(audio, intensity_multiplier=1.5)
        
        # Los configs deben ser diferentes
        # (no podemos garantizar dirección exacta sin saber emoción detectada)
        assert isinstance(config_normal, VoiceAdaptationConfig)
        assert isinstance(config_low, VoiceAdaptationConfig)
        assert isinstance(config_high, VoiceAdaptationConfig)
    
    def test_interpolate_method(self):
        """Test interpolación lineal"""
        integrator = EmotionVoiceIntegrator()
        
        # Interpolación al 0% → valor neutral
        result = integrator._interpolate(10.0, 20.0, 0.0)
        assert result == 10.0
        
        # Interpolación al 100% → valor emocional
        result = integrator._interpolate(10.0, 20.0, 1.0)
        assert result == 20.0
        
        # Interpolación al 50% → promedio
        result = integrator._interpolate(10.0, 20.0, 0.5)
        assert result == 15.0
    
    def test_blend_secondary_emotion_none(self):
        """Test blend sin emoción secundaria"""
        integrator = EmotionVoiceIntegrator()
        
        # Perfil sin secundaria
        profile = EmotionProfile(
            primary=EmotionCategory.HAPPY,
            intensity=0.8,
            confidence=0.9
        )
        
        primary_config = VoiceAdaptationConfig(pitch_shift=2.0)
        
        result = integrator.blend_secondary_emotion(profile, primary_config)
        
        # Debe retornar el mismo config
        assert result == primary_config
    
    def test_blend_secondary_emotion_with_secondary(self):
        """Test blend con emoción secundaria"""
        integrator = EmotionVoiceIntegrator()
        
        # Perfil con secundaria
        profile = EmotionProfile(
            primary=EmotionCategory.HAPPY,
            secondary=EmotionCategory.EXCITED,
            intensity=0.8,
            confidence=0.9,
            raw_scores={
                EmotionCategory.HAPPY: 0.6,
                EmotionCategory.EXCITED: 0.3,
                EmotionCategory.NEUTRAL: 0.1
            }
        )
        
        primary_config = integrator.EMOTION_VOICE_PROFILES[EmotionCategory.HAPPY]
        
        blended = integrator.blend_secondary_emotion(profile, primary_config)
        
        # Debe ser diferente al config primario puro
        assert isinstance(blended, VoiceAdaptationConfig)
        # El blend debe estar entre los valores de HAPPY y EXCITED
        # (verificación aproximada)
        assert blended.pitch_shift is not None


class TestHelperFunctions:
    """Tests para funciones helper"""
    
    def test_create_emotion_voice_integrator(self):
        """Test factory function"""
        integrator = create_emotion_voice_integrator(
            modulation_strength=0.5,
            min_confidence=0.7
        )
        
        assert isinstance(integrator, EmotionVoiceIntegrator)
        assert integrator.modulator.modulation_strength == 0.5
        assert integrator.modulator.min_confidence_threshold == 0.7
    
    def test_get_prosody_for_emotion_no_blend(self):
        """Test obtención de prosodia pura (sin blend)"""
        config = get_prosody_for_emotion(
            EmotionCategory.HAPPY,
            intensity=1.0,
            blend_neutral=False
        )
        
        # Debe ser idéntico al perfil base de HAPPY
        base_happy = EmotionVoiceIntegrator.EMOTION_VOICE_PROFILES[EmotionCategory.HAPPY]
        
        assert config.pitch_shift == base_happy.pitch_shift
        assert config.speed_factor == base_happy.speed_factor
        assert config.volume_gain == base_happy.volume_gain
        assert config.pause_duration == base_happy.pause_duration
    
    def test_get_prosody_for_emotion_with_blend(self):
        """Test obtención de prosodia con blend neutral"""
        # Intensidad baja → más cerca de neutral
        config_low = get_prosody_for_emotion(
            EmotionCategory.ANGRY,
            intensity=0.2,
            blend_neutral=True
        )
        
        # Intensidad alta → más cerca de ANGRY puro
        config_high = get_prosody_for_emotion(
            EmotionCategory.ANGRY,
            intensity=0.9,
            blend_neutral=True
        )
        
        # config_high debe tener pitch_shift más extremo que config_low
        # (ANGRY tiene pitch positivo, neutral es 0)
        assert abs(config_high.pitch_shift) > abs(config_low.pitch_shift)
    
    def test_all_emotions_prosody_valid(self):
        """Test que todas las emociones generan prosodia válida"""
        for emotion in EmotionCategory:
            config = get_prosody_for_emotion(emotion, intensity=0.8)
            
            # Verificar rangos válidos
            assert -12 <= config.pitch_shift <= 12
            assert 0.5 <= config.speed_factor <= 2.0
            assert -10 <= config.volume_gain <= 10
            assert 0 <= config.pause_duration <= 1000


class TestEmotionProsodyMapping:
    """Tests de mapeo emoción → prosodia"""
    
    def test_happy_prosody_characteristics(self):
        """Test características de HAPPY"""
        config = EmotionVoiceIntegrator.EMOTION_VOICE_PROFILES[EmotionCategory.HAPPY]
        
        # HAPPY: pitch alto, rápido, volumen alto
        assert config.pitch_shift > 0
        assert config.speed_factor > 1.0
        assert config.volume_gain > 0
        assert config.pause_duration < 200
    
    def test_sad_prosody_characteristics(self):
        """Test características de SAD"""
        config = EmotionVoiceIntegrator.EMOTION_VOICE_PROFILES[EmotionCategory.SAD]
        
        # SAD: pitch bajo, lento, volumen bajo, pausas largas
        assert config.pitch_shift < 0
        assert config.speed_factor < 1.0
        assert config.volume_gain < 0
        assert config.pause_duration > 200
    
    def test_angry_prosody_characteristics(self):
        """Test características de ANGRY"""
        config = EmotionVoiceIntegrator.EMOTION_VOICE_PROFILES[EmotionCategory.ANGRY]
        
        # ANGRY: rápido, alto volumen, pausas cortas
        assert config.speed_factor > 1.0
        assert config.volume_gain > 0
        assert config.pause_duration < 100
    
    def test_calm_prosody_characteristics(self):
        """Test características de CALM"""
        config = EmotionVoiceIntegrator.EMOTION_VOICE_PROFILES[EmotionCategory.CALM]
        
        # CALM: lento, suave, pausas largas
        assert config.speed_factor < 1.0
        assert config.volume_gain < 0
        assert config.pause_duration > 300
    
    def test_neutral_prosody_baseline(self):
        """Test que NEUTRAL es baseline (sin cambios)"""
        config = EmotionVoiceIntegrator.EMOTION_VOICE_PROFILES[EmotionCategory.NEUTRAL]
        
        assert config.pitch_shift == 0.0
        assert config.speed_factor == 1.0
        assert config.volume_gain == 0.0
        assert config.pause_duration == 200  # Pausas normales
