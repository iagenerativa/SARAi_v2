"""
agents/emotion_integration.py

Integración entre EmotionModulator y Omni Pipeline
Adapta la respuesta vocal según el estado emocional detectado

Flujo:
    Audio input → Emotion Detection → Embedding Modulation → TTS Prosody

Author: SARAi v2.11
Date: 2025-10-28
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from agents.emotion_modulator import (
    EmotionModulator,
    EmotionProfile,
    EmotionCategory,
    create_emotion_modulator
)


@dataclass
class VoiceAdaptationConfig:
    """
    Configuración para adaptar la voz según emoción
    
    Attributes:
        pitch_shift: Ajuste de pitch en semitonos [-12, +12]
        speed_factor: Factor de velocidad [0.5, 2.0]
        volume_gain: Ganancia de volumen en dB [-10, +10]
        pause_duration: Duración de pausas en ms [0, 1000]
    """
    pitch_shift: float = 0.0
    speed_factor: float = 1.0
    volume_gain: float = 0.0
    pause_duration: int = 0
    
    def to_dict(self) -> Dict:
        """Exporta como dict para TTS engine"""
        return {
            "pitch_shift": self.pitch_shift,
            "speed_factor": self.speed_factor,
            "volume_gain": self.volume_gain,
            "pause_duration": self.pause_duration
        }


class EmotionVoiceIntegrator:
    """
    Integrador de emoción → prosodia vocal
    
    Convierte perfiles emocionales en parámetros TTS
    """
    
    # Mapeo de emoción → configuración de voz
    EMOTION_VOICE_PROFILES = {
        EmotionCategory.HAPPY: VoiceAdaptationConfig(
            pitch_shift=+2.0,      # Voz más aguda
            speed_factor=1.1,      # Ligeramente más rápido
            volume_gain=+1.5,      # Más volumen
            pause_duration=100     # Pausas cortas
        ),
        EmotionCategory.SAD: VoiceAdaptationConfig(
            pitch_shift=-2.0,      # Voz más grave
            speed_factor=0.85,     # Más lento
            volume_gain=-2.0,      # Más suave
            pause_duration=300     # Pausas largas
        ),
        EmotionCategory.ANGRY: VoiceAdaptationConfig(
            pitch_shift=+1.0,      # Ligeramente agudo
            speed_factor=1.2,      # Rápido
            volume_gain=+3.0,      # Alto
            pause_duration=50      # Pausas muy cortas
        ),
        EmotionCategory.FEARFUL: VoiceAdaptationConfig(
            pitch_shift=+3.0,      # Muy agudo
            speed_factor=1.15,     # Rápido (nervioso)
            volume_gain=-1.0,      # Suave
            pause_duration=150     # Pausas nerviosas
        ),
        EmotionCategory.SURPRISED: VoiceAdaptationConfig(
            pitch_shift=+4.0,      # Muy agudo (exclamación)
            speed_factor=1.3,      # Muy rápido
            volume_gain=+2.0,      # Alto
            pause_duration=200     # Pausa post-sorpresa
        ),
        EmotionCategory.DISGUSTED: VoiceAdaptationConfig(
            pitch_shift=-1.0,      # Ligeramente grave
            speed_factor=0.9,      # Lento (rechazo)
            volume_gain=0.0,       # Normal
            pause_duration=250     # Pausas de aversión
        ),
        EmotionCategory.CALM: VoiceAdaptationConfig(
            pitch_shift=-0.5,      # Muy ligeramente grave
            speed_factor=0.95,     # Ligeramente lento
            volume_gain=-1.5,      # Suave
            pause_duration=400     # Pausas relajadas
        ),
        EmotionCategory.EXCITED: VoiceAdaptationConfig(
            pitch_shift=+2.5,      # Agudo
            speed_factor=1.25,     # Rápido
            volume_gain=+2.5,      # Alto
            pause_duration=80      # Pausas cortas
        ),
        EmotionCategory.NEUTRAL: VoiceAdaptationConfig(
            pitch_shift=0.0,       # Sin cambio
            speed_factor=1.0,      # Normal
            volume_gain=0.0,       # Normal
            pause_duration=200     # Pausas normales
        ),
    }
    
    def __init__(self, modulator: Optional[EmotionModulator] = None):
        """
        Inicializa integrador
        
        Args:
            modulator: EmotionModulator (si None, crea uno por defecto)
        """
        self.modulator = modulator or create_emotion_modulator()
    
    def detect_and_adapt(
        self,
        audio_features: np.ndarray,
        text: Optional[str] = None,
        intensity_multiplier: float = 1.0
    ) -> Tuple[EmotionProfile, VoiceAdaptationConfig]:
        """
        Detecta emoción y genera configuración de voz adaptada
        
        Args:
            audio_features: Features del audio input (waveform o mel-spec)
            text: Transcripción (opcional, para análisis multi-modal)
            intensity_multiplier: Multiplicador de intensidad [0.0, 2.0]
        
        Returns:
            (EmotionProfile, VoiceAdaptationConfig)
        
        Example:
            >>> integrator = EmotionVoiceIntegrator()
            >>> profile, config = integrator.detect_and_adapt(audio, "Estoy feliz")
            >>> # Usar config para TTS
            >>> tts_engine.set_prosody(**config.to_dict())
        """
        # Detectar emoción
        profile = self.modulator.detect_emotion(audio_features, text)
        
        # Obtener configuración base
        base_config = self.EMOTION_VOICE_PROFILES[profile.primary]
        
        # Ajustar según intensity y confidence
        adaptation_strength = profile.intensity * profile.confidence * intensity_multiplier
        
        # Interpolar entre neutral y emoción específica
        neutral_config = self.EMOTION_VOICE_PROFILES[EmotionCategory.NEUTRAL]
        
        adapted_config = VoiceAdaptationConfig(
            pitch_shift=self._interpolate(
                neutral_config.pitch_shift,
                base_config.pitch_shift,
                adaptation_strength
            ),
            speed_factor=self._interpolate(
                neutral_config.speed_factor,
                base_config.speed_factor,
                adaptation_strength
            ),
            volume_gain=self._interpolate(
                neutral_config.volume_gain,
                base_config.volume_gain,
                adaptation_strength
            ),
            pause_duration=int(self._interpolate(
                neutral_config.pause_duration,
                base_config.pause_duration,
                adaptation_strength
            ))
        )
        
        return profile, adapted_config
    
    def _interpolate(self, neutral_val: float, emotion_val: float, strength: float) -> float:
        """
        Interpolación lineal entre valor neutral y emocional
        
        Args:
            neutral_val: Valor para emoción neutral
            emotion_val: Valor para emoción específica
            strength: Fuerza de interpolación [0, 1]
        
        Returns:
            Valor interpolado
        """
        return neutral_val + (emotion_val - neutral_val) * strength
    
    def blend_secondary_emotion(
        self,
        profile: EmotionProfile,
        primary_config: VoiceAdaptationConfig
    ) -> VoiceAdaptationConfig:
        """
        Mezcla emoción secundaria si existe
        
        Args:
            profile: Perfil con emoción primaria + secundaria
            primary_config: Config de la emoción primaria
        
        Returns:
            Config mezclado si hay secundaria, sino original
        """
        if not profile.secondary:
            return primary_config
        
        secondary_config = self.EMOTION_VOICE_PROFILES[profile.secondary]
        
        # Peso de la secundaria (basado en scores)
        primary_score = profile.raw_scores.get(profile.primary, 0.0)
        secondary_score = profile.raw_scores.get(profile.secondary, 0.0)
        
        # Normalizar a [0, 1]
        total = primary_score + secondary_score
        if total == 0:
            return primary_config
        
        primary_weight = primary_score / total
        secondary_weight = secondary_score / total
        
        # Mezclar configs
        blended_config = VoiceAdaptationConfig(
            pitch_shift=(
                primary_config.pitch_shift * primary_weight +
                secondary_config.pitch_shift * secondary_weight
            ),
            speed_factor=(
                primary_config.speed_factor * primary_weight +
                secondary_config.speed_factor * secondary_weight
            ),
            volume_gain=(
                primary_config.volume_gain * primary_weight +
                secondary_config.volume_gain * secondary_weight
            ),
            pause_duration=int(
                primary_config.pause_duration * primary_weight +
                secondary_config.pause_duration * secondary_weight
            )
        )
        
        return blended_config


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_emotion_voice_integrator(
    modulation_strength: float = 0.3,
    min_confidence: float = 0.5
) -> EmotionVoiceIntegrator:
    """
    Factory para crear integrador con config custom
    
    Args:
        modulation_strength: Fuerza de modulación del modulator
        min_confidence: Confianza mínima para modular
    
    Returns:
        EmotionVoiceIntegrator configurado
    """
    modulator = create_emotion_modulator(modulation_strength, min_confidence)
    return EmotionVoiceIntegrator(modulator)


def get_prosody_for_emotion(
    emotion: EmotionCategory,
    intensity: float = 0.8,
    blend_neutral: bool = True
) -> VoiceAdaptationConfig:
    """
    Obtiene configuración de prosodia para una emoción específica
    
    Args:
        emotion: Categoría emocional
        intensity: Intensidad [0, 1]
        blend_neutral: Si True, mezcla con neutral según intensity
    
    Returns:
        VoiceAdaptationConfig para TTS
    
    Example:
        >>> config = get_prosody_for_emotion(EmotionCategory.HAPPY, intensity=0.9)
        >>> print(f"Pitch: {config.pitch_shift}, Speed: {config.speed_factor}")
    """
    base_config = EmotionVoiceIntegrator.EMOTION_VOICE_PROFILES[emotion]
    
    if not blend_neutral:
        return base_config
    
    # Mezclar con neutral según intensity
    neutral_config = EmotionVoiceIntegrator.EMOTION_VOICE_PROFILES[EmotionCategory.NEUTRAL]
    
    integrator = EmotionVoiceIntegrator()
    
    return VoiceAdaptationConfig(
        pitch_shift=integrator._interpolate(
            neutral_config.pitch_shift,
            base_config.pitch_shift,
            intensity
        ),
        speed_factor=integrator._interpolate(
            neutral_config.speed_factor,
            base_config.speed_factor,
            intensity
        ),
        volume_gain=integrator._interpolate(
            neutral_config.volume_gain,
            base_config.volume_gain,
            intensity
        ),
        pause_duration=int(integrator._interpolate(
            neutral_config.pause_duration,
            base_config.pause_duration,
            intensity
        ))
    )
