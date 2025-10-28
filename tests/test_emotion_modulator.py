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
        
        # Test 1: Audio con energía alta → EXCITED/ANGRY (depende de LibROSA)
        high_energy_audio = np.random.randn(16000) * 0.8
        profile = modulator.detect_emotion(high_energy_audio)
        
        # Con LibROSA puede detectar ANGRY o EXCITED (ambos alta energía)
        assert profile.primary in [EmotionCategory.EXCITED, EmotionCategory.ANGRY]
        assert profile.confidence >= 0.5
        assert 0.0 <= profile.intensity <= 1.0
        
        # Test 2: Audio con energía baja → SAD/CALM/NEUTRAL (depende de LibROSA)
        low_energy_audio = np.random.randn(16000) * 0.1
        profile = modulator.detect_emotion(low_energy_audio)
        
        # LibROSA puede detectar varias emociones de baja energía
        assert profile.primary in [EmotionCategory.SAD, EmotionCategory.CALM, EmotionCategory.NEUTRAL]
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
    
    def test_blend_emotion_vectors(self):
        """Test combinación de vectores emocionales"""
        from agents.emotion_modulator import blend_emotion_vectors
        
        # Vectores sintéticos normalizados
        vec_happy = np.random.randn(768).astype(np.float32)
        vec_happy = vec_happy / np.linalg.norm(vec_happy)
        
        vec_excited = np.random.randn(768).astype(np.float32)
        vec_excited = vec_excited / np.linalg.norm(vec_excited)
        
        vectors = {
            EmotionCategory.HAPPY: vec_happy,
            EmotionCategory.EXCITED: vec_excited
        }
        
        weights = {
            EmotionCategory.HAPPY: 0.7,
            EmotionCategory.EXCITED: 0.3
        }
        
        blended = blend_emotion_vectors(vectors, weights)
        
        # Verificar dimensiones
        assert blended.shape == (768,)
        
        # Verificar normalización
        assert np.isclose(np.linalg.norm(blended), 1.0, atol=1e-5)
    
    def test_blend_emotion_vectors_missing_vector(self):
        """Test blend con emoción sin vector (debe fallar)"""
        from agents.emotion_modulator import blend_emotion_vectors
        
        vectors = {EmotionCategory.HAPPY: np.random.randn(768)}
        weights = {
            EmotionCategory.HAPPY: 0.5,
            EmotionCategory.SAD: 0.5  # ❌ No tiene vector
        }
        
        with pytest.raises(ValueError, match="no tiene vector"):
            blend_emotion_vectors(vectors, weights)
    
    def test_analyze_emotion_trajectory_basic(self):
        """Test análisis de trayectoria emocional"""
        from agents.emotion_modulator import analyze_emotion_trajectory
        
        # Crear secuencia de perfiles
        profiles = [
            EmotionProfile(EmotionCategory.NEUTRAL, intensity=0.3, confidence=0.7),
            EmotionProfile(EmotionCategory.HAPPY, intensity=0.6, confidence=0.8),
            EmotionProfile(EmotionCategory.HAPPY, intensity=0.7, confidence=0.9),
            EmotionProfile(EmotionCategory.EXCITED, intensity=0.8, confidence=0.85),
            EmotionProfile(EmotionCategory.EXCITED, intensity=0.9, confidence=0.9),
        ]
        
        trajectory = analyze_emotion_trajectory(profiles)
        
        # Verificar estructura
        assert "dominant_emotion" in trajectory
        assert "avg_intensity" in trajectory
        assert "volatility" in trajectory
        assert "trend" in trajectory
        
        # Verificar valores lógicos
        assert trajectory["dominant_emotion"] in [EmotionCategory.HAPPY, EmotionCategory.EXCITED]
        assert 0.0 <= trajectory["avg_intensity"] <= 1.0
        assert trajectory["trend"] in ["escalating", "de-escalating", "stable"]
    
    def test_analyze_emotion_trajectory_escalating(self):
        """Test detección de tendencia escalating"""
        from agents.emotion_modulator import analyze_emotion_trajectory
        
        # Secuencia con intensidad creciente
        profiles = [
            EmotionProfile(EmotionCategory.CALM, intensity=0.2, confidence=0.7),
            EmotionProfile(EmotionCategory.CALM, intensity=0.3, confidence=0.7),
            EmotionProfile(EmotionCategory.CALM, intensity=0.4, confidence=0.7),
            EmotionProfile(EmotionCategory.CALM, intensity=0.5, confidence=0.7),
            EmotionProfile(EmotionCategory.CALM, intensity=0.6, confidence=0.7),
            EmotionProfile(EmotionCategory.ANGRY, intensity=0.7, confidence=0.8),
            EmotionProfile(EmotionCategory.ANGRY, intensity=0.8, confidence=0.9),
            EmotionProfile(EmotionCategory.ANGRY, intensity=0.9, confidence=0.9),
            EmotionProfile(EmotionCategory.ANGRY, intensity=0.95, confidence=0.95),
            EmotionProfile(EmotionCategory.ANGRY, intensity=1.0, confidence=1.0),
        ]
        
        trajectory = analyze_emotion_trajectory(profiles, window_size=5)
        
        assert trajectory["trend"] == "escalating"
        assert trajectory["avg_intensity"] > 0.5
    
    def test_analyze_emotion_trajectory_empty(self):
        """Test trayectoria con lista vacía"""
        from agents.emotion_modulator import analyze_emotion_trajectory
        
        trajectory = analyze_emotion_trajectory([])
        
        assert trajectory["dominant_emotion"] == EmotionCategory.NEUTRAL
        assert trajectory["avg_intensity"] == 0.0
        assert trajectory["volatility"] == 0.0
        assert trajectory["trend"] == "stable"


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
    
    @pytest.mark.skip(reason="Test interactivo - ejecutar manualmente con -k real_microphone_emotion")
    def test_emotion_detection_with_real_microphone(self):
        """
        Test INTERACTIVO de detección emocional con micrófono
        
        Uso:
            pytest tests/test_emotion_modulator.py::TestEmotionModulationIntegration::test_emotion_detection_with_real_microphone -s
        
        O activar todos los tests de micrófono:
            pytest tests/ -k real_microphone -s
        """
        try:
            import pyaudio
            import wave
            import tempfile
            from scipy.io import wavfile
        except ImportError as e:
            pytest.skip(f"Dependencia faltante: {e}. Instalar con: pip install pyaudio scipy")
        
        print("\n" + "="*70)
        print("🎭 TEST INTERACTIVO: Detección de emoción con micrófono")
        print("="*70)
        
        # Configuración
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 3
        
        print(f"\n📋 Instrucciones:")
        print(f"   1. Grabarás {RECORD_SECONDS} segundos de audio")
        print(f"   2. Expresa una emoción clara (alegría, tristeza, enojo, etc.)")
        print(f"   3. El sistema detectará tu estado emocional")
        
        input("\n▶️  Presiona ENTER cuando estés listo...")
        
        # Grabar
        p = pyaudio.PyAudio()
        
        print("\n🔴 GRABANDO... (expresa una emoción)")
        print("   💡 Ejemplos:")
        print("      - Feliz: 'Estoy muy contento hoy!'")
        print("      - Triste: 'Me siento mal...'")
        print("      - Enojado: '¡Esto es frustrante!'")
        print("      - Tranquilo: 'Todo está bien'")
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            if i % 8 == 0:
                print("█", end="", flush=True)
        
        print(" ✅ Grabación completada\n")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Guardar y procesar
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wf = wave.open(temp_wav.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Leer como array numpy
            sample_rate, audio_data = wavfile.read(temp_wav.name)
            
            # Normalizar a [-1, 1]
            audio_features = audio_data.astype(np.float32) / 32768.0
        
        # DETECCIÓN EMOCIONAL
        print("🔍 Analizando emoción del audio...")
        
        modulator = EmotionModulator()
        profile = modulator.detect_emotion(audio_features)
        
        # Mostrar resultados
        print("\n" + "="*70)
        print("📊 RESULTADOS DE ANÁLISIS EMOCIONAL")
        print("="*70)
        
        # Emoji por emoción
        emotion_emojis = {
            EmotionCategory.HAPPY: "😊",
            EmotionCategory.SAD: "😢",
            EmotionCategory.ANGRY: "😠",
            EmotionCategory.FEARFUL: "😨",
            EmotionCategory.SURPRISED: "😮",
            EmotionCategory.DISGUSTED: "🤢",
            EmotionCategory.CALM: "😌",
            EmotionCategory.EXCITED: "🤩",
            EmotionCategory.NEUTRAL: "😐"
        }
        
        emoji = emotion_emojis.get(profile.primary, "❓")
        
        print(f"\n🎭 Emoción Primaria: {emoji} {profile.primary.value.upper()}")
        print(f"   Intensidad: {profile.intensity:.2f} / 1.00")
        print(f"   Confianza: {profile.confidence:.2f} / 1.00")
        
        if profile.secondary:
            emoji_sec = emotion_emojis.get(profile.secondary, "❓")
            print(f"\n🎭 Emoción Secundaria: {emoji_sec} {profile.secondary.value}")
        
        # Top 3 scores
        if profile.raw_scores:
            print("\n📈 Top 3 Scores:")
            sorted_scores = sorted(profile.raw_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (emotion, score) in enumerate(sorted_scores, 1):
                bar = "█" * int(score * 30)
                print(f"   {i}. {emotion.value:12} {bar} {score:.3f}")
        
        # Características del audio
        print("\n🔊 Características Acústicas:")
        mean_energy = np.mean(np.abs(audio_features))
        max_energy = np.max(np.abs(audio_features))
        std_energy = np.std(audio_features)
        zcr = np.mean(np.abs(np.diff(np.sign(audio_features)))) / 2.0
        
        print(f"   Energía promedio: {mean_energy:.4f}")
        print(f"   Energía máxima: {max_energy:.4f}")
        print(f"   Desviación std: {std_energy:.4f}")
        print(f"   Zero-crossing rate: {zcr:.4f}")
        
        # Verificación manual
        print("\n" + "="*70)
        print("❓ ¿La detección es correcta? (y/n): ", end="")
        user_confirm = input().strip().lower()
        
        if user_confirm == 'y':
            print("\n✅ Test PASSED - Detección emocional correcta")
            print(f"   Emoción detectada: {profile.primary.value}")
            print(f"   Confianza: {profile.confidence:.1%}")
            assert profile.confidence > 0.3, "Confianza muy baja"
        else:
            print("\n📝 ¿Cuál era la emoción correcta? (happy/sad/angry/calm/etc.): ", end="")
            correct_emotion = input().strip().lower()
            
            print(f"\n❌ Test FAILED - Detección incorrecta")
            print(f"   Detectado: {profile.primary.value}")
            print(f"   Correcto: {correct_emotion}")
            print(f"\n💡 Esto es esperado en Fase 1 (heurísticas básicas)")
            print(f"   Fase 2 integrará modelo real (emoDBert) para mejorar precisión")
            
            pytest.fail(f"Usuario indicó emoción correcta: '{correct_emotion}', detectado: '{profile.primary.value}'")
