"""
tests/test_emotion_cache.py

Test suite para emotion_cache.py

Author: SARAi Team
Date: 2025-10-28
"""

import pytest
import numpy as np
import time
from agents.emotion_cache import (
    EmotionCache,
    CachedEmotion,
    create_emotion_cache
)
from agents.emotion_modulator import EmotionProfile, EmotionCategory


class TestCachedEmotion:
    """Tests para CachedEmotion dataclass"""
    
    def test_create_cached_emotion(self):
        """Test creación de entry"""
        profile = EmotionProfile(
            primary=EmotionCategory.HAPPY,
            intensity=0.8,
            confidence=0.9
        )
        
        entry = CachedEmotion(
            profile=profile,
            timestamp=time.time(),
            audio_hash="abc123"
        )
        
        assert entry.profile == profile
        assert entry.audio_hash == "abc123"
        assert entry.hit_count == 0
    
    def test_is_expired(self):
        """Test verificación de expiración"""
        profile = EmotionProfile(EmotionCategory.NEUTRAL, 0.5, 0.5)
        
        # Entry reciente (no expirado)
        entry = CachedEmotion(profile, time.time(), "hash1")
        assert not entry.is_expired(ttl=10.0)
        
        # Entry viejo (expirado)
        entry_old = CachedEmotion(profile, time.time() - 20, "hash2")
        assert entry_old.is_expired(ttl=10.0)


class TestEmotionCache:
    """Tests para EmotionCache"""
    
    def test_initialization(self):
        """Test inicialización del caché"""
        cache = EmotionCache(max_size=50, base_ttl=30.0)
        
        assert cache.max_size == 50
        assert cache.base_ttl == 30.0
        assert len(cache.cache) == 0
        assert cache.stats["total_requests"] == 0
    
    def test_compute_audio_hash(self):
        """Test cálculo de hash perceptual"""
        cache = EmotionCache()
        
        audio = np.random.randn(16000)
        hash1 = cache._compute_audio_hash(audio)
        
        # Mismo audio → mismo hash
        hash2 = cache._compute_audio_hash(audio)
        assert hash1 == hash2
        
        # Audio diferente → hash diferente
        audio_diff = np.random.randn(16000) * 2.0
        hash3 = cache._compute_audio_hash(audio_diff)
        assert hash1 != hash3
        
        # Hash debe ser string de 16 chars
        assert isinstance(hash1, str)
        assert len(hash1) == 16
    
    def test_put_and_get_hit(self):
        """Test put + get con HIT"""
        cache = EmotionCache()
        
        audio = np.random.randn(16000)
        profile = EmotionProfile(EmotionCategory.HAPPY, 0.8, 0.9)
        
        # Put
        cache.put(audio, profile)
        
        # Get (HIT)
        retrieved = cache.get(audio)
        
        assert retrieved is not None
        assert retrieved.primary == EmotionCategory.HAPPY
        assert cache.stats["cache_hits"] == 1
        assert cache.stats["cache_misses"] == 0
    
    def test_get_miss(self):
        """Test get con MISS"""
        cache = EmotionCache()
        
        audio = np.random.randn(16000)
        
        # Get sin put previo (MISS)
        retrieved = cache.get(audio)
        
        assert retrieved is None
        assert cache.stats["cache_hits"] == 0
        assert cache.stats["cache_misses"] == 1
    
    def test_ttl_expiration(self):
        """Test expiración por TTL"""
        cache = EmotionCache(base_ttl=0.5)  # TTL muy corto
        
        audio = np.random.randn(16000)
        profile = EmotionProfile(EmotionCategory.SAD, 0.5, 0.6)
        
        # Put
        cache.put(audio, profile)
        
        # Get inmediato (HIT)
        assert cache.get(audio) is not None
        
        # Esperar expiración
        time.sleep(0.6)
        
        # Get post-expiración (MISS por expiración)
        assert cache.get(audio) is None
        assert cache.stats["expirations"] == 1
    
    def test_lru_eviction(self):
        """Test eviction LRU cuando caché está lleno"""
        cache = EmotionCache(max_size=3)
        
        audio1 = np.random.randn(16000)
        audio2 = np.random.randn(16000) * 1.5
        audio3 = np.random.randn(16000) * 2.0
        audio4 = np.random.randn(16000) * 2.5
        
        profile = EmotionProfile(EmotionCategory.NEUTRAL, 0.5, 0.5)
        
        # Llenar caché (3 entries)
        cache.put(audio1, profile)
        cache.put(audio2, profile)
        cache.put(audio3, profile)
        
        assert len(cache.cache) == 3
        
        # Añadir 4to → evict el más viejo (audio1)
        cache.put(audio4, profile)
        
        assert len(cache.cache) == 3
        assert cache.stats["evictions"] == 1
        
        # audio1 debe haber sido evictado (MISS)
        assert cache.get(audio1) is None
        
        # Otros deben estar (HIT)
        assert cache.get(audio2) is not None
        assert cache.get(audio3) is not None
        assert cache.get(audio4) is not None
    
    def test_hit_rate_calculation(self):
        """Test cálculo de hit rate"""
        cache = EmotionCache()
        
        audio = np.random.randn(16000)
        profile = EmotionProfile(EmotionCategory.EXCITED, 0.9, 0.95)
        
        cache.put(audio, profile)
        
        # 2 HITs + 1 MISS → hit_rate = 2/3 = 0.666...
        cache.get(audio)  # HIT
        cache.get(audio)  # HIT
        cache.get(np.random.randn(16000))  # MISS
        
        assert cache.stats["cache_hits"] == 2
        assert cache.stats["cache_misses"] == 1
        assert 0.65 < cache.stats["hit_rate"] < 0.70
    
    def test_dynamic_ttl_stable_emotions(self):
        """Test TTL dinámico con emociones estables"""
        cache = EmotionCache(base_ttl=30.0)
        
        # Historial estable (misma emoción)
        for _ in range(10):  # Llenar historial completo
            cache.emotion_history.append(EmotionCategory.CALM)
        
        ttl = cache._compute_ttl()
        
        # Baja volatilidad (0%) → TTL largo (mayor que base)
        # TTL = base * (1 - 0 * 0.5) = 30 * 1.0 = 30
        assert ttl >= 30.0
    
    def test_dynamic_ttl_volatile_emotions(self):
        """Test TTL dinámico con emociones volátiles"""
        cache = EmotionCache(base_ttl=30.0)
        
        # Historial volátil (muchos cambios)
        emotions = [
            EmotionCategory.HAPPY,
            EmotionCategory.SAD,
            EmotionCategory.ANGRY,
            EmotionCategory.CALM,
            EmotionCategory.EXCITED
        ]
        for emotion in emotions:
            cache.emotion_history.append(emotion)
        
        ttl = cache._compute_ttl()
        
        # Alta volatilidad → TTL corto (cerca de 10s)
        assert ttl < 20.0
    
    def test_invalidate_on_opposite_emotion(self):
        """Test invalidación por cambio brusco"""
        cache = EmotionCache()
        
        # Cachear perfil HAPPY
        audio = np.random.randn(16000)
        profile_happy = EmotionProfile(EmotionCategory.HAPPY, 0.8, 0.9)
        cache.put(audio, profile_happy)
        
        assert len(cache.cache) == 1
        
        # Cambio brusco a SAD (opuesto de HAPPY)
        cache.invalidate_if_changed(EmotionCategory.SAD)
        
        # Caché debe estar vacío
        assert len(cache.cache) == 0
        assert len(cache.emotion_history) == 0
    
    def test_get_stats(self):
        """Test obtención de estadísticas"""
        cache = EmotionCache(max_size=100)
        
        stats = cache.get_stats()
        
        assert "total_requests" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats
        assert "cache_size" in stats
        assert stats["max_size"] == 100
    
    def test_clear(self):
        """Test limpieza total del caché"""
        cache = EmotionCache()
        
        # Llenar caché
        for i in range(5):
            audio = np.random.randn(16000) * (i + 1)
            profile = EmotionProfile(EmotionCategory.NEUTRAL, 0.5, 0.5)
            cache.put(audio, profile)
        
        assert len(cache.cache) > 0
        
        # Clear
        cache.clear()
        
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
        assert len(cache.emotion_history) == 0


class TestHelperFunctions:
    """Tests para funciones helper"""
    
    def test_create_emotion_cache(self):
        """Test factory function"""
        cache = create_emotion_cache(max_size=100, base_ttl=45.0)
        
        assert isinstance(cache, EmotionCache)
        assert cache.max_size == 100
        assert cache.base_ttl == 45.0
