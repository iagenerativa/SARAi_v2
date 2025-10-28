"""
Tests for TTS Engine - M3.2 Fase 3
===================================

Test Coverage:
- Unit tests: TTSCache, TTSEngine methods
- Integration tests: Full pipeline with emotion
- Benchmark tests: Latency, cache hit rate

Author: SARAi Team
Date: Oct 28, 2025
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from agents.tts_engine import (
    TTSEngine,
    TTSConfig,
    TTSCache,
    TTSOutput,
    create_tts_engine
)

# Check if pyttsx3 is available
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def temp_cache_dir():
    """Temporary cache directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def tts_config(temp_cache_dir):
    """Test TTS configuration"""
    return TTSConfig(
        enable_omni=False,  # Disable for unit tests
        enable_pyttsx3=True,
        enable_cache=True,
        cache_dir=temp_cache_dir,
        cache_max_size_mb=1,  # Small for testing
        cache_ttl_seconds=60
    )


@pytest.fixture
def tts_engine(tts_config):
    """TTS Engine instance"""
    return TTSEngine(tts_config)


@pytest.fixture
def sample_audio_bytes():
    """Sample WAV audio bytes (16kHz, mono, 1 second)"""
    import wave
    import io
    
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    
    # Generate sine wave
    audio_array = (np.sin(2 * np.pi * 440 * np.arange(samples) / sample_rate) * 32767).astype(np.int16)
    
    # Convert to WAV bytes
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()


@pytest.fixture
def mock_emotion_state():
    """Mock EmotionState"""
    emotion = Mock()
    emotion.label = "happy"
    emotion.valence = 0.8  # Positive
    emotion.arousal = 0.7  # High energy
    emotion.dominance = 0.6  # Confident
    return emotion


# ============================================================
# UNIT TESTS - TTSCache
# ============================================================

class TestTTSCache:
    """Unit tests for TTSCache"""
    
    def test_cache_initialization(self, tts_config):
        """Test cache initialization"""
        cache = TTSCache(tts_config)
        
        assert cache.cache_dir.exists()
        assert len(cache.index) == 0
    
    def test_cache_compute_key(self, tts_config):
        """Test cache key computation"""
        cache = TTSCache(tts_config)
        
        # Same text → same key
        key1 = cache._compute_key("Hello world")
        key2 = cache._compute_key("Hello world")
        assert key1 == key2
        
        # Different text → different key
        key3 = cache._compute_key("Goodbye world")
        assert key1 != key3
        
        # Same text + prosody → consistent key
        prosody = {"pitch_shift_hz": 50}
        key4 = cache._compute_key("Hello world", prosody)
        key5 = cache._compute_key("Hello world", prosody)
        assert key4 == key5
        
        # Different prosody → different key
        prosody2 = {"pitch_shift_hz": 100}
        key6 = cache._compute_key("Hello world", prosody2)
        assert key4 != key6
    
    def test_cache_put_get(self, tts_config, sample_audio_bytes):
        """Test cache put and get operations"""
        cache = TTSCache(tts_config)
        
        text = "Hello world"
        
        # Initially empty
        assert cache.get(text) is None
        
        # Put audio
        cache.put(text, sample_audio_bytes)
        
        # Get should return audio
        retrieved = cache.get(text)
        assert retrieved == sample_audio_bytes
    
    def test_cache_ttl_expiration(self, temp_cache_dir):
        """Test cache TTL expiration"""
        config = TTSConfig(
            cache_dir=temp_cache_dir,
            cache_ttl_seconds=1  # 1 second TTL
        )
        cache = TTSCache(config)
        
        text = "Expire me"
        audio = b"fake_audio_data"
        
        # Put audio
        cache.put(text, audio)
        assert cache.get(text) == audio
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be None (expired)
        assert cache.get(text) is None
    
    def test_cache_lru_eviction(self, temp_cache_dir):
        """Test LRU eviction when cache full"""
        config = TTSConfig(
            cache_dir=temp_cache_dir,
            cache_max_size_mb=0.001  # ~1KB max (very small)
        )
        cache = TTSCache(config)
        
        # Add multiple entries until eviction
        audio = b"x" * 500  # 500 bytes per entry
        
        cache.put("text1", audio)
        cache.put("text2", audio)
        time.sleep(0.1)  # Ensure different timestamps
        cache.put("text3", audio)  # Should trigger eviction
        
        # text1 should be evicted (LRU)
        assert cache.get("text1") is None
        assert cache.get("text2") is not None  # Still there
        assert cache.get("text3") is not None  # Newest
    
    def test_cache_stats(self, tts_config, sample_audio_bytes):
        """Test cache statistics"""
        cache = TTSCache(tts_config)
        
        # Initially empty
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["total_size_mb"] == 0
        
        # Add entry
        cache.put("test", sample_audio_bytes)
        
        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["total_size_mb"] > 0
        assert 0 <= stats["utilization_pct"] <= 100


# ============================================================
# UNIT TESTS - TTSEngine
# ============================================================

class TestTTSEngine:
    """Unit tests for TTSEngine"""
    
    def test_engine_initialization(self, tts_config):
        """Test engine initialization"""
        engine = TTSEngine(tts_config)
        
        assert engine.config == tts_config
        assert engine.cache is not None
        assert engine._omni_model is None  # Lazy loaded
    
    def test_generate_text_only(self):
        """Test text-only fallback (no TTS)"""
        config = TTSConfig(
            enable_omni=False,
            enable_pyttsx3=False  # All disabled
        )
        engine = TTSEngine(config)
        
        result = engine.generate("Hello world")
        
        assert result.text == "Hello world"
        assert result.audio_bytes is None
        assert result.method_used == "text_only"
        assert result.latency_ms >= 0
        assert not result.cached
    
    @pytest.mark.skipif(not PYTTSX3_AVAILABLE, reason="pyttsx3 not installed")
    @patch('pyttsx3.init')
    def test_generate_pyttsx3(self, mock_init, tts_config, sample_audio_bytes, temp_cache_dir):
        """Test pyttsx3 TTS generation"""
        # Mock pyttsx3 engine
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        # Create temp WAV file for mock
        temp_wav = Path(temp_cache_dir) / "test_output.wav"
        temp_wav.write_bytes(sample_audio_bytes)
        
        # Mock save_to_file to write our sample audio
        def mock_save(text, path):
            Path(path).write_bytes(sample_audio_bytes)
        
        mock_engine.save_to_file.side_effect = mock_save
        mock_engine.runAndWait.return_value = None
        mock_engine.getProperty.return_value = 200  # Default rate/volume
        
        engine = TTSEngine(tts_config)
        result = engine.generate("Hello world")
        
        assert result.audio_bytes is not None
        assert result.method_used == "pyttsx3"
        assert result.latency_ms >= 0
        assert not result.cached
    
    def test_extract_prosody(self, tts_engine):
        """Test prosody extraction from EmotionState"""
        # Create mock emotion manually (avoid import issues)
        mock_emotion = Mock()
        mock_emotion.label = "happy"
        mock_emotion.valence = 0.8
        mock_emotion.arousal = 0.7
        mock_emotion.dominance = 0.6
        
        prosody = tts_engine._extract_prosody(mock_emotion)
        
        assert "pitch_shift_hz" in prosody
        assert "rate_multiplier" in prosody
        assert "volume_multiplier" in prosody
        assert "emotion_label" in prosody
        
        # Happy emotion should have positive pitch shift
        assert prosody["pitch_shift_hz"] > 0
        
        # High arousal → faster rate
        assert prosody["rate_multiplier"] > 1.0
    
    def test_cache_integration(self, tts_engine, sample_audio_bytes):
        """Test cache integration in generate()"""
        text = "Cached text"
        
        # Manually populate cache
        tts_engine.cache.put(text, sample_audio_bytes)
        
        # Generate should hit cache
        result = tts_engine.generate(text)
        
        assert result.cached is True
        assert result.method_used == "cache"
        assert result.audio_bytes == sample_audio_bytes
        assert result.latency_ms < 50  # Cache should be fast


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestTTSIntegration:
    """Integration tests with full pipeline"""
    
    @pytest.mark.skipif(not PYTTSX3_AVAILABLE, reason="pyttsx3 not installed")
    @patch('pyttsx3.init')
    def test_full_pipeline_with_emotion(self, mock_init, temp_cache_dir, 
                                       sample_audio_bytes, mock_emotion_state):
        """Test full TTS pipeline with emotion prosody"""
        # Mock pyttsx3
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        def mock_save(text, path):
            Path(path).write_bytes(sample_audio_bytes)
        
        mock_engine.save_to_file.side_effect = mock_save
        mock_engine.runAndWait.return_value = None
        mock_engine.getProperty.return_value = 200
        
        # Create engine
        config = TTSConfig(
            enable_omni=False,
            enable_pyttsx3=True,
            cache_dir=temp_cache_dir
        )
        engine = TTSEngine(config)
        
        # Generate with emotion
        result = engine.generate("Hello!", emotion_state=mock_emotion_state)
        
        assert result.audio_bytes is not None
        assert result.prosody_applied is True
        assert result.method_used == "pyttsx3"
        
        # Verify prosody was applied to engine
        assert mock_engine.setProperty.called
    
    def test_cache_hit_scenario(self, temp_cache_dir, sample_audio_bytes):
        """Test cache hit scenario (second call)"""
        config = TTSConfig(
            enable_omni=False,
            enable_pyttsx3=False,
            enable_cache=True,
            cache_dir=temp_cache_dir
        )
        
        # Manually populate cache
        cache = TTSCache(config)
        cache.put("Test text", sample_audio_bytes)
        
        engine = TTSEngine(config)
        
        # First call should hit cache
        result1 = engine.generate("Test text")
        assert result1.cached is True
        assert result1.audio_bytes == sample_audio_bytes
        
        # Second call should also hit cache
        result2 = engine.generate("Test text")
        assert result2.cached is True
        
        # Both should be fast
        assert result1.latency_ms < 50
        assert result2.latency_ms < 50


# ============================================================
# BENCHMARK TESTS
# ============================================================

class TestTTSBenchmarks:
    """Benchmark tests for performance validation"""
    
    @pytest.mark.benchmark
    def test_cache_latency_benchmark(self, temp_cache_dir, sample_audio_bytes):
        """Benchmark: Cache latency should be <10ms"""
        config = TTSConfig(cache_dir=temp_cache_dir)
        cache = TTSCache(config)
        
        # Populate cache
        cache.put("Benchmark text", sample_audio_bytes)
        
        # Measure cache retrieval
        start = time.time()
        for _ in range(100):
            retrieved = cache.get("Benchmark text")
            assert retrieved is not None
        
        elapsed_ms = (time.time() - start) * 1000
        avg_latency_ms = elapsed_ms / 100
        
        print(f"\n📊 Cache avg latency: {avg_latency_ms:.2f}ms")
        assert avg_latency_ms < 10, "Cache latency too high"
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not PYTTSX3_AVAILABLE, reason="pyttsx3 not installed")
    @patch('pyttsx3.init')
    def test_pyttsx3_latency_benchmark(self, mock_init, temp_cache_dir, sample_audio_bytes):
        """Benchmark: pyttsx3 latency should be <500ms P99"""
        # Mock pyttsx3
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        def mock_save(text, path):
            Path(path).write_bytes(sample_audio_bytes)
            time.sleep(0.05)  # Simulate 50ms generation
        
        mock_engine.save_to_file.side_effect = mock_save
        mock_engine.runAndWait.return_value = None
        mock_engine.getProperty.return_value = 200
        
        config = TTSConfig(
            enable_omni=False,
            enable_pyttsx3=True,
            enable_cache=False,  # Disable cache for benchmark
            cache_dir=temp_cache_dir
        )
        engine = TTSEngine(config)
        
        # Measure latencies
        latencies = []
        for i in range(20):
            result = engine.generate(f"Benchmark text {i}")
            latencies.append(result.latency_ms)
        
        # Calculate P99
        p99_latency = np.percentile(latencies, 99)
        avg_latency = np.mean(latencies)
        
        print(f"\n📊 pyttsx3 latencies:")
        print(f"  - Average: {avg_latency:.1f}ms")
        print(f"  - P99: {p99_latency:.1f}ms")
        
        assert p99_latency < 500, f"P99 latency too high: {p99_latency:.1f}ms"
    
    @pytest.mark.benchmark
    def test_cache_hit_rate_benchmark(self, temp_cache_dir, sample_audio_bytes):
        """Benchmark: Cache hit rate should be >40%"""
        config = TTSConfig(cache_dir=temp_cache_dir)
        engine = TTSEngine(config)
        
        # Pre-populate cache with common phrases
        common_phrases = [
            "Hello", "Goodbye", "Thank you", "Yes", "No"
        ]
        for phrase in common_phrases:
            engine.cache.put(phrase, sample_audio_bytes)
        
        # Simulate realistic query distribution
        # 50% common phrases, 50% unique
        queries = common_phrases * 4 + [f"Unique {i}" for i in range(5)]
        
        hits = 0
        for query in queries:
            result = engine.generate(query)
            if result.cached:
                hits += 1
        
        hit_rate = hits / len(queries)
        
        print(f"\n📊 Cache hit rate: {hit_rate * 100:.1f}%")
        assert hit_rate > 0.4, f"Cache hit rate too low: {hit_rate * 100:.1f}%"


# ============================================================
# FACTORY TESTS
# ============================================================

def test_create_tts_engine():
    """Test factory function"""
    engine = create_tts_engine()
    
    assert isinstance(engine, TTSEngine)
    assert engine.config is not None


def test_create_tts_engine_custom_config(temp_cache_dir):
    """Test factory with custom config"""
    config = TTSConfig(
        enable_omni=False,
        cache_dir=temp_cache_dir
    )
    engine = create_tts_engine(config)
    
    assert engine.config == config


# ============================================================
# TTSOutput TESTS
# ============================================================

def test_tts_output_duration_calculation():
    """Test TTSOutput duration calculation"""
    # Create sample WAV (44 byte header + 16000 samples * 2 bytes = 32044 bytes)
    # Duration: 16000 samples / 16000 Hz = 1.0 second
    import wave
    import io
    
    audio_array = np.zeros(16000, dtype=np.int16)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_array.tobytes())
    
    wav_buffer.seek(0)
    audio_bytes = wav_buffer.read()
    
    output = TTSOutput(
        audio_bytes=audio_bytes,
        text="Test",
        method_used="test",
        latency_ms=100,
        cached=False,
        prosody_applied=False
    )
    
    assert 0.9 < output.duration_seconds < 1.1  # ~1 second
