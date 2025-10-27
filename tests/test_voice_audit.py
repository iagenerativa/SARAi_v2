"""
Test Suite for Voice Audit Logger (M3.1 Fase 4)

Tests HMAC-SHA256 auditing for voice interactions:
- Log voice interactions with HMAC signatures
- Verify integrity of voice logs
- Detect corruption and tampering
- Integration with omni_pipeline and nllb_translator

Author: SARAi Team
License: CC-BY-NC-SA 4.0
"""

import pytest
import sys
import os
import json
import hmac
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.web_audit import (
    VoiceAuditLogger,
    get_voice_audit_logger,
    log_voice_interaction
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_log_dir():
    """Temporary directory for test logs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def voice_logger(temp_log_dir):
    """Voice logger with temporary directory"""
    logger = VoiceAuditLogger(log_dir=temp_log_dir, secret_key="test-secret-key")
    return logger


@pytest.fixture
def sample_audio_input():
    """Sample audio input (WAV bytes)"""
    return b"RIFF....WAVE...." + b"\x00" * 1000  # Simulated WAV


@pytest.fixture
def sample_audio_output():
    """Sample audio output (TTS response)"""
    return b"RIFF....WAVE...." + b"\xff" * 800


# ============================================================================
# Test: VoiceAuditLogger Initialization
# ============================================================================

class TestVoiceAuditLoggerInit:
    """Test logger initialization and configuration"""
    
    def test_init_with_default_key(self, temp_log_dir):
        """Test initialization with default secret key"""
        logger = VoiceAuditLogger(log_dir=temp_log_dir)
        assert logger.secret_key == b"sarai-voice-audit-key"
    
    def test_init_with_custom_key(self, temp_log_dir):
        """Test initialization with custom secret key"""
        logger = VoiceAuditLogger(log_dir=temp_log_dir, secret_key="my-secret")
        assert logger.secret_key == b"my-secret"
    
    def test_init_with_env_key(self, temp_log_dir):
        """Test initialization with key from environment variable"""
        with patch.dict(os.environ, {"HMAC_SECRET_KEY": "env-secret"}):
            logger = VoiceAuditLogger(log_dir=temp_log_dir)
            assert logger.secret_key == b"env-secret"
    
    def test_log_directory_created(self, temp_log_dir):
        """Test that log directory is created if not exists"""
        log_dir = os.path.join(temp_log_dir, "new_logs")
        logger = VoiceAuditLogger(log_dir=log_dir)
        assert os.path.exists(log_dir)


# ============================================================================
# Test: Log Voice Interaction
# ============================================================================

class TestLogVoiceInteraction:
    """Test logging voice interactions with HMAC"""
    
    def test_log_basic_interaction(self, voice_logger, sample_audio_input, temp_log_dir):
        """Test logging a basic voice interaction"""
        voice_logger.log_voice_interaction(
            input_audio=sample_audio_input,
            detected_lang="es",
            engine_used="omni",
            response_text="Hola, ¿en qué puedo ayudarte?"
        )
        
        # Verify log file created
        date = datetime.now().strftime("%Y-%m-%d")
        jsonl_path = os.path.join(temp_log_dir, f"voice_interactions_{date}.jsonl")
        hmac_path = f"{jsonl_path}.hmac"
        
        assert os.path.exists(jsonl_path)
        assert os.path.exists(hmac_path)
        
        # Verify log content
        with open(jsonl_path) as f:
            entry = json.loads(f.readline())
            assert entry["detected_lang"] == "es"
            assert entry["engine_used"] == "omni"
            assert "Hola" in entry["response_text"]
            assert "input_audio_sha256" in entry
    
    def test_log_with_response_audio(self, voice_logger, sample_audio_input, sample_audio_output):
        """Test logging with response audio"""
        voice_logger.log_voice_interaction(
            input_audio=sample_audio_input,
            detected_lang="fr",
            engine_used="nllb",
            response_text="Bonjour!",
            response_audio=sample_audio_output
        )
        
        date = datetime.now().strftime("%Y-%m-%d")
        jsonl_path = voice_logger._get_log_paths()[0]
        
        with open(jsonl_path) as f:
            entry = json.loads(f.readline())
            assert entry["response_audio_sha256"] is not None
            assert len(entry["response_audio_sha256"]) == 64  # SHA256 hex length
    
    def test_log_with_latency(self, voice_logger, sample_audio_input):
        """Test logging with latency measurement"""
        voice_logger.log_voice_interaction(
            input_audio=sample_audio_input,
            detected_lang="ja",
            engine_used="nllb",
            response_text="こんにちは",
            latency_ms=1250.5
        )
        
        jsonl_path = voice_logger._get_log_paths()[0]
        
        with open(jsonl_path) as f:
            entry = json.loads(f.readline())
            assert entry["latency_ms"] == 1250.5
    
    def test_log_with_error(self, voice_logger, sample_audio_input):
        """Test logging interaction with error"""
        voice_logger.log_voice_interaction(
            input_audio=sample_audio_input,
            detected_lang="unknown",
            engine_used="lfm2",
            response_text="",
            error="Language detection failed"
        )
        
        jsonl_path = voice_logger._get_log_paths()[0]
        
        with open(jsonl_path) as f:
            entry = json.loads(f.readline())
            assert entry["error"] == "Language detection failed"
    
    def test_multiple_interactions_same_day(self, voice_logger, sample_audio_input):
        """Test logging multiple interactions in same day"""
        for i in range(5):
            voice_logger.log_voice_interaction(
                input_audio=sample_audio_input,
                detected_lang="es",
                engine_used="omni",
                response_text=f"Respuesta {i}"
            )
        
        jsonl_path = voice_logger._get_log_paths()[0]
        
        # Verify 5 entries
        with open(jsonl_path) as f:
            lines = f.readlines()
            assert len(lines) == 5


# ============================================================================
# Test: HMAC Computation
# ============================================================================

class TestHMACComputation:
    """Test HMAC signature computation"""
    
    def test_hmac_deterministic(self, voice_logger):
        """Test that HMAC is deterministic (same input = same signature)"""
        entry_str = '{"key": "value"}'
        
        hmac1 = voice_logger._compute_hmac(entry_str)
        hmac2 = voice_logger._compute_hmac(entry_str)
        
        assert hmac1 == hmac2
    
    def test_hmac_different_for_different_input(self, voice_logger):
        """Test that different inputs produce different HMACs"""
        entry1 = '{"key": "value1"}'
        entry2 = '{"key": "value2"}'
        
        hmac1 = voice_logger._compute_hmac(entry1)
        hmac2 = voice_logger._compute_hmac(entry2)
        
        assert hmac1 != hmac2
    
    def test_hmac_length(self, voice_logger):
        """Test that HMAC has correct length (SHA256 = 64 hex chars)"""
        entry_str = '{"test": "data"}'
        signature = voice_logger._compute_hmac(entry_str)
        
        assert len(signature) == 64
        assert all(c in '0123456789abcdef' for c in signature)


# ============================================================================
# Test: Integrity Verification
# ============================================================================

class TestIntegrityVerification:
    """Test verification of log integrity"""
    
    def test_verify_valid_logs(self, voice_logger, sample_audio_input):
        """Test verification of valid, uncorrupted logs"""
        # Log some interactions
        for lang in ["es", "fr", "de"]:
            voice_logger.log_voice_interaction(
                input_audio=sample_audio_input,
                detected_lang=lang,
                engine_used="omni",
                response_text=f"Response in {lang}"
            )
        
        # Verify
        date = datetime.now().strftime("%Y-%m-%d")
        is_valid = voice_logger.verify_integrity(date)
        
        assert is_valid is True
    
    def test_verify_missing_logs(self, voice_logger):
        """Test verification when no logs exist for date"""
        is_valid = voice_logger.verify_integrity("2025-01-01")
        
        # Should return True (no logs = no corruption)
        assert is_valid is True
    
    @patch('core.web_audit.activate_safe_mode')
    def test_verify_missing_hmac_file(self, mock_safe_mode, voice_logger, sample_audio_input, temp_log_dir):
        """Test verification when HMAC file is missing"""
        # Log interaction
        voice_logger.log_voice_interaction(
            input_audio=sample_audio_input,
            detected_lang="es",
            engine_used="omni",
            response_text="Test"
        )
        
        # Delete HMAC file
        date = datetime.now().strftime("%Y-%m-%d")
        hmac_path = os.path.join(temp_log_dir, f"voice_interactions_{date}.jsonl.hmac")
        os.remove(hmac_path)
        
        # Verify
        is_valid = voice_logger.verify_integrity(date)
        
        assert is_valid is False
        mock_safe_mode.assert_called_once_with(reason="voice_audit_missing_hmac")
    
    @patch('core.web_audit.activate_safe_mode')
    @patch('core.web_audit.send_critical_webhook')
    def test_verify_corrupted_logs(self, mock_webhook, mock_safe_mode, voice_logger, sample_audio_input, temp_log_dir):
        """Test verification when log is corrupted"""
        # Log interaction
        voice_logger.log_voice_interaction(
            input_audio=sample_audio_input,
            detected_lang="es",
            engine_used="omni",
            response_text="Original text"
        )
        
        # Corrupt log (change content)
        date = datetime.now().strftime("%Y-%m-%d")
        jsonl_path = os.path.join(temp_log_dir, f"voice_interactions_{date}.jsonl")
        
        with open(jsonl_path, 'r') as f:
            entry = json.loads(f.readline())
        
        # Modify entry
        entry["response_text"] = "TAMPERED TEXT"
        
        with open(jsonl_path, 'w') as f:
            f.write(json.dumps(entry) + "\n")
        
        # Verify (should fail)
        is_valid = voice_logger.verify_integrity(date)
        
        assert is_valid is False
        mock_safe_mode.assert_called_once_with(reason="voice_audit_hmac_mismatch")
        mock_webhook.assert_called_once()


# ============================================================================
# Test: Statistics
# ============================================================================

class TestStatistics:
    """Test statistics generation"""
    
    def test_stats_empty_logs(self, voice_logger):
        """Test stats for date with no logs"""
        stats = voice_logger.get_stats("2025-01-01")
        
        assert stats["total_interactions"] == 0
        assert stats["by_language"] == {}
        assert stats["by_engine"] == {}
        assert stats["avg_latency_ms"] == 0.0
        assert stats["errors_count"] == 0
    
    def test_stats_multiple_languages(self, voice_logger, sample_audio_input):
        """Test stats with multiple languages"""
        languages = ["es", "es", "fr", "de", "fr"]
        
        for lang in languages:
            voice_logger.log_voice_interaction(
                input_audio=sample_audio_input,
                detected_lang=lang,
                engine_used="omni",
                response_text="Test"
            )
        
        date = datetime.now().strftime("%Y-%m-%d")
        stats = voice_logger.get_stats(date)
        
        assert stats["total_interactions"] == 5
        assert stats["by_language"]["es"] == 2
        assert stats["by_language"]["fr"] == 2
        assert stats["by_language"]["de"] == 1
    
    def test_stats_multiple_engines(self, voice_logger, sample_audio_input):
        """Test stats with multiple engines"""
        engines = ["omni", "nllb", "omni", "lfm2", "omni"]
        
        for engine in engines:
            voice_logger.log_voice_interaction(
                input_audio=sample_audio_input,
                detected_lang="es",
                engine_used=engine,
                response_text="Test"
            )
        
        date = datetime.now().strftime("%Y-%m-%d")
        stats = voice_logger.get_stats(date)
        
        assert stats["by_engine"]["omni"] == 3
        assert stats["by_engine"]["nllb"] == 1
        assert stats["by_engine"]["lfm2"] == 1
    
    def test_stats_average_latency(self, voice_logger, sample_audio_input):
        """Test average latency calculation"""
        latencies = [100.0, 200.0, 300.0]
        
        for lat in latencies:
            voice_logger.log_voice_interaction(
                input_audio=sample_audio_input,
                detected_lang="es",
                engine_used="omni",
                response_text="Test",
                latency_ms=lat
            )
        
        date = datetime.now().strftime("%Y-%m-%d")
        stats = voice_logger.get_stats(date)
        
        assert stats["avg_latency_ms"] == 200.0  # (100+200+300)/3
    
    def test_stats_error_count(self, voice_logger, sample_audio_input):
        """Test error counting"""
        # Log 3 successful, 2 with errors
        for i in range(3):
            voice_logger.log_voice_interaction(
                input_audio=sample_audio_input,
                detected_lang="es",
                engine_used="omni",
                response_text="OK"
            )
        
        for i in range(2):
            voice_logger.log_voice_interaction(
                input_audio=sample_audio_input,
                detected_lang="es",
                engine_used="omni",
                response_text="",
                error="Failed"
            )
        
        date = datetime.now().strftime("%Y-%m-%d")
        stats = voice_logger.get_stats(date)
        
        assert stats["total_interactions"] == 5
        assert stats["errors_count"] == 2


# ============================================================================
# Test: Singleton Pattern
# ============================================================================

class TestSingleton:
    """Test singleton factory function"""
    
    @patch('core.web_audit.VoiceAuditLogger')
    def test_get_voice_audit_logger_singleton(self, MockLogger):
        """Test that get_voice_audit_logger returns same instance"""
        # Reset singleton
        import core.web_audit
        core.web_audit._voice_audit_logger_instance = None
        
        # First call creates instance
        logger1 = get_voice_audit_logger()
        MockLogger.assert_called_once()
        
        # Second call reuses instance
        logger2 = get_voice_audit_logger()
        assert MockLogger.call_count == 1  # Not called again
        assert logger1 is logger2


# ============================================================================
# Test: Convenience Function
# ============================================================================

class TestConvenienceFunction:
    """Test log_voice_interaction convenience function"""
    
    @patch('core.web_audit.get_voice_audit_logger')
    def test_log_voice_interaction_uses_singleton(self, mock_get_logger, sample_audio_input):
        """Test that convenience function uses singleton logger"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        log_voice_interaction(
            input_audio=sample_audio_input,
            detected_lang="fr",
            engine_used="nllb",
            response_text="Bonjour",
            latency_ms=1500.0
        )
        
        mock_get_logger.assert_called_once()
        mock_logger.log_voice_interaction.assert_called_once()
        
        # Verify arguments passed correctly
        call_args = mock_logger.log_voice_interaction.call_args
        assert call_args[0][0] == sample_audio_input
        assert call_args[0][1] == "fr"
        assert call_args[0][2] == "nllb"
        assert call_args[0][3] == "Bonjour"


# ============================================================================
# Test: Integration with Omni Pipeline
# ============================================================================

class TestIntegrationOmniPipeline:
    """Integration tests with omni_pipeline.py"""
    
    def test_log_omni_interaction(self, voice_logger, sample_audio_input, sample_audio_output):
        """Simulate logging an Omni pipeline interaction"""
        # Simulate Omni-3B processing
        voice_logger.log_voice_interaction(
            input_audio=sample_audio_input,
            detected_lang="es",
            engine_used="omni",
            response_text="Entiendo tu solicitud. ¿En qué más puedo ayudarte?",
            response_audio=sample_audio_output,
            latency_ms=245.8  # Target: <250ms
        )
        
        date = datetime.now().strftime("%Y-%m-%d")
        stats = voice_logger.get_stats(date)
        
        assert stats["total_interactions"] == 1
        assert stats["by_engine"]["omni"] == 1
        assert stats["avg_latency_ms"] < 250  # Within target
    
    def test_log_nllb_interaction(self, voice_logger, sample_audio_input, sample_audio_output):
        """Simulate logging an NLLB translation interaction"""
        # Simulate NLLB pipeline (French)
        voice_logger.log_voice_interaction(
            input_audio=sample_audio_input,
            detected_lang="fr",
            engine_used="nllb",
            response_text="Je comprends votre demande.",
            response_audio=sample_audio_output,
            latency_ms=1450.2  # Target: 1-2s
        )
        
        date = datetime.now().strftime("%Y-%m-%d")
        stats = voice_logger.get_stats(date)
        
        assert stats["by_engine"]["nllb"] == 1
        assert 1000 < stats["avg_latency_ms"] < 2000  # Within target


# ============================================================================
# Test: Thread Safety
# ============================================================================

class TestThreadSafety:
    """Test thread-safe logging"""
    
    def test_concurrent_logging(self, voice_logger, sample_audio_input):
        """Test that concurrent logging doesn't corrupt logs"""
        import threading
        
        def log_interaction(lang):
            for _ in range(10):
                voice_logger.log_voice_interaction(
                    input_audio=sample_audio_input,
                    detected_lang=lang,
                    engine_used="omni",
                    response_text=f"Response {lang}"
                )
        
        # Create 3 threads logging concurrently
        threads = [
            threading.Thread(target=log_interaction, args=("es",)),
            threading.Thread(target=log_interaction, args=("fr",)),
            threading.Thread(target=log_interaction, args=("de",))
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all 30 entries logged
        date = datetime.now().strftime("%Y-%m-%d")
        stats = voice_logger.get_stats(date)
        
        assert stats["total_interactions"] == 30
        assert stats["by_language"]["es"] == 10
        assert stats["by_language"]["fr"] == 10
        assert stats["by_language"]["de"] == 10
        
        # Verify integrity
        assert voice_logger.verify_integrity(date) is True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
