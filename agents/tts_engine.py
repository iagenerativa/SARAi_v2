"""
TTS Engine for SARAi - M3.2 Fase 3
===================================

3-Level Fallback TTS System:
1. Qwen2.5-Omni-3B TTS (native, high quality)
2. pyttsx3 (local, fast fallback)
3. Text-only (graceful degradation)

Features:
- Prosody application from emotion_integration
- Audio encoding: WAV 16kHz mono
- Perceptual hash caching + LRU eviction
- Performance: <500ms P99, >40% cache hit, >4.0 MOS

Author: SARAi Team
Date: Oct 28, 2025
"""

import os
import time
import hashlib
import wave
import io
from typing import Optional, Dict, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import torch

# Lazy imports (cargados bajo demanda)
_omni_model = None
_pyttsx3_engine = None

logger = logging.getLogger(__name__)


@dataclass
class TTSOutput:
    """Output structure for TTS generation"""
    audio_bytes: Optional[bytes]  # WAV audio data
    text: str  # Input text
    method_used: str  # "omni" | "pyttsx3" | "text_only"
    latency_ms: float  # Generation time
    cached: bool  # Was served from cache?
    prosody_applied: bool  # Was emotion prosody applied?
    sample_rate: int = 16000  # Default: 16kHz
    channels: int = 1  # Mono
    
    @property
    def duration_seconds(self) -> float:
        """Calculate audio duration from bytes"""
        if not self.audio_bytes:
            return 0.0
        
        # WAV header is 44 bytes, data is after that
        # bytes_per_sample = 2 (16-bit)
        # duration = (total_bytes - 44) / (sample_rate * channels * bytes_per_sample)
        data_bytes = len(self.audio_bytes) - 44
        return data_bytes / (self.sample_rate * self.channels * 2)


@dataclass
class TTSConfig:
    """Configuration for TTS Engine"""
    enable_omni: bool = True  # Use Qwen2.5-Omni-3B
    enable_pyttsx3: bool = True  # Fallback to pyttsx3
    enable_cache: bool = True  # Cache TTS outputs
    cache_dir: str = "state/tts_cache"
    cache_max_size_mb: int = 100  # 100MB max cache
    cache_ttl_seconds: int = 3600  # 1 hour TTL
    omni_voice: str = "Chelsie"  # "Chelsie" | "Ethan"
    target_sample_rate: int = 16000
    apply_prosody: bool = True  # Apply emotion prosody


class TTSCache:
    """
    Perceptual hash-based cache for TTS outputs
    
    Uses text content hash + prosody hash as key
    LRU eviction when cache size exceeds limit
    """
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory index: {cache_key: (file_path, timestamp, size_bytes)}
        self.index: Dict[str, Tuple[Path, float, int]] = {}
        self._load_index()
    
    def _load_index(self):
        """Load existing cache files into index"""
        if not self.cache_dir.exists():
            return
        
        for wav_file in self.cache_dir.glob("*.wav"):
            key = wav_file.stem  # filename without extension
            stat = wav_file.stat()
            self.index[key] = (wav_file, stat.st_mtime, stat.st_size)
        
        logger.info(f"TTS Cache loaded: {len(self.index)} entries")
    
    def _compute_key(self, text: str, prosody_params: Optional[Dict] = None) -> str:
        """
        Compute cache key from text + prosody
        
        Uses SHA-256 hash of normalized text + prosody JSON
        """
        # Normalize text (lowercase, strip whitespace)
        normalized_text = text.lower().strip()
        
        # Include prosody in hash if available
        if prosody_params:
            import json
            prosody_str = json.dumps(prosody_params, sort_keys=True)
            combined = f"{normalized_text}|{prosody_str}"
        else:
            combined = normalized_text
        
        # SHA-256 hash (first 16 chars for filename safety)
        hash_obj = hashlib.sha256(combined.encode('utf-8'))
        return hash_obj.hexdigest()[:16]
    
    def get(self, text: str, prosody_params: Optional[Dict] = None) -> Optional[bytes]:
        """
        Retrieve cached TTS audio
        
        Returns None if cache miss or expired
        """
        if not self.config.enable_cache:
            return None
        
        key = self._compute_key(text, prosody_params)
        
        if key not in self.index:
            return None  # Cache miss
        
        file_path, timestamp, _ = self.index[key]
        
        # Check TTL
        age_seconds = time.time() - timestamp
        if age_seconds > self.config.cache_ttl_seconds:
            # Expired, remove
            self._remove_entry(key)
            return None
        
        # Read and return audio bytes
        try:
            with open(file_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Update access time (LRU)
            file_path.touch()
            self.index[key] = (file_path, time.time(), len(audio_bytes))
            
            logger.debug(f"TTS Cache HIT: {key[:8]}... ({len(audio_bytes)} bytes)")
            return audio_bytes
        
        except Exception as e:
            logger.error(f"Cache read error for {key}: {e}")
            self._remove_entry(key)
            return None
    
    def put(self, text: str, audio_bytes: bytes, prosody_params: Optional[Dict] = None):
        """
        Store TTS audio in cache
        
        Evicts LRU entries if cache size exceeds limit
        """
        if not self.config.enable_cache:
            return
        
        key = self._compute_key(text, prosody_params)
        file_path = self.cache_dir / f"{key}.wav"
        
        # Write audio to disk
        try:
            with open(file_path, 'wb') as f:
                f.write(audio_bytes)
            
            # Update index
            self.index[key] = (file_path, time.time(), len(audio_bytes))
            
            # Check cache size and evict if needed
            self._evict_if_needed()
            
            logger.debug(f"TTS Cache PUT: {key[:8]}... ({len(audio_bytes)} bytes)")
        
        except Exception as e:
            logger.error(f"Cache write error for {key}: {e}")
    
    def _evict_if_needed(self):
        """Evict LRU entries if cache exceeds max size"""
        total_size_bytes = sum(size for _, _, size in self.index.values())
        max_size_bytes = self.config.cache_max_size_mb * 1024 * 1024
        
        if total_size_bytes <= max_size_bytes:
            return  # No eviction needed
        
        # Sort by access time (oldest first)
        sorted_entries = sorted(
            self.index.items(),
            key=lambda x: x[1][1]  # timestamp
        )
        
        # Evict oldest until under limit
        for key, (file_path, _, size) in sorted_entries:
            if total_size_bytes <= max_size_bytes:
                break
            
            self._remove_entry(key)
            total_size_bytes -= size
            logger.info(f"TTS Cache EVICT: {key[:8]}... (LRU)")
    
    def _remove_entry(self, key: str):
        """Remove cache entry from disk and index"""
        if key not in self.index:
            return
        
        file_path, _, _ = self.index[key]
        
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
        
        del self.index[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_size_bytes = sum(size for _, _, size in self.index.values())
        
        return {
            "entries": len(self.index),
            "total_size_mb": total_size_bytes / (1024 * 1024),
            "max_size_mb": self.config.cache_max_size_mb,
            "utilization_pct": (total_size_bytes / (self.config.cache_max_size_mb * 1024 * 1024)) * 100
        }


class TTSEngine:
    """
    3-Level Fallback TTS Engine
    
    Priority:
    1. Qwen2.5-Omni-3B (native, high quality)
    2. pyttsx3 (local, fast fallback)
    3. Text-only (graceful degradation)
    
    Usage:
        engine = TTSEngine()
        result = engine.generate("Hello world", emotion_state=emotion)
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.cache = TTSCache(self.config)
        
        # Lazy-loaded models
        self._omni_model = None
        self._pyttsx3_engine = None
        
        logger.info(f"TTSEngine initialized (omni={self.config.enable_omni}, "
                   f"pyttsx3={self.config.enable_pyttsx3}, cache={self.config.enable_cache})")
    
    def generate(self, 
                text: str,
                emotion_state: Optional['EmotionState'] = None,
                force_method: Optional[str] = None) -> TTSOutput:
        """
        Generate TTS audio with 3-level fallback
        
        Args:
            text: Input text to synthesize
            emotion_state: Optional EmotionState from emotion_integration
            force_method: Force specific method ("omni", "pyttsx3", "text_only")
        
        Returns:
            TTSOutput with audio bytes and metadata
        """
        start_time = time.time()
        
        # Extract prosody parameters if emotion provided
        prosody_params = None
        if emotion_state and self.config.apply_prosody:
            prosody_params = self._extract_prosody(emotion_state)
        
        # Check cache first
        cached_audio = self.cache.get(text, prosody_params)
        if cached_audio:
            latency_ms = (time.time() - start_time) * 1000
            return TTSOutput(
                audio_bytes=cached_audio,
                text=text,
                method_used="cache",
                latency_ms=latency_ms,
                cached=True,
                prosody_applied=bool(prosody_params)
            )
        
        # Try methods in priority order
        methods = []
        if force_method:
            methods = [force_method]
        else:
            if self.config.enable_omni:
                methods.append("omni")
            if self.config.enable_pyttsx3:
                methods.append("pyttsx3")
            methods.append("text_only")  # Always available
        
        for method in methods:
            try:
                if method == "omni":
                    audio_bytes = self._generate_omni(text, prosody_params)
                elif method == "pyttsx3":
                    audio_bytes = self._generate_pyttsx3(text, prosody_params)
                elif method == "text_only":
                    audio_bytes = None  # No audio
                else:
                    continue
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Cache successful generation (except text_only)
                if audio_bytes and method != "text_only":
                    self.cache.put(text, audio_bytes, prosody_params)
                
                return TTSOutput(
                    audio_bytes=audio_bytes,
                    text=text,
                    method_used=method,
                    latency_ms=latency_ms,
                    cached=False,
                    prosody_applied=bool(prosody_params and audio_bytes)
                )
            
            except Exception as e:
                logger.warning(f"TTS method '{method}' failed: {e}")
                continue  # Try next fallback
        
        # All methods failed, return text-only
        latency_ms = (time.time() - start_time) * 1000
        return TTSOutput(
            audio_bytes=None,
            text=text,
            method_used="text_only",
            latency_ms=latency_ms,
            cached=False,
            prosody_applied=False
        )
    
    def _extract_prosody(self, emotion_state: 'EmotionState') -> Dict:
        """
        Extract prosody parameters from EmotionState
        
        Maps emotion scores to TTS parameters:
        - pitch: valence mapping
        - rate: arousal mapping
        - volume: intensity mapping
        """
        # No need to import, just check if object has required attributes
        if not hasattr(emotion_state, 'valence'):
            return {}
        
        # Map emotion dimensions to prosody
        # valence: -1 (negative) to +1 (positive) → pitch
        # arousal: 0 (calm) to 1 (excited) → rate
        # dominance: 0 (submissive) to 1 (dominant) → volume
        
        pitch_shift = emotion_state.valence * 50  # ±50 Hz
        rate_multiplier = 0.8 + (emotion_state.arousal * 0.4)  # 0.8x to 1.2x
        volume_multiplier = 0.7 + (emotion_state.dominance * 0.3)  # 0.7x to 1.0x
        
        return {
            "pitch_shift_hz": pitch_shift,
            "rate_multiplier": rate_multiplier,
            "volume_multiplier": volume_multiplier,
            "emotion_label": emotion_state.label
        }
    
    def _generate_omni(self, text: str, prosody_params: Optional[Dict]) -> bytes:
        """
        Generate TTS using Qwen2.5-Omni-3B
        
        Uses native audio-to-audio capability with emotion prosody
        """
        # Lazy load Omni model
        if self._omni_model is None:
            from agents.omni_pipeline import create_omni_full
            self._omni_model = create_omni_full(enable_audio=True)
            logger.info("Omni model loaded for TTS")
        
        # Select voice based on config
        voice = self.config.omni_voice  # "Chelsie" or "Ethan"
        
        # Generate audio-to-audio (text → audio)
        # Note: Omni's native TTS doesn't directly support prosody params
        # For emotion, we rely on system prompt hints
        system_prompt = ""
        if prosody_params:
            emotion = prosody_params.get("emotion_label", "neutral")
            system_prompt = f"Speak with a {emotion} tone."
        
        result = self._omni_model.text_to_speech(
            text=text,
            voice=voice,
            system_prompt=system_prompt if system_prompt else None
        )
        
        # Convert to WAV bytes at 16kHz mono
        audio_array = result.audio  # numpy array
        audio_bytes = self._array_to_wav_bytes(audio_array, self.config.target_sample_rate)
        
        return audio_bytes
    
    def _generate_pyttsx3(self, text: str, prosody_params: Optional[Dict]) -> bytes:
        """
        Generate TTS using pyttsx3 (local fallback)
        
        Applies prosody parameters to engine
        """
        # Lazy load pyttsx3
        if self._pyttsx3_engine is None:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            logger.info("pyttsx3 engine loaded for TTS")
        
        # Apply prosody if available
        if prosody_params:
            # pyttsx3 properties: rate, volume (no pitch control)
            current_rate = self._pyttsx3_engine.getProperty('rate')
            current_volume = self._pyttsx3_engine.getProperty('volume')
            
            new_rate = int(current_rate * prosody_params.get('rate_multiplier', 1.0))
            new_volume = current_volume * prosody_params.get('volume_multiplier', 1.0)
            
            self._pyttsx3_engine.setProperty('rate', new_rate)
            self._pyttsx3_engine.setProperty('volume', min(1.0, new_volume))
        
        # Generate to memory buffer
        audio_buffer = io.BytesIO()
        
        # pyttsx3 saves to file, so use temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self._pyttsx3_engine.save_to_file(text, temp_path)
            self._pyttsx3_engine.runAndWait()
            
            # Read generated audio
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Resample to target sample rate if needed
            # (pyttsx3 typically outputs at 22050 Hz)
            audio_bytes = self._resample_wav(audio_bytes, self.config.target_sample_rate)
            
            return audio_bytes
        
        finally:
            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)
    
    def _array_to_wav_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes"""
        # Ensure int16 format
        if audio_array.dtype != np.int16:
            audio_array = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_array.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    def _resample_wav(self, wav_bytes: bytes, target_rate: int) -> bytes:
        """Resample WAV audio to target sample rate"""
        # Read input WAV
        wav_buffer = io.BytesIO(wav_bytes)
        with wave.open(wav_buffer, 'rb') as wav_file:
            params = wav_file.getparams()
            audio_data = wav_file.readframes(params.nframes)
            original_rate = params.framerate
        
        if original_rate == target_rate:
            return wav_bytes  # No resampling needed
        
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Simple linear resampling (for production, use scipy.signal.resample)
        # This is a fallback implementation
        duration_seconds = len(audio_array) / original_rate
        target_length = int(duration_seconds * target_rate)
        
        indices = np.linspace(0, len(audio_array) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.int16)
        
        # Write resampled WAV
        return self._array_to_wav_bytes(resampled, target_rate)
    
    def get_cache_stats(self) -> Dict:
        """Get TTS cache statistics"""
        return self.cache.get_stats()


# Factory function for easy instantiation
def create_tts_engine(config: Optional[TTSConfig] = None) -> TTSEngine:
    """
    Factory function to create TTSEngine instance
    
    Usage:
        from agents.tts_engine import create_tts_engine
        
        tts = create_tts_engine()
        result = tts.generate("Hello world")
    """
    return TTSEngine(config)
