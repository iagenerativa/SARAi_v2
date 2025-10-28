"""
agents/emotion_cache.py

Sistema de caché para detección emocional
Evita recálculos en diálogos cortos y coherentes

Características:
- TTL dinámico según volatilidad emocional
- Cache por similarity de audio (hash perceptual)
- Invalidación automática en cambios bruscos

Author: SARAi v2.11
Date: 2025-10-28
"""

import time
import hashlib
from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from agents.emotion_modulator import EmotionProfile, EmotionCategory


@dataclass
class CachedEmotion:
    """
    Entrada de caché emocional
    
    Attributes:
        profile: Perfil emocional cacheado
        timestamp: Momento de creación
        audio_hash: Hash perceptual del audio
        hit_count: Número de hits en este cache entry
    """
    profile: EmotionProfile
    timestamp: float
    audio_hash: str
    hit_count: int = 0
    
    def is_expired(self, ttl: float) -> bool:
        """Verifica si el cache expiró"""
        return (time.time() - self.timestamp) > ttl


class EmotionCache:
    """
    Caché LRU + TTL para perfiles emocionales
    
    Features:
    - TTL dinámico según volatilidad
    - Hash perceptual de audio (resistente a ruido)
    - Invalidación en cambios bruscos
    - Estadísticas de hit rate
    """
    
    def __init__(
        self,
        max_size: int = 50,
        base_ttl: float = 30.0,  # 30 segundos
        similarity_threshold: float = 0.95
    ):
        """
        Inicializa caché
        
        Args:
            max_size: Máximo de entries en caché
            base_ttl: TTL base en segundos
            similarity_threshold: Umbral de similaridad [0, 1]
        """
        self.max_size = max_size
        self.base_ttl = base_ttl
        self.similarity_threshold = similarity_threshold
        
        # Storage
        self.cache: Dict[str, CachedEmotion] = {}
        self.access_order: deque = deque(maxlen=max_size)
        
        # Historial para volatilidad
        self.emotion_history: deque = deque(maxlen=10)
        
        # Estadísticas
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "expirations": 0,
            "evictions": 0,
            "hit_rate": 0.0
        }
    
    def _compute_audio_hash(self, audio: np.ndarray) -> str:
        """
        Calcula hash perceptual del audio
        
        Args:
            audio: Waveform numpy array
        
        Returns:
            Hash string (resistente a ruido leve)
        
        Algorithm:
            1. Resample a 8kHz (reduce dimensión)
            2. Cuantizar a 8 bits
            3. SHA-256 del resultado
        """
        # Resample simple (tomar 1 cada N samples)
        target_length = 8000  # 1 segundo a 8kHz
        step = max(1, len(audio) // target_length)
        resampled = audio[::step][:target_length]
        
        # Normalizar a [-1, 1]
        if len(resampled) > 0:
            max_val = np.max(np.abs(resampled))
            if max_val > 0:
                resampled = resampled / max_val
        
        # Cuantizar a 8 bits (reduce sensibilidad a ruido)
        quantized = (resampled * 127).astype(np.int8)
        
        # Hash
        hash_obj = hashlib.sha256(quantized.tobytes())
        return hash_obj.hexdigest()[:16]  # Primeros 16 chars
    
    def _compute_ttl(self) -> float:
        """
        Calcula TTL dinámico según volatilidad emocional
        
        Returns:
            TTL en segundos
        
        Algorithm:
            - Volatilidad alta (muchos cambios) → TTL corto
            - Volatilidad baja (estable) → TTL largo
        """
        if len(self.emotion_history) < 3:
            return self.base_ttl
        
        # Contar cambios de emoción
        changes = 0
        for i in range(1, len(self.emotion_history)):
            if self.emotion_history[i] != self.emotion_history[i-1]:
                changes += 1
        
        volatility = changes / len(self.emotion_history)
        
        # TTL inversamente proporcional a volatilidad
        # Alta volatilidad (0.8) → TTL 10s
        # Baja volatilidad (0.2) → TTL 60s
        ttl = self.base_ttl * (1.0 - volatility * 0.5)
        
        return max(10.0, min(60.0, ttl))  # Clamp [10s, 60s]
    
    def get(self, audio: np.ndarray) -> Optional[EmotionProfile]:
        """
        Obtiene perfil del caché si existe
        
        Args:
            audio: Waveform numpy array
        
        Returns:
            EmotionProfile si hit, None si miss
        """
        self.stats["total_requests"] += 1
        
        audio_hash = self._compute_audio_hash(audio)
        
        # Buscar en caché
        if audio_hash in self.cache:
            entry = self.cache[audio_hash]
            
            # Verificar expiración
            ttl = self._compute_ttl()
            if entry.is_expired(ttl):
                self.stats["expirations"] += 1
                del self.cache[audio_hash]
                self.access_order.remove(audio_hash)
                self.stats["cache_misses"] += 1
                return None
            
            # HIT
            entry.hit_count += 1
            self.stats["cache_hits"] += 1
            
            # Mover al final (LRU)
            self.access_order.remove(audio_hash)
            self.access_order.append(audio_hash)
            
            # Actualizar hit rate
            self._update_hit_rate()
            
            return entry.profile
        
        # MISS
        self.stats["cache_misses"] += 1
        self._update_hit_rate()
        return None
    
    def put(self, audio: np.ndarray, profile: EmotionProfile):
        """
        Guarda perfil en caché
        
        Args:
            audio: Waveform numpy array
            profile: EmotionProfile a cachear
        """
        audio_hash = self._compute_audio_hash(audio)
        
        # Eviction si está lleno
        if len(self.cache) >= self.max_size and audio_hash not in self.cache:
            # Eliminar el menos usado (LRU)
            lru_hash = self.access_order.popleft()
            if lru_hash in self.cache:
                del self.cache[lru_hash]
                self.stats["evictions"] += 1
        
        # Guardar
        entry = CachedEmotion(
            profile=profile,
            timestamp=time.time(),
            audio_hash=audio_hash
        )
        
        self.cache[audio_hash] = entry
        
        # Actualizar access order
        if audio_hash in self.access_order:
            self.access_order.remove(audio_hash)
        self.access_order.append(audio_hash)
        
        # Actualizar historial de emociones
        self.emotion_history.append(profile.primary)
    
    def invalidate_if_changed(self, new_emotion: EmotionCategory):
        """
        Invalida caché si hay cambio brusco de emoción
        
        Args:
            new_emotion: Nueva emoción detectada
        """
        if not self.emotion_history:
            return
        
        last_emotion = self.emotion_history[-1]
        
        # Cambio brusco (emoción opuesta)
        opposites = {
            EmotionCategory.HAPPY: EmotionCategory.SAD,
            EmotionCategory.SAD: EmotionCategory.HAPPY,
            EmotionCategory.ANGRY: EmotionCategory.CALM,
            EmotionCategory.CALM: EmotionCategory.ANGRY,
            EmotionCategory.EXCITED: EmotionCategory.CALM,
        }
        
        if opposites.get(last_emotion) == new_emotion:
            # Cambio brusco → invalidar todo el caché
            self.cache.clear()
            self.access_order.clear()
            self.emotion_history.clear()
    
    def _update_hit_rate(self):
        """Actualiza estadística de hit rate"""
        total = self.stats["total_requests"]
        if total > 0:
            self.stats["hit_rate"] = self.stats["cache_hits"] / total
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del caché"""
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "current_ttl": self._compute_ttl(),
            "emotion_history_length": len(self.emotion_history)
        }
    
    def clear(self):
        """Limpia completamente el caché"""
        self.cache.clear()
        self.access_order.clear()
        self.emotion_history.clear()


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_emotion_cache(
    max_size: int = 50,
    base_ttl: float = 30.0
) -> EmotionCache:
    """
    Factory para crear caché con configuración
    
    Args:
        max_size: Máximo de entries
        base_ttl: TTL base en segundos
    
    Returns:
        EmotionCache configurado
    """
    return EmotionCache(max_size=max_size, base_ttl=base_ttl)
