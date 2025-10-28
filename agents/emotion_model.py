"""
agents/emotion_model.py

Wrapper para modelos pre-entrenados de detección emocional

Modelos soportados:
- ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (7 emociones)
- superb/hubert-large-superb-er (4 emociones básicas)
- Custom fine-tuned models (español)

Features:
- Lazy loading (carga bajo demanda)
- Batch inference
- CPU/GPU compatible
- Normalización automática de audio
- Cache de predicciones

Author: SARAi v2.11
Date: 2025-10-28
"""

import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch

# Intentar importar transformers
try:
    from transformers import (
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2Processor,
        AutoModelForAudioClassification,
        AutoFeatureExtractor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EmotionPrediction:
    """
    Predicción de emoción del modelo
    
    Attributes:
        emotion: Emoción predicha (label)
        confidence: Confianza [0, 1]
        all_scores: Scores de todas las emociones
        raw_logits: Logits sin procesar (opcional)
    """
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    raw_logits: Optional[np.ndarray] = None
    
    def __repr__(self):
        return f"EmotionPrediction(emotion='{self.emotion}', confidence={self.confidence:.3f})"


class EmotionModelWrapper:
    """
    Wrapper unificado para modelos de detección emocional
    
    Soporta múltiples arquitecturas:
    - Wav2Vec2 (Facebook)
    - HuBERT (Microsoft)
    - Custom fine-tuned models
    
    Example:
        >>> wrapper = EmotionModelWrapper(model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        >>> audio = np.random.randn(16000)  # 1s audio
        >>> prediction = wrapper.predict(audio)
        >>> print(prediction.emotion, prediction.confidence)
    """
    
    # Modelos pre-configurados
    SUPPORTED_MODELS = {
        "wav2vec2-emotion-en": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "hubert-emotion": "superb/hubert-large-superb-er",
        "wav2vec2-emotion-es": "finiteautomata/wav2vec2-large-xlsr-spanish-emotion",  # Si existe
    }
    
    def __init__(
        self,
        model_name: str = "wav2vec2-emotion-en",
        device: Optional[str] = None,
        cache_dir: Optional[str] = "models/cache/emotion"
    ):
        """
        Inicializa wrapper del modelo
        
        Args:
            model_name: Nombre del modelo (key de SUPPORTED_MODELS o HF model ID)
            device: 'cpu', 'cuda', o None (auto-detect)
            cache_dir: Directorio para cachear modelos descargados
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers no disponible. Instalar con: pip install transformers"
            )
        
        # Resolver nombre del modelo
        self.model_id = self.SUPPORTED_MODELS.get(model_name, model_name)
        self.cache_dir = cache_dir
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Lazy loading
        self._model = None
        self._processor = None
        self._emotion_labels = None
        
        logger.info(f"🧠 EmotionModelWrapper initialized: {self.model_id} (device: {self.device})")
    
    def load_model(self):
        """
        Carga modelo y processor (lazy loading)
        
        Raises:
            OSError: Si falla la descarga del modelo
        """
        if self._model is not None:
            return  # Ya cargado
        
        logger.info(f"⏳ Descargando modelo {self.model_id}...")
        
        try:
            # Intentar con Wav2Vec2 primero (más común para emotion)
            self._model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
            self._processor = Wav2Vec2Processor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir
            )
        except Exception as e_wav2vec:
            # Fallback a AutoModel (compatible con HuBERT, etc.)
            logger.warning(f"Wav2Vec2 falló, intentando AutoModel: {e_wav2vec}")
            try:
                self._model = AutoModelForAudioClassification.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir
                )
                self._processor = AutoFeatureExtractor.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir
                )
            except Exception as e_auto:
                raise OSError(f"No se pudo cargar el modelo: {e_auto}")
        
        # Mover a device
        self._model.to(self.device)
        self._model.eval()  # Modo evaluación
        
        # Extraer labels de emoción
        self._emotion_labels = list(self._model.config.id2label.values())
        
        logger.info(f"✅ Modelo cargado: {len(self._emotion_labels)} emociones - {self._emotion_labels}")
    
    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        return_all_scores: bool = True
    ) -> EmotionPrediction:
        """
        Predice emoción de un audio
        
        Args:
            audio: Waveform numpy array (mono)
            sample_rate: Sample rate del audio (Hz)
            return_all_scores: Si True, retorna scores de todas las emociones
        
        Returns:
            EmotionPrediction con emoción + confianza
        
        Raises:
            ValueError: Si audio está vacío o es inválido
        """
        if len(audio) == 0:
            raise ValueError("Audio vacío")
        
        # Lazy load modelo
        self.load_model()
        
        # Preprocesar audio
        inputs = self._processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Mover a device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inferencia (sin gradientes)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        
        # Softmax para probabilidades
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy()[0]  # [num_emotions]
        
        # Emoción predicha (max probability)
        pred_idx = np.argmax(probs_np)
        pred_emotion = self._emotion_labels[pred_idx]
        pred_confidence = float(probs_np[pred_idx])
        
        # Scores de todas las emociones
        all_scores = {
            label: float(score)
            for label, score in zip(self._emotion_labels, probs_np)
        }
        
        return EmotionPrediction(
            emotion=pred_emotion,
            confidence=pred_confidence,
            all_scores=all_scores if return_all_scores else {},
            raw_logits=logits.cpu().numpy() if return_all_scores else None
        )
    
    def predict_batch(
        self,
        audios: List[np.ndarray],
        sample_rate: int = 16000
    ) -> List[EmotionPrediction]:
        """
        Predice emociones de múltiples audios (batch inference)
        
        Args:
            audios: Lista de waveforms
            sample_rate: Sample rate común
        
        Returns:
            Lista de EmotionPrediction
        
        Note:
            Más eficiente que predict() individual si GPU disponible
        """
        # Lazy load
        self.load_model()
        
        # Preprocesar batch
        inputs = self._processor(
            audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Mover a device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inferencia batch
        with torch.no_grad():
            logits = self._model(**inputs).logits
        
        # Procesar cada predicción
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy()  # [batch_size, num_emotions]
        
        predictions = []
        for i in range(len(audios)):
            pred_idx = np.argmax(probs_np[i])
            pred_emotion = self._emotion_labels[pred_idx]
            pred_confidence = float(probs_np[i][pred_idx])
            
            all_scores = {
                label: float(score)
                for label, score in zip(self._emotion_labels, probs_np[i])
            }
            
            predictions.append(EmotionPrediction(
                emotion=pred_emotion,
                confidence=pred_confidence,
                all_scores=all_scores
            ))
        
        return predictions
    
    def get_emotion_labels(self) -> List[str]:
        """
        Retorna lista de emociones soportadas por el modelo
        
        Returns:
            Lista de labels (e.g., ['angry', 'happy', 'sad', ...])
        """
        self.load_model()
        return self._emotion_labels
    
    def benchmark(
        self,
        audio_duration_s: float = 3.0,
        sample_rate: int = 16000,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark de latencia del modelo
        
        Args:
            audio_duration_s: Duración del audio sintético
            sample_rate: Sample rate
            num_iterations: Número de iteraciones
        
        Returns:
            Dict con métricas: mean_time_ms, std_time_ms, throughput_samples_s
        """
        import time
        
        # Audio sintético
        audio = np.random.randn(int(audio_duration_s * sample_rate)).astype(np.float32)
        
        # Warm-up
        self.predict(audio, sample_rate)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            self.predict(audio, sample_rate)
            elapsed = time.time() - start
            times.append(elapsed)
        
        times_np = np.array(times)
        
        return {
            "mean_time_ms": float(np.mean(times_np) * 1000),
            "std_time_ms": float(np.std(times_np) * 1000),
            "min_time_ms": float(np.min(times_np) * 1000),
            "max_time_ms": float(np.max(times_np) * 1000),
            "throughput_samples_s": sample_rate / np.mean(times_np),
            "device": self.device,
            "model": self.model_id
        }
    
    def unload(self):
        """Descarga modelo de memoria (libera RAM/VRAM)"""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("🗑️  Modelo descargado de memoria")


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_emotion_model(
    model_name: str = "wav2vec2-emotion-en",
    device: Optional[str] = None
) -> EmotionModelWrapper:
    """
    Factory para crear modelo de emoción
    
    Args:
        model_name: Nombre del modelo
        device: Device target
    
    Returns:
        EmotionModelWrapper configurado
    
    Example:
        >>> model = create_emotion_model("wav2vec2-emotion-en")
        >>> prediction = model.predict(audio_waveform)
    """
    return EmotionModelWrapper(model_name=model_name, device=device)


def map_emotion_to_category(emotion_label: str) -> str:
    """
    Mapea label del modelo a EmotionCategory estándar
    
    Args:
        emotion_label: Label del modelo (e.g., 'ang', 'happiness')
    
    Returns:
        EmotionCategory compatible (e.g., 'angry', 'happy')
    
    Algorithm:
        Normaliza labels de diferentes modelos a categorías estándar
    """
    # Normalizar a lowercase
    label = emotion_label.lower().strip()
    
    # Mapeo de variantes
    mappings = {
        # Angry
        "ang": "angry",
        "anger": "angry",
        "angry": "angry",
        "enojado": "angry",
        
        # Happy
        "hap": "happy",
        "happiness": "happy",
        "happy": "happy",
        "feliz": "happy",
        "joy": "happy",
        
        # Sad
        "sad": "sad",
        "sadness": "sad",
        "triste": "sad",
        
        # Fearful
        "fea": "fearful",
        "fear": "fearful",
        "fearful": "fearful",
        "miedo": "fearful",
        
        # Surprised
        "sur": "surprised",
        "surprise": "surprised",
        "surprised": "surprised",
        "sorprendido": "surprised",
        
        # Calm
        "cal": "calm",
        "calm": "calm",
        "tranquilo": "calm",
        
        # Excited
        "exc": "excited",
        "excited": "excited",
        "emocionado": "excited",
        
        # Neutral
        "neu": "neutral",
        "neutral": "neutral",
        
        # Disgusted
        "dis": "disgusted",
        "disgust": "disgusted",
        "disgusted": "disgusted",
        "asco": "disgusted"
    }
    
    return mappings.get(label, "neutral")  # Default a neutral si desconocido
