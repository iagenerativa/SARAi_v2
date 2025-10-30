"""
BERT-es Embedder
Genera embeddings semánticos de texto en español para análisis
"""

import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class BERTEmbedder:
    """
    Generador de embeddings con BERT en español
    
    Modelos soportados:
    - dccuchile/bert-base-spanish-wwm-uncased (110M params)
    - PlanTL-GOB-ES/roberta-base-bne (125M params)
    - hiiamsid/sentence_similarity_spanish_es (lighter)
    
    Uso:
        embedder = BERTEmbedder()
        emb = embedder.encode("¿Cómo estás?")
        # emb.shape = (768,)
    """
    
    def __init__(
        self,
        model_name: str = "hiiamsid/sentence_similarity_spanish_es",
        device: str = "cpu",
        max_length: int = 128
    ):
        """
        Args:
            model_name: Nombre del modelo HuggingFace
            device: 'cpu' o 'cuda'
            max_length: Longitud máxima de tokens (128 suficiente para frases cortas)
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        # Lazy loading
        self._model = None
        self._tokenizer = None
        
        logger.info(f"BERTEmbedder inicializado (modelo: {model_name})")
    
    def _load_model(self):
        """Carga el modelo bajo demanda"""
        if self._model is not None:
            return
        
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        logger.info(f"Cargando {self.model_name}...")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        
        # Mover a CPU/GPU
        self._model.to(self.device)
        self._model.eval()
        
        logger.info(f"✓ BERT-es cargado en {self.device}")
    
    def encode(self, text: str) -> np.ndarray:
        """
        Genera embedding de un texto
        
        Args:
            text: Texto en español
        
        Returns:
            Embedding de 768 dimensiones (np.ndarray)
        """
        self._load_model()
        
        import torch
        
        # Tokenizar
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        # Mover a device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inferencia sin gradientes
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        # Mean pooling sobre tokens (CLS token alternativa: outputs.last_hidden_state[:, 0])
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Convertir a numpy
        embedding = embeddings.cpu().numpy().squeeze()
        
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings de múltiples textos (más eficiente)
        
        Args:
            texts: Lista de textos
        
        Returns:
            Embeddings shape=(len(texts), 768)
        """
        self._load_model()
        
        import torch
        
        # Tokenizar batch
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud coseno entre dos textos
        
        Returns:
            Similitud en rango [0, 1]
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        return float(dot_product / (norm1 * norm2))


# Ejemplo de uso
if __name__ == "__main__":
    embedder = BERTEmbedder()
    
    # Test de embeddings
    texts = [
        "¿Cómo estás?",
        "¿Qué tal estás?",
        "Explícame Python"
    ]
    
    print("Generando embeddings...")
    embeddings = embedder.encode_batch(texts)
    print(f"Shape: {embeddings.shape}")  # (3, 768)
    
    # Test de similitud
    sim = embedder.similarity(texts[0], texts[1])
    print(f"Similitud '{texts[0]}' vs '{texts[1]}': {sim:.3f}")
    
    sim2 = embedder.similarity(texts[0], texts[2])
    print(f"Similitud '{texts[0]}' vs '{texts[2]}': {sim2:.3f}")
