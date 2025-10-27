"""
Módulo de embeddings para SARAi v2
Gestiona EmbeddingGemma-300M con carga optimizada para CPU
"""

import torch
import numpy as np
from typing import List, Union
from transformers import AutoModel, AutoTokenizer
import yaml


class EmbeddingGemma:
    """
    Wrapper para EmbeddingGemma-300M cuantizado
    CRÍTICO: Este modelo permanece en memoria durante toda la ejecución
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['embeddings']
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")  # Forzar CPU
        
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo SIN cuantización (EmbeddingGemma es pequeño, solo 300M)"""
        print(f"🔄 Cargando {self.config['name']}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['source'],
            cache_dir=self.config['cache_dir']
        )
        
        # Cargar sin cuantización (300M es manejable en CPU)
        self.model = AutoModel.from_pretrained(
            self.config['source'],
            dtype=torch.float32,  # CPU usa float32 (parámetro actualizado)
            device_map="cpu",
            cache_dir=self.config['cache_dir'],
            low_cpu_mem_usage=True
        )
        
        self.model.eval()  # Modo evaluación
        print(f"✅ {self.config['name']} cargado en CPU (~{self.config['max_memory_mb']}MB)")
    
    def encode(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Genera embeddings para texto(s)
        
        Args:
            text: String o lista de strings
            normalize: Si True, normaliza vectores a norma 1
            
        Returns:
            Array de shape (embedding_dim,) o (n, embedding_dim)
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        with torch.no_grad():
            # Tokenizar
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Obtener embeddings
            outputs = self.model(**inputs)
            
            # Mean pooling sobre tokens
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            embeddings = embeddings.cpu().numpy()
        
        return embeddings[0] if is_single else embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud coseno entre dos textos
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return float(np.dot(emb1, emb2))
    
    def get_embedding_dim(self) -> int:
        """Retorna dimensión de embeddings (2048)"""
        return self.config['embedding_dim']


# Singleton global (se carga una sola vez)
_embedding_gemma = None

def get_embedding_model() -> EmbeddingGemma:
    """Obtiene instancia singleton de EmbeddingGemma"""
    global _embedding_gemma
    if _embedding_gemma is None:
        _embedding_gemma = EmbeddingGemma()
    return _embedding_gemma
