"""
TRM-Classifier Dual para SARAi v2
Clasificador de intenciones hard/soft usando Tiny Recursive Models (7M params)
"""

import torch
import torch.nn as nn
from typing import Dict
import yaml
import os


class TinyRecursiveLayer(nn.Module):
    """
    Capa recursiva básica del TRM
    Basada en la arquitectura del paper Samsung SAIL
    """
    
    def __init__(self, d_model: int, d_latent: int):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        
        # f_z: actualiza estado latente
        self.f_z = nn.Sequential(
            nn.Linear(d_model + d_model + d_latent, d_latent * 2),
            nn.ReLU(),
            nn.Linear(d_latent * 2, d_latent),
            nn.LayerNorm(d_latent)
        )
        
        # f_y: refina hipótesis
        self.f_y = nn.Sequential(
            nn.Linear(d_model + d_latent, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple:
        """
        Un paso de recursión
        
        Args:
            x: Input embedding [batch, d_model]
            y: Hipótesis actual [batch, d_model]
            z: Estado latente [batch, d_latent]
        
        Returns:
            (y_new, z_new)
        """
        # Actualizar z
        z_input = torch.cat([x, y, z], dim=-1)
        z_new = self.f_z(z_input)
        
        # Actualizar y
        y_input = torch.cat([y, z_new], dim=-1)
        y_new = self.f_y(y_input)
        
        return y_new, z_new


class TRMClassifierDual(nn.Module):
    """
    Clasificador dual hard/soft usando TRM
    
    Produce dos scores independientes (no mutuamente excluyentes):
    - hard: tareas técnicas (código, matemáticas, configuración)
    - soft: tareas emocionales (empatía, creatividad, persuasión)
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        super().__init__()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['models']['trm_classifier']
        self.d_model = self.config['d_model']
        self.d_latent = self.config['d_latent']
        self.H_cycles = self.config['H_cycles']
        self.L_cycles = self.config['L_cycles']
        
        # Proyección de EmbeddingGemma (768-D) a d_model
        self.input_proj = nn.Linear(768, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        
        # Capas recursivas
        self.recursive_layer = TinyRecursiveLayer(self.d_model, self.d_latent)
        
        # Cabezas de clasificación dual + web_query (v2.10)
        self.head_hard = nn.Linear(self.d_model, 1)
        self.head_soft = nn.Linear(self.d_model, 1)
        self.head_web_query = nn.Linear(self.d_model, 1)  # v2.10: Skill RAG
        
        # Inicialización de y₀ y z₀ (aprendibles)
        self.y0 = nn.Parameter(torch.zeros(1, self.d_model))
        self.z0 = nn.Parameter(torch.zeros(1, self.d_latent))
        
        self.device = torch.device("cpu")
        self.to(self.device)
    
    def forward(self, x_embedding: torch.Tensor) -> Dict[str, float]:
        """
        Clasifica intención del input
        
        Args:
            x_embedding: Embedding de EmbeddingGemma [batch, 768]
        
        Returns:
            {"hard": float, "soft": float, "web_query": float}
        """
        batch_size = x_embedding.size(0)
        
        # Proyectar a d_model
        x = self.input_norm(self.input_proj(x_embedding))
        
        # Inicializar y, z
        y = self.y0.expand(batch_size, -1)
        z = self.z0.expand(batch_size, -1)
        
        # Ciclos recursivos (H_cycles × L_cycles pasos)
        for h in range(self.H_cycles):
            for l in range(self.L_cycles):
                y, z = self.recursive_layer(x, y, z)
        
        # Clasificación triple (hard/soft/web_query) - v2.10
        hard_logit = self.head_hard(y)
        soft_logit = self.head_soft(y)
        web_query_logit = self.head_web_query(y)  # v2.10
        
        hard_score = torch.sigmoid(hard_logit).squeeze(-1)
        soft_score = torch.sigmoid(soft_logit).squeeze(-1)
        web_query_score = torch.sigmoid(web_query_logit).squeeze(-1)  # v2.10
        
        # Retornar dict con tensors (para training) o floats (para inference)
        if self.training:
            # Training mode: retorna tensors con grad
            return {
                "hard": hard_score,
                "soft": soft_score,
                "web_query": web_query_score
            }
        else:
            # Inference mode: retorna floats
            return {
                "hard": hard_score.item() if batch_size == 1 else hard_score.tolist(),
                "soft": soft_score.item() if batch_size == 1 else soft_score.tolist(),
                "web_query": web_query_score.item() if batch_size == 1 else web_query_score.tolist()
            }
    
    def invoke(self, embedding: torch.Tensor) -> Dict[str, float]:
        """
        Interfaz compatible con LangChain Runnable
        """
        with torch.no_grad():
            return self.forward(embedding.unsqueeze(0))
    
    def save_checkpoint(self, path: str = None):
        """Guarda checkpoint del modelo"""
        if path is None:
            path = self.config['checkpoint_path']
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config
        }, path)
        print(f"✅ Checkpoint guardado en {path}")
    
    def load_checkpoint(self, path: str = None):
        """Carga checkpoint del modelo"""
        if path is None:
            path = self.config['checkpoint_path']
        
        if not os.path.exists(path):
            print(f"⚠️  Checkpoint no encontrado: {path}")
            print("💡 Usando modelo sin entrenar (scores aleatorios)")
            return False
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"✅ Checkpoint cargado desde {path}")
        return True


def create_trm_classifier() -> TRMClassifierDual:
    """Factory para crear TRM-Classifier con checkpoint si existe"""
    classifier = TRMClassifierDual()
    classifier.load_checkpoint()
    return classifier


# Versión simulada para prototipado rápido (antes de entrenar TRM)
class TRMClassifierSimulated:
    """
    Clasificador simulado basado en keywords
    Útil para prototipado antes de entrenar el TRM real
    """
    
    def invoke(self, text: str) -> Dict[str, float]:
        """Clasificación basada en heurísticas (incluye web_query v2.10)"""
        low = text.lower()
        
        # Palabras clave técnicas
        hard_keywords = [
            "código", "error", "bug", "configurar", "instalar", "ssh", "linux",
            "python", "javascript", "servidor", "api", "base de datos", "sql",
            "git", "docker", "algoritmo", "función", "clase", "variable"
        ]
        
        # Palabras clave emocionales
        soft_keywords = [
            "triste", "feliz", "frustrado", "gracias", "ayuda", "emocionado",
            "preocupado", "cansado", "motivado", "perdido", "confundido",
            "siento", "explicar como", "entender", "difícil de", "no sé"
        ]
        
        # Palabras clave de búsqueda web (v2.10)
        web_keywords = [
            "quién ganó", "quién es", "cuándo fue", "dónde está", "cómo está",
            "clima en", "weather in", "precio de", "noticias de", "resultados del",
            "últimas noticias", "qué pasó", "oscar", "copa del mundo", "bitcoin",
            "stock price", "hoy", "today", "ahora", "now", "actual", "current"
        ]
        
        hard_count = sum(1 for kw in hard_keywords if kw in low)
        soft_count = sum(1 for kw in soft_keywords if kw in low)
        web_count = sum(1 for kw in web_keywords if kw in low)
        
        # Normalizar a [0, 1]
        hard = min(hard_count / 3.0, 1.0) if hard_count > 0 else 0.2
        soft = min(soft_count / 3.0, 1.0) if soft_count > 0 else 0.2
        web_query = min(web_count / 2.0, 1.0) if web_count > 0 else 0.1  # v2.10
        
        return {"hard": hard, "soft": soft, "web_query": web_query}
