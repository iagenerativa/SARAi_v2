"""
TRM-Mini v2.3: Clasificador ligero para prefetching proactivo
Destilado del TRM-Router (7M → 3.5M params)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict


class TinyRecursiveLayerMini(nn.Module):
    """
    Versión ligera de la capa recursiva TRM
    d_model reducido: 256 → 128
    """
    
    def __init__(self, d_model: int = 128):
        super().__init__()
        
        # f_z: actualiza estado latente
        self.f_z = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # Concatena x, y, z
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # f_y: actualiza estado visible
        self.f_y = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Concatena y, z
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple:
        """
        Un paso recursivo
        
        Args:
            x: Input embedding (d_model)
            y: Estado visible (d_model)
            z: Estado latente (d_model)
        
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


class TRMMini(nn.Module):
    """
    TRM-Mini: Clasificador ligero (3.5M params, d=128, K=2)
    Entrenado por distilación del TRM-Router original
    
    Uso: Clasificación rápida en input parcial para prefetching
    """
    
    def __init__(self, d_model: int = 128, K_cycles: int = 2):
        super().__init__()
        
        self.d_model = d_model
        self.K_cycles = K_cycles
        
        # Proyección desde embeddings (2048-D → 128-D)
        self.projection = nn.Linear(2048, d_model)
        
        # Capa recursiva compartida
        self.recursive_layer = TinyRecursiveLayerMini(d_model)
        
        # Cabezas de clasificación
        self.head_hard = nn.Linear(d_model, 1)
        self.head_soft = nn.Linear(d_model, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Clasificación recursiva
        
        Args:
            x: Embedding de entrada [batch, 2048]
        
        Returns:
            Dict con logits: {"hard": Tensor, "soft": Tensor}
        """
        # Proyectar a espacio reducido
        x = self.projection(x)  # [batch, 128]
        
        # Inicializar estados
        batch_size = x.shape[0]
        y = torch.zeros(batch_size, self.d_model, device=x.device)
        z = torch.zeros(batch_size, self.d_model, device=x.device)
        
        # K ciclos recursivos (K=2 para TRM-Mini)
        for _ in range(self.K_cycles):
            y, z = self.recursive_layer(x, y, z)
        
        # Clasificación final
        hard_logit = self.head_hard(y)  # [batch, 1]
        soft_logit = self.head_soft(y)  # [batch, 1]
        
        return {
            "hard": hard_logit.squeeze(-1),
            "soft": soft_logit.squeeze(-1)
        }
    
    def invoke(self, embedding: torch.Tensor) -> Dict[str, float]:
        """
        Interfaz compatible con TRM-Router
        
        Args:
            embedding: Embedding 2048-D (puede ser numpy array)
        
        Returns:
            Dict: {"hard": float, "soft": float} con probabilidades
        """
        with torch.no_grad():
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)  # [1, 2048]
            
            logits = self.forward(embedding)
            
            return {
                "hard": torch.sigmoid(logits["hard"]).item(),
                "soft": torch.sigmoid(logits["soft"]).item()
            }
    
    def save(self, path: str):
        """Guarda checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'd_model': self.d_model,
            'K_cycles': self.K_cycles
        }, path)
        print(f"[TRM-Mini] Checkpoint guardado: {path}")
    
    @classmethod
    def load(cls, path: str):
        """Carga checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        
        model = cls(
            d_model=checkpoint.get('d_model', 128),
            K_cycles=checkpoint.get('K_cycles', 2)
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        print(f"[TRM-Mini] Checkpoint cargado: {path}")
        return model


# Singleton global
_trm_mini_instance = None


def get_trm_mini(model_path: str = "models/trm_mini/trm_mini.pt") -> TRMMini:
    """
    Singleton: retorna instancia única del TRM-Mini
    
    Args:
        model_path: Ruta al checkpoint
    
    Returns:
        TRM-Mini cargado
    """
    global _trm_mini_instance
    
    if _trm_mini_instance is None:
        if Path(model_path).exists():
            _trm_mini_instance = TRMMini.load(model_path)
        else:
            print(f"⚠️ TRM-Mini no encontrado en {model_path}, creando nuevo modelo")
            _trm_mini_instance = TRMMini()
    
    return _trm_mini_instance
