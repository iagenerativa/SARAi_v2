"""
SARAi v2.17 - LoRA Router
Clasificador con LoRA que decide: TRM (cache) vs LLM (generación) vs Traducir
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import time


class LoRARouter(nn.Module):
    """
    Router con LoRA (Low-Rank Adaptation) para clasificación eficiente
    
    Input: BERT embeddings (768-dim)
    Output: Decisión [TRM, LLM, Traducir] + confianza
    
    Arquitectura:
        BERT emb (768) → LoRA Linear (768 → 256) → ReLU →
        → LoRA Linear (256 → 128) → ReLU →
        → Output (128 → 3) → Softmax
    
    LoRA permite ajuste fino con solo ~2M params adicionales
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # BERT embedding dim
        hidden_dim: int = 256,
        lora_rank: int = 16,   # Rank de LoRA (menor = más eficiente)
        lora_alpha: int = 32,  # Scaling factor
        num_classes: int = 3   # TRM, LLM, Traducir
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_classes = num_classes
        
        # Capa 1: Input → Hidden (con LoRA)
        self.fc1_base = nn.Linear(input_dim, hidden_dim)
        self.fc1_lora_A = nn.Linear(input_dim, lora_rank, bias=False)
        self.fc1_lora_B = nn.Linear(lora_rank, hidden_dim, bias=False)
        
        # Capa 2: Hidden → Hidden/2 (con LoRA)
        self.fc2_base = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2_lora_A = nn.Linear(hidden_dim, lora_rank, bias=False)
        self.fc2_lora_B = nn.Linear(lora_rank, hidden_dim // 2, bias=False)
        
        # Capa 3: Output (sin LoRA, es pequeña)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Activaciones
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Scaling de LoRA
        self.lora_scaling = lora_alpha / lora_rank
        
        # Inicializar pesos LoRA
        self._init_lora_weights()
        
        # Congelar capas base (solo entrenar LoRA)
        self.freeze_base_layers()
    
    def _init_lora_weights(self):
        """Inicializa pesos LoRA (Gaussian para A, zeros para B)"""
        nn.init.kaiming_uniform_(self.fc1_lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.fc1_lora_B.weight)
        
        nn.init.kaiming_uniform_(self.fc2_lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.fc2_lora_B.weight)
    
    def freeze_base_layers(self):
        """Congela capas base (solo entrena LoRA adapters)"""
        for param in self.fc1_base.parameters():
            param.requires_grad = False
        for param in self.fc2_base.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Descongela todas las capas (fine-tuning completo)"""
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con LoRA
        
        Args:
            x: BERT embeddings (batch_size, 768)
        
        Returns:
            Logits (batch_size, 3) - [TRM, LLM, Traducir]
        """
        # Capa 1 con LoRA
        h1_base = self.fc1_base(x)
        h1_lora = self.fc1_lora_B(self.fc1_lora_A(x)) * self.lora_scaling
        h1 = self.relu(h1_base + h1_lora)
        h1 = self.dropout(h1)
        
        # Capa 2 con LoRA
        h2_base = self.fc2_base(h1)
        h2_lora = self.fc2_lora_B(self.fc2_lora_A(h1)) * self.lora_scaling
        h2 = self.relu(h2_base + h2_lora)
        h2 = self.dropout(h2)
        
        # Capa de salida
        logits = self.fc3(h2)
        
        return logits
    
    def predict(self, embedding: np.ndarray) -> Dict:
        """
        Predice clase y confianza
        
        Args:
            embedding: BERT embedding (768,)
        
        Returns:
            {
                "decision": str,        # "TRM", "LLM", "Traducir"
                "confidence": float,    # [0, 1]
                "scores": dict          # {"TRM": 0.1, "LLM": 0.85, "Traducir": 0.05}
            }
        """
        self.eval()
        
        with torch.no_grad():
            # A tensor
            x = torch.from_numpy(embedding).float().unsqueeze(0)
            
            # Forward
            logits = self.forward(x)
            
            # Softmax para probabilidades
            probs = torch.softmax(logits, dim=1)[0].numpy()
            
            # Decisión
            class_idx = int(np.argmax(probs))
            class_names = ["TRM", "LLM", "Traducir"]
            decision = class_names[class_idx]
            confidence = float(probs[class_idx])
            
            # Scores detallados
            scores = {
                "TRM": float(probs[0]),
                "LLM": float(probs[1]),
                "Traducir": float(probs[2])
            }
            
            return {
                "decision": decision,
                "confidence": confidence,
                "scores": scores
            }
    
    def get_trainable_params(self) -> int:
        """Retorna número de parámetros entrenables (solo LoRA)"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Retorna número total de parámetros"""
        return sum(p.numel() for p in self.parameters())
    
    def save(self, path: str):
        """Guarda modelo completo"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'lora_rank': self.lora_rank,
                'lora_alpha': self.lora_alpha,
                'num_classes': self.num_classes
            }
        }, path)
        print(f"✅ LoRA Router guardado en {path}")
    
    @classmethod
    def load(cls, path: str):
        """Carga modelo desde archivo"""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Crear modelo con config guardada
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        
        print(f"✅ LoRA Router cargado desde {path}")
        return model


# ============ Training Utils ============

def train_lora_router(
    router: LoRARouter,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32
):
    """
    Entrena LoRA Router
    
    Args:
        router: Modelo LoRARouter
        train_embeddings: Embeddings BERT (N, 768)
        train_labels: Labels (N,) - 0=TRM, 1=LLM, 2=Traducir
        epochs: Número de épocas
        lr: Learning rate
        batch_size: Tamaño de batch
    """
    router.train()
    
    # Optimizer (solo params LoRA)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, router.parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    
    # Convertir a tensores
    X = torch.from_numpy(train_embeddings).float()
    y = torch.from_numpy(train_labels).long()
    
    print(f"\n🏋️ Entrenando LoRA Router...")
    print(f"  Samples: {len(X)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Params entrenables: {router.get_trainable_params():,}")
    print(f"  Params totales: {router.get_total_params():,}")
    print()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Shuffle
        indices = torch.randperm(len(X))
        
        # Batches
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Forward
            logits = router(batch_X)
            loss = criterion(logits, batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Métricas
            epoch_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        # Accuracy
        accuracy = correct / total
        avg_loss = epoch_loss / (len(X) // batch_size)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {accuracy:.3f}")
    
    print(f"\n✅ Entrenamiento completado")
    router.eval()


# ============ Test ============
if __name__ == "__main__":
    print("=== Test LoRA Router ===\n")
    
    # Crear router
    router = LoRARouter(
        input_dim=768,
        hidden_dim=256,
        lora_rank=16,
        lora_alpha=32
    )
    
    print(f"Parámetros entrenables (LoRA): {router.get_trainable_params():,}")
    print(f"Parámetros totales: {router.get_total_params():,}")
    print(f"Ratio LoRA/Total: {router.get_trainable_params()/router.get_total_params()*100:.1f}%")
    
    # Test forward
    print("\n1️⃣ Test forward pass:")
    dummy_emb = np.random.randn(768).astype(np.float32)
    result = router.predict(dummy_emb)
    
    print(f"Decisión: {result['decision']}")
    print(f"Confianza: {result['confidence']:.3f}")
    print(f"Scores: {result['scores']}")
    
    # Test training con datos sintéticos
    print("\n2️⃣ Test entrenamiento con datos sintéticos:")
    
    # Generar datos sintéticos
    np.random.seed(42)
    n_samples = 300
    
    # TRM: embeddings con patrón específico (primeros 100)
    trm_embs = np.random.randn(100, 768) + np.array([1.0] * 384 + [-1.0] * 384)
    trm_labels = np.zeros(100, dtype=np.int64)
    
    # LLM: embeddings diferentes (siguientes 100)
    llm_embs = np.random.randn(100, 768) + np.array([-1.0] * 384 + [1.0] * 384)
    llm_labels = np.ones(100, dtype=np.int64)
    
    # Traducir: otro patrón (últimos 100)
    trans_embs = np.random.randn(100, 768) + np.array([0.5] * 256 + [-0.5] * 256 + [0.0] * 256)
    trans_labels = np.full(100, 2, dtype=np.int64)
    
    # Combinar
    train_embs = np.vstack([trm_embs, llm_embs, trans_embs])
    train_labels = np.concatenate([trm_labels, llm_labels, trans_labels])
    
    # Entrenar
    train_lora_router(router, train_embs, train_labels, epochs=30, batch_size=16)
    
    # Test predicción
    print("\n3️⃣ Test predicción post-entrenamiento:")
    test_trm = trm_embs[0]
    test_llm = llm_embs[0]
    test_trans = trans_embs[0]
    
    result_trm = router.predict(test_trm)
    result_llm = router.predict(test_llm)
    result_trans = router.predict(test_trans)
    
    print(f"Test TRM: {result_trm['decision']} (conf: {result_trm['confidence']:.3f})")
    print(f"Test LLM: {result_llm['decision']} (conf: {result_llm['confidence']:.3f})")
    print(f"Test Traducir: {result_trans['decision']} (conf: {result_trans['confidence']:.3f})")
    
    # Guardar modelo
    print("\n4️⃣ Test guardar/cargar:")
    router.save("models/lora_router_test.pt")
    router_loaded = LoRARouter.load("models/lora_router_test.pt")
    
    # Verificar que funciona igual
    result_loaded = router_loaded.predict(test_trm)
    print(f"Modelo cargado - Decisión: {result_loaded['decision']} (conf: {result_loaded['confidence']:.3f})")
    
    print("\n✅ Tests completados")
