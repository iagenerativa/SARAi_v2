"""
Script para entrenar TRM-Classifier desde dataset sintético
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.trm_classifier import TRMClassifierDual
from core.embeddings import get_embedding_model


class TRMDataset(Dataset):
    """Dataset para entrenar TRM-Classifier (v2.11 con web_query)"""
    
    def __init__(self, data_path: Path, embedding_model):
        # Leer JSONL (una línea por ejemplo)
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        self.embedding_model = embedding_model
        
        # Pre-computar embeddings
        print("📊 Pre-computando embeddings...")
        self.embeddings = []
        for item in tqdm(self.data):
            emb = self.embedding_model.encode(item['text'])
            self.embeddings.append(emb)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        hard = torch.tensor([item['hard']], dtype=torch.float32)
        soft = torch.tensor([item['soft']], dtype=torch.float32)
        web_query = torch.tensor([item['web_query']], dtype=torch.float32)  # v2.11
        
        return embedding, hard, soft, web_query


def train_epoch(model, dataloader, optimizer, device):
    """Entrena una época (v2.11 con web_query)"""
    model.train()
    total_loss = 0.0
    
    for embeddings, hard_labels, soft_labels, web_query_labels in tqdm(dataloader, desc="Training"):
        embeddings = embeddings.to(device)
        hard_labels = hard_labels.to(device)
        soft_labels = soft_labels.to(device)
        web_query_labels = web_query_labels.to(device)  # v2.11
        
        # Forward
        batch_size = embeddings.size(0)
        x = model.input_norm(model.input_proj(embeddings))
        
        # Inicializar y, z
        y = model.y0.expand(batch_size, -1)
        z = model.z0.expand(batch_size, -1)
        
        # Ciclos recursivos
        for h in range(model.H_cycles):
            for l in range(model.L_cycles):
                y, z = model.recursive_layer(x, y, z)
        
        # Clasificación triple (v2.11)
        hard_pred = torch.sigmoid(model.head_hard(y))
        soft_pred = torch.sigmoid(model.head_soft(y))
        web_query_pred = torch.sigmoid(model.head_web_query(y))  # v2.11
        
        # Loss (BCE) con pesos balanceados
        loss_hard = nn.functional.binary_cross_entropy(hard_pred, hard_labels)
        loss_soft = nn.functional.binary_cross_entropy(soft_pred, soft_labels)
        loss_web_query = nn.functional.binary_cross_entropy(web_query_pred, web_query_labels)  # v2.11
        
        # Loss total (igual peso para las 3 cabezas)
        loss = loss_hard + loss_soft + loss_web_query
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Valida el modelo (v2.11 con web_query)"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for embeddings, hard_labels, soft_labels, web_query_labels in dataloader:
            embeddings = embeddings.to(device)
            hard_labels = hard_labels.to(device)
            soft_labels = soft_labels.to(device)
            web_query_labels = web_query_labels.to(device)  # v2.11
            
            # Forward
            batch_size = embeddings.size(0)
            x = model.input_norm(model.input_proj(embeddings))
            y = model.y0.expand(batch_size, -1)
            z = model.z0.expand(batch_size, -1)
            
            for h in range(model.H_cycles):
                for l in range(model.L_cycles):
                    y, z = model.recursive_layer(x, y, z)
            
            hard_pred = torch.sigmoid(model.head_hard(y))
            soft_pred = torch.sigmoid(model.head_soft(y))
            web_query_pred = torch.sigmoid(model.head_web_query(y))  # v2.11
            
            loss_hard = nn.functional.binary_cross_entropy(hard_pred, hard_labels)
            loss_soft = nn.functional.binary_cross_entropy(soft_pred, soft_labels)
            loss_web_query = nn.functional.binary_cross_entropy(web_query_pred, web_query_labels)  # v2.11
            loss = loss_hard + loss_soft + loss_web_query
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Entrenar TRM-Classifier")
    parser.add_argument("--data", type=str, required=True,
                       help="Ruta al dataset de entrenamiento")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--output", type=str,
                       default="models/trm_classifier/checkpoint.pth",
                       help="Ruta de salida del checkpoint")
    
    args = parser.parse_args()
    
    print("🚀 Iniciando entrenamiento de TRM-Classifier")
    print("=" * 60)
    
    # Cargar embedding model
    print("📥 Cargando EmbeddingGemma...")
    embedding_model = get_embedding_model()
    
    # Crear dataset
    print(f"📂 Cargando dataset desde {args.data}...")
    dataset = TRMDataset(Path(args.data), embedding_model)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # CPU-only
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Crear modelo
    print("🧠 Creando TRM-Classifier...")
    model = TRMClassifierDual()
    device = torch.device("cpu")
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Entrenamiento
    print("\n🎓 Comenzando entrenamiento...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nÉpoca {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(args.output)
            print(f"  ✅ Mejor modelo guardado (val_loss: {val_loss:.4f})")
    
    print("\n" + "=" * 60)
    print("✅ Entrenamiento completado")
    print(f"📁 Checkpoint final: {args.output}")
    print(f"📉 Mejor val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
