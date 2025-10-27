#!/usr/bin/env python3
"""
Entrena TRM-Classifier usando embeddings pre-computados de EmbeddingGemma (768-D)
Versión optimizada que NO requiere recomputar embeddings en cada época.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.trm_classifier import TRMClassifierDual


class EmbeddingsDataset(Dataset):
    """Dataset de embeddings pre-computados"""
    
    def __init__(self, embeddings_path: str):
        print(f"📂 Cargando embeddings desde: {embeddings_path}")
        
        data = np.load(embeddings_path)
        
        self.embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
        self.labels_hard = torch.tensor(data['labels_hard'], dtype=torch.float32)
        self.labels_soft = torch.tensor(data['labels_soft'], dtype=torch.float32)
        self.labels_web = torch.tensor(data['labels_web'], dtype=torch.float32)
        
        print(f"✅ Cargados {len(self.embeddings)} ejemplos")
        print(f"   Embedding dim: {self.embeddings.shape[1]}-D")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'hard': self.labels_hard[idx],
            'soft': self.labels_soft[idx],
            'web_query': self.labels_web[idx]
        }


def train_epoch(model, dataloader, optimizer, device):
    """Entrena una época"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        embeddings = batch['embedding'].to(device)
        hard_labels = batch['hard'].to(device)
        soft_labels = batch['soft'].to(device)
        web_labels = batch['web_query'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (modelo en train mode retorna tensors con grad)
        scores = model.forward(embeddings)
        
        hard_pred = scores['hard']
        soft_pred = scores['soft']
        web_pred = scores['web_query']
        
        # Loss (BCE para clasificación binaria multi-label)
        loss_hard = nn.functional.binary_cross_entropy(hard_pred, hard_labels)
        loss_soft = nn.functional.binary_cross_entropy(soft_pred, soft_labels)
        loss_web = nn.functional.binary_cross_entropy(web_pred, web_labels)
        
        loss = loss_hard + loss_soft + loss_web
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Valida el modelo"""
    model.eval()
    total_loss = 0
    correct_hard = 0
    correct_soft = 0
    correct_web = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            embeddings = batch['embedding'].to(device)
            hard_labels = batch['hard'].to(device)
            soft_labels = batch['soft'].to(device)
            web_labels = batch['web_query'].to(device)
            
            batch_size = embeddings.size(0)
            
            # Forward en eval mode (retorna floats pero necesitamos tensors para loss)
            # Temporalmente activar train mode solo para get tensors
            model.train()
            scores = model.forward(embeddings)
            model.eval()
            
            hard_pred = scores['hard']
            soft_pred = scores['soft']
            web_pred = scores['web_query']
            
            # Loss
            loss_hard = nn.functional.binary_cross_entropy(hard_pred, hard_labels)
            loss_soft = nn.functional.binary_cross_entropy(soft_pred, soft_labels)
            loss_web = nn.functional.binary_cross_entropy(web_pred, web_labels)
            
            total_loss += (loss_hard + loss_soft + loss_web).item()
            
            # Accuracy (umbral 0.5)
            correct_hard += ((hard_pred > 0.5) == (hard_labels > 0.5)).sum().item()
            correct_soft += ((soft_pred > 0.5) == (soft_labels > 0.5)).sum().item()
            correct_web += ((web_pred > 0.5) == (web_labels > 0.5)).sum().item()
            total += batch_size
    
    avg_loss = total_loss / len(dataloader)
    acc_hard = correct_hard / total
    acc_soft = correct_soft / total
    acc_web = correct_web / total
    
    return avg_loss, acc_hard, acc_soft, acc_web


def main():
    parser = argparse.ArgumentParser(description="Entrenar TRM con embeddings pre-computados")
    parser.add_argument("--data", type=str, default="data/trm_training_embeddings.npz",
                       help="Ruta al dataset .npz")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--output", type=str, default="models/trm_classifier/checkpoint.pth",
                       help="Ruta de salida del checkpoint")
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 ENTRENAMIENTO TRM CON EMBEDDINGGEMMA (768-D)")
    print("="*80)
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Cargar dataset
    dataset = EmbeddingsDataset(args.data)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\n✅ Train: {train_size} ejemplos")
    print(f"✅ Val: {val_size} ejemplos")
    
    # Crear modelo
    print("\n🧠 Creando TRM-Classifier (768-D input)...")
    model = TRMClassifierDual()
    
    # Verificar que input_proj sea 768→256
    input_features = model.input_proj.in_features
    print(f"✅ TRM configurado para embeddings de {input_features}-D")
    
    if input_features != 768:
        print(f"❌ ERROR: TRM espera {input_features}-D pero dataset es 768-D")
        print("   Actualiza core/trm_classifier.py línea ~88:")
        print("   self.input_proj = nn.Linear(768, self.d_model)")
        return 1
    
    device = torch.device("cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Entrenamiento
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("\n🎓 COMENZANDO ENTRENAMIENTO")
    print("="*80)
    
    for epoch in range(args.epochs):
        print(f"\nÉPOCA {epoch + 1}/{args.epochs}")
        print("-"*80)
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, acc_hard, acc_soft, acc_web = validate(model, val_loader, device)
        
        print(f"\n📊 Resultados:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Acc (hard):      {acc_hard:.3f}")
        print(f"   Acc (soft):      {acc_soft:.3f}")
        print(f"   Acc (web_query): {acc_web:.3f}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': model.config,
                'epoch': epoch + 1,
                'loss': val_loss,
                'metrics': {
                    'accuracy': (acc_hard + acc_soft + acc_web) / 3,
                    'val_loss': val_loss
                }
            }
            
            torch.save(checkpoint, str(output_path))
            print(f"   ✅ Mejor modelo guardado (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   ⏳ Paciencia: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping en época {epoch + 1}")
            break
    
    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"📁 Checkpoint: {args.output}")
    print(f"📉 Mejor val_loss: {best_val_loss:.4f}")
    print(f"🎯 Accuracy promedio: {(acc_hard + acc_soft + acc_web) / 3:.3f}")
    print("="*80)


if __name__ == "__main__":
    sys.exit(main())
