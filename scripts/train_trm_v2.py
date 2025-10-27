#!/usr/bin/env python3
"""
scripts/train_trm_v2.py - Entrenamiento TRM-Classifier v2.11 con web_query

Copyright (c) 2025 Noel
Licencia: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
No se permite uso comercial sin permiso del autor.

---

Versión simplificada que usa embeddings TF-IDF en lugar de EmbeddingGemma
para evitar dependencias pesadas durante el entrenamiento.
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.trm_classifier import TRMClassifierDual


class SimpleTRMDataset(Dataset):
    """Dataset para TRM-Classifier con embeddings TF-IDF proyectados a 2048-D"""
    
    def __init__(self, data_path: Path, vectorizer=None, svd=None):
        # Leer JSONL
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"📊 Cargados {len(self.data)} ejemplos")
        
        # Crear embeddings TF-IDF
        print("🔧 Generando embeddings TF-IDF...")
        texts = [item['text'] for item in self.data]
        
        # Si no hay vectorizador, crear uno nuevo
        if vectorizer is None:
            print("   Creando nuevo vectorizador TF-IDF...")
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        else:
            print("   Usando vectorizador existente...")
            self.vectorizer = vectorizer
            tfidf_matrix = self.vectorizer.transform(texts).toarray()
        
        print(f"   TF-IDF shape: {tfidf_matrix.shape}")
        
        # Si tenemos menos de 2048 features, hacer padding con zeros
        if tfidf_matrix.shape[1] < 2048:
            padding = np.zeros((tfidf_matrix.shape[0], 2048 - tfidf_matrix.shape[1]))
            embeddings_2048 = np.hstack([tfidf_matrix, padding])
            print(f"   Padding aplicado: {tfidf_matrix.shape[1]} → 2048 dims")
            self.svd = None
        else:
            # Si tenemos más, usar SVD para reducir
            if svd is None:
                print("   Creando nuevo SVD...")
                self.svd = TruncatedSVD(n_components=2048, random_state=42)
                embeddings_2048 = self.svd.fit_transform(tfidf_matrix)
            else:
                print("   Usando SVD existente...")
                self.svd = svd
                embeddings_2048 = self.svd.transform(tfidf_matrix)
            print(f"   SVD aplicado: {tfidf_matrix.shape[1]} → 2048 dims")
        
        # Normalizar
        norms = np.linalg.norm(embeddings_2048, axis=1, keepdims=True)
        self.embeddings = embeddings_2048 / (norms + 1e-8)
        
        print(f"✅ Embeddings generados: {self.embeddings.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        hard = torch.tensor([item['hard']], dtype=torch.float32)
        soft = torch.tensor([item['soft']], dtype=torch.float32)
        web_query = torch.tensor([item['web_query']], dtype=torch.float32)
        
        return embedding, hard, soft, web_query


def train_epoch(model, dataloader, optimizer, device):
    """Entrena una época"""
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for embeddings, hard_labels, soft_labels, web_query_labels in progress_bar:
        embeddings = embeddings.to(device)
        hard_labels = hard_labels.to(device)
        soft_labels = soft_labels.to(device)
        web_query_labels = web_query_labels.to(device)
        
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
        
        # Clasificación triple
        hard_pred = torch.sigmoid(model.head_hard(y))
        soft_pred = torch.sigmoid(model.head_soft(y))
        web_query_pred = torch.sigmoid(model.head_web_query(y))
        
        # Loss (BCE) con pesos balanceados
        loss_hard = nn.functional.binary_cross_entropy(hard_pred, hard_labels)
        loss_soft = nn.functional.binary_cross_entropy(soft_pred, soft_labels)
        loss_web_query = nn.functional.binary_cross_entropy(web_query_pred, web_query_labels)
        
        # Loss total
        loss = loss_hard + loss_soft + loss_web_query
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Valida el modelo"""
    model.eval()
    total_loss = 0.0
    total_correct_hard = 0
    total_correct_soft = 0
    total_correct_web = 0
    total_samples = 0
    
    with torch.no_grad():
        for embeddings, hard_labels, soft_labels, web_query_labels in dataloader:
            embeddings = embeddings.to(device)
            hard_labels = hard_labels.to(device)
            soft_labels = soft_labels.to(device)
            web_query_labels = web_query_labels.to(device)
            
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
            web_query_pred = torch.sigmoid(model.head_web_query(y))
            
            # Loss
            loss_hard = nn.functional.binary_cross_entropy(hard_pred, hard_labels)
            loss_soft = nn.functional.binary_cross_entropy(soft_pred, soft_labels)
            loss_web_query = nn.functional.binary_cross_entropy(web_query_pred, web_query_labels)
            loss = loss_hard + loss_soft + loss_web_query
            
            total_loss += loss.item()
            
            # Accuracy (umbral 0.5)
            hard_correct = ((hard_pred > 0.5) == (hard_labels > 0.5)).sum().item()
            soft_correct = ((soft_pred > 0.5) == (soft_labels > 0.5)).sum().item()
            web_correct = ((web_query_pred > 0.5) == (web_query_labels > 0.5)).sum().item()
            
            total_correct_hard += hard_correct
            total_correct_soft += soft_correct
            total_correct_web += web_correct
            total_samples += batch_size
    
    avg_loss = total_loss / len(dataloader)
    acc_hard = total_correct_hard / total_samples
    acc_soft = total_correct_soft / total_samples
    acc_web = total_correct_web / total_samples
    
    return avg_loss, acc_hard, acc_soft, acc_web


def main():
    parser = argparse.ArgumentParser(description="Entrenar TRM-Classifier v2.11")
    parser.add_argument("--data", type=str, default="data/trm_training.jsonl",
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
    
    print("=" * 80)
    print("🚀 ENTRENAMIENTO TRM-CLASSIFIER v2.11 CON WEB_QUERY HEAD")
    print("=" * 80)
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Crear dataset
    print(f"\n📂 Cargando dataset...")
    dataset = SimpleTRMDataset(Path(args.data))
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Train: {len(train_dataset)} ejemplos")
    print(f"✅ Val: {len(val_dataset)} ejemplos")
    
    # Crear modelo
    print("\n🧠 Creando TRM-Classifier...")
    model = TRMClassifierDual()
    device = torch.device("cpu")
    model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Parámetros totales: {total_params:,}")
    print(f"📊 Parámetros entrenables: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Entrenamiento
    print("\n" + "=" * 80)
    print("🎓 COMENZANDO ENTRENAMIENTO")
    print("=" * 80)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"ÉPOCA {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        # Entrenamiento
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validación
        val_loss, acc_hard, acc_soft, acc_web = validate(model, val_loader, device)
        
        print(f"\n📊 Resultados Época {epoch + 1}:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Val Acc (hard):      {acc_hard:.3f}")
        print(f"   Val Acc (soft):      {acc_soft:.3f}")
        print(f"   Val Acc (web_query): {acc_web:.3f}")
        
        # Guardar mejor modelo CON vectorizador (v2.11 CRÍTICO)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Crear directorio si no existe
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar checkpoint con vectorizador y SVD
            checkpoint = {
                'state_dict': model.state_dict(),
                'config': model.config,
                'vectorizer': dataset.vectorizer,  # CRÍTICO para test
                'svd': getattr(dataset, 'svd', None),  # Puede ser None si padding
                'val_loss': val_loss,
                'epoch': epoch + 1
            }
            torch.save(checkpoint, str(output_path))
            
            print(f"   ✅ Mejor modelo guardado (val_loss: {val_loss:.4f})")
            print(f"      Incluye: state_dict + vectorizer + svd")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   ⏳ Paciencia: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping en época {epoch + 1}")
            break
    
    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"📁 Checkpoint final: {args.output}")
    print(f"📉 Mejor val_loss: {best_val_loss:.4f}")
    print(f"🎯 Accuracy hard: {acc_hard:.3f}")
    print(f"🎯 Accuracy soft: {acc_soft:.3f}")
    print(f"🎯 Accuracy web_query: {acc_web:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
