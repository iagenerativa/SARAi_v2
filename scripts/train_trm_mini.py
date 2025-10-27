"""
Script de entrenamiento para TRM-Mini v2.3
Distilación por KL Divergence del TRM-Router (7M → 3.5M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.trm_mini import TRMMini
from core.trm_classifier import TRMClassifier, get_trm_classifier
from core.embeddings import get_embedding_model


class DistillationDataset(Dataset):
    """
    Dataset para distilación:
    - Lee logs de feedback
    - Genera embeddings de cada input
    - Guarda labels (hard/soft) del TRM-Router original
    """
    
    def __init__(self, log_path: str, embedder, teacher_trm):
        self.embedder = embedder
        self.teacher = teacher_trm
        
        # Cargar datos desde logs
        self.samples = []
        
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    self.samples.append(entry)
                except:
                    continue
        
        print(f"[Dataset] Cargadas {len(self.samples)} muestras desde {log_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        entry = self.samples[idx]
        
        # Embedding del input
        text = entry.get('input', '')
        emb = self.embedder.encode(text)
        
        # Labels del teacher (TRM-Router)
        hard_score = entry.get('hard', 0.5)
        soft_score = entry.get('soft', 0.5)
        
        return {
            'embedding': torch.tensor(emb, dtype=torch.float32),
            'hard_label': torch.tensor(hard_score, dtype=torch.float32),
            'soft_label': torch.tensor(soft_score, dtype=torch.float32)
        }


def train_trm_mini(args):
    """
    Entrenamiento por distilación con KL Divergence
    """
    print("=" * 60)
    print("TRM-Mini Training: Distillation from TRM-Router")
    print("=" * 60)
    
    # Cargar embedder y teacher
    print("\n[1/5] Cargando embedder y TRM-Router (teacher)...")
    embedder = get_embedding_model()
    teacher = get_trm_classifier()
    teacher.eval()  # Teacher en modo evaluación
    
    # Crear student (TRM-Mini)
    print("[2/5] Inicializando TRM-Mini (student)...")
    student = TRMMini(d_model=args.d_model, K_cycles=args.K_cycles)
    
    # Dataset y DataLoader
    print(f"[3/5] Cargando dataset desde {args.log_path}...")
    dataset = DistillationDataset(args.log_path, embedder, teacher)
    
    if len(dataset) < args.min_samples:
        raise ValueError(
            f"Dataset muy pequeño ({len(dataset)} muestras). "
            f"Mínimo requerido: {args.min_samples}"
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # CPU-only, evitar overhead
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Entrenamiento
    print(f"[4/5] Entrenando {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        student.train()
        
        total_loss = 0
        total_kl_loss = 0
        total_mse_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            embeddings = batch['embedding']  # [batch, 2048]
            hard_labels = batch['hard_label']  # [batch]
            soft_labels = batch['soft_label']  # [batch]
            
            # Forward student
            student_logits = student(embeddings)
            student_hard = student_logits['hard']
            student_soft = student_logits['soft']
            
            # Probabilidades
            student_hard_prob = torch.sigmoid(student_hard)
            student_soft_prob = torch.sigmoid(student_soft)
            
            # Loss 1: MSE con labels del dataset (ground truth)
            mse_loss = (
                F.mse_loss(student_hard_prob, hard_labels) +
                F.mse_loss(student_soft_prob, soft_labels)
            )
            
            # Loss 2: KL Divergence con teacher (distilación)
            # Esto suaviza el aprendizaje usando las distribuciones del teacher
            with torch.no_grad():
                teacher_logits = teacher.forward_batch(embeddings)
                teacher_hard_prob = torch.sigmoid(teacher_logits['hard'])
                teacher_soft_prob = torch.sigmoid(teacher_logits['soft'])
            
            # KL(P_teacher || P_student)
            kl_loss = (
                F.kl_div(
                    torch.log(student_hard_prob + 1e-8),
                    teacher_hard_prob,
                    reduction='batchmean'
                ) +
                F.kl_div(
                    torch.log(student_soft_prob + 1e-8),
                    teacher_soft_prob,
                    reduction='batchmean'
                )
            )
            
            # Loss combinado
            loss = args.alpha_mse * mse_loss + args.alpha_kl * kl_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_mse_loss += mse_loss.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{mse_loss.item():.4f}",
                'kl': f"{kl_loss.item():.4f}"
            })
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_kl = total_kl_loss / len(dataloader)
        avg_mse = total_mse_loss / len(dataloader)
        
        print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | KL: {avg_kl:.4f}")
        
        # Guardar checkpoint cada 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(args.output_dir) / f"trm_mini_epoch{epoch+1}.pt"
            student.save(str(checkpoint_path))
            print(f"✅ Checkpoint guardado: {checkpoint_path}")
    
    # Guardar modelo final
    print(f"\n[5/5] Guardando modelo final...")
    final_path = Path(args.output_dir) / "trm_mini.pt"
    student.save(str(final_path))
    
    print(f"\n{'='*60}")
    print(f"✅ Entrenamiento completado!")
    print(f"Modelo guardado en: {final_path}")
    print(f"Parámetros: {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar TRM-Mini por distilación")
    
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/feedback_log.jsonl",
        help="Ruta al archivo de logs de feedback"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/trm_mini",
        help="Directorio de salida para checkpoints"
    )
    
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Dimensión del modelo TRM-Mini"
    )
    
    parser.add_argument(
        "--K-cycles",
        type=int,
        default=2,
        help="Número de ciclos recursivos"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--alpha-mse",
        type=float,
        default=0.5,
        help="Peso del MSE loss"
    )
    
    parser.add_argument(
        "--alpha-kl",
        type=float,
        default=0.5,
        help="Peso del KL divergence loss"
    )
    
    parser.add_argument(
        "--min-samples",
        type=int,
        default=500,
        help="Mínimo de muestras requeridas"
    )
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train_trm_mini(args)
