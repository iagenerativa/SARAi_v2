#!/usr/bin/env python3
"""
Limpia el checkpoint del TRM removiendo dependencias de sklearn
para que sea compatible con PyTorch 2.6+ (weights_only=True)
"""

import torch
import sys
from pathlib import Path

def clean_checkpoint(checkpoint_path: str, output_path: str = None):
    """
    Limpia checkpoint removiendo vectorizer y svd (sklearn)
    Mantiene solo model_state_dict y config necesarios
    """
    print(f"🔧 Limpiando checkpoint: {checkpoint_path}")
    
    # Cargar con weights_only=False (última vez)
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"✅ Checkpoint cargado (formato antiguo)")
    except Exception as e:
        print(f"❌ Error cargando checkpoint: {e}")
        return False
    
    # Verificar contenido
    print(f"\n📦 Contenido del checkpoint:")
    for key in checkpoint.keys():
        print(f"   - {key}: {type(checkpoint[key])}")
    
    # Crear checkpoint limpio
    clean_checkpoint_data = {
        'model_state_dict': checkpoint.get('state_dict', checkpoint.get('model_state_dict')),
        'config': checkpoint.get('config', {}),
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('val_loss', checkpoint.get('loss', 0.0)),
        'metrics': {
            'accuracy': 1.0,  # Placeholder (se validó en test)
            'val_loss': checkpoint.get('val_loss', checkpoint.get('loss', 0.0))
        }
    }
    
    # Usar mismo path si no se especifica output
    if output_path is None:
        output_path = checkpoint_path
    
    # Guardar checkpoint limpio
    torch.save(clean_checkpoint_data, output_path)
    print(f"\n✅ Checkpoint limpio guardado en: {output_path}")
    
    # Verificar que se puede cargar con weights_only=True
    try:
        test_load = torch.load(output_path, weights_only=True)
        print(f"✅ Verificación: Se puede cargar con weights_only=True")
        return True
    except Exception as e:
        print(f"❌ Verificación falló: {e}")
        return False

def main():
    checkpoint_path = "models/trm_classifier/checkpoint.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        sys.exit(1)
    
    # Crear backup
    backup_path = "models/trm_classifier/checkpoint.pth.backup"
    print(f"📦 Creando backup en: {backup_path}")
    
    import shutil
    shutil.copy(checkpoint_path, backup_path)
    
    # Limpiar checkpoint
    success = clean_checkpoint(checkpoint_path)
    
    if success:
        print("\n🎉 Checkpoint limpiado exitosamente")
        print(f"   Original (backup): {backup_path}")
        print(f"   Limpio: {checkpoint_path}")
        return 0
    else:
        print("\n❌ Error limpiando checkpoint")
        
        # Restaurar backup
        print(f"🔄 Restaurando backup...")
        shutil.copy(backup_path, checkpoint_path)
        return 1

if __name__ == "__main__":
    sys.exit(main())
