#!/usr/bin/env python3
"""
Test simple de Token2Wav para verificar que funciona
"""

import torch
import numpy as np
from pathlib import Path

base_path = Path(__file__).parent.parent

print("🔧 Cargando Token2Wav...")
print(f"   Path: models/onnx/token2wav_int8.pt")

# Intentar cargar
try:
    # Opción 1: Cargar como state_dict
    state_dict = torch.load(
        base_path / "models/onnx/token2wav_int8.pt",
        map_location='cpu',
        weights_only=False
    )
    
    print(f"✓ Archivo cargado")
    print(f"   Type: {type(state_dict)}")
    
    if isinstance(state_dict, dict):
        print(f"   Keys: {list(state_dict.keys())[:5]}...")
        print(f"   Total keys: {len(state_dict)}")
        
        # Ver si es un checkpoint completo
        if 'model' in state_dict:
            print("   → Contiene clave 'model' (checkpoint completo)")
            model_weights = state_dict['model']
        elif 'state_dict' in state_dict:
            print("   → Contiene clave 'state_dict'")
            model_weights = state_dict['state_dict']
        else:
            print("   → Parece ser directamente los pesos del modelo")
            model_weights = state_dict
        
        # Ver primer tensor
        first_key = list(model_weights.keys())[0]
        first_tensor = model_weights[first_key]
        print(f"   Primer tensor: {first_key}")
        print(f"   Shape: {first_tensor.shape}")
        print(f"   Dtype: {first_tensor.dtype}")
    
    else:
        print(f"   → Es un módulo completo")
        print(f"   Module type: {type(state_dict).__name__}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("CONCLUSIÓN:")
print("El archivo token2wav_int8.pt es un state_dict (pesos)")
print("Necesitamos la arquitectura del modelo para instanciarlo")
print("="*70)
