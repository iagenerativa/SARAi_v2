#!/usr/bin/env python3
"""
Download VisCoder2-7B GGUF model from HuggingFace
=================================================
Este script descarga el modelo VisCoder2-7B Q4_K_M para uso en SARAi.
Modelo: mradermacher/VisCoder2-7B-GGUF
"""

import os
from huggingface_hub import hf_hub_download
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_viscoder2():
    """Descarga VisCoder2-7B Q4_K_M GGUF"""
    
    repo_id = "mradermacher/VisCoder2-7B-GGUF"
    filename = "VisCoder2-7B.Q4_K_M.gguf"
    cache_dir = "models/cache/viscoder2"
    
    logger.info(f"Descargando {filename} desde {repo_id}...")
    logger.info(f"Destino: {cache_dir}")
    
    # Crear directorio si no existe
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Descargar modelo
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True  # Continuar si se interrumpe
        )
        
        logger.info(f"✅ Modelo descargado exitosamente: {model_path}")
        
        # Verificar tamaño
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"📊 Tamaño del modelo: {size_mb:.2f} MB")
        
        # Crear symlink en la ubicación esperada
        expected_path = os.path.join(cache_dir, filename)
        if not os.path.exists(expected_path):
            os.symlink(model_path, expected_path)
            logger.info(f"🔗 Symlink creado: {expected_path}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"❌ Error al descargar modelo: {e}")
        raise

if __name__ == "__main__":
    print("=" * 70)
    print("VisCoder2-7B GGUF Download Script")
    print("=" * 70)
    print()
    
    try:
        model_path = download_viscoder2()
        
        print()
        print("=" * 70)
        print("✅ DESCARGA COMPLETADA")
        print("=" * 70)
        print()
        print(f"Modelo: VisCoder2-7B Q4_K_M")
        print(f"Path: {model_path}")
        print()
        print("Para usar el modelo:")
        print("  from core.unified_model_wrapper import get_model")
        print("  viscoder = get_model('viscoder2')")
        print("  response = viscoder.invoke('Write a Python function to...')")
        print()
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ ERROR EN DESCARGA")
        print("=" * 70)
        print(f"Error: {e}")
        exit(1)
