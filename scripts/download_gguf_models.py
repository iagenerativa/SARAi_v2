#!/usr/bin/env python3
"""
Script de descarga de GGUFs v2.4
Descarga archivos GGUF desde HuggingFace con verificación
Soporte para GGUF dinámico: un solo archivo solar-10.7b sirve para short/long
"""

import os
import sys
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm


# v2.6.1: Repositorios GGUF verificados
# Links oficiales:
# - hf.co/solxxcero/SOLAR-10.7B-Instruct-v1.0-Q4_K_M-GGUF:Q4_K_M
# - hf.co/LiquidAI/LFM2-1.2B-GGUF:Q4_K_M
GGUF_MODELS = {
    "SOLAR-10.7B": {
        "repo_id": "solxxcero/SOLAR-10.7B-Instruct-v1.0-Q4_K_M-GGUF",
        "filename": "solar-10.7b-instruct-v1.0.Q4_K_M.gguf",
        "output_path": "models/gguf/solar-10.7b.gguf",  # Nombre canónico
        "expected_size_mb": 6144,  # ~6GB
        "required": True
    },
    "LFM2-1.2B": {
        "repo_id": "LiquidAI/LFM2-1.2B-GGUF",
        "filename": "lfm2-1.2b.Q4_K_M.gguf",
        "output_path": "models/gguf/lfm2-1.2b.gguf",
        "expected_size_mb": 700,
        "required": True
    },
    "Qwen2.5-Omni-7B": {
        "repo_id": "Qwen/Qwen2.5-Omni-7B-GGUF",
        "filename": "qwen2.5-omni-7b.Q4_K_M.gguf",
        "output_path": "models/gguf/qwen2.5-omni-7b.gguf",
        "expected_size_mb": 4096,
        "required": False  # Opcional, solo para multimodal
    }
}


def verify_file_size(filepath: Path, expected_size_mb: int, tolerance: float = 0.1):
    """
    Verifica que el archivo descargado tenga el tamaño esperado
    
    Args:
        filepath: Ruta al archivo
        expected_size_mb: Tamaño esperado en MB
        tolerance: Tolerancia (±10% por defecto)
    
    Returns:
        bool: True si el tamaño es correcto
    """
    if not filepath.exists():
        return False
    
    actual_size_mb = filepath.stat().st_size / (1024 * 1024)
    min_size = expected_size_mb * (1 - tolerance)
    max_size = expected_size_mb * (1 + tolerance)
    
    return min_size <= actual_size_mb <= max_size


def download_gguf(model_name: str, config: dict):
    """
    Descarga archivo GGUF desde HuggingFace con verificación
    
    Args:
        model_name: Nombre del modelo
        config: Configuración con repo_id, filename, output_path, etc.
    """
    print(f"\n{'='*60}")
    print(f"Descargando: {model_name}")
    print(f"Repositorio: {config['repo_id']}")
    print(f"Archivo: {config['filename']}")
    print(f"Destino: {config['output_path']}")
    print(f"{'='*60}\n")
    
    output_path = Path(config['output_path'])
    
    # Si ya existe y tiene tamaño correcto, skip
    if output_path.exists():
        if verify_file_size(output_path, config['expected_size_mb']):
            print(f"✅ {model_name} ya descargado y verificado: {output_path}")
            return str(output_path)
        else:
            print(f"⚠️ Archivo existe pero tamaño incorrecto, re-descargando...")
            output_path.unlink()
    
    # Crear directorio de salida
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Descargar a ubicación temporal en cache de HF
        print(f"Descargando desde HuggingFace...")
        cached_path = hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['filename'],
            resume_download=True,
            local_dir_use_symlinks=False
        )
        
        # Copiar/mover a ubicación final
        import shutil
        shutil.copy2(cached_path, output_path)
        
        # Verificar tamaño
        if verify_file_size(output_path, config['expected_size_mb']):
            print(f"✅ Descarga exitosa y verificada: {output_path}")
            print(f"   Tamaño: {output_path.stat().st_size / (1024**3):.2f} GB\n")
            return str(output_path)
        else:
            print(f"❌ Error: Tamaño de archivo incorrecto")
            return None
        
    except Exception as e:
        if config.get('fallback_repo'):
            print(f"⚠️ Modelo GGUF no disponible en {config['repo_id']}")
            print(f"   Intentar con repo base: {config['fallback_repo']}")
            print(f"   NOTA: Requiere conversión manual a GGUF con llama.cpp")
            print(f"   Comandos:")
            print(f"     git clone https://github.com/ggerganov/llama.cpp")
            print(f"     cd llama.cpp")
            print(f"     python convert.py /path/to/model --outfile {output_path}")
            print(f"     ./quantize {output_path} {output_path}.Q4_K_M.gguf Q4_K_M")
            print(f"      python llama.cpp/convert.py <modelo_original> --outtype q4_K_M")
            print(f"   3. Usar backend GPU en config/sarai.yaml (backend: 'gpu')\n")
        else:
            print(f"❌ Error descargando {model_name}: {e}\n")
            raise


def main():
    """
    Descarga todos los modelos GGUF necesarios
    """
    print("🚀 SARAi v2.2 - Descargador de Modelos GGUF")
    print("="*60)
    print("⚠️  ADVERTENCIA: Los archivos GGUF ocupan ~10GB en total")
    print("   Asegúrate de tener suficiente espacio en disco\n")
    
    confirm = input("¿Continuar con la descarga? [s/N]: ")
    if confirm.lower() not in ['s', 'si', 'y', 'yes']:
        print("❌ Descarga cancelada")
        sys.exit(0)
    
    # Descargar embeddings (no GGUF, formato estándar)
    print("\n" + "="*60)
    print("Descargando: EmbeddingGemma-300M (formato estándar)")
    print("="*60 + "\n")
    
    try:
        from transformers import AutoModel
        AutoModel.from_pretrained(
            "google/embeddinggemma-300m-qat-q4_0-unquantized",
            cache_dir="./models/cache/embeddings"
        )
        print("✅ EmbeddingGemma descargado\n")
    except Exception as e:
        print(f"❌ Error descargando embeddings: {e}\n")
    
    # Descargar modelos GGUF
    success_count = 0
    failed_models = []
    
    for model_name, config in GGUF_MODELS.items():
        try:
            download_gguf(model_name, config)
            success_count += 1
        except Exception:
            failed_models.append(model_name)
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE DESCARGA")
    print("="*60)
    print(f"✅ Exitosos: {success_count}/{len(GGUF_MODELS)}")
    
    if failed_models:
        print(f"❌ Fallidos: {', '.join(failed_models)}")
        print("\n⚠️  Algunos modelos requieren conversión manual o no están")
        print("   disponibles en formato GGUF. Consulta la documentación.")
    else:
        print("🎉 ¡Todos los modelos descargados exitosamente!")
    
    print("\n📝 Próximos pasos:")
    print("   1. Actualiza config/sarai.yaml con las rutas correctas")
    print("   2. Ejecuta: python main.py")
    print("   3. Monitorea el uso de RAM con: watch -n1 'free -h'\n")


if __name__ == "__main__":
    main()
