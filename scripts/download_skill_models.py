#!/usr/bin/env python3
"""
Script para descargar modelos GGUF de skills MoE
Descarga solo los modelos necesarios para tests de integración
"""
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Modelos necesarios para tests (versiones pequeñas)
SKILL_MODELS = {
    "programming": {
        "repo_id": "TheBloke/CodeLlama-7B-Instruct-GGUF",
        "filename": "codellama-7b-instruct.Q4_K_M.gguf",
        "size_mb": 4100
    },
    "diagnosis": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "size_mb": 4100
    }
}

def download_skill_model(skill_name: str, model_info: dict):
    """Descarga un modelo GGUF con barra de progreso"""
    cache_dir = Path("models/cache/skills") / skill_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Descargando {skill_name} ({model_info['size_mb']} MB)...")
    print(f"   Repo: {model_info['repo_id']}")
    print(f"   Archivo: {model_info['filename']}")
    
    try:
        # Verificar si ya existe
        local_path = cache_dir / model_info['filename']
        if local_path.exists():
            print(f"   ✅ Ya existe en cache: {local_path}")
            return str(local_path)
        
        # Descargar con barra de progreso
        downloaded_path = hf_hub_download(
            repo_id=model_info['repo_id'],
            filename=model_info['filename'],
            cache_dir=str(cache_dir),
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"   ✅ Descargado: {downloaded_path}")
        return downloaded_path
    
    except Exception as e:
        print(f"   ❌ Error descargando {skill_name}: {e}")
        return None

def main():
    """Descarga todos los modelos necesarios"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Descarga modelos GGUF para skills MoE")
    parser.add_argument('--yes', '-y', action='store_true', help='Auto-confirmar descarga')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Descarga de Modelos GGUF para Skills MoE v2.12")
    print("=" * 60)
    print(f"\nModelos a descargar: {len(SKILL_MODELS)}")
    
    total_size_mb = sum(m['size_mb'] for m in SKILL_MODELS.values())
    print(f"Tamaño total estimado: {total_size_mb} MB (~{total_size_mb/1024:.1f} GB)")
    print(f"Ubicación: models/cache/skills/")
    print("\nEsto puede tardar 10-30 minutos según tu conexión...")
    
    # Confirmar si el usuario quiere continuar
    if not args.yes and sys.stdin.isatty():  # Solo preguntar si es interactivo y sin --yes
        response = input("\n¿Continuar con la descarga? (s/n): ")
        if response.lower() not in ['s', 'si', 'y', 'yes']:
            print("❌ Descarga cancelada por el usuario")
            sys.exit(0)
    
    # Descargar cada modelo
    results = {}
    for skill_name, model_info in SKILL_MODELS.items():
        path = download_skill_model(skill_name, model_info)
        results[skill_name] = path is not None
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE DESCARGA")
    print("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for skill, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {skill}")
    
    print(f"\n{success_count}/{total_count} modelos descargados correctamente")
    
    if success_count == total_count:
        print("\n✅ TODOS los modelos descargados. Listo para tests de integración.")
        print("\nEjecutar tests con:")
        print("  pytest tests/test_mcp_skills.py::TestIntegrationWithModelPool -v")
        sys.exit(0)
    else:
        print("\n⚠️ Algunos modelos fallaron. Tests de integración se saltarán.")
        sys.exit(1)

if __name__ == "__main__":
    main()
