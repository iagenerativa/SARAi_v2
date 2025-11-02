#!/usr/bin/env python3
"""
Test DIRECTO usando AudioOmniPipeline existente

Usa el pipeline ya implementado en agents/audio_omni_pipeline.py
que tiene toda la lógica de audio integrada.
"""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyaudio
    import numpy as np
except ImportError as e:
    pytest.skip(f"Dependencias de audio faltantes: {e}", allow_module_level=True)


def test_audio_pipeline():
    """Test simple usando el pipeline existente"""
    
    print("="*60)
    print("🎙️  TEST CON AUDIOOMNIPIPELINE")
    print("="*60)
    
    # Importar el pipeline existente
    try:
        from agents.audio_omni_pipeline import get_audio_omni_pipeline, AudioOmniConfig
        import yaml
    except ImportError as e:
        print(f"❌ Error importando: {e}")
        return
    
    # Cargar configuración
    print("\n📦 Cargando configuración...")
    config_path = Path("config/sarai.yaml")
    if not config_path.exists():
        print(f"❌ No existe: {config_path}")
        return
    
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)
    
    audio_config = AudioOmniConfig.from_yaml(yaml_config.get('audio', {}))
    
    # Obtener pipeline (singleton)
    print("\n📦 Inicializando AudioOmniPipeline...")
    print("    ⏳ Esto puede tomar 30-60 segundos...")
    start = time.time()
    
    pipeline = get_audio_omni_pipeline()
    pipeline.config = audio_config
    pipeline.load()
    
    load_time = time.time() - start
    print(f"✅ Pipeline cargado en {load_time:.1f}s")
    
    # Grabar audio
    print("\n" + "="*60)
    print("GRABANDO AUDIO")
    print("="*60)
    
    audio_obj = pyaudio.PyAudio()
    
    print("\n🎤 Grabando 5s...")
    stream = audio_obj.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )
    
    frames = []
    for _ in range(int(16000 / 1024 * 5)):
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    
    audio_bytes = b''.join(frames)
    print(f"✅ Grabación completa ({len(audio_bytes)} bytes)")
    
    # Procesar con pipeline
    print("\n" + "="*60)
    print("PROCESANDO CON PIPELINE")
    print("="*60)
    
    print("\n🔄 Procesando audio...")
    process_start = time.time()
    
    try:
        result = pipeline.process_audio(audio_bytes)
        process_time = (time.time() - process_start) * 1000
        
        print(f"\n✅ Procesamiento completado ({process_time:.0f}ms)")
        print(f"\nResultado:")
        print(f"  - Texto: {result.get('text', 'N/A')}")
        print(f"  - Metadata: {result.get('metadata', {})}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    audio_obj.terminate()
    
    print("\n" + "="*60)
    print("TEST COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    test_audio_pipeline()
