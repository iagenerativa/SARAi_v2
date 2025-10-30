#!/usr/bin/env python3
"""
Test E2E SIMPLIFICADO: Pipeline Audio + LFM2

Estado actual: Usamos STT básico mientras exportamos componentes ONNX modulares

Flujo:
1. Audio (WAV) → [STT BÁSICO TEMPORAL] → Texto
2. Texto → LFM2-1.2B → Respuesta

Próximos pasos:
- Exportar Whisper-tiny a ONNX para reemplazar STT básico
- Exportar componentes TTS (Encoder + Vocoder) para flujo completo

Arquitectura objetivo: Ver docs/AUDIO_PIPELINE_ARCHITECTURE.md
"""

import pytest
import numpy as np
import time
import os
from pathlib import Path


class TestAudioLFM2Simple:
    """Tests simplificados del pipeline Audio + LFM2"""
    
    def test_lfm2_text_generation(self):
        """Test básico: LFM2 genera respuesta desde texto"""
        from core.model_pool import ModelPool
        
        config_path = str(Path(__file__).parent.parent / "config" / "sarai.yaml")
        
        if not Path(config_path).exists():
            pytest.skip("Configuración sarai.yaml no encontrada")
        
        # Cargar LFM2
        model_pool = ModelPool(config_path)
        lfm2 = model_pool.get("tiny")
        
        # Generar respuesta
        prompt = "Usuario: Hola, ¿cómo estás?\nAsistente:"
        
        start = time.time()
        response = lfm2(prompt, max_tokens=50, temperature=0.7, stop=["\n\n", "Usuario:"])
        latency = time.time() - start
        
        print(f"\n[Test LFM2]")
        print(f"  Prompt: {prompt}")
        print(f"  Respuesta: {response}")
        print(f"  Latencia: {latency*1000:.1f}ms")
        
        assert response, "No se generó respuesta"
        assert len(response) > 5, "Respuesta muy corta"
        assert latency < 2.0, f"Latencia muy alta: {latency*1000:.0f}ms"
        
        print(f"  ✅ Test LFM2 exitoso")
    
    def test_audio_simulation_to_lfm2(self):
        """
        Test E2E simulado: Audio sintético → Texto simulado → LFM2
        
        NOTA: El paso Audio → Texto está simulado hasta que exportemos Whisper ONNX
        """
        from core.model_pool import ModelPool
        
        config_path = str(Path(__file__).parent.parent / "config" / "sarai.yaml")
        
        if not Path(config_path).exists():
            pytest.skip("Configuración sarai.yaml no encontrada")
        
        print("\n" + "="*70)
        print("TEST E2E SIMULADO: Audio → [STT Simulado] → LFM2")
        print("="*70)
        
        # PASO 1: Simular transcripción de audio
        # TODO: Reemplazar con whisper_encoder.onnx cuando esté exportado
        simulated_transcription = "Hola, necesito ayuda con Python"
        audio_latency = 80  # ms (latencia esperada de Whisper-tiny)
        
        print(f"\n[1] Audio → Texto (SIMULADO)")
        print(f"    Transcripción: {simulated_transcription}")
        print(f"    Latencia simulada: {audio_latency}ms")
        
        # PASO 2: LFM2 procesa el texto
        model_pool = ModelPool(config_path)
        lfm2 = model_pool.get("tiny")
        
        prompt = f"Usuario: {simulated_transcription}\nAsistente:"
        
        print(f"\n[2] Texto → LFM2")
        print(f"    Prompt: {prompt}")
        
        start_lfm2 = time.time()
        response = lfm2(prompt, max_tokens=100, temperature=0.7, stop=["\n\n", "Usuario:"])
        lfm2_latency = (time.time() - start_lfm2) * 1000
        
        print(f"    Respuesta: {response}")
        print(f"    Latencia: {lfm2_latency:.1f}ms")
        
        # PASO 3: Métricas totales
        total_latency = audio_latency + lfm2_latency
        
        print(f"\n" + "="*70)
        print(f"📊 MÉTRICAS E2E")
        print(f"="*70)
        print(f"  Audio → Texto:   {audio_latency:>6.1f} ms  (simulado)")
        print(f"  LFM2:            {lfm2_latency:>6.1f} ms  (real)")
        print(f"  {'─'*68}")
        print(f"  TOTAL E2E:       {total_latency:>6.1f} ms")
        print(f"  Objetivo:        {'✅ <500ms' if total_latency < 500 else '⚠️  >500ms'}")
        print(f"  RAM estimada:    ~740 MB  (Whisper 39MB + LFM2 700MB)")
        print(f"="*70)
        
        assert response, "No se generó respuesta de LFM2"
        assert total_latency < 1000, f"Latencia muy alta: {total_latency:.0f}ms"
        
        print(f"\n✅ TEST E2E SIMULADO EXITOSO")
        print(f"\n📝 Próximo paso: Exportar Whisper-tiny a ONNX")
        print(f"   Comando: python scripts/export_whisper_encoder.py")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
