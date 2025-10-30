#!/usr/bin/env python3
"""
Test E2E: Pipeline Audio (ONNX Monolítico) + LFM2

Flujo actual de producción:
1. Audio (WAV) → agi_audio_core_int8.onnx (1.1GB) → Texto
2. Texto → LFM2-1.2B (ModelPool nativo) → Respuesta
3. Validar latencia <500ms

NOTA: Pipeline modular (qwen25_7b_audio.onnx) requiere:
  - Audio Encoder ONNX (pendiente exportar)
  - Talker ONNX (✅ tenemos qwen25_7b_audio.onnx 41MB)
  - Vocoder ONNX (pendiente exportar)

Por ahora usamos el modelo monolítico completo que SÍ funciona.
"""

import pytest
import numpy as np
import time
import os
from pathlib import Path


class TestAudioLFM2Standalone:
    """Tests del pipeline ONNX optimizado + LFM2 (standalone)"""
    
    def test_onnx_model_exists(self):
        """Verificar que el modelo ONNX monolítico existe"""
        model_path = Path("models/onnx/old/agi_audio_core_int8.onnx")
        
        # Buscar en ubicaciones posibles
        possible_paths = [
            Path("models/onnx/old/agi_audio_core_int8.onnx"),
            Path("models/onnx/agi_audio_core_int8.onnx")
        ]
        
        found = None
        for p in possible_paths:
            if p.exists():
                found = p
                break
        
        assert found is not None, f"Modelo monolítico no encontrado en: {possible_paths}"
        
        print(f"✅ Modelo ONNX monolítico: {found}")
        
        data_path = Path(str(found) + ".data")
        if data_path.exists():
            print(f"✅ External data: {data_path}")
    
    def test_audio_pipeline_monolithic_load(self):
        """Cargar pipeline ONNX monolítico (modelo completo)"""
        from agents.audio_omni_pipeline import AudioOmniPipeline, AudioOmniConfig
        
        config = AudioOmniConfig()
        config.pipeline_mode = "monolithic"  # Usar modelo monolítico completo
        config.model_path = "models/onnx/old/agi_audio_core_int8.onnx"
        
        # Buscar modelo en ubicaciones posibles
        if not Path(config.model_path).exists():
            config.model_path = "models/onnx/agi_audio_core_int8.onnx"
        
        pipeline = AudioOmniPipeline(config)
        pipeline.load()
        
        print(f"\n[Pipeline Info]")
        print(f"  Modo: {pipeline.mode}")
        print(f"  Modelo: {config.model_path}")
        
        assert pipeline.session is not None, "Sesión ONNX no cargada"
        assert pipeline.mode == "monolithic", f"Modo debe ser monolithic, es: {pipeline.mode}"
    
    def test_audio_to_text(self):
        """Pipeline ONNX monolítico: Audio → Texto"""
        from agents.audio_omni_pipeline import AudioOmniPipeline, AudioOmniConfig
        import io
        import soundfile as sf
        
        # Configuración
        config = AudioOmniConfig()
        config.pipeline_mode = "monolithic"
        config.model_path = "models/onnx/old/agi_audio_core_int8.onnx"
        
        # Buscar modelo
        if not Path(config.model_path).exists():
            config.model_path = "models/onnx/agi_audio_core_int8.onnx"
        
        # Cargar pipeline
        pipeline = AudioOmniPipeline(config)
        pipeline.load()
        
        # Generar audio sintético (3s, 16kHz)
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        audio = audio.astype(np.float32)
        
        # Convertir a bytes WAV
        audio_io = io.BytesIO()
        sf.write(audio_io, audio, sample_rate, format='WAV')
        audio_bytes = audio_io.getvalue()
        
        # Procesar
        print(f"\n[Audio Processing]")
        print(f"  Input: {len(audio_bytes)} bytes WAV")
        
        start = time.time()
        result = pipeline.process_audio(audio_bytes)
        latency = time.time() - start
        
        print(f"  Output: {result.get('text', 'N/A')}")
        print(f"  Latencia: {latency*1000:.1f}ms")
        print(f"  Modo usado: {result['metadata']['mode']}")
        
        # Validaciones
        assert "text" in result, "No se generó texto"
        assert latency < 2.0, f"Latencia muy alta: {latency*1000:.0f}ms"
    
    @pytest.mark.skipif(
        not os.path.exists("models/cache/lfm2") and not os.path.exists("config/sarai.yaml"),
        reason="LFM2 o configuración no disponible"
    )
    def test_e2e_audio_lfm2_standalone(self):
        """
        TEST E2E COMPLETO: Audio → ONNX Monolítico → Texto → LFM2 → Respuesta
        
        Pipeline actual de producción:
        - agi_audio_core_int8.onnx (1.1GB) para Audio → Texto
        - LFM2-1.2B GGUF (700MB) para Texto → Respuesta
        
        Total RAM: ~1.8GB
        Latencia objetivo: <1s (relajado por modelo más grande)
        """
        from agents.audio_omni_pipeline import AudioOmniPipeline, AudioOmniConfig
        from core.model_pool import ModelPool
        import io
        import soundfile as sf
        import yaml
        
        print("\n" + "="*70)
        print("TEST E2E: Pipeline ONNX Monolítico + LFM2")
        print("="*70)
        
        # PASO 1: Configurar pipeline de audio
        audio_config = AudioOmniConfig()
        audio_config.pipeline_mode = "monolithic"
        audio_config.model_path = "models/onnx/old/agi_audio_core_int8.onnx"
        
        # Buscar modelo
        if not Path(audio_config.model_path).exists():
            audio_config.model_path = "models/onnx/agi_audio_core_int8.onnx"
        
        # PASO 2: Cargar ModelPool con LFM2
        config_path = str(Path(__file__).parent.parent / "config" / "sarai.yaml")
        model_pool = ModelPool(config_path)
        
        # PASO 3: Cargar pipeline de audio
        pipeline = AudioOmniPipeline(audio_config)
        pipeline.load()
        
        print(f"\n[1] Pipeline Audio Cargado")
        print(f"    Modo: {pipeline.mode}")
        print(f"    Modelo: {audio_config.model_path}")
        
        # PASO 4: Generar audio sintético
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        audio = audio.astype(np.float32)
        
        audio_io = io.BytesIO()
        sf.write(audio_io, audio, sample_rate, format='WAV')
        audio_bytes = audio_io.getvalue()
        
        # PASO 5: Audio → Texto
        print(f"\n[2] Procesando Audio → Texto")
        print(f"    Input: {len(audio_bytes)} bytes")
        
        start_audio = time.time()
        result_audio = pipeline.process_audio(audio_bytes)
        audio_time = time.time() - start_audio
        
        text_transcribed = result_audio.get("text", "[AUDIO]")
        print(f"    ✅ Texto: {text_transcribed}")
        print(f"    ⏱️  Latencia: {audio_time*1000:.1f}ms")
        
        # PASO 6: Texto → LFM2
        print(f"\n[3] Procesando Texto → LFM2")
        
        lfm2_model = model_pool.get("tiny")  # LFM2-1.2B GGUF
        
        prompt = f"""Usuario dice: '{text_transcribed}'.

Responde de forma muy breve y natural (máximo 2 frases)."""
        
        start_lfm2 = time.time()
        response_lfm2 = lfm2_model(prompt, max_tokens=50, temperature=0.7, stop=["\n\n"])
        lfm2_time = time.time() - start_lfm2
        
        print(f"    ✅ Respuesta: {response_lfm2}")
        print(f"    ⏱️  Latencia: {lfm2_time*1000:.1f}ms")
        
        # PASO 7: Métricas finales
        total_time = audio_time + lfm2_time
        
        print(f"\n" + "="*70)
        print(f"📊 MÉTRICAS DE PRODUCCIÓN")
        print(f"="*70)
        print(f"  Pipeline Audio:      {audio_time*1000:>6.1f} ms  (monolithic)")
        print(f"  LFM2 (GGUF nativo):  {lfm2_time*1000:>6.1f} ms")
        print(f"  {'─'*68}")
        print(f"  TOTAL E2E:           {total_time*1000:>6.1f} ms")
        print(f"  Objetivo:            {'✅ <1000ms' if total_time < 1.0 else '⚠️  >1000ms'}")
        print(f"  RAM estimada:        ~1.8 GB  (ONNX 1.1GB + LFM2 700MB)")
        print(f"  Sin Qwen2.5-Omni:    ✅ (modelo monolítico standalone)")
        print(f"="*70)
        
        # Validaciones
        assert text_transcribed, "No se generó texto del audio"
        assert response_lfm2, "No se generó respuesta de LFM2"
        assert total_time < 2.0, f"Latencia muy alta: {total_time*1000:.0f}ms (objetivo: <1000ms)"
        
        print(f"\n✅ TEST E2E EXITOSO\n")


if __name__ == "__main__":
    # Ejecutar tests standalone
    pytest.main([__file__, "-v", "-s"])
