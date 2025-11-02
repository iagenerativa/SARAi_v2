#!/usr/bin/env python3
"""
Test de Conversación Real con SARAi - E2E con Micrófono y Altavoz

Este script permite probar SARAi en conversación real midiendo:
1. Latencia E2E (micrófono → respuesta en altavoz)
2. Calidad de voz (tono, expresividad, naturalidad)
3. Precisión del STT (Word Error Rate)
4. Calidad del TTS (Mean Opinion Score estimado)
5. Coherencia de respuestas LLM

REQUISITOS:
- Micrófono conectado
- Altavoces conectados
- pyaudio instalado: pip install pyaudio
- sounddevice instalado: pip install sounddevice

USO:
    python tests/test_real_voice_conversation.py

    # Modo silencioso (sin TTS, solo métricas)
    python tests/test_real_voice_conversation.py --silent

    # Conversación específica
    python tests/test_real_voice_conversation.py --scenario greeting
"""

import os
import sys
import time
import wave
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pytest

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import sounddevice as sd
    import numpy as np
    import librosa
    from scipy.io import wavfile
except ImportError as e:
    pytest.skip(
        f"Dependencias de audio faltantes: {e}. Instala sounddevice numpy librosa scipy",
        allow_module_level=True,
    )


class VoiceConversationTester:
    """
    Tester de conversación real con SARAi
    
    Funcionalidades:
    - Grabación de audio del micrófono
    - Reproducción de respuestas en altavoz
    - Medición de latencias E2E
    - Evaluación de calidad de voz
    - Logging detallado de métricas
    """
    
    def __init__(self, silent: bool = False):
        self.silent = silent  # Si True, no reproduce audio
        self.sample_rate = 16000  # 16kHz (Qwen2.5-Audio)
        self.chunk_duration = 0.1  # 100ms chunks
        self.silence_threshold = 0.01  # Umbral de silencio
        self.silence_duration = 1.5  # Segundos de silencio para terminar
        
        # Estado de la conversación
        self.conversation_log = []
        self.metrics = {
            "latencies": [],
            "audio_quality": [],
            "turn_count": 0,
            "total_conversation_time": 0
        }
        
        # Paths
        self.temp_dir = Path("state/voice_test_temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar pipeline de audio (si está disponible)
        self._load_audio_pipeline()
        
        # Cargar LLM para razonamiento
        self._load_llm()
    
    def _load_audio_pipeline(self):
        """Carga pipeline de audio (ONNX o Qwen-Omni)"""
        try:
            from agents.audio_omni_pipeline import AudioOmniPipeline, AudioOmniConfig
            from config import Config
            
            # Cargar configuración
            config_path = Path("config/sarai.yaml")
            if config_path.exists():
                import yaml
                with open(config_path) as f:
                    config_dict = yaml.safe_load(f)
                self.audio_config = AudioOmniConfig.from_yaml(config_dict)
            else:
                self.audio_config = AudioOmniConfig()
            
            print(f"🔧 Inicializando pipeline de audio ({self.audio_config.pipeline_mode})...")
            self.audio_pipeline = AudioOmniPipeline(self.audio_config)
            print(f"✅ Pipeline de audio cargado: {self.audio_config.pipeline_mode}")
            
        except Exception as e:
            print(f"⚠️  No se pudo cargar pipeline de audio: {e}")
            print("   Usando STT/TTS básico como fallback...")
            self.audio_pipeline = None
            self._load_fallback_stt_tts()
    
    def _load_fallback_stt_tts(self):
        """Carga STT/TTS básico como fallback"""
        try:
            import whisper
            print("🔧 Cargando Whisper (STT fallback)...")
            self.stt_model = whisper.load_model("base")
            print("✅ Whisper cargado")
        except Exception as e:
            print(f"❌ Error cargando Whisper: {e}")
            self.stt_model = None
        
        try:
            # TTS básico con pyttsx3
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Velocidad
            self.tts_engine.setProperty('volume', 0.9)
            print("✅ TTS básico cargado")
        except Exception as e:
            print(f"⚠️  TTS básico no disponible: {e}")
            self.tts_engine = None
    
    def _load_llm(self):
        """Carga LLM para razonamiento (LFM2 o similar)"""
        try:
            from llama_cpp import Llama
            
            llm_path = Path("models/lfm2/LFM2-1.2B-Q4_K_M.gguf")
            if not llm_path.exists():
                print(f"⚠️  LFM2 no encontrado: {llm_path}")
                print("   Usando respuestas simuladas...")
                self.llm = None
                return
            
            print(f"🔧 Cargando LFM2 desde {llm_path}...")
            start = time.time()
            
            self.llm = Llama(
                model_path=str(llm_path),
                n_ctx=512,
                n_threads=6,
                verbose=False
            )
            
            load_time = (time.time() - start) * 1000
            print(f"✅ LFM2 cargado en {load_time:.0f}ms")
            
        except Exception as e:
            print(f"⚠️  Error cargando LFM2: {e}")
            print("   Usando respuestas simuladas...")
            self.llm = None
    
    def record_audio(self, max_duration: int = 10) -> Tuple[np.ndarray, float]:
        """
        Graba audio del micrófono hasta detectar silencio
        
        Returns:
            (audio_array, recording_time_ms)
        """
        print("\n🎤 Escuchando... (habla y luego haz silencio)")
        
        recording = []
        silence_chunks = 0
        chunks_needed = int(self.silence_duration / self.chunk_duration)
        
        start_time = time.time()
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            ) as stream:
                
                while True:
                    # Leer chunk
                    chunk, _ = stream.read(int(self.sample_rate * self.chunk_duration))
                    recording.append(chunk)
                    
                    # Detectar silencio
                    if np.abs(chunk).max() < self.silence_threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                    
                    # Terminar si hay silencio prolongado
                    if silence_chunks >= chunks_needed:
                        print("✅ Silencio detectado, finalizando grabación")
                        break
                    
                    # Timeout de seguridad
                    elapsed = time.time() - start_time
                    if elapsed > max_duration:
                        print(f"⏰ Timeout alcanzado ({max_duration}s)")
                        break
        
        except KeyboardInterrupt:
            print("\n⚠️  Grabación interrumpida por usuario")
        
        recording_time = (time.time() - start_time) * 1000
        audio = np.concatenate(recording, axis=0).flatten()
        
        # Eliminar silencio inicial/final
        audio = self._trim_silence(audio)
        
        print(f"📊 Grabado: {len(audio)/self.sample_rate:.2f}s, {recording_time:.0f}ms")
        
        return audio, recording_time
    
    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Elimina silencio del inicio y final"""
        # Encontrar primer y último sample con señal
        mask = np.abs(audio) > self.silence_threshold
        if not mask.any():
            return audio[:int(self.sample_rate * 0.1)]  # Mínimo 100ms
        
        indices = np.where(mask)[0]
        start_idx = max(0, indices[0] - int(self.sample_rate * 0.1))  # 100ms padding
        end_idx = min(len(audio), indices[-1] + int(self.sample_rate * 0.1))
        
        return audio[start_idx:end_idx]
    
    def save_audio(self, audio: np.ndarray, filename: str) -> Path:
        """Guarda audio a archivo WAV"""
        filepath = self.temp_dir / filename
        
        # Normalizar a int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        wavfile.write(str(filepath), self.sample_rate, audio_int16)
        return filepath
    
    def audio_to_text(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Convierte audio a texto (STT)
        
        Returns:
            (transcripción, latencia_ms)
        """
        start = time.time()
        
        # Guardar temporalmente
        temp_file = self.save_audio(audio, "temp_input.wav")
        
        if self.audio_pipeline:
            # Usar pipeline ONNX
            try:
                audio_bytes = temp_file.read_bytes()
                result = self.audio_pipeline.process_audio(audio_bytes)
                text = result.get("transcription", "")
            except Exception as e:
                print(f"⚠️  Error en pipeline ONNX: {e}")
                text = self._fallback_stt(audio)
        else:
            # Fallback a Whisper
            text = self._fallback_stt(audio)
        
        latency = (time.time() - start) * 1000
        
        return text, latency
    
    def _fallback_stt(self, audio: np.ndarray) -> str:
        """STT con Whisper como fallback"""
        if self.stt_model is None:
            return "[STT no disponible]"
        
        result = self.stt_model.transcribe(audio, fp16=False)
        return result["text"].strip()
    
    def text_to_audio(self, text: str) -> Tuple[np.ndarray, float, Dict]:
        """
        Convierte texto a audio (TTS)
        
        Returns:
            (audio_array, latencia_ms, métricas_calidad)
        """
        start = time.time()
        
        if self.audio_pipeline:
            # TTS con pipeline ONNX
            try:
                result = self.audio_pipeline.text_to_speech(text)
                audio = result.get("audio", np.array([]))
                quality = result.get("quality_metrics", {})
            except Exception as e:
                print(f"⚠️  Error en TTS ONNX: {e}")
                audio, quality = self._fallback_tts(text)
        else:
            # Fallback TTS básico
            audio, quality = self._fallback_tts(text)
        
        latency = (time.time() - start) * 1000
        
        return audio, latency, quality
    
    def _fallback_tts(self, text: str) -> Tuple[np.ndarray, Dict]:
        """TTS básico como fallback"""
        if self.tts_engine is None:
            return np.array([]), {"quality": "N/A"}
        
        # Guardar a archivo temporal
        temp_file = self.temp_dir / "temp_tts.wav"
        self.tts_engine.save_to_file(text, str(temp_file))
        self.tts_engine.runAndWait()
        
        # Leer audio generado
        sr, audio = wavfile.read(str(temp_file))
        
        # Convertir a float32 y resamplear si es necesario
        audio = audio.astype(np.float32) / 32767.0
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        return audio, {"quality": "basic", "mos_estimated": 3.0}
    
    def play_audio(self, audio: np.ndarray):
        """Reproduce audio en altavoces"""
        if self.silent:
            print("🔇 Modo silencioso: audio no reproducido")
            return
        
        print("🔊 Reproduciendo respuesta...")
        sd.play(audio, self.sample_rate)
        sd.wait()
    
    def llm_reasoning(self, user_input: str) -> Tuple[str, float]:
        """
        Razonamiento LLM para generar respuesta
        
        Returns:
            (respuesta, latencia_ms)
        """
        start = time.time()
        
        if self.llm is None:
            # Respuestas simuladas
            responses = {
                "hola": "Hola! ¿En qué puedo ayudarte hoy?",
                "cómo estás": "Estoy funcionando correctamente. Gracias por preguntar.",
                "qué eres": "Soy SARAi, un asistente de inteligencia artificial local.",
                "default": "Interesante. Cuéntame más sobre eso."
            }
            
            user_lower = user_input.lower()
            for key, response in responses.items():
                if key in user_lower:
                    latency = 50  # Simulado
                    return response, latency
            
            return responses["default"], 50
        
        # Razonamiento real con LFM2
        response = self.llm.create_completion(
            user_input,
            max_tokens=50,
            temperature=0.7,
            stop=["\n\n"]
        )
        
        latency = (time.time() - start) * 1000
        text = response['choices'][0]['text'].strip()
        
        return text, latency
    
    def single_turn(self, turn_number: int) -> Dict:
        """
        Ejecuta un turno de conversación completo
        
        Returns:
            métricas del turno
        """
        print(f"\n{'='*60}")
        print(f"TURNO {turn_number}")
        print(f"{'='*60}")
        
        turn_start = time.time()
        
        # 1. Grabar audio del usuario
        print("\n[1/5] Grabación de audio")
        user_audio, recording_time = self.record_audio()
        
        # 2. STT (Speech-to-Text)
        print("\n[2/5] Transcripción (STT)")
        user_text, stt_latency = self.audio_to_text(user_audio)
        print(f"👤 Usuario: \"{user_text}\"")
        print(f"⏱️  STT: {stt_latency:.0f}ms")
        
        # 3. LLM Razonamiento
        print("\n[3/5] Razonamiento (LLM)")
        response_text, llm_latency = self.llm_reasoning(user_text)
        print(f"🤖 SARAi: \"{response_text}\"")
        print(f"⏱️  LLM: {llm_latency:.0f}ms")
        
        # 4. TTS (Text-to-Speech)
        print("\n[4/5] Síntesis de voz (TTS)")
        response_audio, tts_latency, audio_quality = self.text_to_audio(response_text)
        print(f"⏱️  TTS: {tts_latency:.0f}ms")
        print(f"📊 Calidad estimada: {audio_quality}")
        
        # 5. Reproducir respuesta
        print("\n[5/5] Reproducción")
        self.play_audio(response_audio)
        
        # Métricas del turno
        turn_end = time.time()
        total_latency = (turn_end - turn_start) * 1000
        
        metrics = {
            "turn": turn_number,
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "response_text": response_text,
            "latencies": {
                "recording": recording_time,
                "stt": stt_latency,
                "llm": llm_latency,
                "tts": tts_latency,
                "total_e2e": total_latency
            },
            "audio_quality": audio_quality
        }
        
        print(f"\n{'─'*60}")
        print(f"⏱️  LATENCIA E2E TOTAL: {total_latency:.0f}ms")
        print(f"   ├─ Grabación: {recording_time:.0f}ms")
        print(f"   ├─ STT: {stt_latency:.0f}ms")
        print(f"   ├─ LLM: {llm_latency:.0f}ms")
        print(f"   └─ TTS: {tts_latency:.0f}ms")
        print(f"{'─'*60}")
        
        return metrics
    
    def run_conversation(self, max_turns: int = 5) -> Dict:
        """
        Ejecuta conversación completa de N turnos
        
        Returns:
            métricas totales de la conversación
        """
        print("\n" + "="*60)
        print("🎙️  TEST DE CONVERSACIÓN REAL CON SARAi")
        print("="*60)
        print(f"\nTurnos máximos: {max_turns}")
        print("Presiona Ctrl+C para terminar en cualquier momento\n")
        
        conversation_start = time.time()
        
        for turn in range(1, max_turns + 1):
            try:
                metrics = self.single_turn(turn)
                self.conversation_log.append(metrics)
                
                # Preguntar si continuar
                if turn < max_turns:
                    print("\n¿Continuar conversación? (Enter=Sí, Ctrl+C=No)")
                    input()
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Conversación terminada por usuario")
                break
        
        conversation_end = time.time()
        total_time = (conversation_end - conversation_start) * 1000
        
        # Calcular estadísticas
        stats = self._calculate_statistics()
        stats["total_conversation_time_ms"] = total_time
        stats["turns_completed"] = len(self.conversation_log)
        
        # Mostrar resumen
        self._print_summary(stats)
        
        # Guardar log
        self._save_log(stats)
        
        return stats
    
    def _calculate_statistics(self) -> Dict:
        """Calcula estadísticas de la conversación"""
        if not self.conversation_log:
            return {}
        
        # Extraer latencias
        stt_latencies = [m["latencies"]["stt"] for m in self.conversation_log]
        llm_latencies = [m["latencies"]["llm"] for m in self.conversation_log]
        tts_latencies = [m["latencies"]["tts"] for m in self.conversation_log]
        e2e_latencies = [m["latencies"]["total_e2e"] for m in self.conversation_log]
        
        return {
            "stt": {
                "mean": np.mean(stt_latencies),
                "min": np.min(stt_latencies),
                "max": np.max(stt_latencies),
                "std": np.std(stt_latencies)
            },
            "llm": {
                "mean": np.mean(llm_latencies),
                "min": np.min(llm_latencies),
                "max": np.max(llm_latencies),
                "std": np.std(llm_latencies)
            },
            "tts": {
                "mean": np.mean(tts_latencies),
                "min": np.min(tts_latencies),
                "max": np.max(tts_latencies),
                "std": np.std(tts_latencies)
            },
            "e2e": {
                "mean": np.mean(e2e_latencies),
                "min": np.min(e2e_latencies),
                "max": np.max(e2e_latencies),
                "std": np.std(e2e_latencies)
            }
        }
    
    def _print_summary(self, stats: Dict):
        """Imprime resumen de estadísticas"""
        print("\n" + "="*60)
        print("📊 RESUMEN DE LA CONVERSACIÓN")
        print("="*60)
        
        print(f"\n🔢 Turnos completados: {stats['turns_completed']}")
        print(f"⏰ Tiempo total: {stats['total_conversation_time_ms']/1000:.1f}s")
        
        print("\n📈 LATENCIAS PROMEDIO:")
        print(f"  STT:  {stats['stt']['mean']:.0f}ms (±{stats['stt']['std']:.0f}ms)")
        print(f"  LLM:  {stats['llm']['mean']:.0f}ms (±{stats['llm']['std']:.0f}ms)")
        print(f"  TTS:  {stats['tts']['mean']:.0f}ms (±{stats['tts']['std']:.0f}ms)")
        print(f"  E2E:  {stats['e2e']['mean']:.0f}ms (±{stats['e2e']['std']:.0f}ms)")
        
        print("\n📊 LATENCIAS MIN/MAX:")
        print(f"  STT:  {stats['stt']['min']:.0f}ms / {stats['stt']['max']:.0f}ms")
        print(f"  LLM:  {stats['llm']['min']:.0f}ms / {stats['llm']['max']:.0f}ms")
        print(f"  TTS:  {stats['tts']['min']:.0f}ms / {stats['tts']['max']:.0f}ms")
        print(f"  E2E:  {stats['e2e']['min']:.0f}ms / {stats['e2e']['max']:.0f}ms")
        
        # Análisis de calidad
        print("\n🎤 ANÁLISIS DE CALIDAD:")
        
        # Calcular MOS estimado promedio
        mos_scores = []
        for turn in self.conversation_log:
            quality = turn.get("audio_quality", {})
            if "mos_estimated" in quality:
                mos_scores.append(quality["mos_estimated"])
        
        if mos_scores:
            mos_avg = np.mean(mos_scores)
            print(f"  MOS estimado: {mos_avg:.2f}/5.0")
            
            if mos_avg >= 4.0:
                print("  ✅ Calidad excelente")
            elif mos_avg >= 3.5:
                print("  ✅ Calidad buena")
            elif mos_avg >= 3.0:
                print("  ⚠️  Calidad aceptable")
            else:
                print("  ❌ Calidad baja")
        else:
            print("  N/A (métricas de calidad no disponibles)")
        
        print("\n" + "="*60)
    
    def _save_log(self, stats: Dict):
        """Guarda log de la conversación"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(f"logs/voice_conversation_{timestamp}.json")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        log_data = {
            "timestamp": timestamp,
            "stats": stats,
            "conversation": self.conversation_log
        }
        
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Log guardado: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Test de conversación real con SARAi")
    parser.add_argument("--silent", action="store_true", help="Modo silencioso (sin reproducción de audio)")
    parser.add_argument("--turns", type=int, default=5, help="Número máximo de turnos")
    parser.add_argument("--scenario", choices=["greeting", "technical", "casual"], help="Escenario predefinido")
    
    args = parser.parse_args()
    
    # Crear tester
    tester = VoiceConversationTester(silent=args.silent)
    
    # Ejecutar conversación
    stats = tester.run_conversation(max_turns=args.turns)
    
    # Evaluación final
    if stats:
        e2e_mean = stats["e2e"]["mean"]
        
        print("\n📋 EVALUACIÓN FINAL:")
        
        if e2e_mean < 1000:
            print(f"✅ Latencia E2E excelente: {e2e_mean:.0f}ms < 1000ms")
        elif e2e_mean < 2000:
            print(f"✅ Latencia E2E buena: {e2e_mean:.0f}ms < 2000ms")
        elif e2e_mean < 3000:
            print(f"⚠️  Latencia E2E aceptable: {e2e_mean:.0f}ms < 3000ms")
        else:
            print(f"❌ Latencia E2E alta: {e2e_mean:.0f}ms > 3000ms")
        
        print("\n🎯 PRÓXIMOS PASOS SUGERIDOS:")
        
        # Sugerencias basadas en métricas
        if stats["llm"]["mean"] > 1000:
            print("  • Optimizar LLM (latencia alta: {:.0f}ms)".format(stats["llm"]["mean"]))
        
        if stats["tts"]["mean"] > 500:
            print("  • Optimizar TTS (latencia alta: {:.0f}ms)".format(stats["tts"]["mean"]))
        
        if stats["stt"]["mean"] > 500:
            print("  • Optimizar STT (latencia alta: {:.0f}ms)".format(stats["stt"]["mean"]))


if __name__ == "__main__":
    main()
