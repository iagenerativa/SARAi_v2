#!/usr/bin/env python3
"""
SARAi v2.16.3 - Conversación de Voz en Streaming (ONLINE)
==========================================================

Conversación continua y natural sin turnos ni grabaciones.
- Detección automática de actividad de voz (VAD)
- Streaming bidireccional (STT → LLM → TTS)
- Reproducción inmediata del audio generado

Uso:
    python3 tests/test_voice_streaming.py
    
    Habla naturalmente. El sistema detecta cuándo empiezas y terminas.
    Presiona Ctrl+C para salir.
"""

import sys
import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
import queue
import threading
from collections import deque
import subprocess
import tempfile
import wave
import json

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.piper_tts import PiperTTSEngine
from llama_cpp import Llama

# Vosk para STT
from vosk import Model, KaldiRecognizer


class Colors:
    """ANSI color codes"""
    R = '\033[91m'  # Red
    G = '\033[92m'  # Green
    Y = '\033[93m'  # Yellow
    B = '\033[94m'  # Blue
    M = '\033[95m'  # Magenta
    C = '\033[96m'  # Cyan
    E = '\033[0m'   # End


class VoiceActivityDetector:
    """
    Detector de actividad de voz simple basado en energía.
    Detecta cuando el usuario empieza y termina de hablar.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: float = 0.03,  # 30ms
        energy_threshold: float = 0.02,
        silence_duration: float = 1.5   # 1.5s de silencio = fin
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration)
        self.energy_threshold = energy_threshold
        self.silence_frames = int(silence_duration / frame_duration)
        
        self.is_speaking = False
        self.silence_count = 0
        self.speech_frames = []
    
    def get_energy(self, frame: np.ndarray) -> float:
        """Calcula la energía RMS del frame."""
        return np.sqrt(np.mean(frame ** 2))
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Procesa un frame de audio.
        
        Returns:
            (is_speaking, speech_ended, audio_data)
            - is_speaking: True si el usuario está hablando ahora
            - speech_ended: True si acaba de terminar de hablar
            - audio_data: Array con el audio capturado (si speech_ended=True)
        """
        energy = self.get_energy(frame)
        
        if energy > self.energy_threshold:
            # Hay voz
            if not self.is_speaking:
                # Inicio de nueva frase
                self.is_speaking = True
                self.speech_frames = [frame]
                self.silence_count = 0
            else:
                # Continuación
                self.speech_frames.append(frame)
                self.silence_count = 0
            
            return (True, False, None)
        
        else:
            # Silencio
            if self.is_speaking:
                # Estamos hablando pero hay silencio
                self.speech_frames.append(frame)
                self.silence_count += 1
                
                if self.silence_count >= self.silence_frames:
                    # Fin de la frase
                    audio_data = np.concatenate(self.speech_frames)
                    self.is_speaking = False
                    self.speech_frames = []
                    self.silence_count = 0
                    
                    return (False, True, audio_data)
                
                return (True, False, None)
            
            return (False, False, None)


class StreamingVoiceConversation:
    """
    Sistema de conversación por voz en streaming continuo.
    """
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        
        # Audio config
        self.SAMPLE_RATE = 16000
        self.CHUNK = 480  # 30ms @ 16kHz
        
        # Queues para comunicación entre threads
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Control
        self.running = False
        
        # Stats
        self.conversation_count = 0
        
        print(f"{Colors.C}{'=' * 70}")
        print(f"   SARAi v2.16.3 - Conversación de Voz en Streaming")
        print(f"{'=' * 70}{Colors.E}\n")
    
    def load_components(self):
        """Carga todos los componentes del pipeline."""
        
        print(f"{Colors.Y}🔧 Cargando Componentes del Pipeline...{Colors.E}\n")
        
        # 1. Vosk STT (Speech-to-Text)
        print(f"{Colors.C}[1/3] Vosk STT (Español)...{Colors.E}", end=' ', flush=True)
        vosk_model_path = self.base_path / "models/vosk/vosk-model-small-es-0.42"
        self.vosk_model = Model(str(vosk_model_path))
        print(f"{Colors.G}✓{Colors.E}")
        
        # 2. LFM2
        print(f"{Colors.C}[2/3] LFM2-1.2B...{Colors.E}", end=' ', flush=True)
        lfm2_path = self.base_path / "models/lfm2/LFM2-1.2B-Q4_K_M.gguf"
        self.lfm2 = Llama(
            model_path=str(lfm2_path),
            n_ctx=512,
            n_threads=6,
            use_mmap=True,
            use_mlock=False,
            verbose=False
        )
        print(f"{Colors.G}✓{Colors.E}")
        
        # 3. Piper TTS
        print(f"{Colors.C}[3/3] Piper TTS (Voz Española)...{Colors.E}", end=' ', flush=True)
        self.piper_tts = PiperTTSEngine()
        print(f"{Colors.G}✓{Colors.E}")
        
        # 4. VAD
        self.vad = VoiceActivityDetector()
        
        print(f"\n{Colors.G}✅ Pipeline listo para conversación continua{Colors.E}\n")
    
    def audio_capture_thread(self):
        """Thread que captura audio del micrófono continuamente usando arecord."""
        
        print(f"{Colors.G}🎤 Micrófono activo - Habla naturalmente...{Colors.E}\n")
        
        # Usar arecord para captura continua
        cmd = [
            'arecord',
            '-f', 'S16_LE',
            '-c', '1',
            '-r', str(self.SAMPLE_RATE),
            '-t', 'raw',
            '--buffer-size=4800'  # 100ms buffer
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.CHUNK * 2
            )
            
            while self.running:
                # Leer chunk de audio
                data = process.stdout.read(self.CHUNK * 2)  # 2 bytes per sample
                if not data:
                    break
                
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # Procesar con VAD
                is_speaking, speech_ended, speech_audio = self.vad.process_frame(audio_float)
                
                if is_speaking and not speech_ended:
                    # Usuario hablando
                    if not hasattr(self, '_speaking_notified'):
                        print(f"{Colors.Y}🗣️  Escuchando...{Colors.E}", end='\r', flush=True)
                        self._speaking_notified = True
                
                elif speech_ended and speech_audio is not None:
                    # Usuario terminó de hablar
                    print(f"{Colors.G}✓ Frase capturada ({len(speech_audio)/self.SAMPLE_RATE:.1f}s){Colors.E}")
                    self.audio_queue.put(speech_audio)
                    self._speaking_notified = False
            
            process.terminate()
            
        except Exception as e:
            print(f"{Colors.R}❌ Error captura: {e}{Colors.E}")
            print(f"{Colors.Y}Asegúrate de tener un micrófono conectado{Colors.E}")
    
    def processing_thread(self):
        """Thread que procesa el audio capturado."""
        
        while self.running:
            try:
                # Esperar audio del usuario
                speech_audio = self.audio_queue.get(timeout=0.5)
                
                self.conversation_count += 1
                
                print(f"\n{Colors.C}{'─' * 70}")
                print(f"   Procesando frase #{self.conversation_count}")
                print(f"{'─' * 70}{Colors.E}")
                
                start_total = time.perf_counter()
                
                # 1. Vosk STT - Transcribir audio a texto
                print(f"{Colors.C}[1/3] Vosk STT → Texto...{Colors.E}", end=' ', flush=True)
                start = time.perf_counter()
                
                # Crear recognizer para este audio
                recognizer = KaldiRecognizer(self.vosk_model, self.SAMPLE_RATE)
                
                # Convertir audio a int16 para Vosk
                audio_int16 = (speech_audio * 32768).astype(np.int16)
                
                # Procesar audio
                recognizer.AcceptWaveform(audio_int16.tobytes())
                result = json.loads(recognizer.FinalResult())
                
                # Extraer texto transcrito
                transcribed_text = result.get('text', '')
                
                if not transcribed_text:
                    print(f"{Colors.Y}(no detectado){Colors.E}")
                    continue
                
                stt_time = (time.perf_counter() - start) * 1000
                print(f"{Colors.G}{stt_time:.0f}ms{Colors.E}")
                print(f"{Colors.Y}   Usuario dijo: \"{transcribed_text}\"{Colors.E}")
                
                # 2. LFM2 - Ahora procesa el TEXTO REAL transcrito
                print(f"{Colors.C}[2/3] LFM2 Razonamiento...{Colors.E}", end=' ', flush=True)
                start = time.perf_counter()
                
                self.lfm2.reset()
                # El prompt ahora tiene el texto REAL del usuario
                prompt = f"Usuario: {transcribed_text}\nAsistente:"
                
                response = self.lfm2.create_completion(
                    prompt,
                    max_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    echo=False
                )
                
                response_text = response['choices'][0]['text'].strip()
                
                llm_time = (time.perf_counter() - start) * 1000
                print(f"{Colors.G}{llm_time:.0f}ms{Colors.E}")
                
                # 3. Piper TTS - Sintetizar respuesta a voz
                print(f"{Colors.C}[3/3] Piper TTS → Voz...{Colors.E}", end=' ', flush=True)
                start = time.perf_counter()
                
                audio_output = self.piper_tts.synthesize(response_text)
                
                tts_time = (time.perf_counter() - start) * 1000
                print(f"{Colors.G}{tts_time:.0f}ms{Colors.E}")
                
                # Total
                total_time = (time.perf_counter() - start_total) * 1000
                print(f"{Colors.C}Total pipeline: {Colors.G}{total_time:.0f}ms{Colors.E}")
                
                # Enviar a reproducción
                self.response_queue.put((response_text, audio_output))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"{Colors.R}❌ Error procesamiento: {e}{Colors.E}")
                import traceback
                traceback.print_exc()
    
    def playback_thread(self):
        """Thread que reproduce las respuestas de voz usando aplay."""
        
        while self.running:
            try:
                # Esperar respuesta
                response_text, audio_output = self.response_queue.get(timeout=0.5)
                
                # Mostrar texto
                print(f"\n{Colors.M}💬 SARAi:{Colors.E}")
                print(f"   {response_text}")
                
                # Reproducir audio con aplay
                print(f"{Colors.C}🔊 Reproduciendo audio...{Colors.E}", end=' ', flush=True)
                start_play = time.perf_counter()
                
                # Crear archivo temporal WAV
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                    
                    # Escribir WAV
                    with wave.open(tmp_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(22050)
                        
                        audio_int16 = (audio_output * 32767).astype(np.int16)
                        wav_file.writeframes(audio_int16.tobytes())
                
                # Reproducir con aplay
                subprocess.run(
                    ['aplay', '-q', tmp_path],
                    check=False,
                    stderr=subprocess.DEVNULL
                )
                
                # Limpiar
                Path(tmp_path).unlink(missing_ok=True)
                
                play_time = (time.perf_counter() - start_play) * 1000
                print(f"{Colors.G}{play_time:.0f}ms{Colors.E}")
                
                print(f"{Colors.G}✓ Listo para siguiente frase{Colors.E}\n")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"{Colors.R}❌ Error reproducción: {e}{Colors.E}")
    
    def run(self):
        """Inicia la conversación continua."""
        
        self.running = True
        
        # Iniciar threads
        threads = [
            threading.Thread(target=self.audio_capture_thread, daemon=True),
            threading.Thread(target=self.processing_thread, daemon=True),
            threading.Thread(target=self.playback_thread, daemon=True)
        ]
        
        for t in threads:
            t.start()
        
        print(f"{Colors.G}{'=' * 70}")
        print(f"   🎙️  Sistema de Conversación Continua Activo")
        print(f"{'=' * 70}{Colors.E}")
        print(f"\n{Colors.Y}Instrucciones:{Colors.E}")
        print(f"  • Habla naturalmente cuando quieras")
        print(f"  • El sistema detecta automáticamente cuándo empiezas y terminas")
        print(f"  • Espera 1.5s de silencio para que detecte el fin de tu frase")
        print(f"  • Presiona Ctrl+C para salir\n")
        
        try:
            # Mantener vivo el programa
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.Y}Deteniendo conversación...{Colors.E}")
            self.running = False
            
            # Esperar a que terminen los threads
            time.sleep(1)
            
            print(f"\n{Colors.C}{'=' * 70}")
            print(f"   Estadísticas de la Sesión")
            print(f"{'=' * 70}{Colors.E}")
            print(f"   Frases procesadas: {self.conversation_count}")
            print(f"\n{Colors.G}✓ Sesión finalizada{Colors.E}\n")


def main():
    """Función principal."""
    
    conversation = StreamingVoiceConversation()
    conversation.load_components()
    conversation.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
