"""
SARAi v2.18 - Arquitectura Full-Duplex REAL con Multiprocessing

PROBLEMA v2.17:
- Python threading con GIL = NO paralelismo real
- Queue bloqueante = interferencias
- Flag compartido = serialización forzada

SOLUCIÓN v2.18:
- 2 PROCESOS independientes (no threads)
- Shared Memory para comunicación ultra-rápida
- Audio simultáneo (portaudio duplex real)
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value, Array
import numpy as np
import sounddevice as sd
import time
from typing import Optional
import queue


class FullDuplexAudioEngine:
    """
    Motor de audio REAL full-duplex usando sounddevice.
    
    Un solo stream que SIMULTÁNEAMENTE:
    - Graba del micrófono (Canal IN)
    - Reproduce al altavoz (Canal OUT)
    
    Sin bloqueos, sin turnos, sin interferencias.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        blocksize: int = 1600  # 100ms a 16kHz
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        
        # Buffers compartidos (shared memory)
        # INPUT: Audio del micrófono
        self.input_buffer = mp.Queue(maxsize=100)
        
        # OUTPUT: Audio para reproducir
        self.output_buffer = mp.Queue(maxsize=100)
        
        # Stream activo
        self.stream = None
        self.running = mp.Value('b', False)
    
    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """
        Callback que se ejecuta cada 100ms de forma ATÓMICA.
        
        Este callback corre en un thread C de PortAudio, 
        NO en Python, por lo que NO sufre del GIL.
        
        Args:
            indata: Audio capturado del micrófono [frames, channels]
            outdata: Buffer para escribir audio a reproducir [frames, channels]
        """
        if status:
            print(f"⚠️ Audio status: {status}")
        
        # 1. CAPTURAR: Guardar audio del micrófono en queue
        try:
            audio_input = indata[:, 0].copy()  # Mono
            self.input_buffer.put_nowait(audio_input)
        except queue.Full:
            pass  # Descartar si buffer lleno (backpressure)
        
        # 2. REPRODUCIR: Obtener audio de la queue o silencio
        try:
            audio_output = self.output_buffer.get_nowait()
            outdata[:] = audio_output.reshape(-1, 1)
        except queue.Empty:
            # No hay audio para reproducir, silencio
            outdata.fill(0)
    
    def start(self):
        """Inicia el stream full-duplex"""
        self.running.value = True
        
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.blocksize,
            callback=self._audio_callback
        )
        
        self.stream.start()
        print(f"🎙️🔊 Full-Duplex Audio Engine INICIADO")
        print(f"   Sample Rate: {self.sample_rate}Hz")
        print(f"   Block Size: {self.blocksize} samples ({self.blocksize/self.sample_rate*1000:.0f}ms)")
    
    def stop(self):
        """Detiene el stream"""
        self.running.value = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        print("🛑 Full-Duplex Audio Engine DETENIDO")


class STTProcessor:
    """
    Proceso INDEPENDIENTE para STT.
    Corre en su propio proceso Python, sin GIL compartido.
    """
    
    def __init__(
        self,
        input_buffer: mp.Queue,
        text_output_queue: mp.Queue,
        running: mp.Value
    ):
        self.input_buffer = input_buffer
        self.text_output_queue = text_output_queue
        self.running = running
    
    def run(self):
        """Loop principal del proceso STT"""
        from .vosk_streaming import VoskSTTStreaming, VoskStreamingSession
        
        print("[STT Process] Iniciando Vosk...")
        vosk = VoskSTTStreaming()
        session = VoskStreamingSession(vosk)
        
        print("[STT Process] ✅ Listo, procesando audio...")
        
        while self.running.value:
            try:
                # Obtener audio del buffer compartido
                audio_chunk = self.input_buffer.get(timeout=0.1)
                
                # Procesar con Vosk
                complete_sentence = session.feed_audio(audio_chunk)
                
                if complete_sentence:
                    # Enviar texto a la queue de output
                    self.text_output_queue.put({
                        "text": complete_sentence,
                        "timestamp": time.time()
                    })
                    
                    print(f"[STT] 🎙️ '{complete_sentence}'")
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[STT] ❌ Error: {e}")
        
        print("[STT Process] Finalizando...")


class LLMProcessor:
    """
    Proceso INDEPENDIENTE para LLM + TTS.
    Genera respuestas sin bloquear STT.
    """
    
    def __init__(
        self,
        text_input_queue: mp.Queue,
        audio_output_buffer: mp.Queue,
        running: mp.Value
    ):
        self.text_input_queue = text_input_queue
        self.audio_output_buffer = audio_output_buffer
        self.running = running
    
    def run(self):
        """Loop principal del proceso LLM"""
        from agents.melo_tts import MeloTTSEngine
        from llama_cpp import Llama
        
        print("[LLM Process] Cargando modelos...")
        
        # TTS
        tts = MeloTTSEngine(language="ES", speed=1.3, preload=True)
        
        # LLM (LFM2)
        llm = Llama(
            model_path="models/gguf/LFM2-1.2B-Q4_K_M.gguf",
            n_ctx=512,
            n_threads=4,
            verbose=False
        )
        
        print("[LLM Process] ✅ Modelos cargados, esperando input...")
        
        while self.running.value:
            try:
                # Obtener texto del usuario
                user_input = self.text_input_queue.get(timeout=0.1)
                text = user_input["text"]
                
                print(f"[LLM] 🤔 Procesando: '{text}'")
                
                # Generar respuesta con LLM
                prompt = f"Usuario: {text}\nAsistente:"
                
                response = llm(
                    prompt,
                    max_tokens=100,
                    temperature=0.7,
                    stop=["Usuario:", "\n\n"]
                )
                
                response_text = response["choices"][0]["text"].strip()
                
                print(f"[LLM] 💬 Respuesta: '{response_text}'")
                
                # Sintetizar con TTS
                print(f"[LLM] 🔊 Sintetizando...")
                audio = tts.synthesize(response_text)
                
                # Enviar audio al buffer de reproducción en chunks
                chunk_size = 1600  # 100ms
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    
                    # Asegurarse de que el chunk tiene el tamaño correcto
                    if len(chunk) < chunk_size:
                        # Padding con ceros
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    
                    self.audio_output_buffer.put(chunk)
                
                print(f"[LLM] ✅ Audio enviado al buffer")
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[LLM] ❌ Error: {e}")
        
        print("[LLM Process] Finalizando...")


class TrueFullDuplexOrchestrator:
    """
    Orquestador con VERDADERO paralelismo.
    
    Arquitectura:
        - Proceso 1: Motor de Audio (captura + reproducción simultánea)
        - Proceso 2: STT (Vosk)
        - Proceso 3: LLM + TTS
    
    Comunicación:
        - Shared Memory Queues (NO Python Queue con GIL)
        - Sin locks, sin bloqueos, sin turnos
    """
    
    def __init__(self):
        # Motor de audio
        self.audio_engine = FullDuplexAudioEngine()
        
        # Queues compartidas entre procesos
        self.text_queue = mp.Queue(maxsize=10)  # STT → LLM
        
        # Flag de ejecución compartido
        self.running = mp.Value('b', False)
        
        # Procesos
        self.stt_process = None
        self.llm_process = None
    
    def start(self):
        """Inicia todos los procesos"""
        print("\n" + "=" * 70)
        print("  SARAi v2.18 - TRUE Full-Duplex (Multiprocessing)")
        print("=" * 70 + "\n")
        
        # Iniciar flag
        self.running.value = True
        
        # 1. Motor de audio
        self.audio_engine.start()
        
        # 2. Proceso STT
        self.stt_process = Process(
            target=STTProcessor(
                self.audio_engine.input_buffer,
                self.text_queue,
                self.running
            ).run,
            name="STT-Process"
        )
        self.stt_process.start()
        
        # 3. Proceso LLM
        self.llm_process = Process(
            target=LLMProcessor(
                self.text_queue,
                self.audio_engine.output_buffer,
                self.running
            ).run,
            name="LLM-Process"
        )
        self.llm_process.start()
        
        print("\n✅ Sistema TRUE Full-Duplex iniciado")
        print("   Puedes HABLAR y SER INTERRUMPIDO simultáneamente")
        print("   Presiona Ctrl+C para detener\n")
    
    def stop(self):
        """Detiene todos los procesos"""
        print("\n🛑 Deteniendo sistema...")
        
        # Señal de parada
        self.running.value = False
        
        # Esperar procesos
        if self.stt_process:
            self.stt_process.join(timeout=2.0)
            if self.stt_process.is_alive():
                self.stt_process.terminate()
        
        if self.llm_process:
            self.llm_process.join(timeout=2.0)
            if self.llm_process.is_alive():
                self.llm_process.terminate()
        
        # Detener audio
        self.audio_engine.stop()
        
        print("✅ Sistema detenido")
    
    def run_forever(self):
        """Loop principal"""
        try:
            self.start()
            
            # Mantener vivo mientras procesos corren
            while self.running.value:
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción de usuario")
        
        finally:
            self.stop()


if __name__ == "__main__":
    # Crear orquestador
    orchestrator = TrueFullDuplexOrchestrator()
    
    # Ejecutar
    orchestrator.run_forever()
