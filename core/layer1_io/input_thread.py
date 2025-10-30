"""
SARAi v2.17 - Input Thread (Canal IN)
Hilo que procesa audio → STT → Router → Decision
Siempre escuchando, procesamiento streaming
"""

import copy
import threading
import time
import numpy as np
from queue import Queue, Empty
from typing import Dict, Optional
import subprocess

from .vosk_streaming import VoskSTTStreaming, VoskStreamingSession
from .bert_embedder import BERTEmbedder
from .lora_router import LoRARouter
from .audio_emotion_lite import EmotionAudioLite, EmotionResult
from ..layer2_memory.tone_memory import ToneMemoryBuffer
from ..layer3_fluidity.tone_bridge import ToneStyleBridge


class InputThread:
    """
    Canal IN - Procesamiento continuo de entrada de audio
    
    Pipeline:
        Micrófono → VAD → Vosk STT → BERT → LoRA Router → Decisión
        
    Threads:
        1. Audio Capture (arecord)
        2. STT Processing (Vosk streaming)
        3. Routing (BERT + LoRA)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,  # 100ms chunks
        vad_energy_threshold: float = 0.01,  # Más sensible (antes: 0.02)
        silence_timeout: float = 1.5  # 1500ms silencio = fin de frase (antes: 0.5s)
    ):
        """
        Args:
            sample_rate: Sample rate del audio (16kHz para Vosk)
            chunk_duration_ms: Duración de cada chunk en ms
            vad_energy_threshold: Umbral de energía para VAD
            silence_timeout: Tiempo de silencio para detectar fin
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.vad_energy_threshold = vad_energy_threshold
        self.silence_timeout = silence_timeout
        
        # Componentes
        self.vosk_stt = None
        self.vosk_session = None
        self.bert_embedder = None
        self.lora_router = None
        self.audio_emotion = None
        self.current_sentence_audio = []
        self.tone_memory = ToneMemoryBuffer()
        self.tone_bridge = ToneStyleBridge()
        
        # Colas
        self.audio_queue = Queue(maxsize=50)  # Audio chunks
        self.text_queue = Queue(maxsize=20)   # Frases + audio
        self.decision_queue = Queue(maxsize=10)  # Decisiones router
        
        # Estado
        self.running = False
        self.threads = []
        
        # Estadísticas
        self.stats = {
            "chunks_processed": 0,
            "sentences_detected": 0,
            "trm_hits": 0,
            "llm_calls": 0,
            "translate_calls": 0,
            "tone_last_label": None,
            "tone_last_confidence": 0.0,
            "tone_valence_avg": 0.5,
            "tone_arousal_avg": 0.0,
            "tone_samples": 0,
            "tone_counts": {emo: 0 for emo in EmotionAudioLite.EMOTIONS},
            "tone_history": [],
            "tone_active_style": "neutral_support",
            "tone_filler_hint": "neutral_fillers"
        }
    
    def load_components(self):
        """Carga todos los componentes del pipeline"""
        print("\n🔧 Cargando componentes Canal IN...")
        
        # 1. Vosk STT
        self.vosk_stt = VoskSTTStreaming()
        self.vosk_session = VoskStreamingSession(self.vosk_stt)
        
        # 2. BERT Embedder
        self.bert_embedder = BERTEmbedder()
        
        # 3. LoRA Router
        router_path = "models/lora_router.pt"
        try:
            self.lora_router = LoRARouter.load(router_path)
            print(f"✓ LoRA Router cargado desde {router_path}")
        except FileNotFoundError:
            print(f"⚠️  LoRA Router no encontrado en {router_path}")
            print("   Creando router nuevo (sin entrenar)")
            self.lora_router = LoRARouter()

                # 4. Emotion Audio Lite (análisis de tono)
        print("   Cargando Emotion Audio Lite...")
        try:
            self.audio_emotion = EmotionAudioLite()
            
            # TEST: Validar que funciona
            test_audio = np.random.randn(16000).astype(np.float32) * 0.01
            test_result = self.audio_emotion.analyze(test_audio, sr=16000)
            
            print(f"   ✅ EmotionAudio OK: test={test_result.label} ({test_result.confidence:.2f})")
            
        except Exception as e:
            print(f"   ❌ EmotionAudio FALLÓ: {e}")
            print(f"      Sistema continuará sin detección de emociones")
            self.audio_emotion = None
        
        print("✅ Canal IN listo")
    
    def start(self):
        """Inicia todos los threads del canal IN"""
        if self.running:
            print("⚠️  Canal IN ya está corriendo")
            return
        
        self.running = True
        
        # Thread 1: Audio Capture
        t1 = threading.Thread(target=self._audio_capture_thread, daemon=True, name="AudioCapture")
        t1.start()
        self.threads.append(t1)
        
        # Thread 2: STT Processing
        t2 = threading.Thread(target=self._stt_processing_thread, daemon=True, name="STTProcessing")
        t2.start()
        self.threads.append(t2)
        
        # Thread 3: Routing
        t3 = threading.Thread(target=self._routing_thread, daemon=True, name="Routing")
        t3.start()
        self.threads.append(t3)
        
        print(f"✅ Canal IN iniciado ({len(self.threads)} threads)")
    
    def stop(self):
        """Detiene todos los threads"""
        print("\n🛑 Deteniendo Canal IN...")
        self.running = False
        
        for t in self.threads:
            t.join(timeout=2.0)
        
        self.threads.clear()
        print("✅ Canal IN detenido")
    
    def _audio_capture_thread(self):
        """Thread 1: Captura continua de audio con arecord"""
        print(f"[AudioCapture] Iniciado (chunks de {self.chunk_duration_ms}ms)")
        
        # Comando arecord
        cmd = [
            "arecord",
            "-f", "S16_LE",  # 16-bit little-endian
            "-c", "1",        # Mono
            "-r", str(self.sample_rate),
            "-t", "raw",      # Raw PCM
            "-"               # Output a stdout
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.chunk_samples * 2  # 2 bytes por sample (S16)
            )
            
            bytes_per_chunk = self.chunk_samples * 2
            
            while self.running:
                # Leer chunk
                audio_bytes = process.stdout.read(bytes_per_chunk)
                
                if len(audio_bytes) < bytes_per_chunk:
                    break
                
                # Convertir a float32
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                
                # VAD simple (energy-based)
                energy = np.sqrt(np.mean(audio_float32 ** 2))
                
                if energy > self.vad_energy_threshold:
                    # Audio con voz detectada
                    self.audio_queue.put(audio_float32)
                    self.stats["chunks_processed"] += 1
            
            process.terminate()
        
        except Exception as e:
            print(f"❌ [AudioCapture] Error: {e}")
    
    def _stt_processing_thread(self):
        """Thread 2: Procesa chunks de audio → texto"""
        print("[STTProcessing] Iniciado")

        while self.running:
            try:
                # Obtener chunk de audio
                audio_chunk = self.audio_queue.get(timeout=0.5)

                # Acumular audio para análisis de tono
                self.current_sentence_audio.append(audio_chunk)

                # Procesar con Vosk
                complete_sentence = self.vosk_session.feed_audio(audio_chunk)

                if complete_sentence is not None:
                    audio_for_sentence = None
                    if self.current_sentence_audio:
                        audio_for_sentence = np.concatenate(self.current_sentence_audio)
                    self.current_sentence_audio = []

                    if complete_sentence.strip():
                        self.text_queue.put({
                            "text": complete_sentence,
                            "audio": audio_for_sentence,
                            "timestamp": time.time()
                        })
                        self.stats["sentences_detected"] += 1

                        print(f"\n🎙️  Usuario: \"{complete_sentence}\"")

            except Empty:
                continue
            except Exception as e:
                print(f"❌ [STTProcessing] Error: {e}")
    
    def _routing_thread(self):
        """Thread 3: Procesa texto → BERT → LoRA Router → Decisión"""
        print("[Routing] Iniciado")
        
        while self.running:
            try:
                # Obtener texto + audio asociada
                utterance = self.text_queue.get(timeout=0.5)

                text = utterance["text"]
                audio_data = utterance.get("audio")
                timestamp = utterance.get("timestamp", time.time())

                start = time.perf_counter()
                
                # 1. BERT embeddings
                embedding = self.bert_embedder.encode(text)
                
                # 2. LoRA Router
                decision = self.lora_router.predict(embedding)

                # 3. Análisis de tono (audio)
                tone_result: EmotionResult
                if audio_data is not None and audio_data.size > 0 and self.audio_emotion is not None:
                    try:
                        tone_result = self.audio_emotion.analyze(audio_data, sr=self.sample_rate)
                        
                        # DEBUG: Mostrar emoción detectada
                        print(f"🎭 Emoción: {tone_result.label} (conf: {tone_result.confidence:.2f})")
                        print(f"   Valence: {tone_result.valence:.2f}, Arousal: {tone_result.arousal:.2f}")
                        
                    except Exception as e:
                        print(f"⚠️ Error analizando emoción: {e}")
                        tone_result = EmotionResult(
                            label="neutral",
                            confidence=0.5,
                            valence=0.5,
                            arousal=0.0,
                            probabilities={"neutral": 0.5},
                            features={"detected": False, "error": str(e)}
                        )
                else:
                    tone_result = EmotionResult(
                        label="silencio",
                        confidence=1.0,
                        valence=0.5,
                        arousal=0.0,
                        probabilities={"silencio": 1.0},
                        features={"detected": False}
                    )

                tone_profile = self.tone_bridge.snapshot()
                if tone_result.features.get("detected") is not False:
                    tone_entry = {
                        "timestamp": timestamp,
                        "text": text,
                        "label": tone_result.label,
                        "confidence": tone_result.confidence,
                        "valence": tone_result.valence,
                        "arousal": tone_result.arousal
                    }
                    self.tone_memory.append(tone_entry)
                    tone_profile = self.tone_bridge.update(
                        tone_result.label,
                        tone_result.valence,
                        tone_result.arousal
                    )
                
                routing_time = (time.perf_counter() - start) * 1000
                
                # Actualizar stats
                if decision["decision"] == "TRM":
                    self.stats["trm_hits"] += 1
                elif decision["decision"] == "LLM":
                    self.stats["llm_calls"] += 1
                elif decision["decision"] == "Traducir":
                    self.stats["translate_calls"] += 1

                self._update_tone_stats(tone_result)
                self.stats["tone_active_style"] = tone_profile.style
                self.stats["tone_filler_hint"] = tone_profile.filler_hint
                self.stats["tone_valence_avg"] = tone_profile.valence_avg
                self.stats["tone_arousal_avg"] = tone_profile.arousal_avg
                
                # Enviar a cola de decisiones
                audio_features = {
                    key: float(value) if isinstance(value, (int, float, np.floating)) else value
                    for key, value in tone_result.features.items()
                }

                self.decision_queue.put({
                    "text": text,
                    "embedding": embedding,
                    "decision": decision["decision"],
                    "confidence": decision["confidence"],
                    "scores": decision["scores"],
                    "routing_time_ms": routing_time,
                    "tone": {
                        "label": tone_result.label,
                        "confidence": tone_result.confidence,
                        "valence": tone_result.valence,
                        "arousal": tone_result.arousal,
                        "probabilities": tone_result.probabilities
                    },
                    "tone_profile": tone_profile.__dict__,
                    "tone_features": audio_features,
                    "timestamp": timestamp
                })
                
                print(
                    f"🧠 Router: {decision['decision']} (conf: {decision['confidence']:.2f}, "
                    f"{routing_time:.0f}ms) | tono={tone_result.label} ({tone_result.confidence:.2f})"
                )
            
            except Empty:
                continue
            except Exception as e:
                print(f"❌ [Routing] Error: {e}")
    
    def get_decision(self, timeout: float = 0.5) -> Optional[Dict]:
        """
        Obtiene próxima decisión del router (blocking)
        
        Args:
            timeout: Tiempo máximo de espera
        
        Returns:
            Dict con decisión o None si timeout
        """
        try:
            return self.decision_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del canal IN"""
        return copy.deepcopy(self.stats)
    
    def reset_stats(self):
        """Resetea estadísticas"""
        self.stats = {
            "chunks_processed": 0,
            "sentences_detected": 0,
            "trm_hits": 0,
            "llm_calls": 0,
            "translate_calls": 0,
            "tone_last_label": None,
            "tone_last_confidence": 0.0,
            "tone_valence_avg": 0.5,
            "tone_arousal_avg": 0.0,
            "tone_samples": 0,
            "tone_counts": {emo: 0 for emo in EmotionAudioLite.EMOTIONS},
            "tone_history": [],
            "tone_active_style": "neutral_support",
            "tone_filler_hint": "neutral_fillers"
        }

    def _update_tone_stats(self, tone_result: EmotionResult):
        """Actualiza métricas agregadas de tono."""
        if tone_result.features.get("detected") is False:
            return

        self.stats["tone_last_label"] = tone_result.label
        self.stats["tone_last_confidence"] = tone_result.confidence

        samples = self.stats["tone_samples"]
        self.stats["tone_valence_avg"] = (
            (self.stats["tone_valence_avg"] * samples) + tone_result.valence
        ) / (samples + 1)
        self.stats["tone_arousal_avg"] = (
            (self.stats["tone_arousal_avg"] * samples) + tone_result.arousal
        ) / (samples + 1)
        self.stats["tone_samples"] = samples + 1

        tone_counts = self.stats["tone_counts"]
        tone_counts[tone_result.label] = tone_counts.get(tone_result.label, 0) + 1

        history_entry = {
            "label": tone_result.label,
            "confidence": tone_result.confidence,
            "valence": tone_result.valence,
            "arousal": tone_result.arousal,
            "timestamp": time.time()
        }
        history = self.stats["tone_history"]
        history.append(history_entry)
        if len(history) > 10:
            del history[:-10]


# ============ Test ============
if __name__ == "__main__":
    print("=== Test Input Thread (Canal IN) ===\n")
    
    # Crear y cargar
    input_thread = InputThread()
    input_thread.load_components()
    
    # Iniciar
    input_thread.start()
    
    print("\n🎤 Habla naturalmente...")
    print("   El sistema procesará: Audio → STT → Router → Decisión")
    print("   Presiona Ctrl+C para salir\n")
    
    try:
        while True:
            # Obtener decisiones
            decision = input_thread.get_decision(timeout=1.0)
            
            if decision:
                print(f"\n📊 Decisión completa:")
                print(f"   Texto: \"{decision['text']}\"")
                print(f"   → {decision['decision']} (confianza: {decision['confidence']:.2f})")
                print(
                    f"   Scores: TRM={decision['scores']['TRM']:.2f}, "
                    f"LLM={decision['scores']['LLM']:.2f}, Traducir={decision['scores']['Traducir']:.2f}"
                )
                print(f"   Tiempo routing: {decision['routing_time_ms']:.0f}ms")
                tone = decision.get("tone")
                if tone:
                    print(
                        f"   Tono: {tone.get('label')} (conf: {tone.get('confidence', 0.0):.2f}, "
                        f"valencia={tone.get('valence', 0.0):.2f}, arousal={tone.get('arousal', 0.0):.2f})"
                    )
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Deteniendo...")
        input_thread.stop()
        
        # Stats finales
        stats = input_thread.get_stats()
        print(f"\n📊 Estadísticas finales:")
        print(f"   Chunks procesados: {stats['chunks_processed']}")
        print(f"   Frases detectadas: {stats['sentences_detected']}")
        print(f"   TRM hits: {stats['trm_hits']}")
        print(f"   LLM calls: {stats['llm_calls']}")
        print(f"   Traducir calls: {stats['translate_calls']}")
        if stats["tone_samples"]:
            print(
                f"   Tono promedio: valencia={stats['tone_valence_avg']:.2f}, "
                f"arousal={stats['tone_arousal_avg']:.2f}"
            )
            print(f"   Último tono: {stats['tone_last_label']} ({stats['tone_last_confidence']:.2f})")
            print(
                f"   Estilo activo: {stats['tone_active_style']} "
                f"| filler hint: {stats['tone_filler_hint']}"
            )
