"""
SARAi v2.17 - Output Thread (Canal OUT)
Hilo que gestiona respuestas: TRM/LLM → TTS → Audio
Espera silencio del usuario antes de reproducir
"""

import threading
import time
import numpy as np
from queue import Queue, Empty
from typing import Dict, Optional
import subprocess
from pathlib import Path

# Importar agentes existentes
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from agents.melo_tts import MeloTTSEngine
from llama_cpp import Llama

# Importar módulos de tono (Capa 2 + Capa 3)
from core.layer2_memory.tone_memory import ToneMemoryBuffer
from core.layer3_fluidity.tone_bridge import ToneStyleBridge


class OutputThread:
    """
    Canal OUT - Procesamiento de respuestas con espera inteligente
    
    Pipeline:
        Decision Queue → [TRM Cache | LLM | Traducir] → Response Buffer →
        → Wait User Silence → Kitten TTS Streaming → Audio Playback
    
    Threads:
        1. Response Generation (TRM/LLM/NLLB)
        2. TTS Streaming (Kitten)
        3. Audio Playback (aplay)
    """
    
    def __init__(
        self,
        tts_model_path: str = None,  # No usado, MeloTTS se descarga automáticamente
        lfm2_model_path: str = "models/gguf/LFM2-1.2B-Q4_K_M.gguf",
        trm_cache_path: str = "state/trm_cache.json",
        user_silence_threshold: float = 0.3,  # 300ms sin VAD = usuario callado
        tone_memory_path: str = "state/tone_memory.jsonl"
    ):
        """
        Args:
            tts_model_path: No usado (compatibilidad), MeloTTS se descarga automáticamente
            lfm2_model_path: Ruta al modelo LFM2 GGUF
            trm_cache_path: Ruta al cache de respuestas TRM
            user_silence_threshold: Tiempo de silencio antes de reproducir
            tone_memory_path: Ruta al historial de tono emocional
        """
        self.tts_model_path = tts_model_path
        self.lfm2_model_path = lfm2_model_path
        self.trm_cache_path = trm_cache_path
        self.user_silence_threshold = user_silence_threshold
        self.tone_memory_path = tone_memory_path
        
        # Componentes
        self.melo_tts = None
        self.lfm2 = None
        self.trm_cache = {}
        
        # Componentes de tono (Capa 2 + Capa 3)
        self.tone_memory = None  # ToneMemoryBuffer
        self.tone_bridge = None  # ToneStyleBridge
        
        # Colas
        self.decision_queue = None  # Compartida con InputThread
        self.response_queue = Queue(maxsize=10)  # Respuestas generadas
        self.tts_queue = Queue(maxsize=5)        # Audio chunks para playback
        
        # Estado
        self.running = False
        self.user_speaking = False  # Flag compartido con InputThread
        self.threads = []
        
        # Estadísticas
        self.stats = {
            "trm_responses": 0,
            "llm_responses": 0,
            "translate_responses": 0,
            "avg_tts_latency_ms": 0,
            "avg_llm_latency_ms": 0,
            "total_responses": 0
        }
    
    def load_components(self):
        """Carga todos los componentes del pipeline"""
        print("\n🔧 Cargando componentes Canal OUT...")
        
        # 1. MeloTTS
        print("  [1/5] MeloTTS...")
        self.melo_tts = MeloTTSEngine(
            language='ES',
            speaker='ES',
            device='cpu',
            speed=1.3,  # 30% más rápido para reducir latencia
            preload=True  # Precarga el modelo al inicio
        )
        
        # 2. LFM2
        print("  [2/5] LFM2-1.2B...")
        start = time.perf_counter()
        self.lfm2 = Llama(
            model_path=self.lfm2_model_path,
            n_ctx=512,
            n_threads=4,
            verbose=False
        )
        load_time = (time.perf_counter() - start) * 1000
        print(f"    ✓ LFM2 cargado en {load_time:.0f}ms")
        
        # 3. TRM Cache
        print("  [3/5] TRM Cache...")
        self._load_trm_cache()
        
        # 4. Tone Memory (Capa 2)
        print("  [4/5] Tone Memory...")
        self.tone_memory = ToneMemoryBuffer(
            storage_path=self.tone_memory_path,
            max_entries=100
        )
        
        # 5. Tone Bridge (Capa 3)
        print("  [5/5] Tone Bridge...")
        self.tone_bridge = ToneStyleBridge(
            smoothing=0.25  # EMA smoothing
        )
        
        print("✅ Canal OUT listo (con contexto emocional)")
    
    def _load_trm_cache(self):
        """Carga cache de respuestas TRM desde JSON"""
        import json
        from pathlib import Path
        
        cache_file = Path(self.trm_cache_path)
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.trm_cache = json.load(f)
            print(f"    ✓ TRM Cache: {len(self.trm_cache)} respuestas cargadas")
        else:
            # Crear cache por defecto con respuestas comunes
            self.trm_cache = {
                "hola": "¡Hola! ¿En qué puedo ayudarte?",
                "cómo estás": "Muy bien, gracias por preguntar. ¿Y tú?",
                "adiós": "¡Hasta luego! Que tengas un buen día.",
                "gracias": "De nada, es un placer ayudarte.",
                "buenos días": "¡Buenos días! ¿Cómo puedo asistirte hoy?",
                "buenas tardes": "¡Buenas tardes! ¿En qué puedo ayudarte?",
                "buenas noches": "¡Buenas noches! ¿Necesitas algo?",
                "qué tal": "¡Todo bien! ¿Y tú qué tal?",
                "ayuda": "Claro, estoy aquí para ayudarte. ¿Qué necesitas?",
                "quién eres": "Soy SARAi, tu asistente de inteligencia artificial.",
                "qué puedes hacer": "Puedo ayudarte con consultas, conversaciones y tareas variadas. ¿Qué necesitas?",
            }
            
            # Guardar cache por defecto
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.trm_cache, f, ensure_ascii=False, indent=2)
            
            print(f"    ✓ TRM Cache creado con {len(self.trm_cache)} respuestas por defecto")
    
    def set_decision_queue(self, decision_queue: Queue):
        """Conecta con la cola de decisiones del InputThread"""
        self.decision_queue = decision_queue
    
    def set_user_speaking_flag(self, flag_ref):
        """Referencia al flag de usuario hablando (compartido con InputThread)"""
        self.user_speaking = flag_ref
    
    def update_user_tone(self, valencia: float, arousal: float, label: str):
        """
        Actualiza el tono emocional del usuario (llamado desde InputThread)
        
        Args:
            valencia: Valor de valencia [-1.0, 1.0]
            arousal: Valor de arousal [0.0, 1.0]
            label: Etiqueta emocional ('calm', 'happy', 'angry', etc.)
        """
        if self.tone_memory is None or self.tone_bridge is None:
            return
        
        # Registrar en memoria persistente
        entry = {
            "valence": valencia,
            "arousal": arousal,
            "label": label
        }
        self.tone_memory.append(entry)
        
        # Actualizar bridge (suavizado EMA)
        self.tone_bridge.update(label, valencia, arousal)
    
    def start(self):
        """Inicia todos los threads del canal OUT"""
        if self.running:
            print("⚠️  Canal OUT ya está corriendo")
            return
        
        if self.decision_queue is None:
            raise RuntimeError("decision_queue no configurada. Usar set_decision_queue()")
        
        self.running = True
        
        # Thread 1: Response Generation
        t1 = threading.Thread(
            target=self._response_generation_thread,
            daemon=True,
            name="ResponseGeneration"
        )
        t1.start()
        self.threads.append(t1)
        
        # Thread 2: TTS Streaming
        t2 = threading.Thread(
            target=self._tts_streaming_thread,
            daemon=True,
            name="TTSStreaming"
        )
        t2.start()
        self.threads.append(t2)
        
        # Thread 3: Audio Playback
        t3 = threading.Thread(
            target=self._audio_playback_thread,
            daemon=True,
            name="AudioPlayback"
        )
        t3.start()
        self.threads.append(t3)
        
        print(f"✅ Canal OUT iniciado ({len(self.threads)} threads)")
    
    def stop(self):
        """Detiene todos los threads"""
        print("\n🛑 Deteniendo Canal OUT...")
        self.running = False
        
        for t in self.threads:
            t.join(timeout=2.0)
        
        self.threads.clear()
        print("✅ Canal OUT detenido")
    
    def _response_generation_thread(self):
        """Thread 1: Genera respuestas según decisión del Router"""
        print("[ResponseGeneration] Iniciado")
        
        while self.running:
            try:
                # Obtener decisión del Router
                decision = self.decision_queue.get(timeout=0.5)
                
                text = decision["text"]
                route = decision["decision"]
                confidence = decision["confidence"]
                
                print(f"\n🔄 Generando respuesta: {route} (conf: {confidence:.2f})")
                
                start_gen = time.perf_counter()
                
                # Generar respuesta según ruta
                if route == "TRM":
                    response = self._generate_trm_response(text)
                    gen_time = (time.perf_counter() - start_gen) * 1000
                    self.stats["trm_responses"] += 1
                
                elif route == "LLM":
                    response = self._generate_llm_response(text)
                    gen_time = (time.perf_counter() - start_gen) * 1000
                    self.stats["llm_responses"] += 1
                    self.stats["avg_llm_latency_ms"] = (
                        (self.stats["avg_llm_latency_ms"] * (self.stats["llm_responses"] - 1) + gen_time)
                        / self.stats["llm_responses"]
                    )
                
                elif route == "Traducir":
                    response = self._generate_translate_response(text)
                    gen_time = (time.perf_counter() - start_gen) * 1000
                    self.stats["translate_responses"] += 1
                
                else:
                    print(f"⚠️  Ruta desconocida: {route}")
                    continue
                
                # Enviar a cola de respuestas
                self.response_queue.put({
                    "text": response,
                    "generation_time_ms": gen_time,
                    "route": route
                })
                
                print(f"✅ Respuesta generada en {gen_time:.0f}ms")
                self.stats["total_responses"] += 1
            
            except Empty:
                continue
            except Exception as e:
                print(f"❌ [ResponseGeneration] Error: {e}")
    
    def _generate_trm_response(self, text: str) -> str:
        """Genera respuesta desde cache TRM (< 50ms)"""
        # Normalizar texto
        text_normalized = text.lower().strip()
        
        # Buscar coincidencia exacta
        if text_normalized in self.trm_cache:
            return self.trm_cache[text_normalized]
        
        # Buscar coincidencia parcial (contiene)
        for key, response in self.trm_cache.items():
            if key in text_normalized or text_normalized in key:
                return response
        
        # Fallback: respuesta genérica
        return "Entiendo. ¿Puedes darme más detalles?"
    
    def _generate_llm_response(self, text: str) -> str:
        """Genera respuesta con LFM2"""
        prompt = f"Usuario: {text}\nAsistente:"
        
        self.lfm2.reset()  # Reset KV cache
        
        response = self.lfm2.create_completion(
            prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            stop=["Usuario:", "\n\n"]
        )
        
        return response['choices'][0]['text'].strip()
    
    def _generate_translate_response(self, text: str) -> str:
        """Genera respuesta con traducción NLLB (placeholder)"""
        # TODO: Integrar NLLB cuando esté disponible
        # Por ahora, fallback a LLM directo
        print("⚠️  NLLB no implementado, usando LLM directo")
        return self._generate_llm_response(text)
    
    def _tts_streaming_thread(self):
        """Thread 2: Convierte respuestas a audio con Kitten TTS"""
        print("[TTSStreaming] Iniciado")
        
        while self.running:
            try:
                # Obtener respuesta generada
                response = self.response_queue.get(timeout=0.5)
                
                response_text = response["text"]
                
                # Obtener hint de tono actual (si está disponible)
                tone_hint = "neutral"
                if self.tone_bridge is not None:
                    profile = self.tone_bridge.snapshot()
                    tone_hint = profile.filler_hint  # Usar filler_hint, no style_hint
                    print(f"🎭 Tono detectado: {tone_hint} (v={profile.valence_avg:.2f}, a={profile.arousal_avg:.2f})")
                
                print(f"\n💬 SARAi [{tone_hint}]: {response_text}")
                
                # NO ESPERAR silencio de forma bloqueante
                # En su lugar, lanzar TTS en thread separado que puede ser interrumpido
                threading.Thread(
                    target=self._async_tts_with_interruption,
                    args=(response_text, response["route"], tone_hint),
                    daemon=True
                ).start()
            
            except Empty:
                continue
            except Exception as e:
                print(f"❌ [TTSStreaming] Error: {e}")
    
    def _async_tts_with_interruption(self, text: str, route: str, tone_hint: str):
        """
        Genera y reproduce TTS de forma asíncrona
        Puede ser interrumpido si el usuario empieza a hablar
        """
        try:
            # Pequeña pausa para dar chance a usuario de interrumpir inmediatamente
            time.sleep(0.2)
            
            # Si usuario ya empezó a hablar, no reproducir
            if self.user_speaking:
                print("⚠️ Usuario habló primero, cancelando respuesta")
                return
            
            # Generar audio con MeloTTS
            print(f"🔊 Sintetizando audio...")
            start_tts = time.perf_counter()
            
            audio_output = self.melo_tts.synthesize(text)
            
            tts_time = (time.perf_counter() - start_tts) * 1000
            
            # Actualizar estadísticas
            self.stats["total_responses"] += 1
            if self.stats["total_responses"] > 0:
                self.stats["avg_tts_latency_ms"] = (
                    (self.stats["avg_tts_latency_ms"] * (self.stats["total_responses"] - 1) + tts_time)
                    / self.stats["total_responses"]
                )
            
            print(f"✅ Audio sintetizado en {tts_time:.0f}ms")
            
            # Comprobar de nuevo antes de reproducir
            if self.user_speaking:
                print("⚠️ Usuario empezó a hablar, cancelando reproducción")
                return
            
            # Enviar a cola de playback (con capacidad de interrupción)
            self.tts_queue.put({
                "audio": audio_output,
                "text": text,
                "route": route,
                "tone_hint": tone_hint
            })
            
        except Exception as e:
            print(f"❌ [AsyncTTS] Error: {e}")
    
    def _wait_for_user_silence(self):
        """Espera a que el usuario deje de hablar"""
        # Si el usuario está hablando, esperar
        wait_start = time.time()
        while self.user_speaking and (time.time() - wait_start) < 5.0:
            time.sleep(0.1)
        
        # Esperar threshold adicional para asegurar silencio
        time.sleep(self.user_silence_threshold)
    
    def _audio_playback_thread(self):
        """Thread 3: Reproduce audio con aplay (con capacidad de interrupción)"""
        print("[AudioPlayback] Iniciado")
        
        while self.running:
            try:
                # Obtener audio chunk
                audio_data = self.tts_queue.get(timeout=0.5)
                
                audio = audio_data["audio"]
                
                # Dividir audio en chunks de 100ms para poder interrumpir
                sample_rate = 44100
                chunk_duration_ms = 100
                chunk_size = int(sample_rate * chunk_duration_ms / 1000)
                
                print(f"🔊 Reproduciendo audio (con interrupción posible)...")
                
                # Reproducir en chunks pequeños
                for i in range(0, len(audio), chunk_size):
                    
                    # COMPROBAR: ¿Usuario empezó a hablar?
                    if self.user_speaking:
                        print("⚠️ Usuario interrumpió, deteniendo audio")
                        break  # ABORTAR reproducción
                    
                    # Obtener chunk
                    chunk = audio[i:i+chunk_size]
                    
                    # Reproducir chunk inmediatamente
                    self._play_audio_chunk_immediate(chunk, sample_rate)
                
                if not self.user_speaking:
                    print(f"✅ Audio reproducido completamente")
                else:
                    print(f"⏸️ Audio interrumpido por usuario")
            
            except Empty:
                continue
            except Exception as e:
                print(f"❌ [AudioPlayback] Error: {e}")
    
    def _play_audio_chunk_immediate(self, chunk: np.ndarray, sample_rate: int):
        """
        Reproduce un chunk de audio pequeño (<100ms) inmediatamente
        Usa método más rápido que crear archivo temporal
        """
        import tempfile
        import soundfile as sf
        import os
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, chunk, sample_rate)
        
        try:
            # Reproducir sin bloqueo prolongado
            subprocess.run(
                ["aplay", "-q", tmp_path],
                check=False,  # No lanzar excepción si falla
                stderr=subprocess.DEVNULL,
                timeout=0.2  # Timeout corto (200ms)
            )
        finally:
            # Limpiar
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del canal OUT"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Resetea estadísticas"""
        self.stats = {
            "trm_responses": 0,
            "llm_responses": 0,
            "translate_responses": 0,
            "avg_tts_latency_ms": 0,
            "avg_llm_latency_ms": 0,
            "total_responses": 0
        }


# ============ Test ============
if __name__ == "__main__":
    print("=== Test Output Thread (Canal OUT) ===\n")
    
    # Crear y cargar
    output_thread = OutputThread()
    output_thread.load_components()
    
    # Crear cola de decisiones simulada
    decision_queue = Queue()
    output_thread.set_decision_queue(decision_queue)
    
    # Iniciar
    output_thread.start()
    
    print("\n📝 Enviando decisiones de prueba...")
    
    # Test 1: TRM
    print("\n1️⃣ Test TRM (cache):")
    decision_queue.put({
        "text": "hola",
        "decision": "TRM",
        "confidence": 0.95,
        "scores": {"TRM": 0.95, "LLM": 0.03, "Traducir": 0.02}
    })
    
    time.sleep(3)
    
    # Test 2: LLM
    print("\n2️⃣ Test LLM:")
    decision_queue.put({
        "text": "¿Qué es la inteligencia artificial?",
        "decision": "LLM",
        "confidence": 0.88,
        "scores": {"TRM": 0.05, "LLM": 0.88, "Traducir": 0.07}
    })
    
    time.sleep(5)
    
    # Test 3: TRM otra respuesta
    print("\n3️⃣ Test TRM (otra respuesta):")
    decision_queue.put({
        "text": "gracias",
        "decision": "TRM",
        "confidence": 0.92,
        "scores": {"TRM": 0.92, "LLM": 0.05, "Traducir": 0.03}
    })
    
    time.sleep(3)
    
    # Detener
    print("\n🛑 Deteniendo test...")
    output_thread.stop()
    
    # Stats finales
    stats = output_thread.get_stats()
    print(f"\n📊 Estadísticas finales:")
    print(f"   Total respuestas: {stats['total_responses']}")
    print(f"   TRM: {stats['trm_responses']}")
    print(f"   LLM: {stats['llm_responses']}")
    print(f"   Traducir: {stats['translate_responses']}")
    print(f"   Latencia promedio TTS: {stats['avg_tts_latency_ms']:.0f}ms")
    print(f"   Latencia promedio LLM: {stats['avg_llm_latency_ms']:.0f}ms")
