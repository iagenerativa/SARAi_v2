"""
SARAi v2.17 - Orquestador Capa 1
Coordina Canal IN (Input) y Canal OUT (Output) en modo full-duplex
"""

import threading
import time
from queue import Queue
from typing import Dict
from pathlib import Path

from .input_thread import InputThread
from .output_thread import OutputThread


class Layer1Orchestrator:
    """
    Orquestador de Capa 1: I/O Full-Duplex
    
    Coordina:
        - Canal IN: Audio → STT → Router → Decisión
        - Canal OUT: Decisión → [TRM/LLM/NLLB] → TTS → Audio
    
    Sincronización:
        - Cola compartida de decisiones
        - Flag compartido de usuario hablando
        - Estadísticas unificadas
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        vad_energy_threshold: float = 0.01,  # Más sensible (antes: 0.02)
        silence_timeout: float = 1.5,  # 1500ms para fin de frase (antes: 0.5s)
        user_silence_threshold: float = 0.3
    ):
        """
        Args:
            sample_rate: Sample rate del audio
            chunk_duration_ms: Duración de chunks de audio
            vad_energy_threshold: Umbral VAD
            silence_timeout: Timeout para fin de frase
            user_silence_threshold: Espera antes de reproducir
        """
        # Configuración
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.vad_energy_threshold = vad_energy_threshold
        self.silence_timeout = silence_timeout
        self.user_silence_threshold = user_silence_threshold
        
        # Cola compartida (IN → OUT)
        self.decision_queue = Queue(maxsize=10)
        
        # Flag compartido (usuario hablando)
        self.user_speaking = threading.Event()
        
        # Canales
        self.input_thread = None
        self.output_thread = None
        
        # Estado
        self.running = False
        
        # Estadísticas unificadas
        self.stats = {
            "session_start": None,
            "session_duration_s": 0,
            "total_interactions": 0
        }
    
    def load_components(self):
        """Carga y configura ambos canales"""
        print("\n" + "=" * 70)
        print("   SARAi v2.17 - Orquestador Capa 1: I/O Full-Duplex")
        print("=" * 70)
        
        # Crear canales
        print("\n🔧 Inicializando canales...")
        
        # Canal IN
        self.input_thread = InputThread(
            sample_rate=self.sample_rate,
            chunk_duration_ms=self.chunk_duration_ms,
            vad_energy_threshold=self.vad_energy_threshold,
            silence_timeout=self.silence_timeout
        )
        self.input_thread.load_components()
        
        # Conectar cola compartida
        self.input_thread.decision_queue = self.decision_queue
        
        # Canal OUT
        self.output_thread = OutputThread(
            user_silence_threshold=self.user_silence_threshold
        )
        self.output_thread.load_components()
        
        # Conectar cola compartida
        self.output_thread.set_decision_queue(self.decision_queue)
        
        # TODO: Conectar flag de usuario hablando
        # (requiere modificar InputThread para actualizar el flag)
        
        print("\n✅ Orquestador Capa 1 listo")
        print("\n" + "=" * 70)
    
    def start(self):
        """Inicia ambos canales en modo full-duplex"""
        if self.running:
            print("⚠️  Orquestador ya está corriendo")
            return
        
        self.running = True
        self.stats["session_start"] = time.time()
        
        print("\n🚀 Iniciando modo full-duplex...")
        
        # Iniciar Canal IN
        self.input_thread.start()
        
        # Iniciar Canal OUT
        self.output_thread.start()
        
        print("\n✅ Sistema full-duplex activo")
        print("\n" + "=" * 70)
        print("   🎙️  SISTEMA LISTO - Habla naturalmente")
        print("=" * 70)
        print("\nInstrucciones:")
        print("  • El sistema está SIEMPRE escuchando")
        print("  • Habla naturalmente, sin esperar turnos")
        print("  • El sistema responderá cuando detecte silencio")
        print("  • Presiona Ctrl+C para detener")
        print("\n" + "=" * 70 + "\n")
    
    def stop(self):
        """Detiene ambos canales"""
        if not self.running:
            return
        
        print("\n" + "=" * 70)
        print("   🛑 Deteniendo sistema...")
        print("=" * 70)
        
        self.running = False
        
        # Detener canales
        if self.input_thread:
            self.input_thread.stop()
        
        if self.output_thread:
            self.output_thread.stop()
        
        # Calcular duración de sesión
        if self.stats["session_start"]:
            self.stats["session_duration_s"] = time.time() - self.stats["session_start"]
        
        print("✅ Sistema detenido")
    
    def get_unified_stats(self) -> Dict:
        """Retorna estadísticas unificadas de ambos canales"""
        stats_in = self.input_thread.get_stats() if self.input_thread else {}
        stats_out = self.output_thread.get_stats() if self.output_thread else {}
        
        # Calcular duración de sesión
        if self.running and self.stats["session_start"]:
            self.stats["session_duration_s"] = time.time() - self.stats["session_start"]
        
        return {
            "session": {
                "duration_s": self.stats["session_duration_s"],
                "total_interactions": stats_in.get("sentences_detected", 0)
            },
            "input": stats_in,
            "output": stats_out,
            "routing": {
                "trm_hit_rate": (
                    stats_in.get("trm_hits", 0) / max(stats_in.get("sentences_detected", 1), 1)
                ),
                "llm_usage_rate": (
                    stats_in.get("llm_calls", 0) / max(stats_in.get("sentences_detected", 1), 1)
                ),
                "translate_usage_rate": (
                    stats_in.get("translate_calls", 0) / max(stats_in.get("sentences_detected", 1), 1)
                )
            }
        }
    
    def print_stats(self):
        """Imprime estadísticas en formato legible"""
        stats = self.get_unified_stats()
        
        print("\n" + "=" * 70)
        print("   📊 ESTADÍSTICAS DEL SISTEMA")
        print("=" * 70)
        
        # Sesión
        print(f"\n🕐 Sesión:")
        print(f"   Duración: {stats['session']['duration_s']:.1f}s")
        print(f"   Interacciones: {stats['session']['total_interactions']}")
        
        # Canal IN
        print(f"\n📥 Canal IN:")
        print(f"   Chunks procesados: {stats['input'].get('chunks_processed', 0)}")
        print(f"   Frases detectadas: {stats['input'].get('sentences_detected', 0)}")
        tone_samples = stats['input'].get('tone_samples', 0)
        if tone_samples:
            print(
                f"   Último tono: {stats['input'].get('tone_last_label')} "
                f"({stats['input'].get('tone_last_confidence', 0.0):.2f})"
            )
            print(
                f"   Tono medio (n={tone_samples}): valencia={stats['input'].get('tone_valence_avg', 0.5):.2f}, "
                f"arousal={stats['input'].get('tone_arousal_avg', 0.0):.2f}"
            )
            print(
                f"   Estilo activo: {stats['input'].get('tone_active_style', 'neutral_support')} | "
                f"filler hint: {stats['input'].get('tone_filler_hint', 'neutral_fillers')}"
            )
        
        # Routing
        print(f"\n🧠 Routing:")
        print(f"   TRM hits: {stats['input'].get('trm_hits', 0)} ({stats['routing']['trm_hit_rate']*100:.1f}%)")
        print(f"   LLM calls: {stats['input'].get('llm_calls', 0)} ({stats['routing']['llm_usage_rate']*100:.1f}%)")
        print(f"   Traducir: {stats['input'].get('translate_calls', 0)} ({stats['routing']['translate_usage_rate']*100:.1f}%)")
        
        # Canal OUT
        print(f"\n📤 Canal OUT:")
        print(f"   Respuestas totales: {stats['output'].get('total_responses', 0)}")
        print(f"   TRM: {stats['output'].get('trm_responses', 0)}")
        print(f"   LLM: {stats['output'].get('llm_responses', 0)}")
        print(f"   Latencia TTS: {stats['output'].get('avg_tts_latency_ms', 0):.0f}ms")
        print(f"   Latencia LLM: {stats['output'].get('avg_llm_latency_ms', 0):.0f}ms")
        
        print("\n" + "=" * 70)
    
    def run_interactive(self):
        """Ejecuta en modo interactivo (blocking)"""
        try:
            # Mantener vivo mientras los threads trabajan
            while self.running:
                time.sleep(1.0)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupción detectada...")
        
        finally:
            self.stop()
            self.print_stats()


# ============ Main ============
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   SARAi v2.17 - Test Capa 1 Completa")
    print("=" * 70)
    
    # Crear orquestador
    orchestrator = Layer1Orchestrator()
    
    # Cargar componentes
    orchestrator.load_components()
    
    # Iniciar sistema
    orchestrator.start()
    
    # Ejecutar interactivo
    orchestrator.run_interactive()
