"""
Sherpa Coordinator - Supervisor conversacional completo (Capa 3)

Responsabilidades:
1. Gestión de Turnos: Detecta speech_end → evalúa contexto → decide acción
2. Fillers: Despacha muletillas ("un momento", "déjame pensar") para evitar silencios
3. Backpressure: Retiene payloads si el canal OUT está saturado
4. Fault Sentinel: Watchdogs para STT, TTS y traducción, activa plan B

Máquina de Estados:
- IDLE: Esperando evento (speech_end, payload_ready, silence_timer_hit, fault_detected)
- EVALUATE_CONTEXT: Analiza si necesita filler, payload directo o esperar
- DELIVER_PAYLOAD: Publica respuesta al usuario
- QUEUE_WAIT: Backpressure activo, esperando capacidad en OUT
- DISPATCH_FILLER: Emite muletilla por canal TTS de emergencia
- REPAIR: Sentinel activo, ejecutando plan B (STT/TTS/traducción)

Integración:
- Recibe señales: orchestrator.signal_speech_end() → sherpa.on_speech_end()
- Publica: sherpa.dispatch_filler() → tts_queue.put(filler_audio)
- Coordina: decision_queue (router) → response_queue (LLM) → tts_queue (Kitten)
"""

import threading
import time
import json
import logging
import random
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


class SherpaState(Enum):
    """Estados de la máquina Sherpa"""
    IDLE = "idle"
    EVALUATE_CONTEXT = "evaluate_context"
    DELIVER_PAYLOAD = "deliver_payload"
    QUEUE_WAIT = "queue_wait"
    DISPATCH_FILLER = "dispatch_filler"
    REPAIR = "repair"


@dataclass
class SherpaSignal:
    """Evento recibido por Sherpa"""
    signal_type: str  # 'speech_end', 'payload_ready', 'silence_timer_hit', 'fault_detected'
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class FillerCatalogEntry:
    """Entrada del catálogo de fillers"""
    text: str
    style: str  # 'neutral', 'empathetic', 'urgent', etc.
    duration_ms: int  # Duración estimada de síntesis (para planificación)


class SherpaCoordinator:
    """
    Supervisor conversacional que coordina turnos, fillers y backpressure
    
    Usage:
        sherpa = SherpaCoordinator(
            filler_catalog_path="state/fillers_catalog.json",
            decision_queue=decision_queue,
            response_queue=response_queue,
            tts_queue=tts_queue
        )
        sherpa.start()
        
        # Desde orchestrator
        sherpa.on_speech_end()
        
        # Desde output_thread
        sherpa.on_payload_ready(text="Respuesta del LLM")
    """
    
    def __init__(
        self,
        filler_catalog_path: str,
        decision_queue: Any,  # Queue compartida con InputThread
        response_queue: Any,  # Queue compartida con OutputThread
        tts_queue: Any,  # Queue directa a TTS (para fillers de emergencia)
        max_queue_size: int = 5,  # Umbral de backpressure
        silence_threshold_ms: int = 3000,  # Tiempo máximo sin audio antes de filler
    ):
        self.filler_catalog_path = Path(filler_catalog_path)
        self.decision_queue = decision_queue
        self.response_queue = response_queue
        self.tts_queue = tts_queue
        self.max_queue_size = max_queue_size
        self.silence_threshold_ms = silence_threshold_ms
        
        # Estado
        self.state = SherpaState.IDLE
        self.state_lock = threading.RLock()
        self.signal_queue = deque(maxlen=100)  # Eventos pendientes
        
        # Catálogo de fillers
        self.filler_catalog: Dict[str, List[FillerCatalogEntry]] = {}
        self._load_filler_catalog()
        
        # Timers
        self.last_user_speech_time: Optional[float] = None
        self.last_system_output_time: Optional[float] = None
        self.silence_timer: Optional[threading.Timer] = None
        
        # Watchdogs (Fault Sentinel)
        self.stt_watchdog_active = False
        self.tts_watchdog_active = False
        self.translation_watchdog_active = False
        
        # Métricas
        self.metrics = {
            'fillers_dispatched': 0,
            'backpressure_events': 0,
            'faults_detected': 0,
            'avg_turn_latency_ms': 0.0,
        }
        
        # Thread control
        self._running = False
        self._coordinator_thread: Optional[threading.Thread] = None
    
    def _load_filler_catalog(self):
        """Carga catálogo de muletillas desde JSON"""
        if not self.filler_catalog_path.exists():
            logger.warning(f"Catálogo de fillers no encontrado: {self.filler_catalog_path}")
            # Fallback: fillers mínimos hardcoded
            self.filler_catalog = {
                'neutral': [
                    FillerCatalogEntry(text="un momento", style="neutral", duration_ms=800),
                    FillerCatalogEntry(text="déjame pensar", style="neutral", duration_ms=1200),
                ],
                'empathetic': [
                    FillerCatalogEntry(text="entiendo", style="empathetic", duration_ms=700),
                    FillerCatalogEntry(text="claro", style="empathetic", duration_ms=500),
                ],
            }
            return
        
        try:
            with open(self.filler_catalog_path, 'r', encoding='utf-8') as f:
                catalog_raw = json.load(f)
            
            # Parsear a FillerCatalogEntry
            for style, entries in catalog_raw.items():
                self.filler_catalog[style] = [
                    FillerCatalogEntry(**entry) for entry in entries
                ]
            
            logger.info(f"Catálogo de fillers cargado: {sum(len(v) for v in self.filler_catalog.values())} entradas")
        except Exception as e:
            logger.error(f"Error cargando catálogo de fillers: {e}", exc_info=True)
            # Fallback
            self.filler_catalog = {
                'neutral': [FillerCatalogEntry(text="un momento", style="neutral", duration_ms=800)]
            }
    
    def start(self):
        """Inicia el hilo coordinador"""
        if self._running:
            logger.warning("Sherpa ya está corriendo")
            return
        
        self._running = True
        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop,
            daemon=True,
            name="SherpaCoordinator"
        )
        self._coordinator_thread.start()
        logger.info("🧭 Sherpa Coordinator iniciado")
    
    def stop(self):
        """Detiene el coordinador"""
        self._running = False
        if self._coordinator_thread:
            self._coordinator_thread.join(timeout=2.0)
        if self.silence_timer:
            self.silence_timer.cancel()
        logger.info("🧭 Sherpa Coordinator detenido")
    
    # ========== Señales de entrada ==========
    
    def on_speech_end(self):
        """
        Llamado por orchestrator cuando detecta fin de habla del usuario
        
        Dispara evaluación de contexto: ¿necesita filler? ¿payload listo?
        """
        self.last_user_speech_time = time.time()
        self._enqueue_signal(SherpaSignal(
            signal_type='speech_end',
            timestamp=self.last_user_speech_time,
            metadata={}
        ))
        
        # Iniciar silence timer
        self._start_silence_timer()
    
    def on_payload_ready(self, text: str, metadata: Optional[Dict] = None):
        """
        Llamado por OutputThread cuando el LLM generó una respuesta
        
        Sherpa decide si publicar inmediatamente o retener por backpressure
        """
        self._enqueue_signal(SherpaSignal(
            signal_type='payload_ready',
            timestamp=time.time(),
            metadata={'text': text, 'extra': metadata or {}}
        ))
    
    def on_fault_detected(self, fault_type: str, details: Dict):
        """
        Llamado por watchdogs (STT, TTS, translation) cuando detectan fallo
        
        Activa plan B: reintentos, fallback a modo texto, etc.
        """
        logger.error(f"⚠️ Fault detectado: {fault_type} - {details}")
        self.metrics['faults_detected'] += 1
        
        self._enqueue_signal(SherpaSignal(
            signal_type='fault_detected',
            timestamp=time.time(),
            metadata={'fault_type': fault_type, 'details': details}
        ))
    
    def _enqueue_signal(self, signal: SherpaSignal):
        """Añade señal a la cola de procesamiento"""
        with self.state_lock:
            self.signal_queue.append(signal)
    
    # ========== Máquina de estados ==========
    
    def _coordinator_loop(self):
        """Loop principal del coordinador"""
        while self._running:
            try:
                # Procesar señales pendientes
                if self.signal_queue:
                    with self.state_lock:
                        signal = self.signal_queue.popleft()
                    self._process_signal(signal)
                else:
                    time.sleep(0.05)  # 50ms polling
            except Exception as e:
                logger.error(f"Error en Sherpa loop: {e}", exc_info=True)
                time.sleep(0.1)
    
    def _process_signal(self, signal: SherpaSignal):
        """Procesa señal según estado actual"""
        logger.debug(f"Sherpa [{self.state.value}] <- {signal.signal_type}")
        
        if signal.signal_type == 'speech_end':
            self._handle_speech_end(signal)
        
        elif signal.signal_type == 'payload_ready':
            self._handle_payload_ready(signal)
        
        elif signal.signal_type == 'silence_timer_hit':
            self._handle_silence_timer_hit(signal)
        
        elif signal.signal_type == 'fault_detected':
            self._handle_fault(signal)
    
    def _handle_speech_end(self, signal: SherpaSignal):
        """Usuario terminó de hablar → evaluar contexto"""
        with self.state_lock:
            self.state = SherpaState.EVALUATE_CONTEXT
        
        # Verificar si hay payload pendiente en response_queue
        if self.response_queue.qsize() > 0:
            # Payload ya listo, publicar directamente
            self._transition_to_deliver_payload()
        else:
            # Payload aún no listo, monitorear silence timer
            logger.debug("Payload no listo, esperando...")
            self.state = SherpaState.IDLE
    
    def _handle_payload_ready(self, signal: SherpaSignal):
        """LLM generó respuesta → decidir publicar o retener"""
        with self.state_lock:
            # Verificar backpressure
            if self.tts_queue.qsize() >= self.max_queue_size:
                logger.warning(f"⚡ Backpressure activo: TTS queue = {self.tts_queue.qsize()}")
                self.metrics['backpressure_events'] += 1
                self.state = SherpaState.QUEUE_WAIT
                # Retener payload, esperar a que baje la cola
                return
            
            # Canal libre, publicar
            self._transition_to_deliver_payload(signal.metadata['text'])
    
    def _handle_silence_timer_hit(self, signal: SherpaSignal):
        """Pasó el tiempo de silencio sin respuesta → despachar filler"""
        logger.info("⏱️ Silence timer expiró, despachando filler")
        self._transition_to_dispatch_filler()
    
    def _handle_fault(self, signal: SherpaSignal):
        """Watchdog detectó fallo → activar plan B"""
        with self.state_lock:
            self.state = SherpaState.REPAIR
        
        fault_type = signal.metadata['fault_type']
        
        if fault_type == 'stt_timeout':
            logger.warning("STT timeout → reiniciar Vosk")
            # TODO: Implementar reintento de STT
            self._dispatch_filler(style='urgent', text="disculpa, no te escuché bien")
        
        elif fault_type == 'tts_crash':
            logger.error("TTS crash → fallback a texto plano")
            # TODO: Cambiar a modo texto
        
        elif fault_type == 'translation_error':
            logger.error("Traducción falló → usar respuesta en español")
            # TODO: Publicar respuesta sin traducir
        
        # Volver a IDLE
        with self.state_lock:
            self.state = SherpaState.IDLE
    
    # ========== Transiciones ==========
    
    def _transition_to_deliver_payload(self, text: Optional[str] = None):
        """Publica respuesta al canal TTS"""
        with self.state_lock:
            self.state = SherpaState.DELIVER_PAYLOAD
        
        # Cancelar silence timer (ya no es necesario)
        if self.silence_timer:
            self.silence_timer.cancel()
        
        # Si no se pasó texto, obtener de response_queue
        if text is None:
            try:
                response_data = self.response_queue.get(timeout=0.1)
                text = response_data.get('text', '')
            except:
                logger.warning("No hay payload en response_queue")
                self.state = SherpaState.IDLE
                return
        
        # Publicar a TTS
        self.tts_queue.put({'text': text, 'priority': 'normal'})
        self.last_system_output_time = time.time()
        
        # Calcular latencia de turno
        if self.last_user_speech_time:
            latency_ms = (self.last_system_output_time - self.last_user_speech_time) * 1000
            self.metrics['avg_turn_latency_ms'] = (
                0.9 * self.metrics['avg_turn_latency_ms'] + 0.1 * latency_ms
            )
        
        with self.state_lock:
            self.state = SherpaState.IDLE
    
    def _transition_to_dispatch_filler(self, style: Optional[str] = None):
        """Despacha muletilla de emergencia"""
        with self.state_lock:
            self.state = SherpaState.DISPATCH_FILLER
        
        self._dispatch_filler(style=style or 'neutral')
        
        # Reiniciar silence timer (puede necesitar otro filler)
        self._start_silence_timer()
        
        with self.state_lock:
            self.state = SherpaState.IDLE
    
    def _dispatch_filler(self, style: str = 'neutral', text: Optional[str] = None):
        """
        Emite filler por canal TTS de emergencia
        
        Args:
            style: Estilo emocional del filler
            text: Texto específico (opcional, usa catálogo si None)
        """
        if text is None:
            # Seleccionar filler del catálogo
            fillers = self.filler_catalog.get(style, self.filler_catalog.get('neutral', []))
            if not fillers:
                logger.warning(f"No hay fillers para estilo '{style}'")
                return
            
            filler_entry = random.choice(fillers)
            text = filler_entry.text
        
        logger.info(f"🗣️ Filler despachado: '{text}' (estilo: {style})")
        self.metrics['fillers_dispatched'] += 1
        
        # Publicar con prioridad alta (saltar cola)
        self.tts_queue.put({'text': text, 'priority': 'filler'})
    
    # ========== Silence Timer ==========
    
    def _start_silence_timer(self):
        """Inicia timer de silencio máximo"""
        if self.silence_timer:
            self.silence_timer.cancel()
        
        self.silence_timer = threading.Timer(
            self.silence_threshold_ms / 1000.0,
            self._on_silence_timer_callback
        )
        self.silence_timer.daemon = True
        self.silence_timer.start()
    
    def _on_silence_timer_callback(self):
        """Callback cuando expira el silence timer"""
        self._enqueue_signal(SherpaSignal(
            signal_type='silence_timer_hit',
            timestamp=time.time(),
            metadata={}
        ))
    
    # ========== API pública ==========
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas del coordinador"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reinicia contadores de métricas"""
        self.metrics = {
            'fillers_dispatched': 0,
            'backpressure_events': 0,
            'faults_detected': 0,
            'avg_turn_latency_ms': 0.0,
        }


if __name__ == '__main__':
    # Test básico del coordinador
    logging.basicConfig(level=logging.DEBUG)
    from queue import Queue
    
    decision_q = Queue()
    response_q = Queue()
    tts_q = Queue()
    
    sherpa = SherpaCoordinator(
        filler_catalog_path="state/fillers_catalog.json",
        decision_queue=decision_q,
        response_queue=response_q,
        tts_queue=tts_q
    )
    
    sherpa.start()
    
    # Simular speech_end
    print("Simulando speech_end...")
    sherpa.on_speech_end()
    
    time.sleep(2)
    
    # Simular payload_ready
    print("Simulando payload_ready...")
    response_q.put({'text': 'Esta es una respuesta del LLM'})
    sherpa.on_payload_ready(text='Esta es una respuesta del LLM')
    
    time.sleep(1)
    
    # Verificar TTS queue
    if not tts_q.empty():
        item = tts_q.get()
        print(f"✅ TTS Queue recibió: {item}")
    
    print(f"Métricas: {sherpa.get_metrics()}")
    
    sherpa.stop()
    print("Test completado")
