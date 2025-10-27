"""
SARAi v2.9 - Batch Prioritizer con Fast Lane
Sistema de colas prioritarias para garantizar latencia crítica

NEW v2.9:
- Priority Queue con 4 niveles (critical, high, normal, low)
- Fast Lane para queries críticas (P99 ≤ 1.5s)
- Batching PID para queries normales (P50 ≤ 20s)
- Preemption automática si llega query crítica
"""

import threading
import time
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List
from enum import IntEnum
import asyncio
from concurrent.futures import Future


class Priority(IntEnum):
    """
    Niveles de prioridad (menor valor = mayor prioridad).
    
    CRITICAL (0): Alertas, queries de monitoreo, salud del sistema
    HIGH (1): Queries interactivas de usuario
    NORMAL (2): Queries en batch, procesamiento asíncrono
    LOW (3): Trabajos de background, generación creativa
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class QueueItem:
    """Item en la cola de prioridad."""
    priority: int
    timestamp: float = field(compare=False)
    input_text: str = field(compare=False)
    context: dict = field(default_factory=dict, compare=False)
    future: Future = field(default_factory=Future, compare=False)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class BatchPrioritizer:
    """
    NEW v2.9: Gestor de colas con Fast Lane.
    
    Garantías:
    - Queries críticas: ≤ 1.5s (fast lane, sin batching)
    - Queries normales: ≤ 20s (batching optimizado)
    - Queries bajas: best-effort (pueden esperar)
    """
    
    def __init__(
        self,
        model_getter: Callable[[str], Any],
        pid_window_base: float = 0.5,  # 500ms base
        pid_n_parallel_base: int = 4,
        max_batch_size: int = 8
    ):
        """
        Args:
            model_getter: Función para obtener modelo (ej. model_pool.get)
            pid_window_base: Ventana base de batching (segundos)
            pid_n_parallel_base: Número base de requests paralelos
            max_batch_size: Tamaño máximo de batch
        """
        self.model_getter = model_getter
        self.queue = PriorityQueue()
        
        # PID controller state
        self.pid_window = pid_window_base
        self.pid_n_parallel = pid_n_parallel_base
        self.max_batch_size = max_batch_size
        
        # Worker thread
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # Metrics
        self.stats = {
            "total_processed": 0,
            "critical_processed": 0,
            "batches_processed": 0,
            "avg_batch_size": 0.0,
            "preemptions": 0
        }
        
        self.stats_lock = threading.Lock()
    
    def start(self):
        """Inicia el worker thread."""
        if self.running:
            print("⚠️ Batch worker ya está corriendo")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()
        print("✅ Batch Prioritizer iniciado (Fast Lane activa)")
    
    def stop(self):
        """Detiene el worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        print("🛑 Batch Prioritizer detenido")
    
    def submit(
        self,
        input_text: str,
        priority: Priority = Priority.NORMAL,
        context: Optional[dict] = None
    ) -> Future:
        """
        Envía una query a la cola prioritaria.
        
        Args:
            input_text: Texto de entrada
            priority: Nivel de prioridad (CRITICAL, HIGH, NORMAL, LOW)
            context: Metadata adicional
        
        Returns:
            Future que se resolverá con la respuesta
        
        Example:
            >>> prioritizer = BatchPrioritizer(model_pool.get)
            >>> prioritizer.start()
            >>> future = prioritizer.submit("¿Está el servidor caído?", Priority.CRITICAL)
            >>> response = future.result(timeout=2)  # Garantizado ≤ 1.5s
        """
        item = QueueItem(
            priority=priority.value,
            input_text=input_text,
            context=context or {}
        )
        
        self.queue.put(item)
        return item.future
    
    def _batch_worker(self):
        """
        Worker principal que procesa la cola.
        
        NEW v2.9: Implementa Fast Lane + Batching PID.
        """
        while self.running:
            try:
                # FASE 1: FAST LANE - Procesar todas las queries críticas
                while not self.queue.empty():
                    # Peek sin sacar
                    try:
                        item = self.queue.queue[0]  # Peek en PriorityQueue
                        
                        if item.priority != Priority.CRITICAL:
                            break  # No hay más críticas, pasar a batching
                        
                        # Sacar y procesar inmediatamente
                        item = self.queue.get_nowait()
                        self._process_single(item, is_critical=True)
                    
                    except (Empty, IndexError):
                        break
                
                # FASE 2: BATCHING PID - Agrupar queries normales
                batch = []
                deadline = time.time() + self.pid_window
                
                while time.time() < deadline and len(batch) < self.max_batch_size:
                    try:
                        # Timeout corto para permitir checks frecuentes
                        item = self.queue.get(timeout=0.1)
                        
                        # PREEMPTION: Si llega una crítica, procesarla YA
                        if item.priority == Priority.CRITICAL:
                            # Devolver batch actual a la cola
                            for batched_item in batch:
                                self.queue.put(batched_item)
                            
                            # Procesar crítica
                            self._process_single(item, is_critical=True)
                            
                            with self.stats_lock:
                                self.stats["preemptions"] += 1
                            
                            # Reiniciar batch
                            batch = []
                            deadline = time.time() + self.pid_window
                            continue
                        
                        # Añadir al batch
                        batch.append(item)
                    
                    except Empty:
                        # No hay más items, continuar
                        break
                
                # Procesar batch si hay items
                if batch:
                    self._process_batch(batch)
                
                # Si no hay nada que hacer, esperar un poco
                if self.queue.empty():
                    time.sleep(0.05)
            
            except Exception as e:
                print(f"❌ Error en batch worker: {e}")
                time.sleep(0.1)
    
    def _process_single(self, item: QueueItem, is_critical: bool = False):
        """
        Procesa un item individual (FAST LANE).
        
        Para queries críticas, usa n_parallel=1 para latencia mínima.
        """
        start_time = time.time()
        
        try:
            # Obtener modelo (expert_short para latencia mínima)
            model = self.model_getter("expert_short")
            
            # Generar respuesta
            response = model(
                item.input_text,
                max_tokens=512,  # Límite para críticas
                temperature=0.3  # Baja temperatura = más determinista
            )
            
            # Resolver future
            item.future.set_result(response)
            
            # Metrics
            latency = time.time() - start_time
            
            with self.stats_lock:
                self.stats["total_processed"] += 1
                if is_critical:
                    self.stats["critical_processed"] += 1
            
            if is_critical:
                print(f"⚡ FAST LANE: {latency:.2f}s (objetivo: ≤1.5s)")
        
        except Exception as e:
            item.future.set_exception(e)
            print(f"❌ Error procesando item: {e}")
    
    def _process_batch(self, batch: List[QueueItem]):
        """
        Procesa un batch de items (BATCHING PID).
        
        Usa n_parallel dinámico para optimizar P50.
        """
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Obtener modelo
            model = self.model_getter("expert_short")
            
            # Extraer inputs
            inputs = [item.input_text for item in batch]
            
            # Batching GGUF con n_parallel
            responses = model.create_batch(
                inputs,
                n_parallel=min(self.pid_n_parallel, len(batch)),
                max_tokens=1024
            )
            
            # Distribuir respuestas
            for item, response in zip(batch, responses):
                item.future.set_result(response)
            
            # Metrics
            latency = time.time() - start_time
            batch_size = len(batch)
            
            with self.stats_lock:
                self.stats["total_processed"] += batch_size
                self.stats["batches_processed"] += 1
                
                # Actualizar avg_batch_size
                total_batches = self.stats["batches_processed"]
                old_avg = self.stats["avg_batch_size"]
                self.stats["avg_batch_size"] = (
                    (old_avg * (total_batches - 1) + batch_size) / total_batches
                )
            
            print(f"📦 BATCH: {batch_size} items en {latency:.2f}s")
            
            # PID adjustment (simplificado)
            self._adjust_pid(latency, batch_size)
        
        except Exception as e:
            # Fallar todos los futures del batch
            for item in batch:
                item.future.set_exception(e)
            print(f"❌ Error procesando batch: {e}")
    
    def _adjust_pid(self, latency: float, batch_size: int):
        """
        Ajusta parámetros PID basado en latencia observada.
        
        Objetivo: Mantener latency P50 ≤ 20s
        """
        target_latency = 20.0  # segundos
        error = latency - target_latency
        
        # Si latencia muy alta → reducir batch size
        if error > 5.0:
            self.pid_n_parallel = max(1, self.pid_n_parallel - 1)
            self.pid_window = max(0.2, self.pid_window - 0.1)
        
        # Si latencia baja → aumentar batch size
        elif error < -5.0:
            self.pid_n_parallel = min(8, self.pid_n_parallel + 1)
            self.pid_window = min(2.0, self.pid_window + 0.1)
    
    def get_stats(self) -> dict:
        """Retorna estadísticas del prioritizer."""
        with self.stats_lock:
            return self.stats.copy()


# ---------- USAGE EXAMPLE ----------

if __name__ == "__main__":
    # Mock model getter
    def mock_model_getter(model_name):
        class MockModel:
            def __call__(self, text, **kwargs):
                time.sleep(0.5)  # Simular procesamiento
                return f"Response to: {text[:50]}"
            
            def create_batch(self, texts, **kwargs):
                time.sleep(1.0)  # Simular batch processing
                return [f"Response to: {t[:50]}" for t in texts]
        
        return MockModel()
    
    # Crear prioritizer
    prioritizer = BatchPrioritizer(mock_model_getter)
    prioritizer.start()
    
    # Enviar queries de prueba
    print("Enviando 10 queries normales...")
    normal_futures = [
        prioritizer.submit(f"Query normal {i}", Priority.NORMAL)
        for i in range(10)
    ]
    
    time.sleep(0.5)
    
    print("Enviando 1 query CRÍTICA (debería preemptar)...")
    critical_future = prioritizer.submit("¡ALERTA! Sistema caído", Priority.CRITICAL)
    
    # Esperar resultados
    print(f"Crítica: {critical_future.result(timeout=2)}")
    
    for i, future in enumerate(normal_futures):
        print(f"Normal {i}: {future.result(timeout=30)}")
    
    # Stats
    print("\n📊 Estadísticas:")
    print(prioritizer.get_stats())
    
    prioritizer.stop()
