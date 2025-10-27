"""
Prefetcher v2.3: Sistema de precarga proactiva de modelos
Detecta la intención del usuario mientras escribe/habla
"""

import threading
import time
from typing import Optional, Callable


class Prefetcher:
    """
    Sistema de prefetching inteligente:
    - Usa TRM-Mini (3.5M) para clasificación rápida
    - Debounce de 300ms para evitar precarga prematura
    - Carga modelos en hilo daemon sin bloquear UI
    """
    
    def __init__(self, model_pool, trm_mini, embedder):
        """
        Args:
            model_pool: Instancia de ModelPool
            trm_mini: Instancia de TRM-Mini
            embedder: Instancia de EmbeddingGemma
        """
        self.pool = model_pool
        self.trm_mini = trm_mini
        self.embedder = embedder
        
        self.predicted_need: Optional[str] = None
        self.last_input_time: float = 0
        self.debounce_delay: float = 0.3  # 300ms
        self.input_buffer: str = ""
        
        self.timer: Optional[threading.Timer] = None
        
        print("[Prefetcher] Inicializado (debounce=300ms)")
    
    def on_partial_input(self, partial_input: str):
        """
        Llamado en cada keystroke o fragmento de audio
        
        Args:
            partial_input: Texto parcial del usuario
        """
        self.input_buffer = partial_input
        self.last_input_time = time.time()
        
        # Cancelar timer anterior si existe
        if self.timer is not None:
            self.timer.cancel()
        
        # Iniciar nuevo timer de debounce
        self.timer = threading.Timer(self.debounce_delay, self._run_prefetch_check)
        self.timer.daemon = True
        self.timer.start()
    
    def _run_prefetch_check(self):
        """
        Se ejecuta 300ms después del último keystroke
        Clasificación rápida + decisión de precarga
        """
        # Double-check: si hubo más input, cancelar
        if time.time() - self.last_input_time < self.debounce_delay:
            return
        
        # Ignorar inputs muy cortos
        if len(self.input_buffer.strip()) < 10:
            return
        
        try:
            # Clasificación rápida con TRM-Mini
            emb = self.embedder.encode(self.input_buffer)
            scores = self.trm_mini.invoke(emb)
            
            # Decidir qué modelo precargar
            need = self._decide_model_need(scores, self.input_buffer)
            
            # Si cambió la predicción, lanzar precarga
            if need != self.predicted_need:
                self.predicted_need = need
                self._launch_prefetch(need)
        
        except Exception as e:
            print(f"⚠️ Prefetch check fallido: {e}")
    
    def _decide_model_need(self, scores: dict, input_text: str) -> str:
        """
        Decide qué modelo precargar basado en scores y longitud
        
        Args:
            scores: {"hard": float, "soft": float} del TRM-Mini
            input_text: Texto de entrada
        
        Returns:
            Nombre lógico del modelo: 'expert_short', 'expert_long', 'tiny'
        """
        # Threshold más alto que en MCP (queremos prefetch conservador)
        if scores["hard"] > 0.65:
            # Decidir contexto basado en longitud aproximada
            # Heurística: si el input parcial ya es largo, probablemente sea un query complejo
            if len(input_text) > 400:
                return "expert_long"
            else:
                return "expert_short"
        
        elif scores["soft"] > 0.65:
            return "tiny"
        
        else:
            # Híbrido: preferir expert_short (más común)
            return "expert_short"
    
    def _launch_prefetch(self, model_name: str):
        """
        Lanza precarga en hilo daemon
        
        Args:
            model_name: Nombre lógico del modelo
        """
        print(f"[Prefetcher] Predicción: {model_name} (hard={self.predicted_need})")
        
        # Lanzar carga en background
        thread = threading.Thread(
            target=self.pool.prefetch_model,
            args=(model_name,),
            daemon=True
        )
        thread.start()
    
    def reset(self):
        """Resetea estado (útil al cambiar de conversación)"""
        self.input_buffer = ""
        self.predicted_need = None
        self.last_input_time = 0
        
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None


# Singleton global
_prefetcher_instance: Optional[Prefetcher] = None


def get_prefetcher(model_pool=None, trm_mini=None, embedder=None) -> Prefetcher:
    """
    Singleton: retorna instancia única del Prefetcher
    
    Args:
        model_pool: ModelPool (solo necesario en primera llamada)
        trm_mini: TRM-Mini (solo necesario en primera llamada)
        embedder: EmbeddingGemma (solo necesario en primera llamada)
    
    Returns:
        Prefetcher inicializado
    """
    global _prefetcher_instance
    
    if _prefetcher_instance is None:
        if model_pool is None or trm_mini is None or embedder is None:
            raise ValueError(
                "Primera llamada a get_prefetcher() requiere model_pool, trm_mini y embedder"
            )
        _prefetcher_instance = Prefetcher(model_pool, trm_mini, embedder)
    
    return _prefetcher_instance
