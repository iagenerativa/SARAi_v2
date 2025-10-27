"""
Sistema de feedback implícito para SARAi v2
Detecta y registra señales de satisfacción sin etiquetas humanas
"""

import json
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
import yaml


class FeedbackDetector:
    """
    Detecta feedback implícito del usuario basándose en:
    - Reformulación de preguntas
    - Palabras de confirmación/frustración
    - Tiempo entre interacciones
    - Acciones tomadas
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.log_path = Path(config['paths']['feedback_log'])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Keywords para detección
        self.positive_keywords = [
            "gracias", "perfecto", "funciona", "excelente", "genial",
            "entendido", "claro", "ok", "vale", "bien", "correcto",
            "funcionó", "resuelto", "solucionado", "éxito"
        ]
        
        self.negative_keywords = [
            "error", "no funciona", "mal", "fallo", "problema",
            "no entiendo", "confuso", "difícil", "complicado",
            "no sirve", "incorrecto", "equivocado"
        ]
        
        self.neutral_keywords = [
            "otra pregunta", "ahora", "también", "además"
        ]
        
        self.interaction_history: List[Dict] = []
    
    def detect(self, user_input: str, prev_response: Optional[str] = None,
               next_input: Optional[str] = None) -> float:
        """
        Detecta feedback implícito y retorna score [-1, +1]
        
        Args:
            user_input: Input actual del usuario
            prev_response: Respuesta previa del sistema
            next_input: Siguiente input del usuario (si existe)
        
        Returns:
            Score de feedback: -1 (muy negativo) a +1 (muy positivo)
        """
        score = 0.0
        
        # 1. Análisis de palabras clave en next_input
        if next_input:
            low = next_input.lower()
            
            # Positivo
            if any(kw in low for kw in self.positive_keywords):
                score += 0.8
            
            # Negativo
            if any(kw in low for kw in self.negative_keywords):
                score -= 0.7
            
            # Reformulación de la misma pregunta (negativo)
            if self._is_reformulation(user_input, next_input):
                score -= 0.9
        
        # 2. Longitud de la respuesta (muy corta = posible problema)
        if prev_response:
            if len(prev_response.split()) < 10:
                score -= 0.3
        
        # 3. Si no hay next_input, asumir neutral con sesgo negativo
        if next_input is None and score == 0.0:
            score = -0.2  # Abandono leve
        
        # Normalizar a [-1, 1]
        return max(-1.0, min(1.0, score))
    
    def _is_reformulation(self, input1: str, input2: str, threshold: float = 0.7) -> bool:
        """
        Detecta si input2 es reformulación de input1
        Usa similitud de palabras simple (Jaccard)
        """
        words1 = set(input1.lower().split())
        words2 = set(input2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1 & words2
        union = words1 | words2
        
        similarity = len(intersection) / len(union)
        return similarity > threshold
    
    def log_interaction(self, input_text: str, hard: float, soft: float,
                       alpha: float, beta: float, agent_used: str,
                       response: str, feedback: float = 0.0):
        """
        Registra interacción completa en feedback_log.jsonl
        
        Args:
            input_text: Input del usuario
            hard: Score hard-intent
            soft: Score soft-intent
            alpha: Peso hard usado
            beta: Peso soft usado
            agent_used: "expert", "tiny", o "multimodal"
            response: Respuesta generada
            feedback: Score de feedback detectado
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_text,
            "hard": round(hard, 3),
            "soft": round(soft, 3),
            "alpha": round(alpha, 3),
            "beta": round(beta, 3),
            "agent_used": agent_used,
            "response": response[:200],  # Truncar para ahorrar espacio
            "feedback": round(feedback, 3)
        }
        
        # Guardar en archivo JSONL (una línea por interacción)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # Agregar a historial en memoria
        self.interaction_history.append(log_entry)
    
    def get_recent_interactions(self, n: int = 20) -> List[Dict]:
        """Retorna las últimas n interacciones"""
        return self.interaction_history[-n:]
    
    def compute_statistics(self, days: int = 7) -> Dict:
        """
        Calcula estadísticas de feedback de los últimos N días
        
        Returns:
            Dict con métricas de rendimiento
        """
        from datetime import timedelta
        
        # Cargar logs desde archivo
        if not self.log_path.exists():
            return {"error": "No hay logs disponibles"}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_logs = []
        
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                entry_date = datetime.fromisoformat(entry['timestamp'])
                if entry_date > cutoff_date:
                    recent_logs.append(entry)
        
        if not recent_logs:
            return {"error": f"No hay interacciones en los últimos {days} días"}
        
        # Calcular métricas
        total = len(recent_logs)
        positive = sum(1 for x in recent_logs if x['feedback'] > 0.5)
        negative = sum(1 for x in recent_logs if x['feedback'] < -0.5)
        neutral = total - positive - negative
        
        avg_feedback = sum(x['feedback'] for x in recent_logs) / total
        
        # Por agente
        expert_logs = [x for x in recent_logs if x['agent_used'] == 'expert']
        tiny_logs = [x for x in recent_logs if x['agent_used'] == 'tiny']
        
        return {
            "period_days": days,
            "total_interactions": total,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "avg_feedback": round(avg_feedback, 3),
            "expert_agent": {
                "count": len(expert_logs),
                "avg_feedback": round(sum(x['feedback'] for x in expert_logs) / len(expert_logs), 3) if expert_logs else 0
            },
            "tiny_agent": {
                "count": len(tiny_logs),
                "avg_feedback": round(sum(x['feedback'] for x in tiny_logs) / len(tiny_logs), 3) if tiny_logs else 0
            }
        }


# Singleton global
_feedback_detector = None

def get_feedback_detector() -> FeedbackDetector:
    """Obtiene instancia singleton del detector de feedback"""
    global _feedback_detector
    if _feedback_detector is None:
        _feedback_detector = FeedbackDetector()
    return _feedback_detector
