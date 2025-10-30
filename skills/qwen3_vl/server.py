#!/usr/bin/env python3
"""
skills/qwen3_vl/server.py - gRPC Server para Qwen3-VL Vision
============================================================

Servidor gRPC que expone Qwen3-VL-4B como servicio containerizado.
Usa el agente existente en agents/qwen3_vl.py

Endpoints:
- Infer(): Análisis de imagen/video
- Check(): Health check
- GetMetrics(): Métricas de rendimiento
"""

import os
import sys
import time
import logging
import grpc
from concurrent import futures

# Añadir path del proyecto
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports gRPC
import skills_pb2
import skills_pb2_grpc

# Import del agente Qwen3-VL real
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from agents.qwen3_vl import Qwen3VLAgent, Qwen3VLConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Qwen3VLService(skills_pb2_grpc.SkillServiceServicer):
    """
    Implementación del servicio gRPC para Qwen3-VL
    
    Carga el modelo bajo demanda y lo mantiene en cache según TTL
    """
    
    def __init__(self):
        self.agent = None
        self.last_inference_time = 0
        self.total_inferences = 0
        logger.info("🔷 Qwen3-VL Service iniciado")
    
    def _load_agent_if_needed(self):
        """Lazy loading del agente"""
        if self.agent is None:
            logger.info("📥 Cargando Qwen3-VL-4B agent...")
            start = time.time()
            
            # Configuración desde variables de entorno o defaults
            config = Qwen3VLConfig(
                model_path=os.getenv('MODEL_PATH', 'models/gguf/qwen3-vl-4b-q6_k.gguf'),
                n_ctx=int(os.getenv('N_CTX', '2048')),
                n_threads=int(os.getenv('N_THREADS', '4')),
                temperature=float(os.getenv('TEMPERATURE', '0.3')),
                max_tokens=int(os.getenv('MAX_NEW_TOKENS', '512'))
            )
            
            self.agent = Qwen3VLAgent(config)
            logger.info(f"✅ Qwen3-VL cargado en {time.time() - start:.2f}s")
    
    def Infer(self, request, context):
        """
        Inferencia principal: analiza imagen/video
        
        Formato esperado en request.context:
        - image_path: Ruta a la imagen
        - video_path: Ruta al video (opcional)
        """
        try:
            self._load_agent_if_needed()
            start_time = time.time()
            
            # Extraer imagen/video del contexto
            image_path = request.context.get('image_path', '')
            video_path = request.context.get('video_path', '')
            
            if not image_path and not video_path:
                return skills_pb2.InferResponse(
                    text="Error: No se proporcionó imagen o video",
                    confidence=0.0,
                    finish_reason="error"
                )
            
            # Usar el agente para procesar
            if image_path:
                result = self.agent.analyze_image(
                    image_path=image_path,
                    prompt=request.prompt
                )
            else:
                result = self.agent.analyze_video(
                    video_path=video_path,
                    prompt=request.prompt
                )
            
            latency_ms = int((time.time() - start_time) * 1000)
            self.last_inference_time = time.time()
            self.total_inferences += 1
            
            return skills_pb2.InferResponse(
                text=result['response'],
                confidence=result.get('confidence', 0.9),
                latency_ms=latency_ms,
                tokens_generated=result.get('tokens_generated', 0),
                model="Qwen3-VL-4B-Q6_K",
                finish_reason="stop"
            )
        
        except Exception as e:
            logger.error(f"❌ Error en inferencia: {e}")
            return skills_pb2.InferResponse(
                text=f"Error: {str(e)}",
                confidence=0.0,
                finish_reason="error"
            )
    
    def Check(self, request, context):
        """Health check"""
        if self.agent is None:
            return skills_pb2.HealthCheckResponse(
                status=skills_pb2.HealthCheckResponse.NOT_SERVING
            )
        return skills_pb2.HealthCheckResponse(
            status=skills_pb2.HealthCheckResponse.SERVING
        )
    
    def GetMetrics(self, request, context):
        """Métricas Prometheus-compatible"""
        uptime = time.time() - self.last_inference_time if self.last_inference_time > 0 else 0
        
        metrics_text = f"""# HELP qwen3_vl_inferences_total Total de inferencias
# TYPE qwen3_vl_inferences_total counter
qwen3_vl_inferences_total {self.total_inferences}

# HELP qwen3_vl_uptime_seconds Tiempo desde última inferencia
# TYPE qwen3_vl_uptime_seconds gauge
qwen3_vl_uptime_seconds {uptime:.2f}

# HELP qwen3_vl_model_loaded Estado del modelo
# TYPE qwen3_vl_model_loaded gauge
qwen3_vl_model_loaded {1 if self.agent else 0}
"""
        return skills_pb2.MetricsResponse(metrics=metrics_text)


def serve():
    """Inicia el servidor gRPC"""
    port = os.getenv('GRPC_PORT', '50051')
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]
    )
    
    skills_pb2_grpc.add_SkillServiceServicer_to_server(
        Qwen3VLService(), server
    )
    
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    logger.info(f"🚀 Qwen3-VL gRPC Server escuchando en puerto {port}")
    logger.info(f"   Modelo: Qwen3-VL-4B-Q6_K")
    logger.info(f"   Lazy loading: Sí (se carga en primera inferencia)")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("🛑 Servidor detenido por usuario")
        server.stop(0)


if __name__ == '__main__':
    serve()
