#!/usr/bin/env python3
"""
skill_draft Server - Draft LLM Service (Qwen3-VL-4B-Instruct IQ4_NL)

Servicio gRPC containerizado para generación de drafts rápidos.
Target: <0.5s latencia por iteración (vs 6s local).

ARQUITECTURA:
- Modelo: Qwen3-VL-4B-Instruct cuantizado IQ4_NL (~1.8GB)
- Backend: llama-cpp-python (CPU optimizado)
- Protocolo: gRPC (communication pool reutilizable)
- Recursos: 2 CPUs, 3GB RAM max

PHOENIX BENEFITS:
- Aislamiento: No contamina RAM del host
- Latencia: Precarga del modelo (cold-start 0.4s)
- Escalabilidad: Múltiples instancias si necesario

Autor: SARAi Dev Team
Fecha: 29 octubre 2025
Versión: 2.16.0
"""

import os
import sys
import time
import logging
from concurrent import futures
from pathlib import Path

import grpc
import psutil

# Importar proto generado
sys.path.insert(0, str(Path(__file__).parent.parent))
from skills import skills_pb2
from skills import skills_pb2_grpc

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class DraftService(skills_pb2_grpc.SkillsServiceServicer):
    """
    Implementación del servicio skill_draft
    
    Métodos gRPC:
    - Generate(GenReq) -> GenResp: Genera draft de respuesta
    - Health() -> HealthResp: Health check para Kubernetes
    """
    
    def __init__(self):
        """Inicializa el servicio y precarga el modelo"""
        logger.info("🚀 Inicializando skill_draft service...")
        
        # Cargar modelo en __init__ para warm-up
        self.model = self._load_model()
        self.total_requests = 0
        self.total_tokens = 0
        
        logger.info("✅ skill_draft listo para recibir requests")
    
    def _load_model(self):
        """
        Carga modelo Qwen3-VL-4B-Instruct IQ4_NL con llama-cpp-python
        
        Returns:
            Llama model instance
        """
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        
        # Path del modelo en HuggingFace
        repo_id = "Qwen/Qwen3-VL-4B-Instruct-Instruct-GGUF"
        filename = "Qwen3-VL-4B-Instruct-instruct-iq4_nl.gguf"
        
        # Descargar si no existe (cache de HF)
        logger.info(f"Descargando modelo: {repo_id}/{filename}")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=os.getenv("HF_CACHE_DIR", "/models")
        )
        
        # Configuración optimizada para draft (latencia baja)
        logger.info(f"Cargando modelo desde: {model_path}")
        start = time.perf_counter()
        
        model = Llama(
            model_path=model_path,
            n_ctx=512,  # Contexto pequeño (drafts cortos)
            n_threads=int(os.getenv("N_THREADS", "2")),  # 2 CPUs del container
            n_gpu_layers=0,  # CPU-only
            use_mmap=True,
            use_mlock=False,  # Evitar OOM
            verbose=False
        )
        
        elapsed = time.perf_counter() - start
        logger.info(f"✅ Modelo cargado en {elapsed:.2f}s")
        
        return model
    
    def Generate(self, request, context):
        """
        gRPC method: Genera draft de respuesta
        
        Args:
            request (GenReq): Protobuf request con prompt, max_tokens, etc.
            context: gRPC context
        
        Returns:
            GenResp: Protobuf response con texto generado y métricas
        """
        start_time = time.perf_counter()
        
        logger.info(f"📥 Request recibido: {len(request.prompt)} chars")
        
        try:
            # Generar con llama-cpp
            result = self.model(
                request.prompt,
                max_tokens=request.max_tokens or 256,
                temperature=request.temperature or 0.7,
                stop=list(request.stop) if request.stop else [],
                echo=False
            )
            
            # Extraer resultado
            text = result["choices"][0]["text"].strip()
            tokens = result["usage"]["completion_tokens"]
            
            # Calcular métricas
            latency_ms = (time.perf_counter() - start_time) * 1000
            tokens_per_second = tokens / (latency_ms / 1000) if latency_ms > 0 else 0
            
            # RAM usage actual
            process = psutil.Process()
            ram_mb = process.memory_info().rss / (1024 * 1024)
            
            # Actualizar contadores
            self.total_requests += 1
            self.total_tokens += tokens
            
            logger.info(
                f"✅ Response generado: {tokens} tokens, "
                f"{tokens_per_second:.1f} tok/s, "
                f"{latency_ms:.1f}ms, "
                f"RAM: {ram_mb:.1f}MB"
            )
            
            # Construir respuesta protobuf
            return skills_pb2.GenResp(
                text=text,
                tokens_generated=tokens,
                tokens_per_second=tokens_per_second,
                latency_ms=latency_ms,
                ram_mb=ram_mb
            )
        
        except Exception as e:
            logger.error(f"❌ Error en Generate: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
            return skills_pb2.GenResp(
                text=f"Error: {e}",
                tokens_generated=0,
                tokens_per_second=0.0,
                latency_ms=0.0,
                ram_mb=0.0
            )
    
    def Health(self, request, context):
        """
        gRPC method: Health check para Kubernetes
        
        Returns:
            HealthResp: Protobuf response con estado del servicio
        """
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # Determinar status
        if ram_mb > 3000:  # >3GB
            status = "DEGRADED"
        elif cpu_percent > 90:
            status = "DEGRADED"
        else:
            status = "HEALTHY"
        
        return skills_pb2.HealthResp(
            status=status,
            ram_mb=ram_mb,
            cpu_percent=cpu_percent,
            total_requests=self.total_requests,
            total_tokens=self.total_tokens
        )


def serve():
    """
    Inicia servidor gRPC
    
    Configuración:
    - Puerto: 50051 (estándar skill_draft)
    - Workers: 4 threads (para concurrencia)
    - Grace period: 10s para shutdown limpio
    """
    # Configurar servidor gRPC
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    
    # Registrar servicio
    skills_pb2_grpc.add_SkillsServiceServicer_to_server(
        DraftService(), 
        server
    )
    
    # Bind a puerto
    port = os.getenv("GRPC_PORT", "50051")
    server.add_insecure_port(f'[::]:{port}')
    
    logger.info(f"🚀 skill_draft server escuchando en puerto {port}")
    
    # Iniciar servidor
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("⏹️ Shutdown signal recibido")
        server.stop(grace=10)
        logger.info("✅ Server detenido limpiamente")


if __name__ == '__main__':
    serve()
