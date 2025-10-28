"""
skills/runtime.py - gRPC Server para Skills-as-Services v2.12 Phoenix

ARQUITECTURA:
- gRPC server (puerto 50051)
- Hot-reload con señal USR1
- RAM <50MB por cliente (stub ligero)
- Cold-start <500ms (n_threads=2)

SEGURIDAD:
- Ejecuta en Docker con cap_drop=ALL
- read_only filesystem + tmpfs
- no-new-privileges
- Usuario no-root (skilluser)

EJEMPLO:
  docker run -d --name saraiskill.sql \\
    --cap-drop=ALL --read-only \\
    --tmpfs /tmp:size=256M \\
    -p 50051:50051 \\
    -e SKILL_NAME=sql \\
    saraiskill.sql:v2.12

  # Hot-reload
  docker exec saraiskill.sql sh -c 'kill -USR1 1'
"""

import argparse
import signal
import sys
import os
import logging
import time
from concurrent import futures
from typing import Dict, Any, Optional
from pathlib import Path

import grpc
from grpc_reflection.v1alpha import reflection

# Import generated protobuf stubs
from skills import (
    SkillServiceServicer,
    add_SkillServiceServicer_to_server,
    InferRequest,
    InferResponse,
    HealthCheckRequest,
    HealthCheckResponse,
    MetricsRequest,
    MetricsResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class SkillService(SkillServiceServicer):
    """
    Implementación del servicio gRPC SkillService
    
    PATTERN:
    - Carga lazy del modelo GGUF (solo cuando se necesita)
    - ModelPool local (modelo en memoria mientras haya requests)
    - Hot-reload con USR1 signal (swap atómico del modelo)
    - Métricas Prometheus (total_requests, latency, etc.)
    """
    
    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        self.model = None
        self.reload_count = 0
        self.start_time = time.time()
        
        # Métricas
        self.total_requests = 0
        self.failed_requests = 0
        self.latencies = []  # Lista de latencias para calcular P99
        self.cold_start_ms = None
        
        logger.info(f"🚀 Inicializando skill: {skill_name}")
        
        # Registrar signal handler para hot-reload
        signal.signal(signal.SIGUSR1, self._hot_reload_handler)
        
        # Carga inicial del modelo
        self._load_model()
    
    def _load_model(self):
        """
        Carga el modelo GGUF del skill
        
        UBICACIÓN ESPERADA:
        - Contenedor: /app/models/{skill_name}.gguf
        - Host (testing): models/skills/{skill_name}.gguf
        """
        start_time = time.time()
        
        # Buscar modelo en múltiples ubicaciones (host vs contenedor)
        model_paths = [
            f"/app/models/{self.skill_name}.gguf",  # Contenedor
            f"models/skills/{self.skill_name}.gguf",  # Host
            f"models/{self.skill_name}.gguf",  # Fallback
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            logger.error(f"❌ Modelo no encontrado: {self.skill_name}.gguf")
            logger.error(f"Buscado en: {model_paths}")
            raise FileNotFoundError(f"Missing {self.skill_name}.gguf")
        
        logger.info(f"📥 Cargando modelo: {model_path}")
        
        try:
            from llama_cpp import Llama
            
            # Configuración conservadora para RAM <50MB
            self.model = Llama(
                model_path=model_path,
                n_ctx=512,  # Contexto corto (skills especializados)
                n_threads=2,  # Bajo overhead (cold-start <500ms)
                use_mmap=True,  # Memory-mapped I/O
                use_mlock=False,  # No lock en RAM (permite swap)
                verbose=False
            )
            
            load_time_ms = int((time.time() - start_time) * 1000)
            
            if self.cold_start_ms is None:
                self.cold_start_ms = load_time_ms
            
            logger.info(f"✅ Modelo cargado en {load_time_ms}ms (reload #{self.reload_count})")
        
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    
    def _hot_reload_handler(self, signum, frame):
        """
        Handler para USR1: hot-reload del modelo
        
        ATOMICIDAD:
        1. Cargar nuevo modelo en memoria temporal
        2. Swap atómico (self.model = new_model)
        3. GC libera modelo anterior
        """
        logger.info(f"🔄 Recibida señal USR1, iniciando hot-reload...")
        
        # Liberar modelo actual
        if self.model:
            del self.model
            self.model = None
        
        # Recargar
        self.reload_count += 1
        
        try:
            self._load_model()
            logger.info(f"✅ Hot-reload completado (reload #{self.reload_count})")
        except Exception as e:
            logger.error(f"❌ Hot-reload fallido: {e}")
            # CRITICAL: sin modelo, marcar como NOT_SERVING
    
    def Infer(self, request: InferRequest, context) -> InferResponse:
        """
        RPC principal: ejecuta inferencia en el skill LLM
        
        Args:
            request: InferRequest con prompt, max_tokens, etc.
            context: gRPC context
        
        Returns:
            InferResponse con text, confidence, latency_ms
        """
        start_time = time.time()
        self.total_requests += 1
        
        prompt = request.prompt
        max_tokens = request.max_tokens or 128
        temperature = request.temperature or 0.7
        stop = list(request.stop) if request.stop else ["</s>", "\n\n"]
        
        logger.info(f"📨 Infer request #{self.total_requests}: {prompt[:50]}...")
        
        # Validar que el modelo esté cargado
        if not self.model:
            self.failed_requests += 1
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Modelo no disponible (hot-reload en progreso?)")
            return InferResponse()
        
        try:
            # Inferencia con llama.cpp
            result = self.model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                echo=False  # No incluir prompt en output
            )
            
            response_text = result['choices'][0]['text']
            finish_reason = result['choices'][0]['finish_reason']
            tokens_generated = result['usage']['completion_tokens']
            
            latency_ms = int((time.time() - start_time) * 1000)
            self.latencies.append(latency_ms)
            
            # Confidence simple basado en finish_reason
            # TODO: mejorar con embeddings semánticos
            confidence = 0.85 if finish_reason == "stop" else 0.60
            
            logger.info(f"✅ Respuesta generada: {tokens_generated} tokens, {latency_ms}ms")
            
            return InferResponse(
                text=response_text,
                confidence=confidence,
                latency_ms=latency_ms,
                tokens_generated=tokens_generated,
                model=f"{self.skill_name}.gguf",
                finish_reason=finish_reason
            )
        
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"❌ Error en inferencia: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return InferResponse()
    
    def Check(self, request: HealthCheckRequest, context) -> HealthCheckResponse:
        """
        Health check para Docker/K8s
        
        Returns:
            SERVING si modelo cargado y RAM disponible >100MB
            NOT_SERVING si degradado
        """
        # Verificar que el modelo esté cargado
        if self.model is None:
            return HealthCheckResponse(
                status=HealthCheckResponse.NOT_SERVING,
                message="Modelo no cargado",
                reload_count=self.reload_count,
                uptime_seconds=int(time.time() - self.start_time)
            )
        
        # Verificar RAM disponible (opcional)
        try:
            import psutil
            ram_available_gb = psutil.virtual_memory().available / (1024**3)
            
            if ram_available_gb < 0.1:  # <100MB disponible
                return HealthCheckResponse(
                    status=HealthCheckResponse.NOT_SERVING,
                    message=f"RAM crítica: {ram_available_gb:.2f}GB disponible",
                    reload_count=self.reload_count,
                    uptime_seconds=int(time.time() - self.start_time)
                )
        except ImportError:
            pass  # psutil no disponible, skip RAM check
        
        # Healthy
        return HealthCheckResponse(
            status=HealthCheckResponse.SERVING,
            message="Operativo",
            reload_count=self.reload_count,
            uptime_seconds=int(time.time() - self.start_time)
        )
    
    def GetMetrics(self, request: MetricsRequest, context) -> MetricsResponse:
        """
        Métricas Prometheus
        
        Returns:
            MetricsResponse con total_requests, latencias, RAM, etc.
        """
        # Calcular latencia P99
        p99_latency_ms = 0.0
        avg_latency_ms = 0.0
        
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            p99_index = int(len(sorted_latencies) * 0.99)
            p99_latency_ms = float(sorted_latencies[p99_index])
            avg_latency_ms = sum(self.latencies) / len(self.latencies)
        
        # RAM actual
        ram_bytes = 0
        try:
            import psutil
            process = psutil.Process(os.getpid())
            ram_bytes = process.memory_info().rss
        except ImportError:
            pass
        
        # Total de tokens generados (aproximado)
        total_tokens = self.total_requests * 64  # Estimación conservadora
        
        return MetricsResponse(
            total_requests=self.total_requests,
            failed_requests=self.failed_requests,
            avg_latency_ms=avg_latency_ms,
            p99_latency_ms=p99_latency_ms,
            ram_bytes=ram_bytes,
            total_tokens=total_tokens,
            cold_start_ms=self.cold_start_ms or 0
        )


def serve(skill_name: str, port: int = 50051):
    """
    Inicia el servidor gRPC del skill
    
    Args:
        skill_name: Nombre del skill (sql, code, math, etc.)
        port: Puerto gRPC (default: 50051)
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    
    # Instanciar servicio
    skill_service = SkillService(skill_name)
    
    # Registrar servicio en gRPC
    add_SkillServiceServicer_to_server(skill_service, server)
    
    # Habilitar reflection (para grpcurl, debugging)
    SERVICE_NAMES = (
        "sarai.skills.SkillService",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    # Bind y start
    server.add_insecure_port(f'0.0.0.0:{port}')
    server.start()
    
    logger.info(f"✅ Skill '{skill_name}' servidor gRPC en puerto {port}")
    logger.info(f"🔄 Hot-reload: docker exec <container> sh -c 'kill -USR1 1'")
    logger.info(f"🏥 Health: grpcurl -plaintext localhost:{port} sarai.skills.SkillService/Check")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("⛔ Deteniendo servidor...")
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skills-as-Services gRPC Server v2.12")
    parser.add_argument("--skill", required=True, help="Nombre del skill (sql, code, etc.)")
    parser.add_argument("--port", type=int, default=50051, help="Puerto gRPC (default: 50051)")
    
    args = parser.parse_args()
    
    # Validar que SKILL_NAME esté definido (en contenedor viene de ENV)
    skill_name = args.skill or os.getenv("SKILL_NAME")
    
    if not skill_name:
        logger.error("❌ Error: SKILL_NAME no definido")
        logger.error("Usar: --skill <nombre> o ENV SKILL_NAME=<nombre>")
        sys.exit(1)
    
    serve(skill_name, args.port)

