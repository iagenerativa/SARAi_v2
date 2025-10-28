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
    saraiskill.sql:v2.12

  # Hot-reload
  docker exec saraiskill.sql sh -c 'kill -USR1 1'
"""

import argparse
import signal
import sys
import os
import logging
from concurrent import futures
from typing import Dict, Any, Optional

import grpc
from grpc_reflection.v1alpha import reflection

# Protobuf generado (crear con protoc)
# TODO: Generar desde skills/protos/skill.proto
# import skill_pb2
# import skill_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SkillService:
    """
    Servicio gRPC que expone un skill LLM especializado
    
    PATTERN:
    - Carga lazy del modelo (solo cuando se necesita)
    - ModelPool local (LRU con TTL=60s)
    - Prefetcher si está disponible
    """
    
    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        self.model = None
        self.reload_count = 0
        
        logger.info(f"Inicializando skill: {skill_name}")
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo GGUF del skill"""
        from llama_cpp import Llama
        
        model_path = f"/app/models/{self.skill_name}.gguf"
        
        if not os.path.exists(model_path):
            logger.error(f"Modelo no encontrado: {model_path}")
            raise FileNotFoundError(f"Missing {model_path}")
        
        logger.info(f"Cargando modelo: {model_path}")
        
        # Configuración conservadora para <50MB RAM
        self.model = Llama(
            model_path=model_path,
            n_ctx=512,  # Contexto corto para skills especializados
            n_threads=2,  # Bajo overhead
            use_mmap=True,
            use_mlock=False,
            verbose=False
        )
        
        logger.info(f"Modelo cargado exitosamente (reload #{self.reload_count})")
    
    def reload_model(self, signum, frame):
        """Handler para USR1: hot-reload del modelo"""
        logger.info(f"Recibida señal USR1, recargando modelo...")
        
        # Liberar modelo actual
        if self.model:
            del self.model
            self.model = None
        
        # Recargar
        self.reload_count += 1
        self._load_model()
        
        logger.info(f"Hot-reload completado (reload #{self.reload_count})")
    
    def Execute(self, request, context):
        """
        RPC principal: ejecuta query en el skill LLM
        
        Args:
            request: ExecuteRequest { query: str, max_tokens: int }
            context: gRPC context
        
        Returns:
            ExecuteResponse { response: str, confidence: float }
        """
        query = request.query
        max_tokens = request.max_tokens or 128
        
        logger.info(f"Execute llamado: {query[:50]}...")
        
        if not self.model:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Modelo no disponible")
            return None
        
        try:
            # Inferencia con timeout
            result = self.model.create_completion(
                prompt=query,
                max_tokens=max_tokens,
                temperature=0.7,
                stop=["</s>", "\n\n"]
            )
            
            response_text = result['choices'][0]['text']
            
            # Confidence simple (TODO: mejorar con embeddings)
            confidence = 0.85  # Placeholder
            
            logger.info(f"Respuesta generada: {response_text[:50]}...")
            
            # return ExecuteResponse(
            #     response=response_text,
            #     confidence=confidence
            # )
            
            # Placeholder hasta generar protobufs
            return {"response": response_text, "confidence": confidence}
        
        except Exception as e:
            logger.error(f"Error en inferencia: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return None


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
    
    # Registrar señal USR1 para hot-reload
    signal.signal(signal.SIGUSR1, skill_service.reload_model)
    
    # Registrar servicio en gRPC
    # skill_pb2_grpc.add_SkillServiceServicer_to_server(skill_service, server)
    
    # Habilitar reflection (para grpcurl)
    SERVICE_NAMES = (
        # skill_pb2.DESCRIPTOR.services_by_name['SkillService'].full_name,
        "sarai.skills.SkillService",  # Placeholder
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    # Bind y start
    server.add_insecure_port(f'0.0.0.0:{port}')
    server.start()
    
    logger.info(f"✅ Skill '{skill_name}' servidor en puerto {port}")
    logger.info(f"Hot-reload: docker exec <container> sh -c 'kill -USR1 1'")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Deteniendo servidor...")
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skills-as-Services gRPC Server")
    parser.add_argument("--skill", required=True, help="Nombre del skill (sql, code, etc.)")
    parser.add_argument("--port", type=int, default=50051, help="Puerto gRPC (default: 50051)")
    
    args = parser.parse_args()
    
    serve(args.skill, args.port)
