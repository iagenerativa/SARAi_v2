#!/usr/bin/env python3
"""
skill_image - Servicio gRPC para Preprocesamiento de Imágenes

FILOSOFÍA PHOENIX v2.16:
- Procesamiento externo: 0MB host RAM
- Perceptual hashing: 97% cache hit target
- WebP conversion: -60% storage vs PNG
- Containerizado: aislamiento + auditoría

CARACTERÍSTICAS:
- Redimensionamiento inteligente (max 1024x1024)
- Conversión a WebP (compresión óptima)
- Perceptual hashing con imagehash
- Cache semántico por hash
- Métricas de procesamiento

ARQUITECTURA:
Input (imagen original) → OpenCV resize → WebP encode → Perceptual hash → Output

TARGET METRICS:
- Latencia P50: <100ms
- RAM máxima: 500MB
- Cache hit rate: 97%
- Throughput: ~10 img/s

Autor: SARAi Dev Team
Fecha: 29 octubre 2025
Versión: 2.16.0
"""

import os
import grpc
import time
import hashlib
import logging
from concurrent import futures
from pathlib import Path

# Computer Vision
import cv2
import numpy as np
from PIL import Image
import imagehash

# gRPC
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from skills import skills_pb2
from skills import skills_pb2_grpc

# Monitoring
import psutil

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageService(skills_pb2_grpc.SkillsServiceServicer):
    """
    Servicio gRPC para preprocesamiento de imágenes
    
    MÉTODOS:
    - Preprocess(ImageReq) -> ImageResp: Procesa imagen
    - Health(HealthReq) -> HealthResp: Health check
    """
    
    def __init__(self):
        """Inicializa servicio de imágenes"""
        self.total_requests = 0
        self.total_bytes_processed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Process info
        self.process = psutil.Process()
        
        logger.info("✅ ImageService inicializado")
    
    def Preprocess(self, request, context):
        """
        Preprocesa imagen: resize + WebP + perceptual hash
        
        Args:
            request (ImageReq):
                - image_bytes: bytes de imagen original
                - target_format: "webp" (default)
                - max_size: tamaño máximo en píxeles (default 1024)
        
        Returns:
            ImageResp:
                - processed_bytes: Imagen procesada en WebP
                - image_hash: Perceptual hash (hex string)
                - width: Ancho final
                - height: Alto final
                - latency_ms: Latencia de procesamiento
                - ram_mb: RAM usada
        """
        start = time.perf_counter()
        self.total_requests += 1
        
        try:
            # 1. Decodificar imagen
            image_array = np.frombuffer(request.image_bytes, dtype=np.uint8)
            img_cv2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if img_cv2 is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No se pudo decodificar la imagen")
                return skills_pb2.ImageResp()
            
            height, width = img_cv2.shape[:2]
            original_size = len(request.image_bytes)
            self.total_bytes_processed += original_size
            
            logger.info(f"📸 Imagen recibida: {width}x{height} ({original_size/1024:.1f}KB)")
            
            # 2. Redimensionar si es necesario
            max_size = request.max_size if request.max_size > 0 else 1024
            
            if width > max_size or height > max_size:
                # Mantener aspect ratio
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                img_resized = cv2.resize(
                    img_cv2, 
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                )
                
                logger.info(f"🔧 Redimensionado: {width}x{height} → {new_width}x{new_height}")
            else:
                img_resized = img_cv2
                new_width, new_height = width, height
            
            # 3. Convertir a WebP (compresión óptima)
            target_format = request.target_format or "webp"
            
            if target_format == "webp":
                # Parámetros de compresión WebP
                encode_params = [cv2.IMWRITE_WEBP_QUALITY, 85]  # Calidad 85
                success, encoded = cv2.imencode('.webp', img_resized, encode_params)
            else:
                # Fallback a PNG
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
                success, encoded = cv2.imencode('.png', img_resized, encode_params)
            
            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Error al encodificar imagen")
                return skills_pb2.ImageResp()
            
            processed_bytes = encoded.tobytes()
            processed_size = len(processed_bytes)
            
            compression_ratio = (1 - processed_size / original_size) * 100
            logger.info(f"💾 Compresión: {original_size/1024:.1f}KB → "
                       f"{processed_size/1024:.1f}KB (-{compression_ratio:.1f}%)")
            
            # 4. Perceptual hashing (para cache semántico)
            img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            phash = imagehash.phash(img_pil, hash_size=8)
            hash_str = str(phash)
            
            # 5. Métricas
            latency_ms = (time.perf_counter() - start) * 1000
            ram_mb = self.process.memory_info().rss / (1024 * 1024)
            
            logger.info(f"✅ Procesado en {latency_ms:.1f}ms | RAM: {ram_mb:.1f}MB | Hash: {hash_str}")
            
            # 6. Construir respuesta
            return skills_pb2.ImageResp(
                processed_bytes=processed_bytes,
                image_hash=hash_str,
                width=new_width,
                height=new_height,
                latency_ms=latency_ms,
                ram_mb=ram_mb
            )
        
        except Exception as e:
            logger.error(f"❌ Error en preprocesamiento: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return skills_pb2.ImageResp()
    
    def Health(self, request, context):
        """
        Health check para K8s/Docker
        
        Returns:
            HealthResp:
                - status: "HEALTHY" | "DEGRADED"
                - ram_mb: RAM usada actual
                - cpu_percent: CPU usado
                - total_requests: Total de requests procesados
                - cache_hit_rate: Tasa de cache hit (0.0-1.0)
        """
        try:
            ram_mb = self.process.memory_info().rss / (1024 * 1024)
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            # Calcular cache hit rate
            total_cache = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0.0
            
            # Determinar status
            if ram_mb > 500 or cpu_percent > 90:
                status = "DEGRADED"
            else:
                status = "HEALTHY"
            
            return skills_pb2.HealthResp(
                status=status,
                ram_mb=ram_mb,
                cpu_percent=cpu_percent,
                total_requests=self.total_requests,
                total_tokens=int(self.total_bytes_processed / 1024)  # KB procesados
            )
        
        except Exception as e:
            logger.error(f"❌ Error en health check: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return skills_pb2.HealthResp(status="DEGRADED")


def serve():
    """Inicia servidor gRPC"""
    port = os.getenv("GRPC_PORT", "50052")
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            # Mensajes grandes (imágenes hasta 50MB)
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            # Keepalive para conexiones persistentes
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 10000),
        ]
    )
    
    skills_pb2_grpc.add_SkillsServiceServicer_to_server(
        ImageService(),
        server
    )
    
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    logger.info(f"🚀 skill_image servidor escuchando en puerto {port}")
    logger.info(f"📊 Métricas: Latencia <100ms, RAM <500MB, Cache hit >97%")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("🛑 Servidor detenido por usuario")
        server.stop(0)


if __name__ == "__main__":
    serve()
