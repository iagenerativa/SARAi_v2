"""
skills - Skills-as-Services v2.12 Phoenix

Stubs gRPC generados automáticamente desde skills.proto
"""

# Importar mensajes (request/response)
from .skills_pb2 import (
    InferRequest,
    InferResponse,
    HealthCheckRequest,
    HealthCheckResponse,
    MetricsRequest,
    MetricsResponse,
)

# Importar servicios (stub client + servicer server)
from .skills_pb2_grpc import (
    SkillServiceStub,
    SkillServiceServicer,
    add_SkillServiceServicer_to_server,
)

__all__ = [
    # Mensajes
    "InferRequest",
    "InferResponse",
    "HealthCheckRequest",
    "HealthCheckResponse",
    "MetricsRequest",
    "MetricsResponse",
    # Servicios
    "SkillServiceStub",
    "SkillServiceServicer",
    "add_SkillServiceServicer_to_server",
]

__version__ = "2.12.0"
