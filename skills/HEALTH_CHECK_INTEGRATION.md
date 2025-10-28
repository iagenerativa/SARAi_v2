# Skills Health Check Integration - v2.12 Phoenix

## ✅ Cómo implementar HealthServicer en skills/runtime.py

El health check permite que Docker, Kubernetes y otros orquestadores detecten automáticamente si el skill está operativo.

### 1. Implementar grpc.health.v1.Health

```python
# skills/runtime.py

from grpc_health.v1 import health_pb2, health_pb2_grpc
from skills import HealthCheckRequest, HealthCheckResponse

class SkillHealthServicer(health_pb2_grpc.HealthServicer):
    """
    Implementa el protocolo estándar de health checking de gRPC
    
    Compatible con:
    - Docker HEALTHCHECK
    - Kubernetes livenessProbe/readinessProbe
    - grpc_health_probe CLI
    """
    
    def __init__(self, skill_service):
        self.skill_service = skill_service
    
    def Check(self, request, context):
        """
        Health check síncrono
        
        Returns:
            SERVING si el modelo está cargado
            NOT_SERVING si está degradado
        """
        # Verificar que el modelo esté cargado
        if self.skill_service.model is None:
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.NOT_SERVING
            )
        
        # Verificar RAM disponible (opcional)
        import psutil
        ram_available_gb = psutil.virtual_memory().available / (1024**3)
        
        if ram_available_gb < 0.1:  # <100MB disponible
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.NOT_SERVING
            )
        
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )
    
    def Watch(self, request, context):
        """
        Health check streaming (opcional)
        
        Envía updates cada vez que el estado cambia
        """
        # Implementación futura (no crítico)
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Watch not implemented")
        raise NotImplementedError()
```

### 2. Registrar en el servidor gRPC

```python
# skills/runtime.py - función serve()

def serve(skill_name: str, port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    
    # Instanciar skill service
    skill_service = SkillService(skill_name)
    
    # Registrar SkillService (nuestro servicio custom)
    add_SkillServiceServicer_to_server(skill_service, server)
    
    # Registrar Health service (estándar gRPC)
    health_servicer = SkillHealthServicer(skill_service)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    
    # ... resto del código (reflection, etc.)
    
    server.add_insecure_port(f'0.0.0.0:{port}')
    server.start()
    server.wait_for_termination()
```

### 3. Docker HEALTHCHECK (en Dockerfile)

```dockerfile
# Opción 1: grpc_health_probe (CLI oficial de gRPC)
HEALTHCHECK --interval=15s --timeout=2s --start-period=5s --retries=3 \
  CMD grpc_health_probe -addr=localhost:50051 || exit 1

# Opción 2: Python inline (si grpc_health_probe no está disponible)
HEALTHCHECK --interval=15s --timeout=2s --start-period=5s --retries=3 \
  CMD python -c "from grpc_health.v1 import health_pb2, health_pb2_grpc; \
                 import grpc; \
                 channel = grpc.insecure_channel('localhost:50051'); \
                 stub = health_pb2_grpc.HealthStub(channel); \
                 response = stub.Check(health_pb2.HealthCheckRequest()); \
                 exit(0 if response.status == health_pb2.HealthCheckResponse.SERVING else 1)"
```

### 4. Kubernetes Probes (ejemplo)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: saraiskill-sql
spec:
  containers:
  - name: skill-sql
    image: saraiskill.sql:v2.12
    ports:
    - containerPort: 50051
    
    # Liveness: reinicia si falla
    livenessProbe:
      exec:
        command:
        - grpc_health_probe
        - -addr=localhost:50051
      initialDelaySeconds: 10
      periodSeconds: 15
      timeoutSeconds: 2
      failureThreshold: 3
    
    # Readiness: no envía tráfico si falla
    readinessProbe:
      exec:
        command:
        - grpc_health_probe
        - -addr=localhost:50051
      initialDelaySeconds: 5
      periodSeconds: 10
      timeoutSeconds: 2
      failureThreshold: 2
```

### 5. Testing local

```bash
# Opción 1: grpc_health_probe CLI
grpc_health_probe -addr=localhost:50051

# Opción 2: grpcurl
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check

# Opción 3: Python client
python -c "
from grpc_health.v1 import health_pb2, health_pb2_grpc
import grpc

channel = grpc.insecure_channel('localhost:50051')
stub = health_pb2_grpc.HealthStub(channel)
response = stub.Check(health_pb2.HealthCheckRequest())
print(f'Status: {response.status}')
"
```

## 📦 Dependencias requeridas

Añadir a `skills/requirements.txt`:

```txt
grpcio-health-checking==1.60.0
```

## ✅ Beneficios

1. **Auto-recovery**: Docker reinicia automáticamente contenedores unhealthy
2. **Load balancing**: K8s no envía tráfico a pods NOT_SERVING
3. **Observabilidad**: Métricas de health en Prometheus/Grafana
4. **Debugging**: `docker ps` muestra estado "healthy" o "unhealthy"
5. **Compliance**: Estándar gRPC oficial, compatible con todo el ecosistema

## 🎯 Próximos pasos

1. Implementar `SkillHealthServicer` en `skills/runtime.py`
2. Añadir `grpcio-health-checking` a `skills/requirements.txt`
3. Instalar `grpc_health_probe` en la imagen base:
   ```dockerfile
   RUN GRPC_HEALTH_PROBE_VERSION=v0.4.19 && \
       wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
       chmod +x /bin/grpc_health_probe
   ```
4. Actualizar Dockerfile con HEALTHCHECK optimizado

## 🔒 Nota de seguridad

El health check NO debe exponer información sensible (prompts, modelos internos, etc.). Solo debe indicar:
- ✅ SERVING: operativo
- ❌ NOT_SERVING: degradado

Métricas detalladas deben ir en `GetMetrics()`, no en `Check()`.
